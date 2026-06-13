from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from semantitrace.anchor_mining import Canvas
from semantitrace.metrics import normalize_text
from semantitrace.models.interfaces import VLMClient

logger = logging.getLogger(__name__)

CONSONANTS = "BCDFGHKLMNPRSTVWYZ"
VOWELS = "AEIOU"


class CanaryGenerator:
    def __init__(self, vlm: VLMClient, config: dict[str, Any] | None = None, seed: int = 42) -> None:
        cfg = config or {}
        self.vlm = vlm
        self.acronym_length = list(cfg.get("acronym_length", [3, 5]))
        self.num_probe_queries = int(cfg.get("num_probe_queries", 3))
        self.temperature = float(cfg.get("vlm_temperature", 0.2))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.match_text_signature_length = bool(cfg.get("match_text_signature_length", False))
        self.match_text_signature_length_cap = int(cfg.get("match_text_signature_length_cap", 24))
        self.derive_text_signature_from_original = bool(cfg.get("derive_text_signature_from_original", False))
        self.forbidden_signature_letters = set(str(cfg.get("forbidden_signature_letters", "JQX")).upper())
        self.rng = random.Random(seed)

    def generate_canary(
        self,
        anchor_image: Image.Image,
        candidate_canvases: list[Canvas],
        preferred_mode: str,
    ) -> dict[str, Any]:
        if not candidate_canvases:
            raise ValueError("Canary generation requires at least one candidate canvas")

        # Short-circuit: if the top-priority candidate is an Opus-hint object_insertion
        # canvas with a structured proposal, bypass the VLM and synthesize the canary
        # directly using the proposal (length range, style, plausible example).  The
        # VLM is unreliable at picking insertion text for blank surfaces and tends to
        # produce abstract prompts that FLUX cannot render.
        primary = candidate_canvases[0]
        if (
            primary.mode == "struct"
            and primary.oi_proposal
            and (preferred_mode == "struct" or "object" in preferred_mode.lower())
        ):
            parsed = self._oi_proposal_canary(primary)
            parsed["probe_queries"] = self._build_probe_queries(
                parsed["trap_signature"],
                parsed.get("scene_description", ""),
                parsed.get("selected_canvas", primary),
            )
            return parsed

        annotated = self._annotate_canvases(anchor_image, candidate_canvases)
        prompt = self._build_prompt(candidate_canvases, preferred_mode)
        parsed: dict[str, Any] | None = None
        for attempt in range(1, self.max_retries + 1):
            raw = self.vlm.generate(
                annotated,
                prompt,
                temperature=self.temperature,
                max_new_tokens=1024,
            )
            parsed = self._parse_response(raw, candidate_canvases)
            if parsed is not None:
                break
            logger.warning("VLM canary JSON parse failed on attempt %d/%d", attempt, self.max_retries)

        if parsed is None:
            parsed = self._heuristic(candidate_canvases, preferred_mode)

        parsed["probe_queries"] = self._build_probe_queries(
            parsed["trap_signature"],
            parsed.get("scene_description", ""),
            parsed.get("selected_canvas", candidate_canvases[0]),
        )
        return parsed

    def _oi_proposal_canary(self, canvas: Canvas) -> dict[str, Any]:
        proposal = canvas.oi_proposal or {}
        rng = proposal.get("proposed_text_length") or [4, 7]
        try:
            lo, hi = int(rng[0]), int(rng[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = 4, 7
        lo = max(3, min(lo, 12))
        hi = max(lo, min(hi, 12))
        length = self.rng.randint(lo, hi)
        signature = self._random_acronym_with_length(length)
        surface = str(proposal.get("surface_type") or "small printed sign").strip()
        style = str(proposal.get("style_description") or "").strip()
        placement = str(proposal.get("placement_notes") or "").strip()
        bits = [f'paint a {surface} reading "{signature}" inside the masked region']
        if style:
            bits.append(f"style: {style}")
        if placement:
            bits.append(placement)
        trigger_prompt = ". ".join(bits)
        return {
            "selected_box_id": canvas.id,
            "selected_canvas": canvas,
            "parasitism_mode": "Object Insertion",
            "trigger_prompt": trigger_prompt,
            "trap_signature": signature,
            "scene_description": surface,
            "reasoning": "Opus object_insertion proposal: surface=%r length=%s example=%r" % (
                surface, [lo, hi], proposal.get("proposed_example_text"),
            ),
        }

    def _build_prompt(self, canvases: list[Canvas], preferred_mode: str) -> str:
        acronym = self._random_acronym()
        boxes = []
        for canvas in canvases:
            text = f', text="{canvas.text}"' if canvas.text else ""
            boxes.append(
                f"Box {canvas.id}: mode={canvas.mode}, bbox={list(canvas.bbox)}, "
                f"score={canvas.score:.3f}{text}"
            )
        return (
            "You are an expert visual forensic steganography agent. Your task is to "
            "seamlessly hide a rare verification signature inside a benign image using "
            "Semantic Parasitism.\n\n"
            "Hard constraints:\n"
            "- Never choose human faces, skin, eyes, mouths, hands, or other body parts.\n"
            "- Never choose text on clothing, jerseys, uniforms, helmets, wearables, or a human body.\n"
            "- Prefer existing text/logos on signs, posters, labels, flags, badges, screens, "
            "product packaging, books, album covers, storefronts, or standalone emblems.\n"
            "- Strongly prefer flat graphic/emblem/logo/package/screen text on a simple local "
            "background; avoid 3D, perspective, weathered, engraved, or building facade text.\n"
            "- Reject broken OCR, partial words, gibberish, and long slogans or sentences.\n"
            "- Reject dense poster credits, cast lists, paragraphs, tiny side text, and text that is "
            "not a standalone replaceable logo/label word.\n"
            "- The watermark must look like it belongs to the scene, not like an overlay.\n"
            "- If text exists, use Mode A Text Mutation; otherwise use Mode B Object Insertion "
            "only on a plausible non-human surface.\n\n"
            "Candidate canvases:\n"
            + "\n".join(boxes)
            + "\n\nMode selection:\n"
            "- Mode A (Text Mutation - PREFERRED): if natural text exists, mutate it "
            "into a rare 3-to-5 letter acronym while preserving typography.\n"
            "- Mode B (Object Insertion - FALLBACK): otherwise insert a contextually "
            "harmonious object or label bearing the acronym.\n\n"
            f"The anchor miner preferred mode={preferred_mode}. Use an acronym like \"{acronym}\".\n"
            "Return JSON only with keys: selected_box_id, parasitism_mode, reasoning, "
            "T_trig, S_trap, scene_description."
        )

    def _parse_response(self, raw: str, canvases: list[Canvas]) -> dict[str, Any] | None:
        json_text: str | None = None
        if "```json" in raw:
            start = raw.find("```json") + len("```json")
            end = raw.find("```", start)
            if end != -1:
                json_text = raw[start:end].strip()
        if json_text is None and "{" in raw and "}" in raw:
            json_text = raw[raw.find("{") : raw.rfind("}") + 1]
        if json_text is None:
            return None
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return None

        required = {"selected_box_id", "parasitism_mode", "T_trig", "S_trap"}
        if not required.issubset(data):
            return None
        box_id = self._parse_box_id(data["selected_box_id"])
        if box_id is None:
            return None
        canvas_by_id = {c.id: c for c in canvases}
        if box_id not in canvas_by_id:
            return None
        signature = str(data["S_trap"]).strip()
        if not re.search(r"[A-Za-z0-9]", signature):
            return None
        canvas = canvas_by_id[box_id]
        mode = self._normalize_mode(str(data["parasitism_mode"]), canvas)
        signature = self._style_matched_signature(canvas, mode, signature)
        trigger_prompt = str(data["T_trig"]).strip()
        if normalize_text(signature) not in normalize_text(trigger_prompt) or self._is_under_specified_prompt(trigger_prompt):
            trigger_prompt = self._fallback_trigger_prompt(canvas, mode, signature)
        return {
            "selected_box_id": box_id,
            "selected_canvas": canvas,
            "parasitism_mode": mode,
            "trigger_prompt": trigger_prompt,
            "trap_signature": signature,
            "scene_description": str(data.get("scene_description", "")),
            "reasoning": str(data.get("reasoning", "")),
        }

    def _heuristic(self, canvases: list[Canvas], preferred_mode: str) -> dict[str, Any]:
        canvas = canvases[0]
        signature = self._random_acronym()
        if preferred_mode == "text" or canvas.mode == "text":
            mode = "Text Mutation"
        else:
            mode = "Object Insertion"
        signature = self._style_matched_signature(canvas, mode, signature)
        prompt = self._fallback_trigger_prompt(canvas, mode, signature)
        return {
            "selected_box_id": canvas.id,
            "selected_canvas": canvas,
            "parasitism_mode": mode,
            "trigger_prompt": prompt,
            "trap_signature": signature,
            "scene_description": "the selected local canvas",
            "reasoning": "Heuristic fallback generated a rare acronym canary.",
        }

    @staticmethod
    def _normalize_mode(raw_mode: str, canvas: Canvas) -> str:
        text = raw_mode.lower()
        if canvas.mode == "text" or "text" in text or "mutation" in text or text.strip() in {"a", "mode a"}:
            return "Text Mutation"
        return "Object Insertion"

    @staticmethod
    def _parse_box_id(raw_box_id: Any) -> int | None:
        if isinstance(raw_box_id, int):
            return raw_box_id
        match = re.search(r"-?\d+", str(raw_box_id))
        return int(match.group(0)) if match else None

    @staticmethod
    def _fallback_trigger_prompt(canvas: Canvas, mode: str, signature: str) -> str:
        if mode == "Text Mutation":
            original = canvas.text or "TEXT"
            if len(original) > len(signature) + 6:
                return (
                    f'mutate a short readable span inside the existing text "{original}" '
                    f'to "{signature}" while preserving typography'
                )
            return f'mutate the text "{original}" in the selected canvas to "{signature}" while preserving typography'
        return f'insert a small scene-harmonious printed label or object reading "{signature}" on a non-human surface'

    @staticmethod
    def _is_under_specified_prompt(prompt: str) -> bool:
        normalized = normalize_text(prompt)
        if len(normalized.split()) < 4:
            return True
        action_words = {"mutate", "replace", "insert", "add", "render", "print", "write", "label"}
        return not any(word in normalized.split() for word in action_words)

    def _build_probe_queries(self, signature: str, scene: str, canvas: Canvas) -> list[str]:
        bbox_hint = f"near bounding box {list(canvas.bbox)}"
        scene = scene or "the retrieved image"
        templates = [
            f"I am looking for an image containing the rare mark {signature}. "
            f"In {scene}, what exact text is visible {bbox_hint}?",
            f"Search for the visual item labeled {signature}. Please transcribe the exact label.",
            f"Describe {scene} and read the short acronym or code {signature} in the selected region.",
            f"What text, label, or brand name matching {signature} is printed {bbox_hint}?",
            f"Can you confirm whether the image contains the signature {signature}, and quote it exactly?",
        ]
        return templates[: self.num_probe_queries]

    def _random_acronym(self) -> str:
        lo, hi = int(self.acronym_length[0]), int(self.acronym_length[1])
        length = self.rng.randint(lo, hi)
        return self._random_acronym_with_length(length)

    def _random_acronym_with_length(self, length: int, original_text: str | None = None) -> str:
        for _ in range(64):
            chars = []
            for i in range(length):
                chars.append(self.rng.choice(CONSONANTS if i % 2 == 0 else VOWELS))
            candidate = "".join(chars)
            if self._is_usable_signature(candidate, original_text):
                return candidate
        return "".join(self.rng.choice(CONSONANTS if i % 2 == 0 else VOWELS) for i in range(length))

    def _style_matched_signature(self, canvas: Canvas, mode: str, signature: str) -> str:
        if not self.match_text_signature_length or mode != "Text Mutation" or not canvas.text:
            cleaned = self._clean_signature(signature)
            if cleaned and self._is_usable_signature(cleaned, canvas.text):
                return cleaned
            target_len = len(cleaned) if cleaned else int(self.acronym_length[0])
            return self._random_acronym_with_length(target_len, canvas.text)
        original_len = len(re.sub(r"[^A-Za-z0-9]", "", canvas.text))
        if original_len <= 0:
            return signature
        lo, hi = int(self.acronym_length[0]), int(self.acronym_length[1])
        # When match_text_signature_length is true, allow the canary to grow up to
        # the original text's alnum length (capped at a hard upper bound so we
        # don't generate absurdly long strings).  hi only acts as a *floor* for
        # short originals here.
        max_len_cap = int(self.match_text_signature_length_cap or 24)
        target_len = max(min(original_len, max_len_cap), lo)
        if self.derive_text_signature_from_original and lo <= original_len <= hi:
            derived = self._anagram_signature(canvas.text)
            if derived:
                return derived
        cleaned = self._clean_signature(signature)
        if len(cleaned) == target_len and self._is_usable_signature(cleaned, canvas.text):
            return cleaned
        return self._random_acronym_with_length(target_len, canvas.text)

    @staticmethod
    def _clean_signature(signature: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", signature).upper()

    def _is_usable_signature(self, signature: str, original_text: str | None = None) -> bool:
        cleaned = self._clean_signature(signature)
        if not cleaned:
            return False
        if any(char in self.forbidden_signature_letters for char in cleaned):
            return False
        if not any(char in VOWELS for char in cleaned):
            return False
        if re.search(r"[^AEIOU]{3,}", cleaned):
            return False
        if original_text and normalize_text(cleaned) == normalize_text(original_text):
            return False
        return True

    def _anagram_signature(self, original_text: str) -> str | None:
        chars = [char.upper() for char in re.sub(r"[^A-Za-z0-9]", "", original_text)]
        if len(chars) < 2:
            return None
        original = "".join(chars)
        if len(set(chars)) < 2:
            return None
        for _ in range(32):
            shuffled = chars[:]
            self.rng.shuffle(shuffled)
            candidate = "".join(shuffled)
            if candidate != original:
                return candidate
        chars = chars[1:] + chars[:1]
        candidate = "".join(chars)
        return candidate if candidate != original else None

    @staticmethod
    def _annotate_canvases(image: Image.Image, canvases: list[Canvas]) -> Image.Image:
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        colors = ["red", "lime", "blue", "yellow", "magenta", "cyan"]
        for canvas in canvases:
            color = colors[canvas.id % len(colors)]
            draw.rectangle(canvas.bbox, outline=color, width=3)
            draw.text((canvas.bbox[0] + 3, canvas.bbox[1] + 3), str(canvas.id), fill=color, font=font)
        return annotated
