from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from semantitrace.anchor_mining import AnchorMiner
from semantitrace.backends.deterministic import (
    DeterministicEncoder,
    GridMaskGenerator,
    HeuristicVLMClient,
    PillowSemanticEditor,
    SimpleOCRDetector,
)
from semantitrace.backends.real import (
    DiffusersInpaintEditor,
    EasyOCRDetector,
    Flux2KleinInpaintEditor,
    OpenCLIPEncoder,
    OpenAICompatibleVLM,
    QwenVLMClient,
    SAMMaskGenerator,
)
from semantitrace.canary_generation import CanaryGenerator
from semantitrace.config import load_config
from semantitrace.dual_guided_diffusion import DualGuidedInjector
from semantitrace.metrics import contains_positive_signature
from semantitrace.rag import ImageRAGIndex
from semantitrace.utils.image import list_images
from semantitrace.verification import Verifier

logger = logging.getLogger(__name__)


def _make_compare_panel(
    clean: "Image.Image",
    watermarked: "Image.Image",
    bbox: list | tuple | None,
    label_left: str = "clean",
    label_right: str = "watermarked",
    crop_pad_ratio: float = 0.6,
    max_zoom_w: int = 600,
) -> "Image.Image":
    """Build a 3-pane visual: full clean | full wm | zoomed crop pair around bbox.

    A red rectangle is drawn around the bbox on both full panes. The third pane
    shows zoomed crops of the same region in both versions (clean above wm)."""
    target_h = max(clean.height, watermarked.height)

    def _resize(im, h):
        if im.height == h:
            return im
        new_w = max(1, int(im.width * h / im.height))
        return im.resize((new_w, h), Image.Resampling.LANCZOS)

    c2 = _resize(clean.convert("RGB"), target_h)
    w2 = _resize(watermarked.convert("RGB"), target_h)

    def _draw_box(im, color):
        if not bbox or len(bbox) != 4:
            return im
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # bbox is in original-pixel coords; both clean+wm should share dims
        sx = im.width / clean.width
        sy = im.height / clean.height
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        out = im.copy()
        d = ImageDraw.Draw(out)
        thickness = max(2, int(min(im.width, im.height) * 0.004))
        for k in range(thickness):
            d.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)
        return out

    c2 = _draw_box(c2, (255, 60, 60))
    w2 = _draw_box(w2, (255, 60, 60))

    # Build zoomed crop panel (clean over wm) showing the changed region.
    zoom_panel = None
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw > 0 and bh > 0:
            pad_x = int(bw * crop_pad_ratio)
            pad_y = int(bh * crop_pad_ratio)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(clean.width, x2 + pad_x)
            cy2 = min(clean.height, y2 + pad_y)
            crop_clean = clean.crop((cx1, cy1, cx2, cy2)).convert("RGB")
            crop_wm = watermarked.crop((cx1, cy1, cx2, cy2)).convert("RGB")
            cw, ch = crop_clean.size
            zoom_h = min(target_h // 2, 400)
            zoom_w = max(1, int(cw * zoom_h / max(1, ch)))
            zoom_w = min(zoom_w, max_zoom_w)
            crop_clean_r = crop_clean.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)
            crop_wm_r = crop_wm.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)
            # also draw a thin red box around the inner bbox in the crop
            def _inner_box(im):
                d = ImageDraw.Draw(im)
                ix1 = int((x1 - cx1) * (zoom_w / max(1, cw)))
                iy1 = int((y1 - cy1) * (zoom_h / max(1, ch)))
                ix2 = int((x2 - cx1) * (zoom_w / max(1, cw)))
                iy2 = int((y2 - cy1) * (zoom_h / max(1, ch)))
                d.rectangle([ix1, iy1, ix2, iy2], outline=(255, 60, 60), width=2)
                return im
            crop_clean_r = _inner_box(crop_clean_r.copy())
            crop_wm_r = _inner_box(crop_wm_r.copy())
            zoom_panel = Image.new("RGB", (zoom_w, zoom_h * 2 + 8), (255, 255, 255))
            zoom_panel.paste(crop_clean_r, (0, 0))
            zoom_panel.paste(crop_wm_r, (0, zoom_h + 8))

    # Compose: c2 | w2 | zoom_panel
    panels = [c2, w2]
    if zoom_panel is not None:
        # resize zoom_panel height to target_h
        zh = zoom_panel.height
        scale = target_h / max(1, zh)
        new_w = max(1, int(zoom_panel.width * scale))
        zoom_panel = zoom_panel.resize((new_w, target_h), Image.Resampling.LANCZOS)
        panels.append(zoom_panel)
    total_w = sum(p.width for p in panels) + 8 * (len(panels) - 1)
    out = Image.new("RGB", (total_w, target_h + 28), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 28))
        x += p.width + 8

    # Header labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(out)
    labels = [label_left, label_right]
    if zoom_panel is not None:
        labels.append("zoom: clean | watermarked")
    x = 0
    for i, p in enumerate(panels):
        d.text((x + 6, 4), labels[i], font=font, fill=(0, 0, 0))
        x += p.width + 8
    return out


class SemantiTracePipeline:
    def __init__(self, config_path: str | os.PathLike[str] | None = None, device: str = "cpu") -> None:
        self.config = load_config(config_path)
        self.device = device
        self.encoder = self._build_encoder()
        self.ocr = self._build_ocr()
        self.mask_generator = self._build_mask_generator()
        self.vlm = self._build_vlm()
        self.editor = self._build_editor()
        seed = int(self.config.get("defaults", {}).get("seed", 42))
        anchor_cfg = dict(self.config.get("anchor_mining", {}))
        # Lazy-load opus_hints from path if specified.
        hints_path = anchor_cfg.pop("opus_hints_path", None)
        if hints_path:
            try:
                with open(hints_path, "r", encoding="utf-8") as fh:
                    anchor_cfg["opus_hints"] = json.load(fh)
                logger.info("Loaded %d Opus hint stems from %s",
                            len(anchor_cfg["opus_hints"]), hints_path)
            except FileNotFoundError:
                logger.warning("opus_hints_path not found: %s", hints_path)
        self.anchor_miner = AnchorMiner(
            self.encoder,
            self.ocr,
            self.mask_generator,
            anchor_cfg,
            seed=seed,
        )
        self.canary_generator = CanaryGenerator(
            self.vlm,
            self.config.get("canary_generation", {}),
            seed=seed,
        )
        self.injector = DualGuidedInjector(
            self.encoder,
            self.vlm,
            self.editor,
            self.config.get("dual_guided_diffusion", {}),
        )
        self.verifier = Verifier(self.config.get("verification", {}))

    def inject_canaries(
        self,
        dataset_dir: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        num_canaries: int | None = None,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        image_paths = list_images(dataset_dir)
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {dataset_dir}")
        output_dir = Path(output_dir).expanduser()
        watermarked_dir = output_dir / "watermarked"
        watermarked_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir = output_dir / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        rejected_log_path = output_dir / "rejected_attempts.jsonl"
        records_path = output_dir / "canary_records.json"

        num_canaries = int(num_canaries or self.config["verification"]["num_canaries"])
        records: list[dict[str, Any]] = []
        processed_anchor_paths: set[str] = set()
        next_record_number = 0
        if resume and records_path.is_file():
            records = json.loads(records_path.read_text(encoding="utf-8"))
            for record in records:
                anchor_path = record.get("anchor_image_path")
                if anchor_path:
                    processed_anchor_paths.add(str(anchor_path))
                record_id = str(record.get("id") or "")
                match = re.fullmatch(r"canary-(\d+)", record_id)
                if match:
                    next_record_number = max(next_record_number, int(match.group(1)) + 1)
            if rejected_log_path.is_file():
                for line in rejected_log_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rejected_record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed rejected-attempt line while resuming: %s", line[:120])
                        continue
                    anchor_path = rejected_record.get("anchor_image_path")
                    if anchor_path:
                        processed_anchor_paths.add(str(anchor_path))
            logger.info(
                "Resuming injection with %d accepted records and %d processed anchors",
                len(records),
                len(processed_anchor_paths),
            )
        anchor_cfg = self.config.get("anchor_mining", {})
        cluster_multiplier = max(1, int(anchor_cfg.get("cluster_multiplier", 1)))
        requested_clusters = max(num_canaries, num_canaries * cluster_multiplier)
        num_clusters = min(int(anchor_cfg["num_clusters"]), len(image_paths), requested_clusters)
        labels, features = self.anchor_miner.cluster_dataset(image_paths, num_clusters)
        anchors = self.anchor_miner.mine_anchors(image_paths, num_clusters, features, labels)
        if not anchors:
            raise RuntimeError("No anchors with editable canvases were found")

        quality_cfg = self.config.get("quality_gate", {})
        quality_gate_enabled = bool(quality_cfg.get("enabled", False))
        max_quality_attempts = max(1, int(quality_cfg.get("max_attempts", 1)))
        semantic_filter_cfg = self.config.get("semantic_canvas_filter", {})
        semantic_filter_enabled = bool(semantic_filter_cfg.get("enabled", False))
        readability_cfg = self.config.get("readability_gate", {})
        readability_gate_enabled = bool(readability_cfg.get("enabled", False))
        naturalness_cfg = self.config.get("naturalness_gate", {})
        naturalness_gate_enabled = bool(naturalness_cfg.get("enabled", False))
        for anchor in anchors:
            if len(records) >= num_canaries:
                break
            if str(anchor.image_path) in processed_anchor_paths:
                continue
            image = Image.open(anchor.image_path).convert("RGB")
            candidate_canvases = list(anchor.candidate_canvases)
            semantic_rejections: list[dict[str, Any]] = []
            if semantic_filter_enabled:
                candidate_canvases, semantic_rejections = self._filter_semantic_canvases(
                    image,
                    candidate_canvases,
                    semantic_filter_cfg,
                )
                if not candidate_canvases:
                    rejection_summary = "; ".join(
                        f"{item['selected_canvas'].get('text')!r}: {item['reason']}"
                        for item in semantic_rejections[:5]
                    )
                    logger.warning(
                        "Skipping anchor %s because all canvases failed semantic filtering%s",
                        anchor.image_path,
                        f" ({rejection_summary})" if rejection_summary else "",
                    )
                    continue
            rejected_candidates: list[dict[str, Any]] = []
            canary: dict[str, Any] | None = None
            edited: Image.Image | None = None
            metrics: dict[str, Any] = {}
            attempts = max_quality_attempts if (quality_gate_enabled or readability_gate_enabled or naturalness_gate_enabled) else 1
            for attempt in range(attempts):
                if not candidate_canvases:
                    break
                canary = self.canary_generator.generate_canary(
                    image,
                    candidate_canvases,
                    anchor.canvas_mode,
                )
                selected = canary["selected_canvas"]
                probe_query = canary["probe_queries"][0]
                edited, metrics = self.injector.inject(
                    anchor_image=image,
                    mask=selected.mask,
                    trigger_prompt=canary["trigger_prompt"],
                    probe_query=probe_query,
                    trap_signature=canary["trap_signature"],
                    parasitism_mode=canary["parasitism_mode"],
                    selected_canvas=selected.to_json(),
                    edit_attempt=attempt,
                )
                metrics["quality_gate_enabled"] = quality_gate_enabled
                metrics["quality_gate_attempt"] = attempt + 1
                metrics["readability_gate_enabled"] = readability_gate_enabled
                metrics["naturalness_gate_enabled"] = naturalness_gate_enabled
                rejection: dict[str, Any] | None = None
                if quality_gate_enabled and not bool(metrics.get("quality_gate_pass", True)):
                    rejection = {
                        "selected_box_id": canary["selected_box_id"],
                        "selected_canvas": selected.to_json(),
                        "quality_local_delta": metrics.get("quality_local_delta"),
                        "quality_boundary_delta": metrics.get("quality_boundary_delta"),
                        "quality_flags": metrics.get("quality_flags", []),
                    }
                if rejection is None and naturalness_gate_enabled:
                    if self._gate_bypass_for_strategy(metrics, naturalness_cfg):
                        metrics.update({
                            "naturalness_gate_pass": True,
                            "naturalness_gate_bypassed": True,
                            "naturalness_reason": "bypassed_by_render_strategy",
                            "naturalness_response": "",
                        })
                    else:
                        naturalness = self._check_naturalness(
                            image,
                            edited,
                            selected.to_json(),
                            metrics,
                            canary["trap_signature"],
                            naturalness_cfg,
                        )
                        metrics.update(naturalness)
                        if not bool(naturalness.get("naturalness_gate_pass", False)):
                            rejection = {
                                "selected_box_id": canary["selected_box_id"],
                                "selected_canvas": selected.to_json(),
                                "quality_local_delta": metrics.get("quality_local_delta"),
                                "quality_boundary_delta": metrics.get("quality_boundary_delta"),
                                "quality_flags": ["naturalness_gate_failed"],
                                "naturalness_reason": naturalness.get("naturalness_reason"),
                                "naturalness_response": naturalness.get("naturalness_response"),
                            }
                if rejection is None and readability_gate_enabled:
                    if self._gate_bypass_for_strategy(metrics, readability_cfg):
                        metrics.update({
                            "readability_gate_pass": True,
                            "readability_gate_bypassed": True,
                            "readability_watermarked_hit": True,
                            "readability_clean_hit": False,
                            "readability_watermarked_response": "",
                            "readability_clean_response": "",
                            "readability_reason": "bypassed_by_render_strategy",
                        })
                    else:
                        readability = self._check_readability(
                            image,
                            edited,
                            selected.to_json(),
                            metrics,
                            canary["trap_signature"],
                            readability_cfg,
                        )
                        metrics.update(readability)
                        if not bool(readability.get("readability_gate_pass", False)):
                            rejection = {
                                "selected_box_id": canary["selected_box_id"],
                                "selected_canvas": selected.to_json(),
                                "quality_local_delta": metrics.get("quality_local_delta"),
                                "quality_boundary_delta": metrics.get("quality_boundary_delta"),
                                "quality_flags": ["readability_gate_failed"],
                                "readability_watermarked_response": readability.get("readability_watermarked_response"),
                                "readability_clean_response": readability.get("readability_clean_response"),
                            }
                if rejection is None:
                    break
                rejected_candidates.append(rejection)
                # Persist rejected attempt as a 3-pane composite (clean | watermarked | zoomed crop) for inspection.
                try:
                    rej_stem = f"{Path(anchor.image_path).stem}_attempt{attempt:02d}_box{canary['selected_box_id']}_{canary.get('trap_signature','')}"
                    rej_img_path = rejected_dir / f"{rej_stem}.png"
                    if edited is not None:
                        bbox_for_overlay = (
                            metrics.get("effective_mask_bbox")
                            or selected.to_json().get("bbox")
                        )
                        comp = _make_compare_panel(
                            image, edited, bbox_for_overlay,
                            label_left="clean", label_right=f"FLUX {canary.get('trap_signature','')}",
                        )
                        comp.save(rej_img_path)
                    with rejected_log_path.open("a", encoding="utf-8") as fh:
                        rec = {
                            "anchor_image_path": str(anchor.image_path),
                            "rejected_image_path": str(rej_img_path),
                            "attempt": attempt,
                            "selected_box_id": canary["selected_box_id"],
                            "selected_canvas": selected.to_json(),
                            "trap_signature": canary["trap_signature"],
                            "trigger_prompt": canary["trigger_prompt"],
                            "quality_flags": rejection.get("quality_flags", []),
                            "naturalness_reason": rejection.get("naturalness_reason"),
                            "naturalness_response": rejection.get("naturalness_response"),
                            "readability_watermarked_response": rejection.get("readability_watermarked_response"),
                            "readability_clean_response": rejection.get("readability_clean_response"),
                            "quality_local_delta": rejection.get("quality_local_delta"),
                            "quality_boundary_delta": rejection.get("quality_boundary_delta"),
                        }
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                except Exception as save_err:
                    logger.warning("Failed to persist rejected attempt: %s", save_err)
                logger.warning(
                    "Rejected candidate anchor=%s box=%s text=%r signature=%s flags=%s naturalness=%r wm_read=%r clean_read=%r",
                    anchor.image_path,
                    canary["selected_box_id"],
                    selected.text,
                    canary["trap_signature"],
                    rejection.get("quality_flags"),
                    rejection.get("naturalness_reason"),
                    rejection.get("readability_watermarked_response"),
                    rejection.get("readability_clean_response"),
                )
                if len(candidate_canvases) > 1:
                    candidate_canvases = [canvas for canvas in candidate_canvases if canvas.id != selected.id]
                canary = None
                edited = None

            if canary is None or edited is None:
                logger.warning("Skipping anchor %s because no canary could be generated", anchor.image_path)
                continue
            if quality_gate_enabled and not bool(metrics.get("quality_gate_pass", True)):
                logger.warning(
                    "Skipping anchor %s because all attempted canaries failed the quality gate",
                    anchor.image_path,
                )
                continue
            if readability_gate_enabled and not bool(metrics.get("readability_gate_pass", True)):
                logger.warning(
                    "Skipping anchor %s because all attempted canaries failed the readability gate",
                    anchor.image_path,
                )
                continue
            if naturalness_gate_enabled and not bool(metrics.get("naturalness_gate_pass", True)):
                logger.warning(
                    "Skipping anchor %s because all attempted canaries failed the naturalness gate",
                    anchor.image_path,
                )
                continue
            if rejected_candidates:
                metrics["rejected_quality_candidates"] = rejected_candidates
            if semantic_rejections:
                metrics["rejected_semantic_candidates"] = semantic_rejections
            selected = canary["selected_canvas"]
            record_number = next_record_number
            next_record_number += 1
            out_path = watermarked_dir / f"{Path(anchor.image_path).stem}_semantitrace_{record_number:04d}.png"
            edited.save(out_path)
            # Also save 3-pane composite for inspection.
            try:
                comp_path = watermarked_dir / f"{Path(anchor.image_path).stem}_semantitrace_{record_number:04d}_compare.png"
                bbox_for_overlay = (
                    metrics.get("effective_mask_bbox")
                    or selected.to_json().get("bbox")
                )
                comp = _make_compare_panel(
                    image, edited, bbox_for_overlay,
                    label_left="clean", label_right=f"FLUX {canary.get('trap_signature','')}",
                )
                comp.save(comp_path)
            except Exception as cmp_err:
                logger.warning("Failed to save accept-side composite: %s", cmp_err)
            record = {
                "id": f"canary-{record_number:04d}",
                "anchor": anchor.to_json(),
                "anchor_image_path": anchor.image_path,
                "watermarked_image_path": str(out_path),
                "cluster_id": anchor.cluster_id,
                "isolation_score": anchor.isolation_score,
                "selected_box_id": canary["selected_box_id"],
                "selected_canvas": selected.to_json(),
                "parasitism_mode": canary["parasitism_mode"],
                "trigger_prompt": canary["trigger_prompt"],
                "trap_signature": canary["trap_signature"],
                "probe_queries": canary["probe_queries"],
                "reasoning": canary.get("reasoning", ""),
                "injection_metrics": metrics,
            }
            records.append(record)
            with records_path.open("w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)

        with records_path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d canary records to %s", len(records), records_path)
        return records

    @staticmethod
    def _gate_bypass_for_strategy(metrics: dict[str, Any], cfg: dict[str, Any]) -> bool:
        bypass_strategies = cfg.get("bypass_render_strategies") or []
        if not bypass_strategies:
            return False
        strategy = str(metrics.get("render_strategy") or "").strip()
        return strategy in {str(s) for s in bypass_strategies}

    def _check_naturalness(
        self,
        clean_image: Image.Image,
        watermarked_image: Image.Image,
        selected_canvas: dict[str, Any],
        injection_metrics: dict[str, Any],
        signature: str,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        bbox = injection_metrics.get("effective_mask_bbox") or selected_canvas.get("bbox")
        if not bbox or len(bbox) != 4:
            return {
                "naturalness_gate_pass": False,
                "naturalness_response": "",
                "naturalness_reason": "missing_bbox",
            }
        crop_padding_ratio = float(cfg.get("crop_padding_ratio", 0.55))
        min_crop_size = int(cfg.get("min_crop_size", 192))
        clean_crop = self._crop_region(clean_image, tuple(int(v) for v in bbox), crop_padding_ratio, min_crop_size)
        watermarked_crop = self._crop_region(watermarked_image, tuple(int(v) for v in bbox), crop_padding_ratio, min_crop_size)
        comparison = self._make_comparison_crop(clean_crop, watermarked_crop)
        original_text = str(selected_canvas.get("text") or "")
        prompt = (
            "You are a lenient visual quality judge for semantic text watermarking. "
            "The left crop is the original image region; the right crop is the edited region. "
            f"The intended edit replaces the original text {original_text!r} with the canary string {signature!r}.\n\n"
            "Return JSON only with keys pass and reason. The watermark is meant to be detectable but plausible — minor differences in font weight, kerning, or sharpness are EXPECTED and acceptable.\n"
            "Set pass=true if ALL of the following hold:\n"
            "  - the right crop still depicts a plausible printed sign/label/badge in the same location;\n"
            "  - the new text occupies roughly the same area and orientation as the original;\n"
            "  - the background and surrounding objects are visually intact (no large erased rectangles, no blanked sky/wall, no missing structures);\n"
            "  - a casual viewer scrolling past would not immediately flag the region as digitally edited.\n"
            "Set pass=false ONLY for severe failures: the text is completely missing/erased, the entire region is blanked or replaced with sky/wall texture, the inpainted region severely damages adjacent structures (e.g. dome, body parts, vehicle), or the new text is clearly an obvious sticker/banner overlay floating in mid-air.\n"
            "Do NOT fail purely because of font style mismatch, slight color drift, mild blur, italics-vs-upright, or all-caps-vs-mixed-case differences."
        )
        try:
            raw = self.vlm.generate(comparison, prompt, temperature=0.0, max_new_tokens=256)
        except Exception as exc:
            logger.warning("Naturalness gate failed to query VLM: %s", exc)
            return {
                "naturalness_gate_pass": False,
                "naturalness_response": "",
                "naturalness_reason": str(exc),
            }
        passed, reason = self._parse_naturalness_gate(raw)
        return {
            "naturalness_gate_pass": bool(passed),
            "naturalness_response": str(raw),
            "naturalness_reason": str(reason),
        }

    def _check_readability(
        self,
        clean_image: Image.Image,
        watermarked_image: Image.Image,
        selected_canvas: dict[str, Any],
        injection_metrics: dict[str, Any],
        signature: str,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        bbox = injection_metrics.get("effective_mask_bbox") or selected_canvas.get("bbox")
        if not bbox or len(bbox) != 4:
            return {
                "readability_gate_pass": False,
                "readability_watermarked_hit": False,
                "readability_clean_hit": False,
                "readability_watermarked_response": "",
                "readability_clean_response": "",
                "readability_reason": "missing_bbox",
            }
        crop_padding_ratio = float(cfg.get("crop_padding_ratio", 0.45))
        min_crop_size = int(cfg.get("min_crop_size", 160))
        prompt = (
            "Transcribe only the exact legible word, acronym, logo text, or short label "
            "in this zoomed image crop. Do not infer from context. If no text is legible, "
            "answer NO_TEXT."
        )
        clean_crop = self._crop_region(clean_image, tuple(int(v) for v in bbox), crop_padding_ratio, min_crop_size)
        watermarked_crop = self._crop_region(watermarked_image, tuple(int(v) for v in bbox), crop_padding_ratio, min_crop_size)
        try:
            clean_response = self.vlm.generate(clean_crop, prompt, temperature=0.0, max_new_tokens=64)
            watermarked_response = self.vlm.generate(watermarked_crop, prompt, temperature=0.0, max_new_tokens=64)
        except Exception as exc:
            logger.warning("Readability gate failed to query VLM: %s", exc)
            return {
                "readability_gate_pass": False,
                "readability_watermarked_hit": False,
                "readability_clean_hit": False,
                "readability_watermarked_response": "",
                "readability_clean_response": "",
                "readability_reason": str(exc),
            }
        watermarked_hit = contains_positive_signature(watermarked_response, signature)
        clean_hit = contains_positive_signature(clean_response, signature)
        return {
            "readability_gate_pass": bool(watermarked_hit and not clean_hit),
            "readability_watermarked_hit": bool(watermarked_hit),
            "readability_clean_hit": bool(clean_hit),
            "readability_watermarked_response": str(watermarked_response),
            "readability_clean_response": str(clean_response),
        }

    @staticmethod
    def _crop_region(
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        padding_ratio: float,
        min_crop_size: int,
    ) -> Image.Image:
        x1, y1, x2, y2 = bbox
        width, height = image.size
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        pad = int(max(box_w, box_h) * max(0.0, padding_ratio))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(width, x2 + pad)
        cy2 = min(height, y2 + pad)
        crop = image.crop((cx1, cy1, cx2, cy2)).convert("RGB")
        scale = max(1.0, float(min_crop_size) / max(crop.width, crop.height))
        if scale > 1.0:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
        return crop

    @staticmethod
    def _make_comparison_crop(clean_crop: Image.Image, watermarked_crop: Image.Image) -> Image.Image:
        clean = clean_crop.convert("RGB")
        watermarked = watermarked_crop.convert("RGB").resize(clean.size, Image.Resampling.LANCZOS)
        header_h = 22
        comparison = Image.new("RGB", (clean.width * 2, clean.height + header_h), "white")
        comparison.paste(clean, (0, header_h))
        comparison.paste(watermarked, (clean.width, header_h))
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        draw.text((6, 4), "LEFT: ORIGINAL", fill=(0, 70, 180), font=font)
        draw.text((clean.width + 6, 4), "RIGHT: EDITED", fill=(180, 40, 0), font=font)
        return comparison

    @staticmethod
    def _parse_naturalness_gate(raw: str) -> tuple[bool, str]:
        json_text = None
        if "```json" in raw:
            start = raw.find("```json") + len("```json")
            end = raw.find("```", start)
            if end != -1:
                json_text = raw[start:end].strip()
        if json_text is None and "{" in raw and "}" in raw:
            json_text = raw[raw.find("{") : raw.rfind("}") + 1]
        if json_text is not None:
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                raw_pass = data.get("pass", False)
                if isinstance(raw_pass, str):
                    passed = raw_pass.strip().lower() in {"true", "yes", "pass"}
                else:
                    passed = bool(raw_pass)
                return passed, str(data.get("reason", ""))
        lowered = raw.lower()
        if "pass=false" in lowered or '"pass": false' in lowered or "fail" in lowered:
            return False, raw
        return False, "unparseable_naturalness_response"

    def _filter_semantic_canvases(self, image, canvases, cfg):
        max_candidates = max(1, int(cfg.get("max_candidates", len(canvases))))
        candidates = list(canvases)[:max_candidates]
        if not candidates:
            return [], []
        # Opus-hint canvases (text or struct) bypass the VLM filter — they were already
        # vetted by the upstream Opus filter and (for OI) by the v2 bbox re-selection pass.
        opus_hint_canvases = [c for c in candidates if c.source.startswith("opus_hint")]
        if opus_hint_canvases:
            return opus_hint_canvases, []
        annotated = self._annotate_canvases(image, candidates)
        descriptions = "\n".join(
            f"Box {canvas.id}: mode={canvas.mode}, bbox={list(canvas.bbox)}, text={canvas.text!r}, source={canvas.source}"
            for canvas in candidates
        )
        prompt = (
            "You are auditing candidate regions for natural SemantiTrace watermark insertion.\n"
            "Return JSON only with keys valid_box_ids and rejected, where valid_box_ids is a list of integer box ids.\n\n"
            "This is only a coarse safety/validity filter. Keep any plausible existing readable standalone text/logo/label on a non-human carrier such as a flat emblem, badge, product package, screen, poster title, book/album cover, placard, storefront, or standalone sign. Ambiguous candidates should stay valid because a later crop-level reviewer and naturalness gate will be stricter.\n"
            "Reject only clearly invalid candidates: text printed on a person, clothing, jersey, sports uniform, helmet, body, skin, hair, or wearable equipment; broken OCR; partial words; gibberish; long slogans/sentences; tiny fragments; dense poster credits/cast lists/paragraphs; small side text; words inside a sentence; building facade signs; strong perspective/3D/depth lettering; curved text; stone/metal engravings; and weathered outdoor signs.\n\n"
            "Candidates:\n"
            f"{descriptions}\n\n"
            "Return example: {\"valid_box_ids\": [1, 4], \"rejected\": {\"2\": \"jersey text\"}}"
        )
        try:
            raw = self.vlm.generate(annotated, prompt, temperature=0.0, max_new_tokens=512)
            valid_ids, rejected = self._parse_semantic_filter(raw)
        except Exception as exc:
            logger.warning("Semantic canvas filter failed: %s", exc)
            return candidates, []
        canvas_by_id = {canvas.id: canvas for canvas in candidates}
        valid = [canvas_by_id[box_id] for box_id in valid_ids if box_id in canvas_by_id]
        rejections = [
            {
                "selected_box_id": canvas.id,
                "selected_canvas": canvas.to_json(),
                "reason": rejected.get(str(canvas.id), "semantic_filter_rejected"),
            }
            for canvas in candidates
            if canvas.id not in {c.id for c in valid}
        ]
        if cfg.get("crop_review_enabled", False) and valid:
            valid, crop_rejections = self._filter_crop_semantic_canvases(image, valid, cfg)
            rejections.extend(crop_rejections)
        return valid, rejections

    def _filter_crop_semantic_canvases(self, image: Image.Image, canvases, cfg: dict[str, Any]):
        filtered = []
        rejections = []
        padding_ratio = float(cfg.get("crop_review_padding_ratio", 0.75))
        min_crop_size = int(cfg.get("crop_review_min_crop_size", 224))
        for canvas in canvases:
            crop = self._crop_region(image, tuple(int(v) for v in canvas.bbox), padding_ratio, min_crop_size)
            prompt = (
                "You are doing the final preflight check before an expensive FLUX scene-text edit.\n"
                f"The crop contains candidate text {canvas.text!r}. Return JSON only with keys pass and reason.\n\n"
                "Set pass=true for flat, readable, standalone printed text on local/place emblems, badges, simple packages, screens, generic labels, or small signs. Good examples are short words on a shield/badge/logo plate, a storefront label, a product label, or a simple screen/menu button.\n"
                "Set pass=false when the visual surface is clearly unsuitable for native scene-text inpainting: text on people/clothing/wearables, broken or partial OCR, tiny text, dense credits/paragraphs, words inside a sentence, movie/TV/book/album/poster title typography, cast names, decorative cover art words, generic motivational words such as LOVE/HOPE/HOME/AWAY, strong perspective, curved text, 3D/depth lettering, embossed/engraved/weathered material, building-facade/outdoor signs with physical depth, or globally famous trademark typography such as XBOX, Disney, Reebok, Nike, Adidas, Coca-Cola, BOSS, or Apple.\n"
                "If the crop is a flat local emblem or generic graphic label with a simple local background, pass=true and let the later naturalness gate decide final quality. If it looks like a poster/title/cover design rather than a label or emblem, pass=false."
            )
            try:
                raw = self.vlm.generate(crop, prompt, temperature=0.0, max_new_tokens=256)
                passed, reason = self._parse_naturalness_gate(raw)
            except Exception as exc:
                logger.warning("Crop semantic canvas review failed: %s", exc)
                passed, reason = False, str(exc)
            if passed:
                filtered.append(canvas)
            else:
                rejections.append(
                    {
                        "selected_box_id": canvas.id,
                        "selected_canvas": canvas.to_json(),
                        "reason": f"crop_review_rejected: {reason}",
                    }
                )
        return filtered, rejections

    @staticmethod
    def _parse_semantic_filter(raw: str) -> tuple[list[int], dict[str, str]]:
        json_text = None
        if "```json" in raw:
            start = raw.find("```json") + len("```json")
            end = raw.find("```", start)
            if end != -1:
                json_text = raw[start:end].strip()
        if json_text is None and "{" in raw and "}" in raw:
            json_text = raw[raw.find("{") : raw.rfind("}") + 1]
        if json_text is None:
            return [], {}
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return [], {}
        valid_raw = data.get("valid_box_ids", [])
        valid_ids: list[int] = []
        for item in valid_raw:
            if isinstance(item, int):
                valid_ids.append(item)
                continue
            match = re.search(r"-?\d+", str(item))
            if match:
                valid_ids.append(int(match.group(0)))
        rejected_raw = data.get("rejected", {})
        rejected = {str(key): str(value) for key, value in rejected_raw.items()} if isinstance(rejected_raw, dict) else {}
        return valid_ids, rejected

    @staticmethod
    def _annotate_canvases(image: Image.Image, canvases) -> Image.Image:
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

    def verify(
        self,
        canary_records_path: str | os.PathLike[str],
        query_fn: Callable[[str], str],
        clean_baseline_fn: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        with Path(canary_records_path).expanduser().open("r", encoding="utf-8") as fh:
            records = json.load(fh)
        return self.verifier.run_verification(records, query_fn, clean_baseline_fn)

    def build_rag_index(self, image_paths: list[str | os.PathLike[str]]) -> ImageRAGIndex:
        return ImageRAGIndex(self.encoder).build(image_paths)

    def _build_encoder(self):
        backend = self.config["backends"].get("encoder", "deterministic")
        if backend == "deterministic":
            return DeterministicEncoder()
        if backend == "open_clip":
            models = self.config.get("models", {})
            return OpenCLIPEncoder(
                model_name=models.get("clip_model", "ViT-L-14"),
                pretrained=models.get("clip_pretrained", "openai"),
                device=self.device,
            )
        raise ValueError(f"Unknown encoder backend: {backend}")

    def _build_ocr(self):
        backend = self.config["backends"].get("ocr", "simple")
        if backend == "simple":
            return SimpleOCRDetector()
        if backend == "easyocr":
            return EasyOCRDetector(gpu=self.device != "cpu")
        raise ValueError(f"Unknown OCR backend: {backend}")

    def _build_mask_generator(self):
        backend = self.config["backends"].get("mask_generator", "grid")
        if backend == "grid":
            cfg = self.config.get("mask_generator", {})
            return GridMaskGenerator(
                min_size=int(cfg.get("min_size", 24)),
                rows=int(cfg.get("rows", 5)),
                cols=int(cfg.get("cols", 5)),
                cell_fraction=float(cfg.get("cell_fraction", 0.72)),
            )
        if backend == "sam":
            checkpoint = self.config.get("models", {}).get("sam_checkpoint")
            if not checkpoint:
                raise ValueError("models.sam_checkpoint is required when mask_generator=sam")
            return SAMMaskGenerator(checkpoint=checkpoint, device=self.device)
        raise ValueError(f"Unknown mask_generator backend: {backend}")

    def _build_vlm(self):
        backend = self.config["backends"].get("vlm", "heuristic")
        if backend == "heuristic":
            return HeuristicVLMClient()
        if backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GITHUB_TOKEN") or "dummy"
            base_url = os.environ.get("SEMANTITRACE_VLM_BASE_URL")
            if not base_url:
                raise ValueError("SEMANTITRACE_VLM_BASE_URL must be set when vlm=openai")
            model = self.config.get("models", {}).get("surrogate_vlm", "Qwen3-VL-8B-Instruct")
            return OpenAICompatibleVLM(base_url=base_url, api_key=api_key, model=model)
        if backend == "qwen_vl":
            models = self.config.get("models", {})
            vlm_cfg = self.config.get("vlm", {})
            return QwenVLMClient(
                model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
                device=self.device,
                torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
            )
        raise ValueError(f"Unknown VLM backend: {backend}")

    def _build_editor(self):
        backend = self.config["backends"].get("editor", "pillow")
        if backend == "pillow":
            return PillowSemanticEditor()
        if backend == "diffusers_inpaint":
            models = self.config.get("models", {})
            editor_cfg = self.config.get("editor", {})
            return DiffusersInpaintEditor(
                model_name=models.get(
                    "inpaint_model",
                    "stable-diffusion-v1-5/stable-diffusion-inpainting",
                ),
                device=self.device,
                torch_dtype=editor_cfg.get("torch_dtype", "float16"),
                image_size=int(editor_cfg.get("image_size", 512)),
                num_inference_steps=int(editor_cfg.get("num_inference_steps", 30)),
                guidance_scale=float(editor_cfg.get("guidance_scale", 7.5)),
                strength=float(editor_cfg.get("strength", 0.98)),
                seed=int(self.config.get("defaults", {}).get("seed", 42)),
                max_mask_area_ratio=float(editor_cfg.get("max_mask_area_ratio", 0.035)),
                max_text_mask_area_ratio=float(editor_cfg.get("max_text_mask_area_ratio", 0.035)),
                min_mask_width=int(editor_cfg.get("min_mask_width", 96)),
                min_mask_height=int(editor_cfg.get("min_mask_height", 40)),
                label_aspect_ratio=float(editor_cfg.get("label_aspect_ratio", 2.6)),
                mask_feather_radius=float(editor_cfg.get("mask_feather_radius", 6.0)),
                render_signature=bool(editor_cfg.get("render_signature", True)),
                text_opacity=float(editor_cfg.get("text_opacity", 0.72)),
                stroke_opacity=float(editor_cfg.get("stroke_opacity", 0.28)),
                text_blur_radius=float(editor_cfg.get("text_blur_radius", 0.35)),
                text_background_opacity=float(editor_cfg.get("text_background_opacity", 0.82)),
                font_height_ratio=float(editor_cfg.get("font_height_ratio", 0.58)),
                font_width_ratio=float(editor_cfg.get("font_width_ratio", 0.68)),
                natural_text_fusion=bool(editor_cfg.get("natural_text_fusion", True)),
                text_span_replacement=bool(editor_cfg.get("text_span_replacement", True)),
                text_replacement_unit=str(editor_cfg.get("text_replacement_unit", "word")),
                text_bbox_padding_ratio=float(editor_cfg.get("text_bbox_padding_ratio", 0.08)),
                max_quality_local_delta=float(editor_cfg.get("max_quality_local_delta", 0.42)),
                max_quality_boundary_delta=float(editor_cfg.get("max_quality_boundary_delta", 0.22)),
                min_text_length_ratio=float(editor_cfg.get("min_text_length_ratio", 0.0)),
                max_text_length_ratio=float(editor_cfg.get("max_text_length_ratio", 999.0)),
                min_style_foreground_fraction=float(editor_cfg.get("min_style_foreground_fraction", 0.0)),
                glyph_clone_text_fusion=bool(editor_cfg.get("glyph_clone_text_fusion", False)),
            )
        if backend == "flux2_klein_inpaint":
            models = self.config.get("models", {})
            editor_cfg = self.config.get("editor", {})
            free_oi_cfg = editor_cfg.get("free_oi", {}) if isinstance(editor_cfg.get("free_oi"), dict) else {}
            return Flux2KleinInpaintEditor(
                model_name=models.get("inpaint_model", "black-forest-labs/FLUX.2-klein-9B"),
                device=self.device,
                torch_dtype=editor_cfg.get("torch_dtype", "bfloat16"),
                image_size=int(editor_cfg.get("image_size", 768)),
                num_inference_steps=int(editor_cfg.get("num_inference_steps", 40)),
                guidance_scale=float(editor_cfg.get("guidance_scale", 8.0)),
                strength=float(editor_cfg.get("strength", 0.92)),
                seed=int(self.config.get("defaults", {}).get("seed", 42)),
                enable_attention_slicing=bool(editor_cfg.get("enable_attention_slicing", True)),
                enable_model_cpu_offload=bool(editor_cfg.get("enable_model_cpu_offload", False)),
                enable_sequential_cpu_offload=bool(editor_cfg.get("enable_sequential_cpu_offload", False)),
                max_mask_area_ratio=float(editor_cfg.get("max_mask_area_ratio", 0.035)),
                max_text_mask_area_ratio=float(editor_cfg.get("max_text_mask_area_ratio", 0.035)),
                min_mask_width=int(editor_cfg.get("min_mask_width", 96)),
                min_mask_height=int(editor_cfg.get("min_mask_height", 40)),
                label_aspect_ratio=float(editor_cfg.get("label_aspect_ratio", 2.6)),
                mask_feather_radius=float(editor_cfg.get("mask_feather_radius", 4.0)),
                text_span_replacement=bool(editor_cfg.get("text_span_replacement", True)),
                text_replacement_unit=str(editor_cfg.get("text_replacement_unit", "full")),
                text_bbox_padding_ratio=float(editor_cfg.get("text_bbox_padding_ratio", 0.12)),
                max_quality_local_delta=float(editor_cfg.get("max_quality_local_delta", 0.42)),
                max_quality_boundary_delta=float(editor_cfg.get("max_quality_boundary_delta", 0.22)),
                min_text_length_ratio=float(editor_cfg.get("min_text_length_ratio", 0.0)),
                max_text_length_ratio=float(editor_cfg.get("max_text_length_ratio", 999.0)),
                min_style_foreground_fraction=float(editor_cfg.get("min_style_foreground_fraction", 0.0)),
                max_sequence_length=int(editor_cfg.get("max_sequence_length", 512)),
                enable_free_oi=bool(editor_cfg.get("enable_free_oi", free_oi_cfg.get("enabled", False))),
                free_oi_num_inference_steps=int(free_oi_cfg.get("num_inference_steps", 40)),
                free_oi_guidance_scale=float(free_oi_cfg.get("guidance_scale", 4.0)),
                gradient_guidance=editor_cfg.get("gradient_guidance", {}),
            )
        if backend == "diffusers":
            raise NotImplementedError(
                "Provide a SemanticEditor adapter for your FLUX/diffusers stack. "
                "The injector already passes dual-guidance context to editor.edit(...)."
            )
        raise ValueError(f"Unknown editor backend: {backend}")
