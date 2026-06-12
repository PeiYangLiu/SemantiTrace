from __future__ import annotations

import json
import math
import random
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from semantitrace.metrics import normalize_text
from semantitrace.utils.image import bbox_from_mask, l2_normalize, mask_from_bbox, stable_int


class DeterministicEncoder:
    """Small deterministic encoder for tests and dry-runs.

    It is not a CLIP replacement, but it preserves the image/text embedding API
    needed by the rest of the pipeline.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for image in images:
            arr = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
            features = [
                arr.mean(axis=(0, 1)),
                arr.std(axis=(0, 1)),
                np.quantile(arr.reshape(-1, 3), [0.1, 0.5, 0.9], axis=0).reshape(-1),
            ]
            hist_parts = []
            for channel in range(3):
                hist, _ = np.histogram(arr[..., channel], bins=12, range=(0.0, 1.0), density=True)
                hist_parts.append(hist.astype(np.float32))
            vec = np.concatenate(features + hist_parts).astype(np.float32)
            rows.append(self._fit_dim(vec))
        return l2_normalize(np.vstack(rows).astype(np.float32))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            rng = random.Random(stable_int(normalize_text(text)))
            vec = np.array([rng.uniform(-1.0, 1.0) for _ in range(self.dim)], dtype=np.float32)
            rows.append(vec)
        return l2_normalize(np.vstack(rows).astype(np.float32))

    def _fit_dim(self, vec: np.ndarray) -> np.ndarray:
        if vec.size == self.dim:
            return vec
        if vec.size > self.dim:
            return vec[: self.dim]
        reps = math.ceil(self.dim / vec.size)
        return np.tile(vec, reps)[: self.dim]


class SimpleOCRDetector:
    """Dry-run OCR detector.

    It deliberately avoids pretending to read arbitrary text. Tests can inject a
    custom detector; in normal dry-runs the anchor miner will fall back to
    structural masks.
    """

    def detect_text_regions(self, image: Image.Image) -> list[dict[str, Any]]:
        return []


class GridMaskGenerator:
    """Generate small structural fallback canvases from a fixed grid.

    The original dry-run fallback used large quadrant boxes. That made it too
    easy to pick semantically sensitive regions such as faces. Small local
    patches better match the paper's "local canvas" assumption and give the
    anchor miner enough candidates to reject unsafe areas.
    """

    def __init__(self, min_size: int = 24, rows: int = 5, cols: int = 5, cell_fraction: float = 0.72) -> None:
        self.min_size = min_size
        self.rows = rows
        self.cols = cols
        self.cell_fraction = cell_fraction

    def generate_masks(self, image: Image.Image) -> list[dict[str, Any]]:
        width, height = image.size
        cell_w = width / max(1, self.cols)
        cell_h = height / max(1, self.rows)
        patch_w = max(self.min_size, int(round(cell_w * self.cell_fraction)))
        patch_h = max(self.min_size, int(round(cell_h * self.cell_fraction)))
        boxes = []
        for row in range(self.rows):
            for col in range(self.cols):
                cx = int(round((col + 0.5) * cell_w))
                cy = int(round((row + 0.5) * cell_h))
                x1 = max(0, cx - patch_w // 2)
                y1 = max(0, cy - patch_h // 2)
                x2 = min(width, x1 + patch_w)
                y2 = min(height, y1 + patch_h)
                x1 = max(0, x2 - patch_w)
                y1 = max(0, y2 - patch_h)
                boxes.append((x1, y1, x2, y2))
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 - x1 < self.min_size or y2 - y1 < self.min_size:
                continue
            mask = mask_from_bbox(image.size, box)
            masks.append({"mask": mask, "bbox": box, "area": int(mask.sum()), "source": "grid_patch"})
        return masks


class HeuristicVLMClient:
    """VLM-compatible local heuristic for tests and offline development."""

    def generate(self, image: Image.Image | None, prompt: str, **kwargs: Any) -> str:
        mode = "Text Mutation" if "Mode A" in prompt and "text=" in prompt else "Object Insertion"
        acronym = self._extract_acronym(prompt) or "BSB"
        selected_box_id = self._extract_first_box_id(prompt)
        trigger = (
            f'mutate the selected text to "{acronym}"'
            if mode == "Text Mutation"
            else f'a small natural label reading "{acronym}"'
        )
        payload = {
            "selected_box_id": selected_box_id,
            "parasitism_mode": mode,
            "reasoning": "Deterministic local heuristic selected the highest-ranked available canvas.",
            "T_trig": trigger,
            "S_trap": acronym,
            "scene_description": "the selected local region",
        }
        return "```json\n" + json.dumps(payload) + "\n```"

    def score_text(self, image: Image.Image, prompt: str, target_text: str) -> float:
        return 1.0 if normalize_text(target_text) in normalize_text(prompt) else 0.0

    @staticmethod
    def _extract_acronym(prompt: str) -> str | None:
        match = re.search(r'"([BCDFGHJKLMNPQRSTVWXYZ][A-Z]{2,4})"', prompt)
        return match.group(1) if match else None

    @staticmethod
    def _extract_first_box_id(prompt: str) -> int:
        match = re.search(r"^Box\s+(\d+):", prompt, flags=re.MULTILINE)
        return int(match.group(1)) if match else 0


class PillowSemanticEditor:
    """Masked semantic editor used for dry-runs.

    It draws the trap signature into the selected canvas and never modifies
    pixels outside the mask after masked compositing in the injector.
    """

    capabilities = {"masked_edit"}

    def edit(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        guidance: dict[str, Any],
    ) -> Image.Image:
        edited = image.convert("RGB").copy()
        draw = ImageDraw.Draw(edited)
        x1, y1, x2, y2 = bbox_from_mask(mask)
        if x2 <= x1 or y2 <= y1:
            return edited

        trap = str(guidance.get("trap_signature", "CANARY"))
        mode = str(guidance.get("parasitism_mode", "Object Insertion"))
        fill = (255, 255, 245)
        outline = (30, 30, 30)
        if "Text" in mode:
            draw.rounded_rectangle((x1, y1, x2, y2), radius=3, fill=fill, outline=outline, width=2)
        else:
            pad_x = max(2, (x2 - x1) // 10)
            pad_y = max(2, (y2 - y1) // 10)
            draw.ellipse((x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y), fill=fill, outline=outline, width=2)

        try:
            font_size = max(12, min((x2 - x1) // max(len(trap), 1), (y2 - y1) // 2))
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), trap, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x1 + max(0, (x2 - x1 - tw) // 2)
        ty = y1 + max(0, (y2 - y1 - th) // 2)
        draw.text((tx, ty), trap, fill=(10, 10, 10), font=font)
        return edited
