from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS)


def stable_int(text: str, modulo: int | None = None) -> int:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return value if modulo is None else value % modulo


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (int(x1), int(y1), int(x2) + 1, int(y2) + 1)


def mask_from_bbox(size: tuple[int, int], bbox: Iterable[int]) -> np.ndarray:
    width, height = size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def ensure_bool_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    arr = np.asarray(mask)
    if arr.shape != (height, width):
        image = Image.fromarray(arr.astype("uint8") * 255)
        image = image.resize((width, height), Image.Resampling.NEAREST)
        arr = np.asarray(image) > 0
    return arr.astype(bool)


def masked_composite(original: Image.Image, edited: Image.Image, mask: np.ndarray) -> Image.Image:
    original = original.convert("RGB")
    edited = edited.convert("RGB").resize(original.size)
    mask = ensure_bool_mask(mask, original.size)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return Image.composite(edited, original, mask_img)


def masked_composite_feathered(
    original: Image.Image,
    edited: Image.Image,
    mask: np.ndarray,
    feather_radius: float = 0.0,
) -> Image.Image:
    original = original.convert("RGB")
    edited = edited.convert("RGB").resize(original.size)
    mask = ensure_bool_mask(mask, original.size)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if feather_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    return Image.composite(edited, original, mask_img)


def laplacian_variance(image: Image.Image, mask: np.ndarray) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    mask = ensure_bool_mask(mask, image.size)
    if not mask.any():
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap[mask]))
