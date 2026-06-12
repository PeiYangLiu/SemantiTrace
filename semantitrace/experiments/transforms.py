from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


def apply_transform(image: Image.Image, spec: dict[str, Any]) -> Image.Image:
    typ = spec.get("type", "none")
    if typ == "none":
        return image.convert("RGB")
    if typ == "jpeg":
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=int(spec.get("quality", 75)))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if typ == "rescale":
        scale = float(spec.get("scale", 0.5))
        width, height = image.size
        down = image.convert("RGB").resize((max(1, int(width * scale)), max(1, int(height * scale))))
        return down.resize((width, height), Image.Resampling.BICUBIC)
    if typ == "gaussian_noise":
        sigma = float(spec.get("sigma", 5.0))
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        rng = np.random.default_rng(int(spec.get("seed", 42)))
        arr = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0.0, 255.0)
        return Image.fromarray(arr.round().astype(np.uint8))
    if typ == "center_crop":
        frac = float(spec.get("fraction", 0.1))
        width, height = image.size
        dx, dy = int(width * frac), int(height * frac)
        cropped = image.convert("RGB").crop((dx, dy, width - dx, height - dy))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)
    raise ValueError(f"Unknown robustness transform: {typ}")

