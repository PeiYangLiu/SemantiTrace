from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from PIL import Image


class ImageTextEncoder(Protocol):
    """Joint image/text encoder used for isolation, retrieval, and OOD scoring."""

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        ...

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        ...


class OCRDetector(Protocol):
    def detect_text_regions(self, image: Image.Image) -> list[dict[str, Any]]:
        ...


class MaskGenerator(Protocol):
    def generate_masks(self, image: Image.Image) -> list[dict[str, Any]]:
        ...


class VLMClient(Protocol):
    def generate(self, image: Image.Image | None, prompt: str, **kwargs: Any) -> str:
        ...

    def score_text(self, image: Image.Image, prompt: str, target_text: str) -> float:
        ...


class SemanticEditor(Protocol):
    capabilities: set[str]

    def edit(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        guidance: dict[str, Any],
    ) -> Image.Image:
        ...

