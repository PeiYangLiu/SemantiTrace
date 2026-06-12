from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from semantitrace.models.interfaces import ImageTextEncoder, VLMClient
from semantitrace.utils.image import l2_normalize


@dataclass
class RetrievalHit:
    image_id: str
    image_path: str
    score: float
    rank: int


class ImageRAGIndex:
    def __init__(self, encoder: ImageTextEncoder) -> None:
        self.encoder = encoder
        self.image_ids: list[str] = []
        self.image_paths: list[str] = []
        self.embeddings: np.ndarray | None = None

    def build(self, image_paths: list[str | Path], image_ids: list[str] | None = None) -> "ImageRAGIndex":
        self.image_paths = [str(p) for p in image_paths]
        self.image_ids = image_ids or [Path(p).stem for p in self.image_paths]
        images = [Image.open(path).convert("RGB") for path in self.image_paths]
        self.embeddings = l2_normalize(self.encoder.encode_images(images))
        return self

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        if self.embeddings is None:
            raise RuntimeError("Index has not been built")
        query_emb = l2_normalize(self.encoder.encode_texts([query]))[0]
        scores = self.embeddings @ query_emb
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievalHit(
                image_id=self.image_ids[int(idx)],
                image_path=self.image_paths[int(idx)],
                score=float(scores[int(idx)]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(order)
        ]


class BlackBoxRAGSimulator:
    def __init__(self, index: ImageRAGIndex, generator: VLMClient, top_k: int = 3) -> None:
        self.index = index
        self.generator = generator
        self.top_k = top_k

    def query(self, query_text: str) -> dict[str, Any]:
        hits = self.index.search(query_text, self.top_k)
        images = [Image.open(hit.image_path).convert("RGB") for hit in hits]
        prompt = (
            "Answer the user using the retrieved visual evidence. "
            "If text is visible, transcribe it exactly.\n\n"
            f"User query: {query_text}"
        )
        response = self.generator.generate(images[0] if images else None, prompt, temperature=0.0)
        return {
            "query": query_text,
            "response": response,
            "hits": [hit.__dict__ for hit in hits],
        }

