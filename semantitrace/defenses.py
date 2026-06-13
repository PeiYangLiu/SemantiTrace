from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from semantitrace.models.interfaces import ImageTextEncoder, VLMClient


ANOMALY_FILTER_PROMPT = """You are a strict Data Sanitization Inspector for a commercial Multimodal RAG platform.
Flag the image as ANOMALY if it exhibits spatial hallucinations, semantic dissonance,
artificial overlaid text, rare acronyms without context, texture artifacts, localized
noise, or AI inpainting seams. Conclude with exactly one verdict line:
[VERDICT: SAFE] or [VERDICT: ANOMALY]."""


class MahalanobisOODDetector:
    def __init__(
        self,
        percentile: float = 99.0,
        regularization: float = 1e-4,
        max_components: int = 64,
        variance_keep: float = 0.95,
    ) -> None:
        self.percentile = percentile
        self.regularization = regularization
        self.max_components = max_components
        self.variance_keep = variance_keep
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.variances: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(self, clean_embeddings: np.ndarray) -> "MahalanobisOODDetector":
        emb = np.asarray(clean_embeddings, dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] < 2:
            raise ValueError("Mahalanobis detector requires at least two clean embeddings")
        self.mean = emb.mean(axis=0)
        centered = emb - self.mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        variances = (singular_values**2) / max(1, emb.shape[0] - 1)
        positive = variances > self.regularization
        if not positive.any():
            self.components = np.eye(emb.shape[1], dtype=np.float64)[:1]
            self.variances = np.array([self.regularization], dtype=np.float64)
        else:
            max_rank = min(int(self.max_components), int(positive.sum()), emb.shape[0] - 1)
            cumulative = np.cumsum(variances[:max_rank]) / max(float(np.sum(variances[positive])), 1e-12)
            keep_by_variance = int(np.searchsorted(cumulative, self.variance_keep, side="left") + 1)
            keep = max(1, min(max_rank, keep_by_variance))
            self.components = vt[:keep]
            self.variances = variances[:keep] + self.regularization
        scores = self.score_embeddings(emb)
        self.threshold = float(np.percentile(scores, self.percentile))
        return self

    def score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None or self.variances is None:
            raise RuntimeError("Detector must be fit before scoring")
        centered = np.asarray(embeddings, dtype=np.float64) - self.mean
        projected = centered @ self.components.T
        return np.sum((projected**2) / self.variances, axis=1)

    def reject_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Detector must be fit before rejection")
        return self.score_embeddings(embeddings) > self.threshold


@dataclass
class VLMFilterResult:
    rejected: bool
    raw_response: str


class VLMAnomalyFilter:
    def __init__(self, vlm: VLMClient, prompt: str = ANOMALY_FILTER_PROMPT) -> None:
        self.vlm = vlm
        self.prompt = prompt

    def inspect(self, image: Image.Image) -> VLMFilterResult:
        raw = self.vlm.generate(image, self.prompt, temperature=0.0, max_new_tokens=512)
        verdict = raw.upper()
        rejected = "[VERDICT: ANOMALY]" in verdict or ("ANOMALY" in verdict and "SAFE" not in verdict)
        return VLMFilterResult(rejected=rejected, raw_response=raw)


class AdaptiveSanitizer:
    def __init__(
        self,
        encoder: ImageTextEncoder,
        mahalanobis: MahalanobisOODDetector,
        vlm_filter: VLMAnomalyFilter | None = None,
    ) -> None:
        self.encoder = encoder
        self.mahalanobis = mahalanobis
        self.vlm_filter = vlm_filter

    def sanitize(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        embeddings = self.encoder.encode_images(images)
        maha_reject = self.mahalanobis.reject_embeddings(embeddings)
        results: list[dict[str, Any]] = []
        for idx, image in enumerate(images):
            vlm_result = self.vlm_filter.inspect(image) if self.vlm_filter else None
            rejected = bool(maha_reject[idx] or (vlm_result.rejected if vlm_result else False))
            results.append(
                {
                    "index": idx,
                    "rejected": rejected,
                    "mahalanobis_rejected": bool(maha_reject[idx]),
                    "mahalanobis_score": float(self.mahalanobis.score_embeddings(embeddings[[idx]])[0]),
                    "vlm_rejected": bool(vlm_result.rejected) if vlm_result else None,
                    "vlm_response": vlm_result.raw_response if vlm_result else None,
                }
            )
        return results
