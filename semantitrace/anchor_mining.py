from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from semantitrace.models.interfaces import ImageTextEncoder, MaskGenerator, OCRDetector
from semantitrace.utils.image import bbox_from_mask, ensure_bool_mask, l2_normalize, laplacian_variance

logger = logging.getLogger(__name__)


@dataclass
class Canvas:
    id: int
    mode: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    text: str | None = None
    source: str = ""
    oi_proposal: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "bbox": list(self.bbox),
            "score": self.score,
            "text": self.text,
            "source": self.source,
            "oi_proposal": self.oi_proposal,
        }


@dataclass
class Anchor:
    image_path: str
    cluster_id: int
    isolation_score: float
    canvas_mode: str
    candidate_canvases: list[Canvas] = field(default_factory=list)
    joint_score: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "cluster_id": self.cluster_id,
            "isolation_score": self.isolation_score,
            "canvas_mode": self.canvas_mode,
            "joint_score": self.joint_score,
            "candidate_canvases": [canvas.to_json() for canvas in self.candidate_canvases],
        }


class AnchorMiner:
    """Target-aware anchor image mining from the paper.

    The miner first estimates CLIP-like latent isolation, then evaluates local
    editability using OCR text canvases and structural masks. Within each
    cluster, text canvases are strictly prioritized; structural canvases are used
    only when no viable text canvas exists.
    """

    def __init__(
        self,
        encoder: ImageTextEncoder,
        ocr: OCRDetector,
        mask_generator: MaskGenerator,
        config: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        cfg = config or {}
        self.encoder = encoder
        self.ocr = ocr
        self.mask_generator = mask_generator
        self.knn_k = int(cfg.get("knn_k", 10))
        self.min_canvas_area_ratio = float(cfg.get("min_canvas_area_ratio", 0.01))
        self.max_canvas_area_ratio = float(cfg.get("max_canvas_area_ratio", 0.20))
        self.ocr_confidence_threshold = float(cfg.get("ocr_confidence_threshold", 0.5))
        self.top_candidates_per_cluster = int(cfg.get("top_candidates_per_cluster", 8))
        self.max_anchors_per_cluster = max(1, int(cfg.get("max_anchors_per_cluster", 1)))
        self.kmeans_iters = int(cfg.get("kmeans_iters", 25))
        self.epsilon = float(cfg.get("epsilon", 1e-6))
        self.enable_face_filter = bool(cfg.get("enable_face_filter", True))
        self.max_face_overlap_ratio = float(cfg.get("max_face_overlap_ratio", 0.0))
        self.enable_skin_filter = bool(cfg.get("enable_skin_filter", True))
        self.max_skin_ratio = float(cfg.get("max_skin_ratio", 0.38))
        self.max_text_skin_ratio = float(cfg.get("max_text_skin_ratio", 0.80))
        self.enable_body_zone_filter = bool(cfg.get("enable_body_zone_filter", False))
        self.max_body_zone_overlap_ratio = float(cfg.get("max_body_zone_overlap_ratio", 0.0))
        self.body_zone_x_scale = float(cfg.get("body_zone_x_scale", 2.8))
        self.body_zone_y_scale = float(cfg.get("body_zone_y_scale", 4.5))
        self.body_zone_y_offset = float(cfg.get("body_zone_y_offset", -0.15))
        self.allow_structural_fallback = bool(cfg.get("allow_structural_fallback", True))
        self.min_text_alnum = int(cfg.get("min_text_alnum", 1))
        self.max_text_alnum = int(cfg.get("max_text_alnum", 10_000))
        self.max_text_words = int(cfg.get("max_text_words", 10_000))
        self.max_non_alnum_ratio = float(cfg.get("max_non_alnum_ratio", 1.0))
        self.min_letter_ratio = float(cfg.get("min_letter_ratio", 0.0))
        self.max_consecutive_consonants = int(cfg.get("max_consecutive_consonants", 10_000))
        self.reject_numeric_text = bool(cfg.get("reject_numeric_text", False))
        self.reject_short_lowercase_text = bool(cfg.get("reject_short_lowercase_text", False))
        self.short_text_length = int(cfg.get("short_text_length", 4))
        self.min_short_text_uppercase_ratio = float(cfg.get("min_short_text_uppercase_ratio", 0.0))
        self.reject_edge_text = bool(cfg.get("reject_edge_text", False))
        self.edge_margin_ratio = float(cfg.get("edge_margin_ratio", 0.0))
        self.max_short_text_bbox_area_ratio = float(cfg.get("max_short_text_bbox_area_ratio", 1.0))
        self.max_text_bbox_area_per_alnum_ratio = float(cfg.get("max_text_bbox_area_per_alnum_ratio", 1.0))
        self.prefer_short_text = bool(cfg.get("prefer_short_text", False))
        self.seed = seed
        # ---- Opus hint integration ----
        self.opus_hints: dict[str, dict] = cfg.get("opus_hints") or {}
        self.opus_padding_ratio = float(cfg.get("opus_padding_ratio", 0.08))
        self.opus_search_pad_ratio = float(cfg.get("opus_search_pad_ratio", 0.30))
        self.opus_snap_pad_ratio = float(cfg.get("opus_snap_pad_ratio", 0.06))
        self.opus_snap_min_confidence = float(cfg.get("opus_snap_min_confidence", 0.05))
        self.opus_hint_score = float(cfg.get("opus_hint_score", 1e10))
        self.opus_min_pad_px = int(cfg.get("opus_min_pad_px", 8))

    def encode_image_paths(self, image_paths: list[str | Path]) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return l2_normalize(self.encoder.encode_images(images).astype(np.float32))

    def cluster_dataset(self, image_paths: list[str | Path], num_clusters: int) -> tuple[np.ndarray, np.ndarray]:
        features = self.encode_image_paths(image_paths)
        labels = self._kmeans(features, min(num_clusters, len(image_paths)))
        return labels, features

    def compute_isolation_scores(self, features: np.ndarray) -> np.ndarray:
        features = l2_normalize(features)
        n = features.shape[0]
        if n <= 1:
            return np.ones(n, dtype=np.float64)
        sim = features @ features.T
        k = min(self.knn_k + 1, n)
        nearest = np.sort(sim, axis=1)[:, -k:]
        neighbour_sims = nearest[:, :-1]
        return np.mean(1.0 - neighbour_sims, axis=1).astype(np.float64)

    def compute_editability(
        self,
        image: Image.Image,
        image_path: str | Path | None = None,
    ) -> tuple[list[Canvas], list[Canvas]]:
        width, height = image.size
        min_area = width * height * self.min_canvas_area_ratio
        max_area = width * height * self.max_canvas_area_ratio
        face_bboxes = self._detect_faces(image) if self.enable_face_filter else []
        body_bboxes = self._infer_body_zones(face_bboxes, image.size) if self.enable_body_zone_filter else []
        skin_mask = self._skin_mask(image) if self.enable_skin_filter else None

        text_canvases: list[Canvas] = []
        struct_canvases: list[Canvas] = []

        # ---- (A) Inject Opus hint canvases first (highest priority). ----
        # Strategy: pad Opus bbox by `opus_search_pad_ratio` (generous, e.g. 30%),
        # then look for OCR detections overlapping the padded region. If found, use
        # the OCR-detected bbox + recognised text (tighter and more accurate). If
        # no OCR hits, fall back to using the Opus bbox itself (modestly padded).
        opus_rec = None
        if image_path is not None and self.opus_hints:
            stem = Path(str(image_path)).stem
            opus_rec = self.opus_hints.get(stem)
        ocr_dets_for_hints: list[dict[str, Any]] | None = None
        if opus_rec:
            for hint in opus_rec.get("candidates", []):
                bb_norm = hint.get("bbox_norm_1000") or []
                if len(bb_norm) != 4:
                    continue
                x1n, y1n, x2n, y2n = [max(0.0, min(1000.0, float(v))) for v in bb_norm]
                if x2n <= x1n or y2n <= y1n:
                    continue
                # to pixels
                ox1, oy1, ox2, oy2 = (
                    int(x1n / 1000.0 * width),
                    int(y1n / 1000.0 * height),
                    int(x2n / 1000.0 * width),
                    int(y2n / 1000.0 * height),
                )
                obw, obh = ox2 - ox1, oy2 - oy1
                # Generous search pad to absorb Opus bbox inaccuracy.
                spad_x = max(self.opus_min_pad_px, int(obw * self.opus_search_pad_ratio))
                spad_y = max(self.opus_min_pad_px, int(obh * self.opus_search_pad_ratio))
                sx1 = max(0, ox1 - spad_x)
                sy1 = max(0, oy1 - spad_y)
                sx2 = min(width, ox2 + spad_x)
                sy2 = min(height, oy2 + spad_y)
                hint_mode = str(hint.get("mode") or opus_rec.get("selected_mode") or "text_mutation")
                visible_text = str(hint.get("visible_text") or "").strip()

                snap_used = False
                if hint_mode == "text_mutation":
                    if ocr_dets_for_hints is None:
                        try:
                            ocr_dets_for_hints = list(self.ocr.detect_text_regions(image))
                        except Exception:
                            ocr_dets_for_hints = []
                    # Find OCR detections inside the search region with relaxed conf threshold.
                    matches: list[tuple[float, dict[str, Any]]] = []
                    for det in ocr_dets_for_hints:
                        bbox_d = det.get("bbox")
                        if not bbox_d or len(bbox_d) != 4:
                            continue
                        dx1, dy1, dx2, dy2 = [int(v) for v in bbox_d]
                        # require >= 50% of OCR detection inside the search rect
                        ix1 = max(sx1, dx1); iy1 = max(sy1, dy1)
                        ix2 = min(sx2, dx2); iy2 = min(sy2, dy2)
                        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                        det_area = max(1, (dx2 - dx1) * (dy2 - dy1))
                        inside_ratio = (iw * ih) / det_area
                        if inside_ratio < 0.5:
                            continue
                        conf = float(det.get("confidence", 0.0))
                        if conf < self.opus_snap_min_confidence:
                            continue
                        score = conf * inside_ratio
                        matches.append((score, det))
                    if matches:
                        # Snap: union of all matched OCR boxes (keeps multi-line text together).
                        ux1 = min(int(d["bbox"][0]) for _, d in matches)
                        uy1 = min(int(d["bbox"][1]) for _, d in matches)
                        ux2 = max(int(d["bbox"][2]) for _, d in matches)
                        uy2 = max(int(d["bbox"][3]) for _, d in matches)
                        snap_pad_x = max(self.opus_min_pad_px, int((ux2 - ux1) * self.opus_snap_pad_ratio))
                        snap_pad_y = max(self.opus_min_pad_px, int((uy2 - uy1) * self.opus_snap_pad_ratio))
                        px1 = max(0, ux1 - snap_pad_x)
                        py1 = max(0, uy1 - snap_pad_y)
                        px2 = min(width, ux2 + snap_pad_x)
                        py2 = min(height, uy2 + snap_pad_y)
                        # Prefer Opus's visible_text (more reliable) for length-matching, fallback to OCR text.
                        if visible_text:
                            text_for_canvas = visible_text
                        else:
                            ordered = sorted(matches, key=lambda m: (int(m[1]["bbox"][1]) // 8, int(m[1]["bbox"][0])))
                            text_for_canvas = " ".join(str(d.get("text", "")).strip() for _, d in ordered if str(d.get("text","")).strip()) or " "
                        snap_used = True
                    else:
                        # Fallback: pad Opus bbox a smaller amount.
                        pad_x = max(self.opus_min_pad_px, int(obw * self.opus_padding_ratio))
                        pad_y = max(self.opus_min_pad_px, int(obh * self.opus_padding_ratio))
                        px1 = max(0, ox1 - pad_x)
                        py1 = max(0, oy1 - pad_y)
                        px2 = min(width, ox2 + pad_x)
                        py2 = min(height, oy2 + pad_y)
                        text_for_canvas = visible_text or " "
                else:
                    # struct / object_insertion: keep Opus bbox with modest pad.
                    pad_x = max(self.opus_min_pad_px, int(obw * self.opus_padding_ratio))
                    pad_y = max(self.opus_min_pad_px, int(obh * self.opus_padding_ratio))
                    px1 = max(0, ox1 - pad_x)
                    py1 = max(0, oy1 - pad_y)
                    px2 = min(width, ox2 + pad_x)
                    py2 = min(height, oy2 + pad_y)
                    text_for_canvas = visible_text or " "
                if px2 <= px1 or py2 <= py1:
                    continue
                m = np.zeros((height, width), dtype=bool)
                m[py1:py2, px1:px2] = True
                area = int(m.sum())
                if area < min_area:
                    continue
                bbox = (px1, py1, px2, py2)
                logger.info(
                    "Opus hint %s mode=%s snap=%s opus_bbox=%s -> canvas_bbox=%s text=%r",
                    image_path, hint_mode, snap_used,
                    (ox1, oy1, ox2, oy2), bbox, text_for_canvas[:40],
                )
                if hint_mode == "text_mutation":
                    text_canvases.append(
                        Canvas(
                            id=len(text_canvases),
                            mode="text",
                            mask=m,
                            bbox=bbox,
                            score=self.opus_hint_score,
                            text=text_for_canvas,
                            source="opus_hint" + ("_snap" if snap_used else ""),
                        )
                    )
                else:
                    struct_canvases.append(
                        Canvas(
                            id=len(struct_canvases),
                            mode="struct",
                            mask=m,
                            bbox=bbox,
                            score=self.opus_hint_score,
                            source="opus_hint",
                            oi_proposal=opus_rec.get("oi_proposal"),
                        )
                    )

        # ---- (B) Standard OCR text canvases. ----
        for det in self.ocr.detect_text_regions(image):
            confidence = float(det.get("confidence", 1.0))
            if confidence < self.ocr_confidence_threshold:
                continue
            mask = ensure_bool_mask(np.asarray(det["mask"]), image.size)
            area = int(mask.sum())
            if area < min_area:
                continue
            if area > max_area:
                continue
            bbox = tuple(det.get("bbox") or bbox_from_mask(mask))
            text = str(det.get("text", ""))
            if not self._is_valid_text_canvas(text, bbox, image.size):
                continue
            if not self._is_safe_canvas(mask, bbox, face_bboxes, body_bboxes, skin_mask, mode="text"):
                continue
            score = float(area * confidence)
            text_canvases.append(
                Canvas(
                    id=len(text_canvases),
                    mode="text",
                    mask=mask,
                    bbox=bbox,
                    score=score,
                    text=text,
                    source=str(det.get("source", "ocr")),
                )
            )
        text_canvases.sort(key=lambda c: c.score, reverse=True)

        for seg in self.mask_generator.generate_masks(image):
            raw_mask = seg.get("mask", seg.get("segmentation"))
            if raw_mask is None:
                continue
            mask = ensure_bool_mask(np.asarray(raw_mask), image.size)
            area = int(mask.sum())
            if area < min_area:
                continue
            if area > max_area:
                continue
            bbox = tuple(seg.get("bbox") or bbox_from_mask(mask))
            if not self._is_safe_canvas(mask, bbox, face_bboxes, body_bboxes, skin_mask, mode="struct"):
                continue
            score = float(area / (laplacian_variance(image, mask) + self.epsilon))
            struct_canvases.append(
                Canvas(
                    id=len(struct_canvases),
                    mode="struct",
                    mask=mask,
                    bbox=bbox,
                    score=score,
                    source=str(seg.get("source", "mask_generator")),
                )
            )
        struct_canvases.sort(key=lambda c: c.score, reverse=True)
        return text_canvases, struct_canvases

    def mine_anchors(
        self,
        image_paths: list[str | Path],
        num_clusters: int,
        features: np.ndarray | None = None,
        cluster_labels: np.ndarray | None = None,
    ) -> list[Anchor]:
        if not image_paths:
            return []
        if features is None or cluster_labels is None:
            cluster_labels, features = self.cluster_dataset(image_paths, num_clusters)
        isolation_scores = self.compute_isolation_scores(features)

        anchors: list[Anchor] = []
        for cid in sorted(int(c) for c in np.unique(cluster_labels)):
            idxs = np.where(cluster_labels == cid)[0]
            if idxs.size == 0:
                continue
            top_idxs = idxs[np.argsort(-isolation_scores[idxs])[: self.top_candidates_per_cluster]]

            text_options: list[tuple[float, Anchor]] = []
            struct_options: list[tuple[float, Anchor]] = []
            for idx in top_idxs:
                image_path = str(image_paths[int(idx)])
                image = Image.open(image_path).convert("RGB")
                text_canvases, struct_canvases = self.compute_editability(image, image_path=image_path)
                iso = float(isolation_scores[int(idx)])

                if text_canvases:
                    joint = self._joint_text_score(iso, text_canvases[0])
                    text_options.append(
                        (
                            joint,
                            Anchor(
                                image_path=image_path,
                                cluster_id=cid,
                                isolation_score=iso,
                                canvas_mode="text",
                                candidate_canvases=text_canvases,
                                joint_score=joint,
                            ),
                        )
                    )
                if struct_canvases:
                    joint = iso * struct_canvases[0].score
                    struct_options.append(
                        (
                            joint,
                            Anchor(
                                image_path=image_path,
                                cluster_id=cid,
                                isolation_score=iso,
                                canvas_mode="struct",
                                candidate_canvases=struct_canvases,
                                joint_score=joint,
                            ),
                        )
                    )

            # An Opus OI hint canvas should win even if OCR also found text in the image —
            # the upstream filter already determined this image is best handled as
            # object_insertion (e.g. existing text is unsuitable for mutation).
            has_opus_oi_hint = any(
                opt[1].candidate_canvases and opt[1].candidate_canvases[0].source == "opus_hint"
                for opt in struct_options
            )
            if has_opus_oi_hint:
                chosen_pool = [
                    opt for opt in struct_options
                    if opt[1].candidate_canvases and opt[1].candidate_canvases[0].source == "opus_hint"
                ]
            else:
                chosen_pool = text_options if text_options or not self.allow_structural_fallback else struct_options
            if not chosen_pool:
                logger.warning("Cluster %d has no editable canvas.", cid)
                continue
            chosen_pool.sort(key=lambda item: item[0], reverse=True)
            for _score, chosen in chosen_pool[: self.max_anchors_per_cluster]:
                logger.info(
                    "Cluster %d anchor=%s mode=%s iso=%.4f joint=%.4f",
                    cid,
                    chosen.image_path,
                    chosen.canvas_mode,
                    chosen.isolation_score,
                    chosen.joint_score,
                )
                anchors.append(chosen)
        return anchors

    def _joint_text_score(self, isolation_score: float, canvas: Canvas) -> float:
        joint = isolation_score * canvas.score
        if not self.prefer_short_text or not canvas.text:
            return joint
        alnum_len = len(re.sub(r"[^A-Za-z0-9]", "", canvas.text))
        if alnum_len <= 0:
            return joint
        short_bonus = 1.0 / max(1.0, abs(alnum_len - 4) + 1.0)
        area = max(1, canvas.bbox[2] - canvas.bbox[0]) * max(1, canvas.bbox[3] - canvas.bbox[1])
        compactness = canvas.score / area
        return joint * (1.0 + short_bonus) * (1.0 + compactness)

    def _is_valid_text_canvas(self, text: str, bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> bool:
        normalized = re.sub(r"\s+", " ", text.strip())
        alnum = re.sub(r"[^A-Za-z0-9]", "", normalized)
        if len(alnum) < self.min_text_alnum:
            return False
        if len(alnum) > self.max_text_alnum:
            return False
        words = re.findall(r"[A-Za-z0-9]+", normalized)
        if len(words) > self.max_text_words:
            return False
        non_space = re.sub(r"\s+", "", normalized)
        non_alnum_ratio = 1.0 - (len(alnum) / max(1, len(non_space)))
        if non_alnum_ratio > self.max_non_alnum_ratio:
            return False
        letters = re.sub(r"[^A-Za-z]", "", normalized)
        if len(letters) / max(1, len(alnum)) < self.min_letter_ratio:
            return False
        if self._max_consecutive_consonants(letters) > self.max_consecutive_consonants:
            return False
        if self.reject_numeric_text and alnum.isdigit():
            return False
        if self.reject_short_lowercase_text and len(alnum) <= self.short_text_length and normalized.islower():
            return False
        uppercase_letters = sum(1 for char in letters if char.isupper())
        if (
            self.min_short_text_uppercase_ratio > 0
            and len(alnum) <= self.short_text_length + 2
            and uppercase_letters / max(1, len(letters)) < self.min_short_text_uppercase_ratio
        ):
            return False
        width, height = image_size
        x1, y1, x2, y2 = bbox
        if self.reject_edge_text:
            margin = max(1, int(round(min(width, height) * self.edge_margin_ratio)))
            if x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin:
                return False
        bbox_area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1, width * height)
        if len(alnum) <= self.short_text_length and bbox_area_ratio > self.max_short_text_bbox_area_ratio:
            return False
        if bbox_area_ratio / max(1, len(alnum)) > self.max_text_bbox_area_per_alnum_ratio:
            return False
        return True

    @staticmethod
    def _max_consecutive_consonants(text: str) -> int:
        best = 0
        current = 0
        for char in text.upper():
            if char.isalpha() and char not in "AEIOUY":
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def _kmeans(self, features: np.ndarray, k: int) -> np.ndarray:
        n = features.shape[0]
        if k <= 1 or n <= 1:
            return np.zeros(n, dtype=np.int64)
        rng = random.Random(self.seed)
        initial = rng.sample(range(n), k)
        centroids = features[initial].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(self.kmeans_iters):
            sim = features @ l2_normalize(centroids).T
            new_labels = np.argmax(sim, axis=1).astype(np.int64)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cid in range(k):
                members = features[labels == cid]
                if members.size:
                    centroids[cid] = members.mean(axis=0)
        return labels

    def _is_safe_canvas(
        self,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        face_bboxes: list[tuple[int, int, int, int]],
        body_bboxes: list[tuple[int, int, int, int]],
        skin_mask: np.ndarray | None,
        mode: str,
    ) -> bool:
        area = int(mask.sum())
        if area <= 0:
            return False
        for face_bbox in face_bboxes:
            if self._overlap_ratio(bbox, face_bbox) > self.max_face_overlap_ratio:
                return False
        for body_bbox in body_bboxes:
            if self._overlap_ratio(bbox, body_bbox) > self.max_body_zone_overlap_ratio:
                return False
        if skin_mask is not None:
            skin_ratio = float(np.logical_and(mask, skin_mask).sum() / max(area, 1))
            limit = self.max_text_skin_ratio if mode == "text" else self.max_skin_ratio
            if skin_ratio > limit:
                return False
        return True

    def _infer_body_zones(
        self,
        face_bboxes: list[tuple[int, int, int, int]],
        image_size: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        width, height = image_size
        zones: list[tuple[int, int, int, int]] = []
        for fx1, fy1, fx2, fy2 in face_bboxes:
            face_w = max(1, fx2 - fx1)
            face_h = max(1, fy2 - fy1)
            cx = (fx1 + fx2) / 2.0
            zone_w = face_w * self.body_zone_x_scale
            zx1 = int(max(0, round(cx - zone_w / 2.0)))
            zx2 = int(min(width, round(cx + zone_w / 2.0)))
            zy1 = int(max(0, round(fy2 + face_h * self.body_zone_y_offset)))
            zy2 = int(min(height, round(fy2 + face_h * self.body_zone_y_scale)))
            if zx2 > zx1 and zy2 > zy1:
                zones.append((zx1, zy1, zx2, zy2))
        return zones

    @staticmethod
    def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        return (ix * iy) / max(area_a, 1)

    @staticmethod
    def _skin_mask(image: Image.Image) -> np.ndarray | None:
        try:
            import cv2
        except ImportError:
            return None
        rgb = np.asarray(image.convert("RGB"))
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[..., 0]
        cr = ycrcb[..., 1]
        cb = ycrcb[..., 2]
        return (y > 35) & (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)

    @staticmethod
    def _detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
        try:
            import cv2
        except ImportError:
            return []
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        face_boxes: list[tuple[int, int, int, int]] = []
        cascades = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_profileface.xml",
        ]
        for cascade_path in cascades:
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            face_boxes.extend((int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces)
            flipped = cv2.flip(gray, 1)
            flipped_faces = cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            width = gray.shape[1]
            face_boxes.extend((int(width - x - w), int(y), int(width - x), int(y + h)) for x, y, w, h in flipped_faces)
        deduped: list[tuple[int, int, int, int]] = []
        for box in face_boxes:
            if all(AnchorMiner._overlap_ratio(box, kept) < 0.35 for kept in deduped):
                deduped.append(box)
        return deduped
