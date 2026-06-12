from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from semantitrace.anchor_mining import AnchorMiner
from semantitrace.backends.deterministic import DeterministicEncoder, GridMaskGenerator
from semantitrace.utils.image import mask_from_bbox


class FakeOCR:
    def detect_text_regions(self, image: Image.Image):
        mask = mask_from_bbox(image.size, (10, 10, 70, 36))
        return [
            {
                "text": "CAFE",
                "confidence": 0.99,
                "bbox": (10, 10, 70, 36),
                "mask": mask,
                "area": int(mask.sum()),
                "source": "fake",
            }
        ]


class MixedQualityOCR:
    def detect_text_regions(self, image: Image.Image):
        good = mask_from_bbox(image.size, (10, 10, 70, 36))
        short = mask_from_bbox(image.size, (2, 60, 50, 90))
        numeric = mask_from_bbox(image.size, (70, 60, 120, 100))
        lowercase = mask_from_bbox(image.size, (10, 105, 50, 125))
        noisy = mask_from_bbox(image.size, (55, 105, 125, 125))
        sentence = mask_from_bbox(image.size, (10, 128, 150, 138))
        return [
            {"text": "EX", "confidence": 0.99, "bbox": (2, 60, 50, 90), "mask": short, "source": "fake"},
            {"text": "38", "confidence": 0.99, "bbox": (70, 60, 120, 100), "mask": numeric, "source": "fake"},
            {"text": "des", "confidence": 0.99, "bbox": (10, 105, 50, 125), "mask": lowercase, "source": "fake"},
            {"text": "(GRDON", "confidence": 0.99, "bbox": (55, 105, 125, 125), "mask": noisy, "source": "fake"},
            {
                "text": "Death is their way of life",
                "confidence": 0.99,
                "bbox": (10, 128, 150, 138),
                "mask": sentence,
                "source": "fake",
            },
            {"text": "CAFE", "confidence": 0.99, "bbox": (10, 10, 70, 36), "mask": good, "source": "fake"},
        ]


class AnchorMiningTests(unittest.TestCase):
    def test_text_canvas_is_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for idx, color in enumerate([(220, 30, 30), (30, 220, 30), (30, 30, 220)]):
                path = Path(td) / f"img{idx}.png"
                Image.new("RGB", (96, 96), color).save(path)
                paths.append(path)

            miner = AnchorMiner(
                DeterministicEncoder(),
                FakeOCR(),
                GridMaskGenerator(),
                {
                    "num_clusters": 1,
                    "knn_k": 1,
                    "min_canvas_area_ratio": 0.01,
                    "ocr_confidence_threshold": 0.5,
                },
            )
            labels, features = miner.cluster_dataset(paths, 1)
            anchors = miner.mine_anchors(paths, 1, features, labels)
            self.assertEqual(len(anchors), 1)
            self.assertEqual(anchors[0].canvas_mode, "text")
            self.assertEqual(anchors[0].candidate_canvases[0].text, "CAFE")

    def test_isolation_scores_shape(self) -> None:
        miner = AnchorMiner(DeterministicEncoder(), FakeOCR(), GridMaskGenerator())
        features = np.eye(4, dtype=np.float32)
        scores = miner.compute_isolation_scores(features)
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.all(scores >= 0))

    def test_multiple_anchors_per_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for idx, color in enumerate([(220, 30, 30), (30, 220, 30), (30, 30, 220)]):
                path = Path(td) / f"img{idx}.png"
                Image.new("RGB", (96, 96), color).save(path)
                paths.append(path)

            miner = AnchorMiner(
                DeterministicEncoder(),
                FakeOCR(),
                GridMaskGenerator(),
                {
                    "num_clusters": 1,
                    "knn_k": 1,
                    "min_canvas_area_ratio": 0.01,
                    "ocr_confidence_threshold": 0.5,
                    "top_candidates_per_cluster": 3,
                    "max_anchors_per_cluster": 2,
                },
            )
            labels, features = miner.cluster_dataset(paths, 1)
            anchors = miner.mine_anchors(paths, 1, features, labels)
            self.assertEqual(len(anchors), 2)
            self.assertTrue(all(anchor.canvas_mode == "text" for anchor in anchors))

    def test_rejects_low_quality_ocr_fragments(self) -> None:
        image = Image.new("RGB", (160, 140), (240, 240, 240))
        miner = AnchorMiner(
            DeterministicEncoder(),
            MixedQualityOCR(),
            GridMaskGenerator(),
            {
                "min_canvas_area_ratio": 0.001,
                "max_canvas_area_ratio": 0.5,
                "ocr_confidence_threshold": 0.5,
                "min_text_alnum": 4,
                "max_text_alnum": 14,
                "max_text_words": 2,
                "max_non_alnum_ratio": 0.12,
                "min_letter_ratio": 0.75,
                "max_consecutive_consonants": 4,
                "reject_numeric_text": True,
                "reject_short_lowercase_text": True,
                "short_text_length": 3,
                "reject_edge_text": True,
                "edge_margin_ratio": 0.004,
                "max_short_text_bbox_area_ratio": 0.020,
                "max_text_bbox_area_per_alnum_ratio": 0.020,
            },
        )
        text_canvases, _ = miner.compute_editability(image)
        self.assertEqual([canvas.text for canvas in text_canvases], ["CAFE"])

    def test_body_zone_filter_rejects_torso_text(self) -> None:
        miner = AnchorMiner(
            DeterministicEncoder(),
            FakeOCR(),
            GridMaskGenerator(),
            {
                "enable_body_zone_filter": True,
                "max_body_zone_overlap_ratio": 0.0,
                "body_zone_x_scale": 3.0,
                "body_zone_y_scale": 5.0,
                "body_zone_y_offset": -0.1,
            },
        )
        zones = miner._infer_body_zones([(80, 40, 120, 80)], (200, 200))
        torso_text = mask_from_bbox((200, 200), (70, 130, 130, 155))
        safe_sign_text = mask_from_bbox((200, 200), (165, 130, 195, 155))
        self.assertFalse(
            miner._is_safe_canvas(
                torso_text,
                (70, 130, 130, 155),
                [],
                zones,
                None,
                mode="text",
            )
        )
        self.assertTrue(
            miner._is_safe_canvas(
                safe_sign_text,
                (165, 130, 195, 155),
                [],
                zones,
                None,
                mode="text",
            )
        )


if __name__ == "__main__":
    unittest.main()
