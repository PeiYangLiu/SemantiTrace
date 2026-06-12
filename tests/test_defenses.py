from __future__ import annotations

import unittest

import numpy as np

from semantitrace.defenses import MahalanobisOODDetector


class DefenseTests(unittest.TestCase):
    def test_mahalanobis_rejects_far_point(self) -> None:
        clean = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [0.1, 0.1],
            ],
            dtype=float,
        )
        detector = MahalanobisOODDetector(percentile=95).fit(clean)
        far = np.array([[10.0, 10.0]], dtype=float)
        self.assertTrue(bool(detector.reject_embeddings(far)[0]))


if __name__ == "__main__":
    unittest.main()

