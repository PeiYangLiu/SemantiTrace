from __future__ import annotations

import unittest

import numpy as np

from semantitrace.metrics import contains_positive_signature
from semantitrace.mode_verification import detail_target_gated_hit, target_rank_in_topk
from semantitrace.verification import Verifier


class VerificationTests(unittest.TestCase):
    def test_cer_and_rejection(self) -> None:
        records = [
            {"trap_signature": "BSB", "probe_queries": ["q1", "q2", "q3"]},
            {"trap_signature": "SSV", "probe_queries": ["q4", "q5", "q6"]},
        ]
        responses = {
            "q1": "The label is BSB.",
            "q2": "I can read BSB.",
            "q3": "BSB appears there.",
            "q4": "The red flag says SSV.",
            "q5": "SSV",
            "q6": "It is SSV.",
        }
        verifier = Verifier({"num_probes_per_canary": 3, "significance_level": 0.01})
        report = verifier.run_verification(records, lambda q: responses[q])
        self.assertEqual(report["suspect_cer"], 1.0)
        self.assertTrue(report["test_result"]["reject_h0"])

    def test_partial_cer(self) -> None:
        verifier = Verifier({"num_probes_per_canary": 3})
        cer = verifier.compute_cer(["The label is ABC.", "miss", "ABC"], ["ABC"])
        self.assertAlmostEqual(cer, 2 / 3)

    def test_negative_echo_does_not_count_as_hit(self) -> None:
        verifier = Verifier({"num_probes_per_canary": 3})
        cer = verifier.compute_cer(
            [
                "The text FALEK is not visible.",
                "I cannot read the mark FALEK in the image.",
                "FALEK",
            ],
            ["FALEK"],
        )
        self.assertAlmostEqual(cer, 1 / 3)

    def test_panel_negative_echo_does_not_count_as_hit(self) -> None:
        self.assertFalse(
            contains_positive_signature('The label "ZIRO" is not visible in any of the provided panels.', "ZIRO")
        )
        self.assertFalse(
            contains_positive_signature("None of the panels show the acronym DAWA.", "DAWA")
        )
        self.assertFalse(
            contains_positive_signature("The panels show storefronts, but none contain the specific text CEDE.", "CEDE")
        )
        self.assertFalse(
            contains_positive_signature('Regarding the short acronym or code "CEDE" in the selected region.', "CEDE")
        )
        self.assertTrue(contains_positive_signature("The exact text is VUKI.", "VUKI"))

    def test_bootstrap_rate_test_reports_effect_size(self) -> None:
        verifier = Verifier({"significance_level": 0.01})
        suspect = np.array([1.0, 2 / 3, 1.0, 1 / 3, 1.0, 2 / 3, 1.0, 1.0])
        clean = np.zeros_like(suspect)
        result = verifier.bootstrap_rate_test(suspect, clean, iterations=2000, seed=7)
        self.assertAlmostEqual(result["effect_size"], float((suspect - clean).mean()))
        self.assertLess(result["p_value"], 0.01)
        self.assertLessEqual(result["effect_ci95_low"], result["effect_size"])
        self.assertGreaterEqual(result["effect_ci95_high"], result["effect_size"])

    def test_mode_aware_hit_requires_target_in_topk(self) -> None:
        detail = {
            "target_rank": 252777,
            "clean_target_rank": 879303,
            "watermarked_response_strict_hit": True,
            "clean_response_strict_hit": True,
        }
        self.assertFalse(target_rank_in_topk(detail["target_rank"], 3))
        self.assertFalse(detail_target_gated_hit(detail, "watermarked", 3, strict=True))
        self.assertFalse(detail_target_gated_hit(detail, "clean", 3, strict=True))

        detail["target_rank"] = 2
        self.assertTrue(detail_target_gated_hit(detail, "watermarked", 3, strict=True))


if __name__ == "__main__":
    unittest.main()
