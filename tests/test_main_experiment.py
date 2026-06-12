from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from semantitrace.experiments import MainExperimentRunner
from semantitrace.experiments.datasets import ImageCorpus, ImageRecord
from semantitrace.experiments.runner import MethodArtifacts


class ConstantEncoder:
    def encode_images(self, images):
        return np.ones((len(images), 2), dtype=np.float64)

    def encode_texts(self, texts):
        return np.ones((len(texts), 2), dtype=np.float64)


class CleanEchoGenerator:
    name = "clean-echo"

    def __init__(self, signature: str, clean_echo: bool) -> None:
        self.signature = signature
        self.clean_echo = clean_echo

    def answer(self, query, retrieved, records_by_path):
        image_path = Path(retrieved[0]["image_path"])
        if image_path.name == "anchor.png" and self.clean_echo:
            return self.signature
        if image_path.name == "watermarked.png":
            return self.signature
        return "no matching mark is visible"


class MainExperimentTests(unittest.TestCase):
    def test_dry_run_main_experiment_outputs_tables(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "main"
            runner = MainExperimentRunner(
                config_path="configs/main_experiment.yaml",
                output_dir=out,
                device="cpu",
                dry_run_sample=True,
            )
            rows = runner.run(["efficacy", "stealth", "ood", "robustness"])
            self.assertTrue(rows["efficacy"])
            self.assertTrue(rows["stealth"])
            self.assertTrue(rows["ood"])
            self.assertTrue(rows["robustness"])
            self.assertTrue((out / "main_experiment_report.json").is_file())
            report = json.loads((out / "main_experiment_report.json").read_text())
            self.assertIn("efficacy", report)
            for name in ["efficacy.csv", "stealth.csv", "ood.csv", "robustness.csv", "skipped.csv"]:
                self.assertTrue((out / name).is_file(), name)

    def test_efficacy_uses_clean_control_for_cgsr(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            anchor = root / "anchor.png"
            watermarked = root / "watermarked.png"
            Image.new("RGB", (8, 8), (20, 20, 20)).save(anchor)
            Image.new("RGB", (8, 8), (240, 240, 240)).save(watermarked)
            record = {
                "id": "canary-0000",
                "anchor_image_path": str(anchor),
                "watermarked_image_path": str(watermarked),
                "index_image_path": str(watermarked),
                "trap_signature": "FUD",
                "probe_queries": ["what exact rare mark says FUD?"],
            }
            corpus = ImageCorpus("Unit", [ImageRecord("anchor", str(anchor), {})])
            artifacts = MethodArtifacts("semantitrace", [record], [str(watermarked)], ["anchor"])
            runner = MainExperimentRunner(
                config_path="configs/main_experiment.yaml",
                output_dir=root / "out",
                device="cpu",
                dry_run_sample=True,
            )

            leaked = runner._evaluate_efficacy(
                corpus,
                artifacts,
                "constant",
                ConstantEncoder(),
                CleanEchoGenerator("FUD", clean_echo=True),
            )
            self.assertEqual(leaked["raw_cgsr"], 1.0)
            self.assertEqual(leaked["clean_cgsr"], 1.0)
            self.assertEqual(leaked["adjusted_cgsr"], 0.0)
            self.assertEqual(leaked["cgsr"], 0.0)

            controlled = runner._evaluate_efficacy(
                corpus,
                artifacts,
                "constant",
                ConstantEncoder(),
                CleanEchoGenerator("FUD", clean_echo=False),
            )
            self.assertEqual(controlled["raw_cgsr"], 1.0)
            self.assertEqual(controlled["clean_cgsr"], 0.0)
            self.assertEqual(controlled["adjusted_cgsr"], 1.0)
            self.assertEqual(controlled["cgsr"], 1.0)


if __name__ == "__main__":
    unittest.main()
