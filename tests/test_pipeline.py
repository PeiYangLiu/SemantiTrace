from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from semantitrace import SemantiTracePipeline


class PipelineTests(unittest.TestCase):
    def test_parse_semantic_filter_box_label_ids(self) -> None:
        raw = '{"valid_box_ids": ["Box 1", 3], "rejected": {"2": "bad perspective"}}'
        valid_ids, rejected = SemantiTracePipeline._parse_semantic_filter(raw)
        self.assertEqual(valid_ids, [1, 3])
        self.assertEqual(rejected["2"], "bad perspective")

    def test_dryrun_injection_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root / "images"
            out = root / "out"
            images.mkdir()
            for idx, color in enumerate([(220, 40, 40), (40, 220, 40), (40, 40, 220), (180, 180, 40)]):
                image = Image.new("RGB", (128, 128), color)
                draw = ImageDraw.Draw(image)
                draw.rectangle((30, 30, 98, 98), outline=(255, 255, 255), width=2)
                image.save(images / f"img{idx}.png")

            pipeline = SemantiTracePipeline("configs/default.yaml", device="cpu")
            records = pipeline.inject_canaries(images, out, num_canaries=2)
            self.assertGreaterEqual(len(records), 1)
            self.assertTrue((out / "canary_records.json").is_file())
            loaded = json.loads((out / "canary_records.json").read_text())
            self.assertEqual(len(loaded), len(records))
            for record in loaded:
                self.assertTrue(Path(record["watermarked_image_path"]).is_file())
                self.assertIn("trap_signature", record)
                self.assertEqual(len(record["probe_queries"]), 3)


if __name__ == "__main__":
    unittest.main()
