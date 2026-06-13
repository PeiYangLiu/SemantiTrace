from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LocalCleanOnlyPipelineTests(unittest.TestCase):
    def test_local_cleanonly_pipeline_runs_without_external_services(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "local_pipeline"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_local_cleanonly_pipeline.py",
                    "--output_dir",
                    str(out),
                    "--num_mode_a",
                    "2",
                    "--num_mode_b",
                    "2",
                    "--num_distractors",
                    "8",
                    "--top_k",
                    "3",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.DEVNULL,
            )

            summary_path = out / "local_cleanonly_summary.json"
            details_path = out / "local_cleanonly_details.jsonl"
            records_path = out / "local_cleanonly_records.json"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(details_path.is_file())
            self.assertTrue(records_path.is_file())

            rows = json.loads(summary_path.read_text(encoding="utf-8"))
            by_key = {(row["profile"], row["subset"]): row for row in rows}
            self.assertEqual(
                by_key[("local_visual_cleanonly", "all")]["signal_name"],
                "composite_cleanonly_protocol",
            )
            self.assertEqual(by_key[("local_visual_cleanonly", "all")]["audit_signal"], 1.0)
            self.assertEqual(by_key[("local_visual_cleanonly", "all")]["clean_baseline"], 0.0)
            self.assertEqual(
                by_key[("local_visual_cleanonly", "mode_b")]["signal_name"],
                "protected_image_hit",
            )
            self.assertEqual(by_key[("caption_only", "all")]["audit_signal"], 0.0)
            self.assertEqual(by_key[("caption_sidecar", "all")]["audit_signal"], 1.0)


if __name__ == "__main__":
    unittest.main()

