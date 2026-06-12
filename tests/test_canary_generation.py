from __future__ import annotations

import unittest
from typing import Any

from PIL import Image

from semantitrace.anchor_mining import Canvas
from semantitrace.canary_generation import CanaryGenerator
from semantitrace.utils.image import mask_from_bbox


class BoxLabelVLM:
    def generate(self, image: Image.Image | None, prompt: str, **kwargs: Any) -> str:
        return """
        {
          "selected_box_id": "Box 1",
          "parasitism_mode": "Text Mutation",
          "reasoning": "Use the second box.",
          "T_trig": "mutate the text KLUB to GAWA while preserving typography",
          "S_trap": "GAWA",
          "scene_description": "a sports badge"
        }
        """

    def score_text(self, image: Image.Image, prompt: str, target_text: str) -> float:
        return 0.0


class CanaryGenerationTests(unittest.TestCase):
    def test_parse_box_label_id(self) -> None:
        image = Image.new("RGB", (128, 128), "white")
        canvases = [
            Canvas(0, "text", mask_from_bbox(image.size, (10, 10, 50, 30)), (10, 10, 50, 30), 1.0, text="RIga"),
            Canvas(1, "text", mask_from_bbox(image.size, (60, 10, 110, 30)), (60, 10, 110, 30), 1.0, text="KLUB"),
        ]
        generator = CanaryGenerator(BoxLabelVLM(), {"vlm_temperature": 0.0})

        canary = generator.generate_canary(image, canvases, "text")

        self.assertEqual(canary["selected_box_id"], 1)
        self.assertEqual(canary["selected_canvas"].text, "KLUB")
        self.assertEqual(canary["trap_signature"], "GAWA")

    def test_style_matched_signature_rejects_source_echo_and_hard_letters(self) -> None:
        image = Image.new("RGB", (128, 128), "white")
        canvas = Canvas(0, "text", mask_from_bbox(image.size, (10, 10, 80, 30)), (10, 10, 80, 30), 1.0, text="COLOR")
        generator = CanaryGenerator(
            BoxLabelVLM(),
            {
                "match_text_signature_length": True,
                "acronym_length": [3, 5],
                "forbidden_signature_letters": "JQX",
            },
            seed=7,
        )

        signature = generator._style_matched_signature(canvas, "Text Mutation", "QXJ")
        echoed = generator._style_matched_signature(canvas, "Text Mutation", "COLOR")

        self.assertEqual(len(signature), 5)
        self.assertNotIn("Q", signature)
        self.assertNotIn("X", signature)
        self.assertNotIn("J", signature)
        self.assertNotEqual(echoed, "COLOR")
        self.assertEqual(len(echoed), 5)


if __name__ == "__main__":
    unittest.main()
