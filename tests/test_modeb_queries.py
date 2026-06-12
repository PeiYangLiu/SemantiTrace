from __future__ import annotations

import unittest

from semantitrace.modeb_queries import (
    build_modeb_scene_hook_queries,
    filter_modeb_audit_queries,
    modeb_query_leaks_signature,
)


class ModeBQueryTests(unittest.TestCase):
    def test_scene_hook_queries_avoid_full_signature_leak(self) -> None:
        record = {
            "scene_caption": "street scene with a yellow taxi",
            "nontext_plan": {
                "object_class": "water bottle",
                "color": "blue",
                "position_region": "lower left",
                "surface": "pavement beside the yellow taxi",
                "placement_notes": "on the pavement next to the driver's side door of the yellow taxi",
            },
        }

        queries = build_modeb_scene_hook_queries(record)

        self.assertEqual(len(queries), 3)
        joined = " ".join(queries).lower()
        self.assertIn("yellow taxi", joined)
        self.assertIn("pavement", joined)
        self.assertNotIn("blue", joined)
        self.assertNotIn("water bottle", joined)
        self.assertNotIn("lower left", joined)
        self.assertTrue(any("small" in query.lower() for query in queries))

    def test_query_filter_rejects_signature_leaks(self) -> None:
        record = {
            "trap_signature": "lower left blue water bottle",
            "scene_caption": "street scene with a yellow taxi",
            "nontext_plan": {
                "object_class": "water bottle",
                "color": "blue",
                "position_region": "lower left",
                "surface": "pavement beside the yellow taxi",
                "placement_notes": "on the pavement next to the driver's side door of the yellow taxi",
            },
        }

        self.assertTrue(modeb_query_leaks_signature("Is there a blue water bottle in the lower left?", record))
        self.assertTrue(
            modeb_query_leaks_signature(
                "What color is the water bottle near the taxi?",
                record,
                allow_object_term=False,
            )
        )
        self.assertFalse(
            modeb_query_leaks_signature(
                "What color is the water bottle near the taxi?",
                record,
                allow_object_term=True,
            )
        )
        self.assertTrue(
            modeb_query_leaks_signature(
                "What color is the blue water bottle near the taxi?",
                record,
                allow_object_term=True,
            )
        )
        filtered = filter_modeb_audit_queries(
            [
                "Is there a blue water bottle in the lower left?",
                "In the taxi street scene, what small standalone item is visible on the pavement near the vehicle?",
            ],
            record,
            num_queries=1,
        )

        self.assertEqual(len(filtered), 1)
        self.assertNotIn("blue", filtered[0].lower())
        self.assertNotIn("water bottle", filtered[0].lower())


if __name__ == "__main__":
    unittest.main()
