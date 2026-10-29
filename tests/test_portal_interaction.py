"""Tests for portal growth and the two-hand merge energy burst."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portal_app.portal_interaction import PortalInteractionController  # noqa: E402


class PortalInteractionControllerTests(unittest.TestCase):
    def test_portal_grows_with_elapsed_time_and_ease_out(self) -> None:
        controller = PortalInteractionController()
        centers = [(100, 100), (300, 100)]

        start = controller.update(centers, 0)
        halfway = controller.update(centers, 300)
        complete = controller.update(centers, 600)

        self.assertEqual(start.portal_scale, 0.0)
        self.assertAlmostEqual(halfway.portal_scale, 0.875)
        self.assertEqual(complete.portal_scale, 1.0)

    def test_merge_triggers_once_until_hands_separate_again(self) -> None:
        controller = PortalInteractionController()
        far = [(100, 100), (260, 100)]
        merged = [(150, 100), (200, 100)]

        self.assertFalse(controller.update(far, 0).burst_triggered)
        first = controller.update(merged, 100)
        repeated = controller.update(merged, 200)
        controller.update(far, 300)
        second = controller.update(merged, 400)

        self.assertTrue(first.burst_triggered)
        self.assertEqual(first.burst.center, (175, 100))
        self.assertFalse(repeated.burst_triggered)
        self.assertTrue(second.burst_triggered)

    def test_burst_expands_then_expires(self) -> None:
        controller = PortalInteractionController()
        controller.update([(100, 100), (260, 100)], 0)
        controller.update([(150, 100), (200, 100)], 100)

        halfway = controller.update([], 475)
        expired = controller.update([], 851)

        self.assertIsNotNone(halfway.burst)
        self.assertEqual(halfway.burst.progress, 0.5)
        self.assertIsNone(expired.burst)

    def test_losing_two_hands_resets_growth(self) -> None:
        controller = PortalInteractionController()
        centers = [(100, 100), (300, 100)]
        controller.update(centers, 0)
        controller.update(centers, 600)
        controller.update([], 700)

        returned = controller.update(centers, 800)

        self.assertEqual(returned.portal_scale, 0.0)

    def test_invalid_configuration_and_timestamp_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PortalInteractionController(merge_distance=100, rearm_distance=90)
        controller = PortalInteractionController()
        with self.assertRaises(ValueError):
            controller.update([], -1)


if __name__ == "__main__":
    unittest.main()
