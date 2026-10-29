"""Time-based gesture-state tests using synthetic MediaPipe-like data."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portal_app.gesture_control import (  # noqa: E402
    GestureController,
    GesturePhase,
)


def make_hand(*, is_open: bool, x: float = 0.5):
    return SimpleNamespace(is_open=is_open, center=(x, 0.5))


def make_result(*hands, labels=()):
    handedness = [
        [SimpleNamespace(category_name=label)]
        for label in labels
    ]
    return SimpleNamespace(hand_landmarks=list(hands), handedness=handedness)


def make_controller() -> GestureController:
    return GestureController(
        lambda hand: hand.is_open,
        lambda hand: hand.center,
        charge_duration_ms=800.0,
        cooldown_duration_ms=300.0,
        track_timeout_ms=700.0,
    )


class GestureControllerTests(unittest.TestCase):
    def test_open_hand_charges_using_elapsed_time_then_activates(self) -> None:
        controller = make_controller()
        hand = make_hand(is_open=True)
        result = make_result(hand, labels=("Left",))

        first = controller.update(result, 0)
        halfway = controller.update(result, 400)
        active = controller.update(result, 800)

        self.assertEqual(first.statuses[0].phase, GesturePhase.CHARGING)
        self.assertEqual(first.statuses[0].progress, 0.0)
        self.assertEqual(halfway.statuses[0].progress, 0.5)
        self.assertEqual(active.statuses[0].phase, GesturePhase.ACTIVE)
        self.assertEqual(active.active_landmarks, (hand,))

    def test_repeated_calls_without_elapsed_time_do_not_charge(self) -> None:
        controller = make_controller()
        hand = make_hand(is_open=True)
        result = make_result(hand, labels=("Left",))

        frames = [controller.update(result, 100) for _ in range(20)]

        self.assertTrue(
            all(status.statuses[0].phase is GesturePhase.CHARGING for status in frames)
        )
        self.assertEqual(frames[-1].statuses[0].progress, 0.0)

    def test_closed_hand_cancels_charge(self) -> None:
        controller = make_controller()
        open_hand = make_hand(is_open=True)
        closed_hand = make_hand(is_open=False)
        controller.update(make_result(open_hand, labels=("Left",)), 0)
        controller.update(make_result(open_hand, labels=("Left",)), 400)

        result = controller.update(make_result(closed_hand, labels=("Left",)), 500)

        self.assertEqual(result.statuses[0].phase, GesturePhase.INACTIVE)
        self.assertFalse(result.active_landmarks)

    def test_active_hand_uses_cooldown_before_deactivation(self) -> None:
        controller = make_controller()
        open_hand = make_hand(is_open=True)
        closed_hand = make_hand(is_open=False)
        controller.update(make_result(open_hand, labels=("Left",)), 0)
        controller.update(make_result(open_hand, labels=("Left",)), 800)

        cooling = controller.update(make_result(closed_hand, labels=("Left",)), 900)
        inactive = controller.update(make_result(closed_hand, labels=("Left",)), 1200)

        self.assertEqual(cooling.statuses[0].phase, GesturePhase.COOLDOWN)
        self.assertEqual(cooling.active_landmarks, (closed_hand,))
        self.assertEqual(inactive.statuses[0].phase, GesturePhase.INACTIVE)
        self.assertFalse(inactive.active_landmarks)

    def test_reopening_during_cooldown_restores_active_state(self) -> None:
        controller = make_controller()
        open_hand = make_hand(is_open=True)
        closed_hand = make_hand(is_open=False)
        controller.update(make_result(open_hand, labels=("Left",)), 0)
        controller.update(make_result(open_hand, labels=("Left",)), 800)
        controller.update(make_result(closed_hand, labels=("Left",)), 900)

        result = controller.update(make_result(open_hand, labels=("Left",)), 1000)

        self.assertEqual(result.statuses[0].phase, GesturePhase.ACTIVE)

    def test_two_hands_charge_independently_by_handedness(self) -> None:
        controller = make_controller()
        left = make_hand(is_open=True, x=0.3)
        right = make_hand(is_open=False, x=0.7)
        result = make_result(left, right, labels=("Left", "Right"))

        controller.update(result, 0)
        updated = controller.update(result, 800)

        phases = {status.key: status.phase for status in updated.statuses}
        self.assertEqual(phases["left"], GesturePhase.ACTIVE)
        self.assertEqual(phases["right"], GesturePhase.INACTIVE)
        self.assertEqual(updated.active_landmarks, (left,))

    def test_timed_out_track_must_charge_again(self) -> None:
        controller = make_controller()
        hand = make_hand(is_open=True)
        result = make_result(hand, labels=("Left",))
        controller.update(result, 0)
        controller.update(result, 800)
        controller.update(make_result(), 1600)

        returned = controller.update(result, 1700)

        self.assertEqual(returned.statuses[0].phase, GesturePhase.CHARGING)
        self.assertFalse(returned.active_landmarks)


if __name__ == "__main__":
    unittest.main()
