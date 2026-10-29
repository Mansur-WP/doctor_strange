"""Unit checks for hand visualization that do not require a webcam."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portal_app.hand_tracking import (  # noqa: E402
    HAND_CONNECTIONS,
    calculate_arc_ranges,
    calculate_palm_center,
    calculate_palm_centers,
    calculate_ring_radius,
    calculate_rotation_angle,
    calculate_two_hand_portal,
    calculate_tick_segments,
    draw_hand_landmarks,
    draw_gesture_status,
    draw_energy_arcs,
    draw_magic_glow,
    draw_magic_burst,
    draw_magic_rings,
    draw_palm_centers,
    draw_rotating_ring_details,
    draw_two_hand_portal,
    is_open_palm,
    normalized_to_pixel,
    select_open_hands,
)
from portal_app.particles import ParticleSystem  # noqa: E402
from portal_app.gesture_control import GesturePhase, GestureStatus  # noqa: E402
from portal_app.portal_interaction import BurstVisual  # noqa: E402


class CoordinateConversionTests(unittest.TestCase):
    def test_converts_normalized_coordinates_to_pixels(self) -> None:
        self.assertEqual(normalized_to_pixel(0.0, 0.0, 640, 480), (0, 0))
        self.assertEqual(normalized_to_pixel(1.0, 1.0, 640, 480), (639, 479))
        self.assertEqual(normalized_to_pixel(0.5, 0.5, 101, 81), (50, 40))

    def test_clamps_coordinates_outside_the_image(self) -> None:
        self.assertEqual(normalized_to_pixel(-0.2, 1.2, 100, 50), (0, 49))


class LandmarkDrawingTests(unittest.TestCase):
    def test_connection_map_references_all_21_landmarks(self) -> None:
        connected_indices = {index for pair in HAND_CONNECTIONS for index in pair}

        self.assertEqual(connected_indices, set(range(21)))

    def test_no_hands_leaves_frame_unchanged(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        draw_hand_landmarks(frame, SimpleNamespace(hand_landmarks=[]))

        self.assertFalse(frame.any())

    def test_draws_synthetic_landmarks_and_connections(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [
            SimpleNamespace(x=0.1 + index * 0.035, y=0.2 + index * 0.02)
            for index in range(21)
        ]

        draw_hand_landmarks(
            frame,
            SimpleNamespace(hand_landmarks=[landmarks]),
        )

        self.assertTrue(frame.any())
        self.assertGreater(np.count_nonzero(frame), 21)


def make_landmarks(
    palm_coordinates: dict[int, tuple[float, float]],
) -> list[SimpleNamespace]:
    """Build 21 synthetic landmarks with selected palm coordinates."""

    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    for index, (x, y) in palm_coordinates.items():
        landmarks[index] = SimpleNamespace(x=x, y=y)
    return landmarks


def make_gesture_landmarks(extended_fingers: int) -> list[SimpleNamespace]:
    """Build a synthetic upright hand with a chosen number of open fingers."""

    landmarks = [SimpleNamespace(x=0.5, y=0.65) for _ in range(21)]
    landmarks[0] = SimpleNamespace(x=0.5, y=0.85)
    finger_pairs = ((6, 8), (10, 12), (14, 16), (18, 20))
    finger_x_values = (0.35, 0.45, 0.55, 0.65)
    for finger_number, ((joint_index, tip_index), finger_x) in enumerate(
        zip(finger_pairs, finger_x_values)
    ):
        landmarks[joint_index] = SimpleNamespace(x=finger_x, y=0.55)
        tip_y = 0.2 if finger_number < extended_fingers else 0.68
        landmarks[tip_index] = SimpleNamespace(x=finger_x, y=tip_y)
    return landmarks


class GestureActivationTests(unittest.TestCase):
    def test_open_palm_requires_at_least_three_extended_fingers(self) -> None:
        self.assertTrue(is_open_palm(make_gesture_landmarks(4)))
        self.assertTrue(is_open_palm(make_gesture_landmarks(3)))
        self.assertFalse(is_open_palm(make_gesture_landmarks(2)))
        self.assertFalse(is_open_palm(make_gesture_landmarks(0)))

    def test_invalid_landmarks_are_not_treated_as_open(self) -> None:
        self.assertFalse(is_open_palm([]))
        invalid = make_gesture_landmarks(4)
        invalid[8] = SimpleNamespace(x=float("nan"), y=0.2)
        self.assertFalse(is_open_palm(invalid))

    def test_selects_open_hands_independently(self) -> None:
        open_hand = make_gesture_landmarks(4)
        closed_hand = make_gesture_landmarks(0)

        selected = select_open_hands(
            SimpleNamespace(hand_landmarks=[closed_hand, open_hand])
        )

        self.assertEqual(selected.hand_landmarks, (open_hand,))

    def test_missing_hands_produce_empty_active_result(self) -> None:
        self.assertEqual(select_open_hands(SimpleNamespace()).hand_landmarks, ())

    @patch("portal_app.hand_tracking.cv2.putText")
    @patch("portal_app.hand_tracking.cv2.ellipse")
    def test_draws_charging_progress_and_status(self, ellipse, put_text) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        status = GestureStatus(
            key="left",
            phase=GesturePhase.CHARGING,
            progress=0.5,
            center=(0.5, 0.5),
            landmarks=(),
        )

        draw_gesture_status(frame, (status,))

        self.assertEqual(ellipse.call_count, 1)
        self.assertEqual(ellipse.call_args.args[5], 90.0)
        self.assertEqual(put_text.call_count, 2)
        self.assertEqual(put_text.call_args_list[0].args[1], "CHARGING 50%")


class PalmCenterTests(unittest.TestCase):
    def test_averages_wrist_and_four_finger_bases(self) -> None:
        landmarks = make_landmarks(
            {
                0: (0.4, 0.8),
                5: (0.3, 0.4),
                9: (0.4, 0.3),
                13: (0.5, 0.3),
                17: (0.6, 0.4),
            }
        )

        center = calculate_palm_center(landmarks)

        self.assertIsNotNone(center)
        self.assertAlmostEqual(center[0], 0.44)
        self.assertAlmostEqual(center[1], 0.44)

    def test_returns_none_when_required_landmarks_are_missing(self) -> None:
        incomplete_landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(17)]

        self.assertIsNone(calculate_palm_center(incomplete_landmarks))

    def test_returns_none_for_invalid_coordinates(self) -> None:
        landmarks = make_landmarks({index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)})
        landmarks[9].x = float("nan")

        self.assertIsNone(calculate_palm_center(landmarks))

    def test_calculates_two_hands_independently(self) -> None:
        first_hand = make_landmarks(
            {index: (0.25, 0.4) for index in (0, 5, 9, 13, 17)}
        )
        second_hand = make_landmarks(
            {index: (0.75, 0.6) for index in (0, 5, 9, 13, 17)}
        )
        result = SimpleNamespace(hand_landmarks=[first_hand, second_hand])

        self.assertEqual(
            calculate_palm_centers(result),
            [(0.25, 0.4), (0.75, 0.6)],
        )

    def test_skips_invalid_hand_but_keeps_valid_hand(self) -> None:
        valid_hand = make_landmarks(
            {index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        result = SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=0.1, y=0.1)], valid_hand])

        self.assertEqual(calculate_palm_centers(result), [(0.5, 0.5)])

    def test_draws_one_marker_for_each_valid_hand(self) -> None:
        first_hand = make_landmarks(
            {index: (0.25, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        second_hand = make_landmarks(
            {index: (0.75, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        draw_palm_centers(
            frame,
            SimpleNamespace(hand_landmarks=[first_hand, second_hand]),
        )

        self.assertTrue(frame[50, 50].any())
        self.assertTrue(frame[50, 149].any())


class MagicRingTests(unittest.TestCase):
    def test_frame_relative_radius_uses_shorter_dimension(self) -> None:
        self.assertEqual(calculate_ring_radius(1280, 720, (640, 360)), 94)
        self.assertEqual(calculate_ring_radius(640, 480, (320, 240)), 62)

    def test_radius_shrinks_near_image_boundary(self) -> None:
        self.assertEqual(calculate_ring_radius(100, 100, (17, 17)), 1)
        self.assertEqual(calculate_ring_radius(100, 100, (0, 50)), 0)

    def test_rejects_invalid_radius_inputs(self) -> None:
        with self.assertRaises(ValueError):
            calculate_ring_radius(0, 100, (0, 0))
        with self.assertRaises(ValueError):
            calculate_ring_radius(100, 100, (50, 50), radius_scale=0.0)

    @patch("portal_app.hand_tracking.cv2.ellipse")
    def test_draws_ring_at_calculated_palm_center(self, ellipse) -> None:
        hand = make_landmarks(
            {index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_magic_rings(frame, SimpleNamespace(hand_landmarks=[hand]))

        self.assertEqual(ellipse.call_count, 8)
        self.assertEqual(ellipse.call_args_list[0].args[1:3], ((320, 240), (62, 62)))
        self.assertEqual(ellipse.call_args_list[1].args[1:3], ((320, 240), (45, 45)))
        self.assertEqual(ellipse.call_args_list[0].args[6:8], ((80, 240, 255), 1))

    @patch("portal_app.hand_tracking.cv2.ellipse")
    def test_draws_two_independent_rings(self, ellipse) -> None:
        first_hand = make_landmarks(
            {index: (0.25, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        second_hand = make_landmarks(
            {index: (0.75, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_magic_rings(
            frame,
            SimpleNamespace(hand_landmarks=[first_hand, second_hand]),
        )

        centers = [ring_call.args[1] for ring_call in ellipse.call_args_list]
        self.assertEqual(centers.count((160, 240)), 8)
        self.assertEqual(centers.count((479, 240)), 8)

    @patch("portal_app.hand_tracking.cv2.ellipse")
    def test_missing_or_invalid_hands_draw_no_ring(self, ellipse) -> None:
        invalid_hand = [SimpleNamespace(x=0.5, y=0.5)]

        draw_magic_rings(
            np.zeros((100, 100, 3), dtype=np.uint8),
            SimpleNamespace(hand_landmarks=[invalid_hand]),
        )

        ellipse.assert_not_called()


class MagicGlowTests(unittest.TestCase):
    def test_glow_changes_synthetic_frame_and_preserves_image_format(self) -> None:
        hand = make_landmarks(
            {index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        original_shape = frame.shape
        original_dtype = frame.dtype

        glow_layer = draw_magic_glow(
            frame,
            SimpleNamespace(hand_landmarks=[hand]),
            elapsed_ms=500,
        )

        self.assertTrue(frame.any())
        self.assertEqual(frame.shape, original_shape)
        self.assertEqual(frame.dtype, original_dtype)
        self.assertEqual(glow_layer.shape, original_shape)
        self.assertEqual(glow_layer.dtype, original_dtype)

    def test_glow_renders_two_palm_effects(self) -> None:
        first_hand = make_landmarks(
            {index: (0.3, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        second_hand = make_landmarks(
            {index: (0.7, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((240, 400, 3), dtype=np.uint8)

        draw_magic_glow(
            frame,
            SimpleNamespace(hand_landmarks=[first_hand, second_hand]),
            elapsed_ms=1000,
        )

        self.assertTrue(frame[:, :200].any())
        self.assertTrue(frame[:, 200:].any())

    def test_empty_or_invalid_hands_leave_frame_unchanged(self) -> None:
        frame = np.full((100, 100, 3), 25, dtype=np.uint8)
        before = frame.copy()
        invalid_hand = [SimpleNamespace(x=0.5, y=0.5)]

        glow_layer = draw_magic_glow(
            frame,
            SimpleNamespace(hand_landmarks=[invalid_hand]),
            elapsed_ms=0,
        )

        np.testing.assert_array_equal(frame, before)
        self.assertFalse(glow_layer.any())

    def test_reuses_matching_glow_layer(self) -> None:
        hand = make_landmarks(
            {index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        result = SimpleNamespace(hand_landmarks=[hand])
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first_layer = draw_magic_glow(frame, result, elapsed_ms=0)
        second_layer = draw_magic_glow(
            frame,
            result,
            elapsed_ms=100,
            glow_layer=first_layer,
        )

        self.assertIs(second_layer, first_layer)

    def test_lingering_particles_glow_after_hand_disappears(self) -> None:
        particles = ParticleSystem(spawn_rate_per_emitter=40.0)
        particles.update([((80, 60), 25)], 0)
        particles.update([((80, 60), 25)], 100)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        draw_magic_glow(
            frame,
            SimpleNamespace(hand_landmarks=[]),
            elapsed_ms=150,
            particles=particles,
        )

        self.assertTrue(frame.any())


class RotatingRingDetailTests(unittest.TestCase):
    def test_broken_arc_ranges_rotate_and_leave_gaps(self) -> None:
        ranges = calculate_arc_ranges(10.0)

        self.assertEqual(len(ranges), 6)
        self.assertEqual(ranges[0], (10.0, 38.0))
        self.assertEqual(ranges[1], (70.0, 98.0))

    def test_invalid_arc_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_arc_ranges(0.0, segment_count=0)
        with self.assertRaises(ValueError):
            calculate_arc_ranges(0.0, segment_count=6, arc_span=60.0)

    @patch("portal_app.hand_tracking.cv2.ellipse")
    def test_draws_outer_and_inner_energy_arcs(self, ellipse) -> None:
        hand = make_landmarks(
            {index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_energy_arcs(
            frame,
            SimpleNamespace(hand_landmarks=[hand]),
            elapsed_ms=1000,
        )

        self.assertEqual(ellipse.call_count, 11)
        self.assertEqual(ellipse.call_args_list[0].args[1:3], ((320, 240), (62, 62)))
        self.assertEqual(ellipse.call_args_list[6].args[1:3], ((320, 240), (45, 45)))

    def test_rotation_angle_changes_with_elapsed_time(self) -> None:
        self.assertEqual(calculate_rotation_angle(0), 0.0)
        self.assertEqual(calculate_rotation_angle(500), 45.0)
        self.assertEqual(calculate_rotation_angle(1000), 90.0)
        self.assertEqual(calculate_rotation_angle(4000), 0.0)

    def test_rotation_depends_on_time_not_number_of_calls(self) -> None:
        repeated_angles = [calculate_rotation_angle(750) for _ in range(20)]

        self.assertEqual(repeated_angles, [67.5] * 20)

    def test_tick_geometry_uses_sine_and_cosine(self) -> None:
        zero_degree_ticks = calculate_tick_segments((100, 100), 40, 0.0)
        ninety_degree_ticks = calculate_tick_segments((100, 100), 40, 90.0)

        self.assertEqual(zero_degree_ticks[0], ((122, 100), (131, 100)))
        self.assertEqual(ninety_degree_ticks[0], ((100, 122), (100, 131)))

    def test_invalid_timing_and_tick_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_rotation_angle(-1)
        with self.assertRaises(ValueError):
            calculate_rotation_angle(100, float("inf"))
        with self.assertRaises(ValueError):
            calculate_tick_segments((50, 50), 10, 0.0, tick_count=0)

    @patch("portal_app.hand_tracking.cv2.line")
    def test_draws_rotating_detail_at_expected_center(self, line) -> None:
        hand = make_landmarks(
            {index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_rotating_ring_details(
            frame,
            SimpleNamespace(hand_landmarks=[hand]),
            elapsed_ms=0,
        )

        self.assertEqual(line.call_count, 8)
        self.assertEqual(line.call_args_list[0].args[1:3], ((354, 240), (368, 240)))

    @patch("portal_app.hand_tracking.cv2.line")
    def test_draws_details_for_two_hands_independently(self, line) -> None:
        first_hand = make_landmarks(
            {index: (0.25, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        second_hand = make_landmarks(
            {index: (0.75, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_rotating_ring_details(
            frame,
            SimpleNamespace(hand_landmarks=[first_hand, second_hand]),
            elapsed_ms=0,
        )

        self.assertEqual(line.call_count, 16)
        self.assertEqual(line.call_args_list[0].args[1], (194, 240))
        self.assertEqual(line.call_args_list[8].args[1], (513, 240))

    @patch("portal_app.hand_tracking.cv2.line")
    def test_invalid_or_missing_hand_draws_no_detail(self, line) -> None:
        invalid_hand = [SimpleNamespace(x=0.5, y=0.5)]

        draw_rotating_ring_details(
            np.zeros((100, 100, 3), dtype=np.uint8),
            SimpleNamespace(hand_landmarks=[invalid_hand]),
            elapsed_ms=1000,
        )

        line.assert_not_called()


class TwoHandPortalTests(unittest.TestCase):
    def test_midpoint_and_radius_follow_two_hand_distance(self) -> None:
        first_hand = make_landmarks(
            {index: (0.25, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        second_hand = make_landmarks(
            {index: (0.75, 0.5) for index in (0, 5, 9, 13, 17)}
        )

        geometry = calculate_two_hand_portal(
            640,
            480,
            SimpleNamespace(hand_landmarks=[first_hand, second_hand]),
        )

        self.assertIsNotNone(geometry)
        self.assertEqual(geometry.center, (320, 240))
        self.assertEqual(geometry.radius, 144)

    def test_requires_two_separated_active_hands(self) -> None:
        hand = make_landmarks(
            {index: (0.5, 0.5) for index in (0, 5, 9, 13, 17)}
        )
        close_hand = make_landmarks(
            {index: (0.55, 0.5) for index in (0, 5, 9, 13, 17)}
        )

        self.assertIsNone(
            calculate_two_hand_portal(
                640,
                480,
                SimpleNamespace(hand_landmarks=[hand]),
            )
        )
        self.assertIsNone(
            calculate_two_hand_portal(
                640,
                480,
                SimpleNamespace(hand_landmarks=[hand, close_hand]),
            )
        )

    @patch("portal_app.hand_tracking.cv2.ellipse")
    def test_draws_three_broken_portal_layers(self, ellipse: Mock) -> None:
        geometry = SimpleNamespace(center=(200, 150), radius=100)

        rendered = draw_two_hand_portal(
            np.zeros((300, 400, 3), dtype=np.uint8),
            geometry,
            elapsed_ms=1000,
        )

        self.assertTrue(rendered)
        self.assertEqual(ellipse.call_count, 25)
        self.assertEqual(ellipse.call_args_list[0].args[1:3], ((200, 150), (100, 100)))
        self.assertEqual(ellipse.call_args_list[10].args[2], (78, 78))

    @patch("portal_app.hand_tracking.cv2.ellipse")
    def test_missing_portal_draws_nothing(self, ellipse: Mock) -> None:
        self.assertFalse(
            draw_two_hand_portal(
                np.zeros((100, 100, 3), dtype=np.uint8),
                None,
                elapsed_ms=0,
            )
        )
        ellipse.assert_not_called()

    @patch("portal_app.hand_tracking.cv2.line")
    @patch("portal_app.hand_tracking.cv2.circle")
    def test_draws_expanding_merge_burst(self, circle: Mock, line: Mock) -> None:
        rendered = draw_magic_burst(
            np.zeros((300, 400, 3), dtype=np.uint8),
            BurstVisual(center=(200, 150), progress=0.5),
        )

        self.assertTrue(rendered)
        self.assertEqual(circle.call_count, 2)
        self.assertEqual(circle.call_args_list[0].args[2], 106)
        self.assertEqual(line.call_count, 16)

    @patch("portal_app.hand_tracking.cv2.circle")
    def test_missing_merge_burst_draws_nothing(self, circle: Mock) -> None:
        self.assertFalse(
            draw_magic_burst(
                np.zeros((100, 100, 3), dtype=np.uint8),
                None,
            )
        )
        circle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
