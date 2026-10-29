"""MediaPipe hand detection, landmark drawing, and palm-center estimation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, isfinite, radians, sin
from pathlib import Path
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np

from portal_app.particles import Emitter, ParticleSystem
from portal_app.gesture_control import GestureController, GesturePhase, GestureStatus
from portal_app.portal_interaction import (
    BurstVisual,
    PortalInteractionController,
)
from portal_app.styles import EffectStyle, GOLD_STYLE, STYLE_PRESETS


# Landmark indices follow MediaPipe's documented 21-point hand layout.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),                                # Outer palm edge
)

# Wrist and the four finger-base knuckles form a stable outline of the palm.
PALM_LANDMARK_INDICES: tuple[int, ...] = (0, 5, 9, 13, 17)
FINGER_JOINT_AND_TIP_INDICES: tuple[tuple[int, int], ...] = (
    (6, 8),
    (10, 12),
    (14, 16),
    (18, 20),
)
OPEN_FINGER_DISTANCE_RATIO = 1.18

RING_RADIUS_SCALE = 0.13
RING_STROKE_MARGIN = 16
INNER_RING_SCALE = 0.72
ROTATING_TICK_COUNT = 8
DEFAULT_ROTATION_SPEED = 90.0


@dataclass(frozen=True)
class HandTrackingConfig:
    """Configuration for the pretrained MediaPipe Hand Landmarker."""

    model_path: Path
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    rotation_speed_degrees_per_second: float = DEFAULT_ROTATION_SPEED

    def validate(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {self.model_path}"
            )
        if self.max_hands <= 0:
            raise ValueError("max_hands must be greater than zero")
        for name, value in (
            ("min_detection_confidence", self.min_detection_confidence),
            ("min_presence_confidence", self.min_presence_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if not isfinite(self.rotation_speed_degrees_per_second):
            raise ValueError("rotation speed must be a finite number")


@dataclass(frozen=True)
class HandLandmarkResult:
    """Minimal result view used to draw effects for selected hands."""

    hand_landmarks: tuple


@dataclass(frozen=True)
class PortalGeometry:
    """Pixel center and radius of a two-hand portal."""

    center: tuple[int, int]
    radius: int


def normalized_to_pixel(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """Convert normalized image coordinates to a valid pixel coordinate."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero")

    pixel_x = round(x * (frame_width - 1))
    pixel_y = round(y * (frame_height - 1))
    return (
        min(max(pixel_x, 0), frame_width - 1),
        min(max(pixel_y, 0), frame_height - 1),
    )


def calculate_palm_center(landmarks) -> tuple[float, float] | None:
    """Return one normalized palm center, or None for incomplete data."""

    if not landmarks or len(landmarks) <= max(PALM_LANDMARK_INDICES):
        return None

    coordinates: list[tuple[float, float]] = []
    for index in PALM_LANDMARK_INDICES:
        try:
            x = float(landmarks[index].x)
            y = float(landmarks[index].y)
        except (AttributeError, TypeError, ValueError):
            return None

        if not isfinite(x) or not isfinite(y):
            return None
        coordinates.append((x, y))

    count = len(coordinates)
    center_x = sum(x for x, _ in coordinates) / count
    center_y = sum(y for _, y in coordinates) / count
    return center_x, center_y


def calculate_palm_centers(result) -> list[tuple[float, float]]:
    """Calculate normalized centers for every valid detected hand."""

    hands = getattr(result, "hand_landmarks", None) or []
    centers = []
    for landmarks in hands:
        center = calculate_palm_center(landmarks)
        if center is not None:
            centers.append(center)
    return centers


def is_open_palm(landmarks) -> bool:
    """Return whether at least three non-thumb fingers appear extended."""

    if not landmarks or len(landmarks) < 21:
        return False

    try:
        wrist_x = float(landmarks[0].x)
        wrist_y = float(landmarks[0].y)
    except (AttributeError, TypeError, ValueError):
        return False
    if not isfinite(wrist_x) or not isfinite(wrist_y):
        return False

    extended_fingers = 0
    ratio_squared = OPEN_FINGER_DISTANCE_RATIO**2
    for joint_index, tip_index in FINGER_JOINT_AND_TIP_INDICES:
        try:
            joint_x = float(landmarks[joint_index].x)
            joint_y = float(landmarks[joint_index].y)
            tip_x = float(landmarks[tip_index].x)
            tip_y = float(landmarks[tip_index].y)
        except (AttributeError, TypeError, ValueError):
            return False
        if not all(isfinite(value) for value in (joint_x, joint_y, tip_x, tip_y)):
            return False

        joint_distance_squared = (
            (joint_x - wrist_x) ** 2 + (joint_y - wrist_y) ** 2
        )
        tip_distance_squared = (
            (tip_x - wrist_x) ** 2 + (tip_y - wrist_y) ** 2
        )
        if tip_distance_squared > joint_distance_squared * ratio_squared:
            extended_fingers += 1

    return extended_fingers >= 3


def select_open_hands(result) -> HandLandmarkResult:
    """Return a lightweight result containing only open-palm hands."""

    hands = getattr(result, "hand_landmarks", None) or []
    return HandLandmarkResult(
        hand_landmarks=tuple(hand for hand in hands if is_open_palm(hand))
    )


def calculate_ring_radius(
    frame_width: int,
    frame_height: int,
    center: tuple[int, int],
    radius_scale: float = RING_RADIUS_SCALE,
) -> int:
    """Return a frame-relative radius that stays inside the image."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero")
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be greater than zero")

    center_x, center_y = center
    default_radius = max(1, round(min(frame_width, frame_height) * radius_scale))
    edge_radius = min(
        center_x,
        frame_width - 1 - center_x,
        center_y,
        frame_height - 1 - center_y,
    ) - RING_STROKE_MARGIN
    return max(0, min(default_radius, edge_radius))


def calculate_rotation_angle(
    elapsed_ms: int | float,
    speed_degrees_per_second: float = DEFAULT_ROTATION_SPEED,
) -> float:
    """Calculate a repeatable rotation angle from elapsed real time."""

    if not isfinite(float(elapsed_ms)) or elapsed_ms < 0:
        raise ValueError("elapsed_ms must be a finite, non-negative number")
    if not isfinite(speed_degrees_per_second):
        raise ValueError("rotation speed must be a finite number")

    elapsed_seconds = elapsed_ms / 1000.0
    return (elapsed_seconds * speed_degrees_per_second) % 360.0


def calculate_tick_segments(
    center: tuple[int, int],
    radius: int,
    rotation_angle: float,
    tick_count: int = ROTATING_TICK_COUNT,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return inner and outer pixel endpoints for rotating radial ticks."""

    if radius <= 0:
        return []
    if tick_count <= 0:
        raise ValueError("tick_count must be greater than zero")

    center_x, center_y = center
    inner_radius = max(1, round(radius * 0.55))
    outer_radius = max(inner_radius + 1, round(radius * 0.78))
    angle_step = 360.0 / tick_count
    segments = []

    for tick_index in range(tick_count):
        angle = radians(rotation_angle + tick_index * angle_step)
        start = (
            center_x + round(inner_radius * cos(angle)),
            center_y + round(inner_radius * sin(angle)),
        )
        end = (
            center_x + round(outer_radius * cos(angle)),
            center_y + round(outer_radius * sin(angle)),
        )
        segments.append((start, end))

    return segments


def calculate_arc_ranges(
    rotation_angle: float,
    segment_count: int = 6,
    arc_span: float = 28.0,
) -> list[tuple[float, float]]:
    """Return evenly spaced incomplete arc angles for an energy ring."""

    if segment_count <= 0:
        raise ValueError("segment_count must be greater than zero")
    if not 0.0 < arc_span < 360.0 / segment_count:
        raise ValueError("arc_span must fit inside one segment")
    step = 360.0 / segment_count
    return [
        (rotation_angle + index * step, rotation_angle + index * step + arc_span)
        for index in range(segment_count)
    ]


def draw_energy_arcs(
    frame: np.ndarray,
    result,
    elapsed_ms: int | float,
    speed_degrees_per_second: float = DEFAULT_ROTATION_SPEED,
    *,
    glow: bool = False,
    style: EffectStyle = GOLD_STYLE,
    radius_scale: float = RING_RADIUS_SCALE,
) -> None:
    """Draw broken outer and counter-rotating inner energy arcs."""

    angle = calculate_rotation_angle(elapsed_ms, speed_degrees_per_second)
    frame_height, frame_width = frame.shape[:2]
    outer_color = style.glow if glow else style.core
    inner_color = style.secondary
    outer_thickness = 7 if glow else 2
    inner_thickness = 5 if glow else 1

    for center_x, center_y in calculate_palm_centers(result):
        center = normalized_to_pixel(center_x, center_y, frame_width, frame_height)
        radius = calculate_ring_radius(
            frame_width, frame_height, center, radius_scale=radius_scale
        )
        if radius <= 0:
            continue
        for start_angle, end_angle in calculate_arc_ranges(angle):
            cv2.ellipse(
                frame,
                center,
                (radius, radius),
                0.0,
                start_angle,
                end_angle,
                outer_color,
                outer_thickness,
                cv2.LINE_AA,
            )
        inner_radius = max(1, round(radius * INNER_RING_SCALE))
        for start_angle, end_angle in calculate_arc_ranges(
            -angle * 0.7,
            segment_count=5,
            arc_span=35.0,
        ):
            cv2.ellipse(
                frame,
                center,
                (inner_radius, inner_radius),
                0.0,
                start_angle,
                end_angle,
                inner_color,
                inner_thickness,
                cv2.LINE_AA,
            )


def calculate_two_hand_portal(
    frame_width: int,
    frame_height: int,
    result,
    size_multiplier: float = 1.0,
) -> PortalGeometry | None:
    """Calculate a midpoint portal whose radius follows active-hand distance."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero")
    centers = calculate_palm_centers(result)
    if len(centers) < 2:
        return None
    first = normalized_to_pixel(*centers[0], frame_width, frame_height)
    second = normalized_to_pixel(*centers[1], frame_width, frame_height)
    distance = hypot(second[0] - first[0], second[1] - first[1])
    if distance < 80.0:
        return None

    center = (
        round((first[0] + second[0]) / 2.0),
        round((first[1] + second[1]) / 2.0),
    )
    if size_multiplier <= 0.0:
        raise ValueError("size_multiplier must be greater than zero")
    desired_radius = max(30, round(distance * 0.48 * size_multiplier))
    frame_limit = round(min(frame_width, frame_height) * 0.3)
    edge_limit = min(
        center[0],
        frame_width - 1 - center[0],
        center[1],
        frame_height - 1 - center[1],
    ) - RING_STROKE_MARGIN
    radius = min(desired_radius, frame_limit, edge_limit)
    if radius < 30:
        return None
    return PortalGeometry(center, radius)


def draw_two_hand_portal(
    frame: np.ndarray,
    geometry: PortalGeometry | None,
    elapsed_ms: int | float,
    speed_degrees_per_second: float = DEFAULT_ROTATION_SPEED,
    *,
    glow: bool = False,
    style: EffectStyle = GOLD_STYLE,
) -> bool:
    """Draw three counter-rotating broken arc layers at the hand midpoint."""

    if geometry is None:
        return False
    angle = calculate_rotation_angle(elapsed_ms, speed_degrees_per_second)
    layers = (
        (1.0, angle * 0.65, 10, 21.0),
        (0.78, -angle * 0.9, 8, 27.0),
        (0.55, angle * 1.25, 7, 31.0),
    )
    colors = (
        (style.glow, style.secondary, style.accent)
        if glow
        else (style.core, style.secondary, style.accent)
    )

    for layer_index, (scale, layer_angle, count, span) in enumerate(layers):
        radius = max(1, round(geometry.radius * scale))
        thickness = (7 - layer_index * 2) if glow else (2 if layer_index == 0 else 1)
        for start_angle, end_angle in calculate_arc_ranges(
            layer_angle,
            segment_count=count,
            arc_span=span,
        ):
            cv2.ellipse(
                frame,
                geometry.center,
                (radius, radius),
                0.0,
                start_angle,
                end_angle,
                colors[layer_index],
                thickness,
                cv2.LINE_AA,
            )
    return True


def draw_magic_burst(
    frame: np.ndarray,
    burst: BurstVisual | None,
    *,
    glow: bool = False,
    style: EffectStyle = GOLD_STYLE,
) -> bool:
    """Draw an expanding shockwave, bright core, and radial energy rays."""

    if burst is None:
        return False
    fade = max(0.0, 1.0 - burst.progress)
    shock_radius = 18 + round(175 * burst.progress)
    core_radius = max(2, round(24 * fade))

    if glow:
        shock_color = tuple(round(channel * fade) for channel in style.glow)
        ray_color = tuple(round(channel * fade) for channel in style.accent)
        cv2.circle(
            frame,
            burst.center,
            shock_radius,
            shock_color,
            12,
            cv2.LINE_AA,
        )
        cv2.circle(frame, burst.center, core_radius * 2, ray_color, -1, cv2.LINE_AA)
        ray_thickness = 6
    else:
        shock_color = tuple(round(channel * fade) for channel in style.core)
        ray_color = tuple(round(channel * fade) for channel in style.accent)
        cv2.circle(
            frame,
            burst.center,
            shock_radius,
            shock_color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            burst.center,
            core_radius,
            tuple(round(channel * fade) for channel in style.spark_head),
            -1,
            cv2.LINE_AA,
        )
        ray_thickness = 1

    for ray_index in range(16):
        angle = radians(ray_index * 22.5 + burst.progress * 35.0)
        inner_radius = max(2, round(shock_radius * 0.2))
        outer_radius = max(inner_radius + 1, round(shock_radius * 0.88))
        start = (
            burst.center[0] + round(inner_radius * cos(angle)),
            burst.center[1] + round(inner_radius * sin(angle)),
        )
        end = (
            burst.center[0] + round(outer_radius * cos(angle)),
            burst.center[1] + round(outer_radius * sin(angle)),
        )
        cv2.line(frame, start, end, ray_color, ray_thickness, cv2.LINE_AA)
    return True


def draw_hand_landmarks(frame: np.ndarray, result) -> None:
    """Draw all detected hand points and their anatomical connections."""

    hands = getattr(result, "hand_landmarks", None) or []
    frame_height, frame_width = frame.shape[:2]

    for landmarks in hands:
        points = [
            normalized_to_pixel(point.x, point.y, frame_width, frame_height)
            for point in landmarks
        ]

        for start_index, end_index in HAND_CONNECTIONS:
            if start_index < len(points) and end_index < len(points):
                cv2.line(
                    frame,
                    points[start_index],
                    points[end_index],
                    (80, 220, 80),
                    2,
                    cv2.LINE_AA,
                )

        for point in points:
            cv2.circle(frame, point, 4, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_palm_centers(frame: np.ndarray, result) -> None:
    """Draw a temporary high-contrast marker at each valid palm center."""

    frame_height, frame_width = frame.shape[:2]
    for center_x, center_y in calculate_palm_centers(result):
        center = normalized_to_pixel(
            center_x,
            center_y,
            frame_width,
            frame_height,
        )
        cv2.circle(frame, center, 12, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(frame, center, 8, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.drawMarker(
            frame,
            center,
            (255, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
            cv2.LINE_AA,
        )


def draw_gesture_status(
    frame: np.ndarray,
    statuses: tuple[GestureStatus, ...],
    style: EffectStyle = GOLD_STYLE,
) -> None:
    """Draw charging progress and concise state labels near visible palms."""

    frame_height, frame_width = frame.shape[:2]
    for status in statuses:
        center = normalized_to_pixel(
            status.center[0],
            status.center[1],
            frame_width,
            frame_height,
        )
        if status.phase is GesturePhase.CHARGING:
            end_angle = -90.0 + 360.0 * status.progress
            cv2.ellipse(
                frame,
                center,
                (24, 24),
                0.0,
                -90.0,
                end_angle,
                style.secondary,
                3,
                cv2.LINE_AA,
            )
            label = f"CHARGING {round(status.progress * 100):d}%"
        elif status.phase is GesturePhase.ACTIVE:
            label = "ACTIVE"
        elif status.phase is GesturePhase.COOLDOWN:
            label = "HOLD OPEN"
        else:
            label = "OPEN PALM"

        label_origin = (
            min(max(center[0] - 55, 5), max(5, frame_width - 145)),
            min(max(center[1] + 42, 20), frame_height - 8),
        )
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )


def draw_runtime_overlay(
    frame: np.ndarray,
    style: EffectStyle,
    *,
    show_help: bool,
    debug_lines: tuple[str, ...] = (),
) -> None:
    """Draw concise controls and optional diagnostic values."""

    lines = [f"STYLE: {style.name}"]
    if show_help:
        lines.extend(
            (
                "1-5 Style   +/- Intensity   [/] Size",
                "Open palm: charge   Two hands: portal",
                "Separate + merge: burst   S/R: capture",
                "H: Help   D: Debug   Q/Esc: Quit",
            )
        )
    lines.extend(debug_lines)
    x = max(10, frame.shape[1] - 385)
    for index, line in enumerate(lines):
        y = 25 + index * 21
        cv2.putText(
            frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, style.core, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            style.accent,
            1,
            cv2.LINE_AA,
        )


def draw_magic_glow(
    frame: np.ndarray,
    result,
    elapsed_ms: int | float,
    speed_degrees_per_second: float = DEFAULT_ROTATION_SPEED,
    glow_layer: np.ndarray | None = None,
    particles: ParticleSystem | None = None,
    portal_geometry: PortalGeometry | None = None,
    burst: BurstVisual | None = None,
    style: EffectStyle = GOLD_STYLE,
    glow_strength: float = 0.45,
    radius_scale: float = RING_RADIUS_SCALE,
) -> np.ndarray:
    """Blend a reusable, softly blurred ring-and-tick overlay into a frame."""

    frame_height, frame_width = frame.shape[:2]
    if (
        glow_layer is None
        or glow_layer.shape != frame.shape
        or glow_layer.dtype != frame.dtype
    ):
        glow_layer = np.zeros_like(frame)
    else:
        glow_layer.fill(0)

    rotation_angle = calculate_rotation_angle(
        elapsed_ms,
        speed_degrees_per_second,
    )
    has_effect = False

    for center_x, center_y in calculate_palm_centers(result):
        center = normalized_to_pixel(
            center_x,
            center_y,
            frame_width,
            frame_height,
        )
        radius = calculate_ring_radius(
            frame_width, frame_height, center, radius_scale=radius_scale
        )
        if radius <= 0:
            continue

        has_effect = True

        # Only broken arcs receive glow; complete circles made the effect look
        # like a thick tube instead of separate blades of magical energy.
        for start, end in calculate_tick_segments(center, radius, rotation_angle):
            cv2.line(glow_layer, start, end, style.glow, 3, cv2.LINE_AA)

    draw_energy_arcs(
        glow_layer,
        result,
        elapsed_ms,
        speed_degrees_per_second,
        glow=True,
        style=style,
        radius_scale=radius_scale,
    )
    has_effect = draw_two_hand_portal(
        glow_layer,
        portal_geometry,
        elapsed_ms,
        speed_degrees_per_second,
        glow=True,
        style=style,
    ) or has_effect
    has_effect = draw_magic_burst(
        glow_layer, burst, glow=True, style=style
    ) or has_effect

    if particles is not None:
        has_effect = particles.draw_glow(glow_layer) or has_effect

    if not has_effect:
        return glow_layer

    cv2.GaussianBlur(
        glow_layer,
        (0, 0),
        sigmaX=10,
        sigmaY=10,
        dst=glow_layer,
    )
    cv2.addWeighted(frame, 1.0, glow_layer, glow_strength, 0.0, dst=frame)
    return glow_layer


def draw_magic_rings(
    frame: np.ndarray,
    result,
    style: EffectStyle = GOLD_STYLE,
    radius_scale: float = RING_RADIUS_SCALE,
) -> None:
    """Draw sparse one-pixel arc foundations instead of complete circles."""

    frame_height, frame_width = frame.shape[:2]
    for center_x, center_y in calculate_palm_centers(result):
        center = normalized_to_pixel(
            center_x,
            center_y,
            frame_width,
            frame_height,
        )
        radius = calculate_ring_radius(
            frame_width, frame_height, center, radius_scale=radius_scale
        )
        if radius <= 0:
            continue

        inner_radius = max(1, round(radius * INNER_RING_SCALE))
        # Four tiny stationary accents provide structure without a solid band.
        for start_angle in (15.0, 105.0, 195.0, 285.0):
            cv2.ellipse(
                frame,
                center,
                (radius, radius),
                0.0,
                start_angle,
                start_angle + 12.0,
                style.core,
                1,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                frame,
                center,
                (inner_radius, inner_radius),
                0.0,
                start_angle + 35.0,
                start_angle + 44.0,
                style.secondary,
                1,
                cv2.LINE_AA,
            )


def draw_rotating_ring_details(
    frame: np.ndarray,
    result,
    elapsed_ms: int | float,
    speed_degrees_per_second: float = DEFAULT_ROTATION_SPEED,
    style: EffectStyle = GOLD_STYLE,
    radius_scale: float = RING_RADIUS_SCALE,
) -> None:
    """Draw evenly spaced radial ticks rotated according to elapsed time."""

    rotation_angle = calculate_rotation_angle(
        elapsed_ms,
        speed_degrees_per_second,
    )
    frame_height, frame_width = frame.shape[:2]

    for center_x, center_y in calculate_palm_centers(result):
        center = normalized_to_pixel(
            center_x,
            center_y,
            frame_width,
            frame_height,
        )
        radius = calculate_ring_radius(
            frame_width, frame_height, center, radius_scale=radius_scale
        )
        for start, end in calculate_tick_segments(center, radius, rotation_angle):
            cv2.line(frame, start, end, style.accent, 1, cv2.LINE_AA)


class HandTracker:
    """Detect and draw up to two hands using MediaPipe video tracking."""

    def __init__(self, config: HandTrackingConfig) -> None:
        config.validate()
        self.config = config
        self._last_timestamp_ms = -1
        self._glow_layer: np.ndarray | None = None
        self._particles = ParticleSystem()
        self._gesture_controller = GestureController(
            is_open_palm,
            calculate_palm_center,
        )
        self._portal_interaction = PortalInteractionController()

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(config.model_path.resolve())
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """Detect hands in a BGR frame and draw the results in place."""

        animation_timestamp_ms = max(0, timestamp_ms)

        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(media_image, timestamp_ms)
        gesture_frame = self._gesture_controller.update(
            result,
            animation_timestamp_ms,
        )
        active_result = HandLandmarkResult(gesture_frame.active_landmarks)
        frame_height, frame_width = frame.shape[:2]
        raw_portal_geometry = calculate_two_hand_portal(
            frame_width,
            frame_height,
            active_result,
        )
        emitters: list[Emitter] = []
        active_centers: list[tuple[int, int]] = []
        for center_x, center_y in calculate_palm_centers(active_result):
            center = normalized_to_pixel(
                center_x,
                center_y,
                frame_width,
                frame_height,
            )
            active_centers.append(center)
            radius = calculate_ring_radius(frame_width, frame_height, center)
            if radius > 0:
                emitters.append((center, radius))
        interaction = self._portal_interaction.update(
            active_centers,
            animation_timestamp_ms,
        )
        portal_geometry = None
        if raw_portal_geometry is not None and interaction.portal_scale > 0.0:
            portal_geometry = PortalGeometry(
                raw_portal_geometry.center,
                max(1, round(raw_portal_geometry.radius * interaction.portal_scale)),
            )
        if interaction.burst_triggered and interaction.burst is not None:
            self._particles.emit_burst(interaction.burst.center)
        if portal_geometry is not None:
            emitters.append((portal_geometry.center, portal_geometry.radius))
        self._particles.update(emitters, animation_timestamp_ms)
        self._glow_layer = draw_magic_glow(
            frame,
            active_result,
            animation_timestamp_ms,
            self.config.rotation_speed_degrees_per_second,
            self._glow_layer,
            self._particles,
            portal_geometry,
            interaction.burst,
        )
        self._particles.draw_sharp(frame)
        draw_hand_landmarks(frame, result)
        draw_magic_rings(frame, active_result)
        draw_energy_arcs(
            frame,
            active_result,
            animation_timestamp_ms,
            self.config.rotation_speed_degrees_per_second,
        )
        draw_two_hand_portal(
            frame,
            portal_geometry,
            animation_timestamp_ms,
            self.config.rotation_speed_degrees_per_second,
        )
        draw_magic_burst(frame, interaction.burst)
        draw_rotating_ring_details(
            frame,
            active_result,
            animation_timestamp_ms,
            self.config.rotation_speed_degrees_per_second,
        )
        draw_palm_centers(frame, result)
        draw_gesture_status(frame, gesture_frame.statuses)

    def close(self) -> None:
        """Release native resources held by MediaPipe."""

        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

