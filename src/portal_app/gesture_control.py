"""Time-based per-hand gesture states for deliberate spell activation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable


class GesturePhase(str, Enum):
    INACTIVE = "INACTIVE"
    CHARGING = "CHARGING"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class GestureStatus:
    """One hand's current state for rendering and diagnostics."""

    key: str
    phase: GesturePhase
    progress: float
    center: tuple[float, float]
    landmarks: object


@dataclass(frozen=True)
class GestureFrame:
    """Gesture output for one processed camera frame."""

    active_landmarks: tuple
    statuses: tuple[GestureStatus, ...]


@dataclass
class _TrackedGesture:
    phase: GesturePhase = GesturePhase.INACTIVE
    phase_elapsed_ms: float = 0.0
    last_update_ms: float = 0.0
    last_seen_ms: float = 0.0


class GestureController:
    """Convert noisy open-palm observations into stable timed states."""

    def __init__(
        self,
        open_palm_detector: Callable[[object], bool],
        palm_center_calculator: Callable[[object], tuple[float, float] | None],
        *,
        charge_duration_ms: float = 800.0,
        cooldown_duration_ms: float = 300.0,
        track_timeout_ms: float = 700.0,
    ) -> None:
        for name, value in (
            ("charge_duration_ms", charge_duration_ms),
            ("cooldown_duration_ms", cooldown_duration_ms),
            ("track_timeout_ms", track_timeout_ms),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")

        self.open_palm_detector = open_palm_detector
        self.palm_center_calculator = palm_center_calculator
        self.charge_duration_ms = charge_duration_ms
        self.cooldown_duration_ms = cooldown_duration_ms
        self.track_timeout_ms = track_timeout_ms
        self._tracks: dict[str, _TrackedGesture] = {}

    def update(self, result, timestamp_ms: int | float) -> GestureFrame:
        """Advance every visible hand using elapsed milliseconds."""

        if not isfinite(float(timestamp_ms)) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be finite and non-negative")

        hands = getattr(result, "hand_landmarks", None) or []
        statuses: list[GestureStatus] = []
        active_landmarks = []
        used_keys: set[str] = set()

        for index, landmarks in enumerate(hands):
            center = self.palm_center_calculator(landmarks)
            if center is None:
                continue
            key = self._hand_key(result, index, center)
            if key in used_keys:
                key = f"{key}-{index}"
            used_keys.add(key)

            track = self._tracks.get(key)
            if track is None:
                track = _TrackedGesture(
                    last_update_ms=float(timestamp_ms),
                    last_seen_ms=float(timestamp_ms),
                )
                self._tracks[key] = track

            elapsed_ms = max(0.0, float(timestamp_ms) - track.last_update_ms)
            track.last_update_ms = float(timestamp_ms)
            track.last_seen_ms = float(timestamp_ms)
            is_open = self.open_palm_detector(landmarks)
            self._advance_track(track, is_open, elapsed_ms)

            progress = self._progress(track)
            status = GestureStatus(key, track.phase, progress, center, landmarks)
            statuses.append(status)
            if track.phase in (GesturePhase.ACTIVE, GesturePhase.COOLDOWN):
                active_landmarks.append(landmarks)

        stale_keys = [
            key
            for key, track in self._tracks.items()
            if float(timestamp_ms) - track.last_seen_ms > self.track_timeout_ms
        ]
        for key in stale_keys:
            del self._tracks[key]

        return GestureFrame(tuple(active_landmarks), tuple(statuses))

    def _advance_track(
        self,
        track: _TrackedGesture,
        is_open: bool,
        elapsed_ms: float,
    ) -> None:
        if track.phase is GesturePhase.INACTIVE:
            if is_open:
                track.phase = GesturePhase.CHARGING
                track.phase_elapsed_ms = 0.0
            return

        if track.phase is GesturePhase.CHARGING:
            if not is_open:
                track.phase = GesturePhase.INACTIVE
                track.phase_elapsed_ms = 0.0
                return
            track.phase_elapsed_ms += elapsed_ms
            if track.phase_elapsed_ms >= self.charge_duration_ms:
                track.phase = GesturePhase.ACTIVE
                track.phase_elapsed_ms = 0.0
            return

        if track.phase is GesturePhase.ACTIVE:
            if not is_open:
                track.phase = GesturePhase.COOLDOWN
                track.phase_elapsed_ms = 0.0
            return

        if is_open:
            track.phase = GesturePhase.ACTIVE
            track.phase_elapsed_ms = 0.0
            return
        track.phase_elapsed_ms += elapsed_ms
        if track.phase_elapsed_ms >= self.cooldown_duration_ms:
            track.phase = GesturePhase.INACTIVE
            track.phase_elapsed_ms = 0.0

    def _progress(self, track: _TrackedGesture) -> float:
        if track.phase is GesturePhase.CHARGING:
            return min(1.0, track.phase_elapsed_ms / self.charge_duration_ms)
        if track.phase is GesturePhase.ACTIVE:
            return 1.0
        if track.phase is GesturePhase.COOLDOWN:
            return max(0.0, 1.0 - track.phase_elapsed_ms / self.cooldown_duration_ms)
        return 0.0

    @staticmethod
    def _hand_key(
        result,
        index: int,
        center: tuple[float, float],
    ) -> str:
        handedness = getattr(result, "handedness", None) or []
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            label = getattr(category, "category_name", None)
            if label:
                return str(label).lower()
        # MediaPipe normally supplies handedness. This spatial fallback avoids
        # relying directly on result-list order when a synthetic result does not.
        return "screen-left" if center[0] < 0.5 else "screen-right"
