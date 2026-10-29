"""Time-based portal growth and two-hand merge burst state."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class BurstVisual:
    center: tuple[int, int]
    progress: float


@dataclass(frozen=True)
class PortalInteractionFrame:
    portal_scale: float
    burst: BurstVisual | None
    burst_triggered: bool


class PortalInteractionController:
    """Animate portal growth and trigger one burst for each hand merge."""

    def __init__(
        self,
        *,
        growth_duration_ms: float = 600.0,
        burst_duration_ms: float = 750.0,
        merge_distance: float = 70.0,
        rearm_distance: float = 140.0,
    ) -> None:
        if growth_duration_ms <= 0.0 or burst_duration_ms <= 0.0:
            raise ValueError("animation durations must be greater than zero")
        if merge_distance <= 0.0 or rearm_distance <= merge_distance:
            raise ValueError("rearm distance must exceed merge distance")
        self.growth_duration_ms = growth_duration_ms
        self.burst_duration_ms = burst_duration_ms
        self.merge_distance = merge_distance
        self.rearm_distance = rearm_distance
        self._portal_started_ms: float | None = None
        self._burst_started_ms: float | None = None
        self._burst_center: tuple[int, int] | None = None
        self._merge_armed = False

    def update(
        self,
        centers: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        timestamp_ms: int | float,
    ) -> PortalInteractionFrame:
        if not isfinite(float(timestamp_ms)) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be finite and non-negative")
        now = float(timestamp_ms)
        portal_scale = 0.0
        burst_triggered = False

        if len(centers) >= 2:
            first, second = centers[0], centers[1]
            distance = hypot(second[0] - first[0], second[1] - first[1])
            midpoint = (
                round((first[0] + second[0]) / 2.0),
                round((first[1] + second[1]) / 2.0),
            )
            if self._portal_started_ms is None:
                self._portal_started_ms = now
            linear_progress = min(
                1.0,
                max(0.0, (now - self._portal_started_ms) / self.growth_duration_ms),
            )
            # Cubic ease-out: quick initial response with a gentle finish.
            portal_scale = 1.0 - (1.0 - linear_progress) ** 3

            if distance >= self.rearm_distance:
                self._merge_armed = True
            elif distance <= self.merge_distance and self._merge_armed:
                self._merge_armed = False
                self._burst_started_ms = now
                self._burst_center = midpoint
                burst_triggered = True
        else:
            self._portal_started_ms = None

        burst = None
        if self._burst_started_ms is not None and self._burst_center is not None:
            progress = (now - self._burst_started_ms) / self.burst_duration_ms
            if progress <= 1.0:
                burst = BurstVisual(
                    self._burst_center,
                    min(1.0, max(0.0, progress)),
                )
            else:
                self._burst_started_ms = None
                self._burst_center = None

        return PortalInteractionFrame(portal_scale, burst, burst_triggered)
