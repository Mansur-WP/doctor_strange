"""Small, persistent golden dust and spark particles for the palm rings."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, isfinite, pi, sin
from random import Random

import cv2
import numpy as np

from portal_app.styles import EffectStyle, GOLD_STYLE


Emitter = tuple[tuple[int, int], int]


@dataclass
class Particle:
    """One animated point stored in camera-frame pixel coordinates."""

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    lifetime: float
    initial_lifetime: float
    size: int
    brightness: float
    is_spark: bool
    turbulence_phase: float = 0.0
    turbulence_strength: float = 0.0

    @property
    def opacity(self) -> float:
        """Return the remaining life as a safe value between zero and one."""

        if self.initial_lifetime <= 0.0:
            return 0.0
        return min(1.0, max(0.0, self.lifetime / self.initial_lifetime))


class ParticleSystem:
    """Update and render a bounded collection of persistent particles."""

    def __init__(
        self,
        *,
        max_particles: int = 260,
        spawn_rate_per_emitter: float = 60.0,
        max_movement_step: float = 0.05,
        rng: Random | None = None,
        style: EffectStyle = GOLD_STYLE,
    ) -> None:
        if max_particles <= 0:
            raise ValueError("max_particles must be greater than zero")
        if spawn_rate_per_emitter < 0.0:
            raise ValueError("spawn rate must not be negative")
        if max_movement_step <= 0.0:
            raise ValueError("max movement step must be greater than zero")

        self.max_particles = max_particles
        self.spawn_rate_per_emitter = spawn_rate_per_emitter
        self.max_movement_step = max_movement_step
        self._rng = rng or Random()
        self.style = style
        self._particles: list[Particle] = []
        self._last_timestamp_ms: int | float | None = None
        self._spawn_credit = 0.0
        self._spawn_cursor = 0

    @property
    def particles(self) -> tuple[Particle, ...]:
        """Expose a read-only snapshot for diagnostics and unit tests."""

        return tuple(self._particles)

    def set_style(self, style: EffectStyle) -> None:
        self.style = style

    def update(
        self,
        emitters: list[Emitter] | tuple[Emitter, ...] | None,
        timestamp_ms: int | float,
    ) -> None:
        """Move old particles, remove expired ones, and emit new particles."""

        if not isfinite(float(timestamp_ms)) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be finite and non-negative")

        if self._last_timestamp_ms is None:
            elapsed_seconds = 0.0
        else:
            elapsed_seconds = max(
                0.0,
                (float(timestamp_ms) - float(self._last_timestamp_ms)) / 1000.0,
            )
        self._last_timestamp_ms = timestamp_ms

        # Position changes are clamped after a pause, but real elapsed time still
        # expires old particles instead of making a frozen trail reappear.
        movement_seconds = min(elapsed_seconds, self.max_movement_step)
        animation_seconds = float(timestamp_ms) / 1000.0
        drag = 0.985 ** (movement_seconds * 60.0)
        survivors = []
        for particle in self._particles:
            turbulence_angle = (
                animation_seconds * 3.5 + particle.turbulence_phase
            )
            particle.velocity_x += (
                cos(turbulence_angle)
                * particle.turbulence_strength
                * movement_seconds
            )
            particle.velocity_y += (
                sin(turbulence_angle)
                * particle.turbulence_strength
                * movement_seconds
            )
            particle.x += particle.velocity_x * movement_seconds
            particle.y += particle.velocity_y * movement_seconds
            particle.velocity_x *= drag
            particle.velocity_y = particle.velocity_y * drag - 5.0 * movement_seconds
            particle.lifetime -= elapsed_seconds
            if particle.lifetime > 0.0:
                survivors.append(particle)
        self._particles = survivors

        valid_emitters: list[Emitter] = []
        for emitter in emitters or ():
            try:
                center, radius = emitter
                center_x, center_y = center
                center_x = int(center_x)
                center_y = int(center_y)
                radius = int(radius)
            except (TypeError, ValueError):
                continue
            if radius > 0:
                valid_emitters.append(((center_x, center_y), radius))
        if not valid_emitters or movement_seconds <= 0.0:
            return

        self._spawn_credit += (
            self.spawn_rate_per_emitter
            * len(valid_emitters)
            * movement_seconds
        )
        available = self.max_particles - len(self._particles)
        # This cap prevents a delayed frame from producing a visible burst.
        per_frame_limit = max(1, 4 * len(valid_emitters))
        spawn_count = min(int(self._spawn_credit), available, per_frame_limit)
        self._spawn_credit -= spawn_count

        for _ in range(spawn_count):
            emitter = valid_emitters[self._spawn_cursor % len(valid_emitters)]
            self._spawn_cursor += 1
            self._particles.append(self._spawn_particle(*emitter))

    def _spawn_particle(self, center: tuple[int, int], radius: int) -> Particle:
        angle = self._rng.uniform(0.0, 2.0 * pi)
        radial_x = cos(angle)
        radial_y = sin(angle)
        distance = radius * self._rng.uniform(0.82, 1.05)
        is_spark = self._rng.random() < 0.36

        if is_spark:
            radial_speed = self._rng.uniform(95.0, 175.0)
            sideways_speed = self._rng.uniform(-14.0, 14.0)
            lifetime = self._rng.uniform(0.48, 0.85)
            size = self._rng.randint(2, 3)
            brightness = self._rng.uniform(0.9, 1.0)
            turbulence_strength = self._rng.uniform(8.0, 18.0)
        else:
            radial_speed = self._rng.uniform(5.0, 24.0)
            sideways_speed = self._rng.uniform(-28.0, 28.0)
            lifetime = self._rng.uniform(0.8, 1.55)
            size = self._rng.randint(1, 3)
            brightness = self._rng.uniform(0.68, 1.0)
            turbulence_strength = self._rng.uniform(14.0, 34.0)

        # (-sin, cos) is perpendicular to the radius and supplies the swirl.
        velocity_x = radial_x * radial_speed - radial_y * sideways_speed
        velocity_y = radial_y * radial_speed + radial_x * sideways_speed - 7.0
        return Particle(
            x=center[0] + radial_x * distance,
            y=center[1] + radial_y * distance,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            lifetime=lifetime,
            initial_lifetime=lifetime,
            size=size,
            brightness=brightness,
            is_spark=is_spark,
            turbulence_phase=self._rng.uniform(0.0, 2.0 * pi),
            turbulence_strength=turbulence_strength,
        )

    def emit_burst(self, center: tuple[int, int], count: int = 72) -> int:
        """Emit a bounded radial group of bright, short-lived spark particles."""

        if count <= 0:
            return 0
        burst_count = min(count, self.max_particles)
        keep_count = max(0, self.max_particles - burst_count)
        if len(self._particles) > keep_count:
            self._particles.sort(key=lambda particle: particle.opacity, reverse=True)
            self._particles = self._particles[:keep_count]

        for _ in range(burst_count):
            angle = self._rng.uniform(0.0, 2.0 * pi)
            speed = self._rng.uniform(120.0, 290.0)
            lifetime = self._rng.uniform(0.45, 0.95)
            start_distance = self._rng.uniform(0.0, 14.0)
            radial_x = cos(angle)
            radial_y = sin(angle)
            self._particles.append(
                Particle(
                    x=center[0] + radial_x * start_distance,
                    y=center[1] + radial_y * start_distance,
                    velocity_x=radial_x * speed,
                    velocity_y=radial_y * speed - self._rng.uniform(0.0, 18.0),
                    lifetime=lifetime,
                    initial_lifetime=lifetime,
                    size=self._rng.randint(2, 4),
                    brightness=self._rng.uniform(0.9, 1.0),
                    is_spark=True,
                    turbulence_phase=self._rng.uniform(0.0, 2.0 * pi),
                    turbulence_strength=self._rng.uniform(6.0, 16.0),
                )
            )
        return burst_count

    def draw_glow(self, overlay: np.ndarray) -> bool:
        """Draw soft particle energy onto the shared, soon-to-be-blurred layer."""

        drew_particle = False
        for particle in self._particles:
            color = self._color(particle, glow=True)
            if color == (0, 0, 0):
                continue
            point = (round(particle.x), round(particle.y))
            cv2.circle(
                overlay,
                point,
                max(2, particle.size * 3),
                color,
                -1,
                cv2.LINE_AA,
            )
            if particle.is_spark:
                tail = self._tail_point(particle, 18.0 + particle.size * 3.0)
                cv2.line(
                    overlay,
                    tail,
                    point,
                    color,
                    7,
                    cv2.LINE_AA,
                )
            drew_particle = True
        return drew_particle

    def draw_sharp(self, frame: np.ndarray) -> None:
        """Draw crisp golden dots and short spark streaks after glow blending."""

        for particle in self._particles:
            color = self._color(particle, glow=False)
            if color == (0, 0, 0):
                continue
            point = (round(particle.x), round(particle.y))
            if particle.is_spark:
                tail = self._tail_point(particle, 16.0 + particle.size * 4.0)
                cv2.line(
                    frame,
                    tail,
                    point,
                    tuple(
                        round(channel * particle.opacity)
                        for channel in self.style.spark_trail
                    ),
                    particle.size + 2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    frame,
                    tail,
                    point,
                    color,
                    max(1, particle.size - 1),
                    cv2.LINE_AA,
                )
                head_color = tuple(
                    round(channel * particle.opacity)
                    for channel in self.style.spark_head
                )
                cv2.circle(
                    frame,
                    point,
                    particle.size,
                    head_color,
                    -1,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(frame, point, particle.size, color, -1, cv2.LINE_AA)

    @staticmethod
    def _tail_point(particle: Particle, length: float) -> tuple[int, int]:
        speed = hypot(particle.velocity_x, particle.velocity_y)
        if speed <= 0.001:
            return round(particle.x), round(particle.y)
        return (
            round(particle.x - particle.velocity_x / speed * length),
            round(particle.y - particle.velocity_y / speed * length),
        )

    @staticmethod
    def _color(particle: Particle, *, glow: bool) -> tuple[int, int, int]:
        strength = min(1.0, particle.brightness * particle.opacity)
        if strength <= 0.0:
            return 0, 0, 0
        base = self.style.spark if particle.is_spark else self.style.dust
        if glow:
            strength *= 0.8
        return tuple(round(channel * strength) for channel in base)
