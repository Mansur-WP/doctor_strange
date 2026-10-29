"""Deterministic unit checks for Stage 7 particles without a webcam."""

import sys
import unittest
from pathlib import Path
from random import Random

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portal_app.particles import Particle, ParticleSystem  # noqa: E402


class ParticleSystemTests(unittest.TestCase):
    def make_system(self, **overrides) -> ParticleSystem:
        settings = {
            "max_particles": 40,
            "spawn_rate_per_emitter": 40.0,
            "rng": Random(1234),
        }
        settings.update(overrides)
        return ParticleSystem(**settings)

    def test_spawns_deterministically_around_one_palm_ring(self) -> None:
        system = self.make_system()
        system.update([((100, 80), 30)], 0)
        system.update([((100, 80), 30)], 100)

        # Only 50 ms is simulated because long frame gaps are clamped.
        self.assertEqual(len(system.particles), 2)
        for particle in system.particles:
            distance = ((particle.x - 100) ** 2 + (particle.y - 80) ** 2) ** 0.5
            self.assertGreaterEqual(distance, 30 * 0.82)
            self.assertLessEqual(distance, 30 * 1.05)

    def test_two_emitters_both_receive_particles(self) -> None:
        system = self.make_system()
        emitters = [((80, 100), 25), ((320, 100), 25)]
        system.update(emitters, 0)
        system.update(emitters, 100)

        self.assertTrue(any(particle.x < 200 for particle in system.particles))
        self.assertTrue(any(particle.x > 200 for particle in system.particles))

    def test_motion_depends_on_elapsed_time(self) -> None:
        system = self.make_system()
        system.update([((100, 100), 30)], 0)
        system.update([((100, 100), 30)], 100)
        before = [(particle.x, particle.y) for particle in system.particles]

        system.update([], 150)
        after = [(particle.x, particle.y) for particle in system.particles]

        self.assertEqual(len(before), len(after))
        self.assertNotEqual(before, after)

    def test_particles_receive_deterministic_turbulence(self) -> None:
        first = self.make_system()
        second = self.make_system()
        for system in (first, second):
            system.update([((100, 100), 30)], 0)
            system.update([((100, 100), 30)], 100)
            system.update([], 150)

        first_state = [
            (particle.x, particle.y, particle.velocity_x, particle.velocity_y)
            for particle in first.particles
        ]
        second_state = [
            (particle.x, particle.y, particle.velocity_x, particle.velocity_y)
            for particle in second.particles
        ]
        self.assertEqual(first_state, second_state)

    def test_particles_fade_and_expire(self) -> None:
        particle = Particle(0, 0, 0, 0, 0.5, 1.0, 1, 1.0, False)
        self.assertEqual(particle.opacity, 0.5)

        system = self.make_system()
        system.update([((100, 100), 30)], 0)
        system.update([((100, 100), 30)], 100)
        self.assertTrue(system.particles)

        system.update([], 2000)
        self.assertFalse(system.particles)

    def test_particle_budget_is_never_exceeded(self) -> None:
        system = self.make_system(max_particles=12, spawn_rate_per_emitter=1000.0)
        emitters = [((80, 80), 20), ((160, 80), 20)]
        for timestamp_ms in range(0, 1001, 50):
            system.update(emitters, timestamp_ms)

        self.assertLessEqual(len(system.particles), 12)

    def test_empty_emitters_do_not_spawn_or_clear_existing_particles(self) -> None:
        system = self.make_system()
        system.update([], 0)
        system.update([], 100)
        self.assertFalse(system.particles)

        emitters = [((80, 80), 20), ((300, 80), 20)]
        system.update(emitters, 200)
        existing_ids = {id(particle) for particle in system.particles}
        system.update([emitters[1]], 210)

        self.assertTrue(existing_ids.intersection(id(p) for p in system.particles))

    def test_missing_and_invalid_emitters_are_skipped_safely(self) -> None:
        system = self.make_system()
        system.update(None, 0)
        system.update(
            [None, (None, 20), ((10, 10), 0), (("bad", 10), 20)],  # type: ignore[list-item]
            100,
        )

        self.assertFalse(system.particles)

    def test_rendering_preserves_frame_shape_and_dtype(self) -> None:
        system = self.make_system()
        system.update([((80, 60), 25)], 0)
        system.update([((80, 60), 25)], 100)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        overlay = np.zeros_like(frame)
        overlay_id = id(overlay)

        drew_glow = system.draw_glow(overlay)
        system.draw_sharp(frame)
        system.draw_glow(overlay)

        self.assertTrue(drew_glow)
        self.assertTrue(overlay.any())
        self.assertTrue(frame.any())
        self.assertEqual(frame.shape, (120, 160, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(id(overlay), overlay_id)

    def test_visible_sparks_have_long_bright_streaks(self) -> None:
        system = self.make_system(spawn_rate_per_emitter=200.0)
        system.update([((80, 60), 25)], 0)
        for timestamp_ms in range(50, 501, 50):
            system.update([((80, 60), 25)], timestamp_ms)
        sparks = [particle for particle in system.particles if particle.is_spark]
        self.assertTrue(sparks)
        self.assertTrue(all(particle.size >= 2 for particle in sparks))
        self.assertTrue(all(particle.initial_lifetime >= 0.48 for particle in sparks))

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        system.draw_sharp(frame)
        self.assertGreater(np.count_nonzero(frame), len(system.particles) * 3)

    def test_merge_burst_is_radial_and_respects_particle_budget(self) -> None:
        system = self.make_system(max_particles=50)

        emitted = system.emit_burst((100, 80), count=72)

        self.assertEqual(emitted, 50)
        self.assertEqual(len(system.particles), 50)
        self.assertTrue(all(particle.is_spark for particle in system.particles))
        self.assertTrue(any(particle.velocity_x < 0 for particle in system.particles))
        self.assertTrue(any(particle.velocity_x > 0 for particle in system.particles))

    def test_long_delay_expires_particles_without_large_position_jump(self) -> None:
        system = self.make_system()
        system.update([((100, 100), 30)], 0)
        system.update([((100, 100), 30)], 100)
        particle = system.particles[0]
        before = (particle.x, particle.y)

        system.update([], 600)

        if system.particles:
            moved = ((particle.x - before[0]) ** 2 + (particle.y - before[1]) ** 2) ** 0.5
            speed = (particle.velocity_x**2 + particle.velocity_y**2) ** 0.5
            self.assertLessEqual(moved, speed * 0.06)


if __name__ == "__main__":
    unittest.main()
