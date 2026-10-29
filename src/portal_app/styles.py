"""Central visual palettes for selectable magical-energy styles."""

from __future__ import annotations

from dataclasses import dataclass


Color = tuple[int, int, int]


@dataclass(frozen=True)
class EffectStyle:
    name: str
    core: Color
    accent: Color
    secondary: Color
    glow: Color
    dust: Color
    spark: Color
    spark_trail: Color
    spark_head: Color
    rotation_multiplier: float = 1.0


GOLD_STYLE = EffectStyle(
    "Golden Sorcery",
    (80, 240, 255),
    (80, 235, 255),
    (0, 200, 255),
    (0, 150, 255),
    (0, 175, 255),
    (80, 235, 255),
    (0, 100, 255),
    (180, 255, 255),
)
BLUE_STYLE = EffectStyle(
    "Cosmic Blue",
    (255, 255, 180),
    (255, 210, 80),
    (255, 150, 30),
    (255, 110, 0),
    (255, 150, 20),
    (255, 245, 150),
    (255, 120, 0),
    (255, 255, 220),
    0.85,
)
RED_STYLE = EffectStyle(
    "Chaos Red",
    (180, 190, 255),
    (80, 90, 255),
    (40, 40, 220),
    (20, 20, 190),
    (20, 45, 240),
    (120, 170, 255),
    (20, 20, 220),
    (210, 220, 255),
    1.35,
)
GREEN_STYLE = EffectStyle(
    "Time Green",
    (190, 255, 190),
    (80, 255, 100),
    (20, 210, 50),
    (0, 155, 30),
    (20, 200, 60),
    (160, 255, 180),
    (0, 170, 40),
    (220, 255, 220),
    0.65,
)
PURPLE_STYLE = EffectStyle(
    "Dimensional Purple",
    (255, 210, 255),
    (255, 100, 245),
    (210, 40, 200),
    (155, 20, 150),
    (220, 60, 220),
    (255, 170, 255),
    (180, 20, 190),
    (255, 225, 255),
    1.1,
)

STYLE_PRESETS: tuple[EffectStyle, ...] = (
    GOLD_STYLE,
    BLUE_STYLE,
    RED_STYLE,
    GREEN_STYLE,
    PURPLE_STYLE,
)
