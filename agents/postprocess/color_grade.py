"""
Anime Color Grading — enhances generated frames with anime-style color.

Applies an S-curve contrast on luminance and a saturation boost for
vivid anime visuals.  CPU-only, no GPU needed.
"""

from typing import List

import numpy as np
from PIL import Image, ImageEnhance


class AnimeColorGrader:
    """Applies anime-optimised color grading to video frames."""

    def __init__(
        self,
        contrast_strength: float = 1.15,
        saturation_boost: float = 1.12,
        brightness_adjust: float = 1.02,
    ):
        self.contrast_strength = contrast_strength
        self.saturation_boost = saturation_boost
        self.brightness_adjust = brightness_adjust

    def grade_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply anime color grading to a list of PIL Images."""
        return [self._grade_single(f) for f in frames]

    def _grade_single(self, frame: Image.Image) -> Image.Image:
        """Grade a single frame."""
        # Contrast (S-curve approximation via PIL enhancer)
        frame = ImageEnhance.Contrast(frame).enhance(self.contrast_strength)
        # Saturation boost for vivid anime colors
        frame = ImageEnhance.Color(frame).enhance(self.saturation_boost)
        # Slight brightness lift
        frame = ImageEnhance.Brightness(frame).enhance(self.brightness_adjust)
        return frame

    def grade_with_palette(
        self, frames: List[Image.Image], palette: str = "warm"
    ) -> List[Image.Image]:
        """
        Grade frames with an anime color palette.

        Palettes:
          warm   — golden-hour sunset tones (orange shift)
          cool   — moonlight / night scene (blue shift)
          vibrant — maximum saturation for action scenes
          muted  — soft pastel for slice-of-life
        """
        shifts = {
            "warm":    (10, -2, -8),
            "cool":    (-8, -2, 12),
            "vibrant": (0, 0, 0),      # extra saturation only
            "muted":   (0, 0, 0),       # reduced saturation
        }
        sat_overrides = {
            "vibrant": 1.25,
            "muted":   0.90,
        }

        r_shift, g_shift, b_shift = shifts.get(palette, (0, 0, 0))
        sat = sat_overrides.get(palette, self.saturation_boost)

        graded = []
        for frame in frames:
            # Base grading
            f = ImageEnhance.Contrast(frame).enhance(self.contrast_strength)
            f = ImageEnhance.Color(f).enhance(sat)
            f = ImageEnhance.Brightness(f).enhance(self.brightness_adjust)

            # Color shift
            if r_shift or g_shift or b_shift:
                arr = np.array(f).astype(np.int16)
                arr[:, :, 0] = np.clip(arr[:, :, 0] + r_shift, 0, 255)
                arr[:, :, 1] = np.clip(arr[:, :, 1] + g_shift, 0, 255)
                arr[:, :, 2] = np.clip(arr[:, :, 2] + b_shift, 0, 255)
                f = Image.fromarray(arr.astype(np.uint8))

            graded.append(f)
        return graded
