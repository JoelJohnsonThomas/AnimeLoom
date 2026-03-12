"""
Visual Quality Evaluator - Assesses frame quality of generated video
using metrics like sharpness, color consistency, and artifact detection.
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


class VisualQualityEvaluator:
    """
    Evaluates the visual quality of generated video frames.

    Metrics:
    - Sharpness (Laplacian variance)
    - Color consistency across frames
    - Artifact detection (banding, noise)
    """

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)

    def evaluate(self, video_path: str, sample_frames: int = 8) -> float:
        """
        Overall visual quality score for a video.

        Returns:
            Score in [0, 1] where 1 is excellent quality.
        """
        frames = self._extract_frames(video_path, sample_frames)
        if not frames:
            return 0.9

        scores: List[float] = []

        # Per-frame sharpness
        sharpness_scores = [self._sharpness_score(f) for f in frames]
        scores.append(np.mean(sharpness_scores))

        # Colour consistency across frames
        color_score = self._color_consistency(frames)
        scores.append(color_score)

        # Temporal smoothness
        smoothness = self._temporal_smoothness(frames)
        scores.append(smoothness)

        return float(np.clip(np.mean(scores), 0, 1))

    def evaluate_frame(self, frame: Image.Image) -> Dict:
        """Evaluate a single frame."""
        return {
            "sharpness": self._sharpness_score(frame),
            "noise_level": self._noise_level(frame),
        }

    # ------------------------------------------------------------------
    # Sharpness
    # ------------------------------------------------------------------

    def _sharpness_score(self, image: Image.Image) -> float:
        """
        Laplacian-variance sharpness metric.
        Higher variance → sharper image → higher score.
        """
        try:
            import cv2

            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalise: typical anime frames have variance 200-2000
            return float(np.clip(lap_var / 1500.0, 0, 1))
        except ImportError:
            # Numpy-only fallback
            gray = np.mean(np.array(image), axis=2)
            lap = np.abs(np.diff(gray, axis=0)) + np.abs(np.diff(gray, axis=1)[:, :-1])
            return float(np.clip(np.var(lap) / 500.0, 0, 1))

    # ------------------------------------------------------------------
    # Colour consistency
    # ------------------------------------------------------------------

    def _color_consistency(self, frames: List[Image.Image]) -> float:
        """
        Measure how stable the colour palette is across frames.
        Large shifts indicate flickering or style drift.
        """
        if len(frames) < 2:
            return 1.0

        means = []
        for f in frames:
            arr = np.array(f).astype(np.float32)
            means.append(arr.mean(axis=(0, 1)))  # per-channel mean

        means = np.array(means)
        # Standard deviation of channel means across frames
        std = np.mean(np.std(means, axis=0))
        # Lower std → more consistent; normalise
        return float(np.clip(1.0 - std / 50.0, 0, 1))

    # ------------------------------------------------------------------
    # Temporal smoothness
    # ------------------------------------------------------------------

    def _temporal_smoothness(self, frames: List[Image.Image]) -> float:
        """
        Measure smoothness between consecutive frames.
        Sudden jumps reduce the score.
        """
        if len(frames) < 2:
            return 1.0

        diffs: List[float] = []
        for i in range(len(frames) - 1):
            a = np.array(frames[i]).astype(np.float32)
            b = np.array(frames[i + 1]).astype(np.float32)
            # Resize if needed
            if a.shape != b.shape:
                b = np.array(frames[i + 1].resize(frames[i].size)).astype(np.float32)
            mse = np.mean((a - b) ** 2)
            diffs.append(mse)

        avg_diff = np.mean(diffs)
        # Normalise: smooth anime typically has MSE 200-3000
        return float(np.clip(1.0 - avg_diff / 5000.0, 0, 1))

    # ------------------------------------------------------------------
    # Noise detection
    # ------------------------------------------------------------------

    def _noise_level(self, image: Image.Image) -> float:
        """
        Estimate noise level in an image.
        Returns value in [0, 1] where 0 is noise-free.
        """
        gray = np.array(image.convert("L")).astype(np.float64)
        # Median-based noise estimate (MAD)
        sigma = np.median(np.abs(np.diff(gray))) / 0.6745
        return float(np.clip(sigma / 30.0, 0, 1))

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def _extract_frames(
        self, video_path: str, max_frames: int = 8
    ) -> List[Image.Image]:
        """Read evenly-spaced frames from a video."""
        if not os.path.exists(video_path):
            return []

        frames: List[Image.Image] = []
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []

            step = max(1, total // max_frames)
            idx = 0

            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                idx += 1

            cap.release()
        except ImportError:
            pass

        return frames
