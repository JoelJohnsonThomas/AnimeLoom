"""
Temporal Coherence Evaluator — detects frame-to-frame artifacts
and sudden visual discontinuities in generated video.

Uses SSIM (Structural Similarity) between consecutive frames to identify
temporal incoherence. Sudden SSIM drops indicate artifacts, flicker,
or visual glitches that break the illusion of smooth animation.
"""

import os
from typing import List

import numpy as np
from PIL import Image


class TemporalCoherenceEvaluator:
    """
    Evaluates temporal coherence of a generated video by measuring
    frame-to-frame SSIM consistency.
    """

    # Frames with SSIM below this vs the previous frame are flagged as artifacts
    ARTIFACT_THRESHOLD = 0.7

    def __init__(self, warehouse_path: str = ""):
        pass

    def evaluate(
        self,
        video_path: str,
        sample_frames: int = 16,
    ) -> float:
        """
        Score temporal coherence of a video.

        Returns:
            Score in [0, 1] where 1 means perfectly smooth temporal transitions.
        """
        frames = self._extract_frames(video_path, sample_frames)
        if len(frames) < 2:
            return 0.5  # uncertain — not enough frames

        ssim_scores = self._compute_pairwise_ssim(frames)
        if not ssim_scores:
            return 0.5

        # Penalize videos with sudden SSIM drops (artifacts)
        artifact_count = sum(1 for s in ssim_scores if s < self.ARTIFACT_THRESHOLD)
        artifact_ratio = artifact_count / len(ssim_scores)

        # Base score is mean SSIM, penalized by artifact ratio
        mean_ssim = float(np.mean(ssim_scores))
        penalty = artifact_ratio * 0.3  # up to 0.3 penalty for all-artifact video

        return float(np.clip(mean_ssim - penalty, 0.0, 1.0))

    def _compute_pairwise_ssim(self, frames: List[Image.Image]) -> List[float]:
        """Compute SSIM between consecutive frames."""
        try:
            scores = []
            for i in range(len(frames) - 1):
                ssim = self._ssim(frames[i], frames[i + 1])
                scores.append(ssim)
            return scores
        except Exception:
            return []

    def _ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute simplified SSIM between two images."""
        # Resize to small size for efficiency
        size = (128, 128)
        a = np.array(img1.resize(size)).astype(np.float64)
        b = np.array(img2.resize(size)).astype(np.float64)

        if a.shape != b.shape:
            return 0.0

        # Flatten to grayscale
        if a.ndim == 3:
            a = np.mean(a, axis=2)
            b = np.mean(b, axis=2)

        # SSIM constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu_a = np.mean(a)
        mu_b = np.mean(b)
        sigma_a_sq = np.var(a)
        sigma_b_sq = np.var(b)
        sigma_ab = np.mean((a - mu_a) * (b - mu_b))

        numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)

        return float(numerator / denominator)

    def _extract_frames(
        self, video_path: str, max_frames: int = 16
    ) -> List[Image.Image]:
        """Read evenly-spaced frames from a video file."""
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
