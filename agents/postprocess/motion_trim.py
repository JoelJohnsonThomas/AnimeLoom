"""
Motion Trimmer — trims static trailing frames from video clips.

CogVideoX often generates 49 frames but motion decays in the last 10-15.
This module detects where motion dies and trims accordingly.
"""

import numpy as np
from typing import List

from PIL import Image


class MotionTrimmer:
    """Trim trailing static frames using optical flow analysis."""

    def trim_static_tail(
        self,
        frames: List[Image.Image],
        threshold: float = 1.0,
        min_frames: int = 24,
    ) -> List[Image.Image]:
        """
        Trim trailing frames where optical flow drops below threshold.

        Args:
            frames: List of PIL Images (a single video clip).
            threshold: Minimum mean optical flow magnitude to consider
                       a frame as "in motion".
            min_frames: Never trim below this many frames.

        Returns:
            Trimmed list of PIL Images.
        """
        if len(frames) <= min_frames:
            return frames

        flow_magnitudes = self._compute_flow_magnitudes(frames)
        if not flow_magnitudes:
            return frames

        # Find the last frame with significant motion
        last_motion_idx = len(frames) - 1
        for i in range(len(flow_magnitudes) - 1, -1, -1):
            if flow_magnitudes[i] > threshold:
                # flow_magnitudes[i] is between frames[i] and frames[i+1]
                last_motion_idx = i + 1
                break
        else:
            # No motion at all — keep min_frames
            return frames[:min_frames]

        # Keep at least min_frames, and add a small buffer (3 frames)
        trim_point = max(min_frames, last_motion_idx + 3)
        trim_point = min(trim_point, len(frames))

        return frames[:trim_point]

    def extend_last_clip_with_pingpong(
        self,
        frames: List[Image.Image],
        target_frames: int = 49,
    ) -> List[Image.Image]:
        """
        Extend a short clip by ping-ponging (forward + reverse) to reach
        target frame count. Useful for the last clip to avoid abrupt endings.
        """
        if len(frames) >= target_frames:
            return frames[:target_frames]

        result = list(frames)
        reversed_frames = list(reversed(frames[1:-1]))  # skip first/last to avoid doubles

        while len(result) < target_frames and reversed_frames:
            result.extend(reversed_frames[:target_frames - len(result)])
            if len(result) < target_frames:
                result.extend(frames[1:-1][:target_frames - len(result)])

        return result[:target_frames]

    def _compute_flow_magnitudes(
        self, frames: List[Image.Image]
    ) -> List[float]:
        """Compute mean optical flow magnitude between consecutive frames."""
        try:
            import cv2

            magnitudes: List[float] = []
            prev_gray = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY)

            for frame in frames[1:]:
                curr_gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
                # Resize to 256px wide for speed
                scale = 256.0 / max(prev_gray.shape[1], 1)
                if scale < 1.0:
                    h, w = prev_gray.shape[:2]
                    new_w, new_h = int(w * scale), int(h * scale)
                    prev_small = cv2.resize(prev_gray, (new_w, new_h))
                    curr_small = cv2.resize(curr_gray, (new_w, new_h))
                else:
                    prev_small, curr_small = prev_gray, curr_gray

                flow = cv2.calcOpticalFlowFarneback(
                    prev_small, curr_small, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes.append(float(np.mean(mag)))
                prev_gray = curr_gray

            return magnitudes
        except ImportError:
            return []
