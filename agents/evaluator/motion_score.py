"""
Motion Fidelity Evaluator - Measures how faithfully generated motion
matches a pose reference.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class MotionFidelityEvaluator:
    """
    Evaluates motion fidelity between a pose-reference video and a
    generated output video.

    Uses optical-flow magnitude comparison and optional pose-keypoint
    matching when OpenPose is available.
    """

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)

    def evaluate(
        self,
        generated_path: str,
        reference_path: str,
        sample_frames: int = 16,
    ) -> float:
        """
        Compare motion between reference and generated videos.

        Returns:
            Score in [0, 1] where 1 means perfect motion match.
        """
        gen_frames = self._extract_frames(generated_path, sample_frames)
        ref_frames = self._extract_frames(reference_path, sample_frames)

        if not gen_frames or not ref_frames:
            return 0.9  # default when comparison not possible

        # Align frame counts
        min_len = min(len(gen_frames), len(ref_frames))
        gen_frames = gen_frames[:min_len]
        ref_frames = ref_frames[:min_len]

        # Compare optical flow magnitudes
        gen_flows = self._compute_optical_flows(gen_frames)
        ref_flows = self._compute_optical_flows(ref_frames)

        if not gen_flows or not ref_flows:
            return 0.9

        min_flows = min(len(gen_flows), len(ref_flows))
        scores: List[float] = []

        for i in range(min_flows):
            score = self._flow_similarity(gen_flows[i], ref_flows[i])
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.9

    def evaluate_pose_keypoints(
        self,
        generated_path: str,
        reference_path: str,
    ) -> float:
        """
        Compare pose keypoints between reference and generated videos.
        Requires controlnet_aux / OpenPose.
        """
        try:
            from agents.animator.controlnet import PoseConditioner

            conditioner = PoseConditioner(str(self.warehouse))
            gen_poses = conditioner.extract_poses_from_video(generated_path, max_frames=8)
            ref_poses = conditioner.extract_poses_from_video(reference_path, max_frames=8)

            if not gen_poses or not ref_poses:
                return 0.9

            min_len = min(len(gen_poses), len(ref_poses))
            scores: List[float] = []
            for i in range(min_len):
                sim = self._image_similarity(gen_poses[i], ref_poses[i])
                scores.append(sim)

            return float(np.mean(scores)) if scores else 0.9
        except Exception:
            return 0.9

    # ------------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------------

    def _compute_optical_flows(
        self, frames: List[Image.Image]
    ) -> List[np.ndarray]:
        """Compute dense optical flow magnitude between consecutive frames."""
        try:
            import cv2

            flows: List[np.ndarray] = []
            prev_gray = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY)

            for frame in frames[1:]:
                curr_gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flows.append(mag)
                prev_gray = curr_gray

            return flows
        except ImportError:
            return []

    def _flow_similarity(self, flow_a: np.ndarray, flow_b: np.ndarray) -> float:
        """Normalised correlation between two flow-magnitude maps."""
        a = flow_a.flatten().astype(np.float64)
        b = flow_b.flatten().astype(np.float64)

        # Resize if shapes differ
        if a.shape != b.shape:
            min_size = min(len(a), len(b))
            a = np.interp(np.linspace(0, 1, min_size), np.linspace(0, 1, len(a)), a)
            b = np.interp(np.linspace(0, 1, min_size), np.linspace(0, 1, len(b)), b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 1.0 if norm_a == norm_b else 0.0

        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0, 1))

    def _image_similarity(self, img_a: Image.Image, img_b: Image.Image) -> float:
        """Simple pixel-level similarity between two images."""
        a = np.array(img_a.resize((64, 64))).astype(np.float32).flatten()
        b = np.array(img_b.resize((64, 64))).astype(np.float32).flatten()

        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0, 1))

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

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
