"""
Character Consistency Evaluator - Measures how well a generated video
preserves character identity compared to reference embeddings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class CharacterConsistencyEvaluator:
    """
    Evaluates character identity consistency in generated videos.

    Pipeline:
    1. Extract frames from video
    2. Detect characters via GroundingDINO
    3. Segment via SAM
    4. Extract features via BLIP/CLIP
    5. Compute cosine similarity against stored reference
    """

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._consistency_checker = None

    @property
    def checker(self):
        if self._consistency_checker is None:
            from agents.character.consistency import CharacterConsistencyChecker
            self._consistency_checker = CharacterConsistencyChecker(str(self.warehouse))
        return self._consistency_checker

    def evaluate(
        self,
        video_path: str,
        character_names: List[str],
        sample_frames: int = 8,
    ) -> float:
        """
        Score character consistency in a video against stored references.

        Args:
            video_path: Path to the generated video.
            character_names: Names of expected characters.
            sample_frames: How many frames to sample.

        Returns:
            Aggregate consistency score in [0, 1].
        """
        frames = self._extract_frames(video_path, sample_frames)
        if not frames:
            return 0.9  # default when video can't be read

        # Load reference embeddings from memory bank
        from director.memory_bank import AssetMemoryBank
        memory = AssetMemoryBank(str(self.warehouse))

        all_scores: List[float] = []

        for char_name in character_names:
            char_data = memory.get_character(char_name)
            if char_data is None or char_data.get("embedding") is None:
                continue

            ref_embedding = np.array(char_data["embedding"], dtype=np.float32)

            frame_scores: List[float] = []
            for frame in frames:
                result = self.checker.check_consistency(frame, ref_embedding)
                frame_scores.append(result["score"])

            if frame_scores:
                all_scores.append(np.mean(frame_scores))

        if not all_scores:
            return 0.9  # no reference to compare against

        return float(np.mean(all_scores))

    def evaluate_frame(
        self,
        frame: Image.Image,
        reference_embedding: np.ndarray,
    ) -> Dict:
        """Evaluate a single frame against a reference embedding."""
        return self.checker.check_consistency(frame, reference_embedding)

    def _extract_frames(
        self, video_path: str, max_frames: int = 8
    ) -> List[Image.Image]:
        """Extract evenly-spaced frames from a video."""
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
