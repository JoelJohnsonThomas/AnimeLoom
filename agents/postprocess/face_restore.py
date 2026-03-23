"""
Anime Face Restorer — detects and sharpens anime faces in video frames.

Replaces GFPGAN (trained on real photos) with anime-appropriate face
enhancement using YOLO face detection + OpenCV sharpening.
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


class AnimeFaceRestorer:
    """
    Detect anime faces and sharpen them using OpenCV operations.

    Falls back gracefully if YOLO/ultralytics is unavailable — applies
    whole-frame sharpening instead.
    """

    def __init__(self):
        self._detector = None
        self._detector_failed = False

    def restore_frames(
        self,
        frames: List[Image.Image],
        face_padding: float = 0.25,
    ) -> List[Image.Image]:
        """
        Restore/sharpen faces in a list of video frames.

        Args:
            frames: List of PIL Images.
            face_padding: Fractional padding around detected face bbox.

        Returns:
            List of PIL Images with enhanced faces.
        """
        detector = self._load_detector()
        restored = []

        for frame in frames:
            if detector is not None:
                faces = self._detect_faces(detector, frame)
                if faces:
                    frame = self._enhance_face_regions(frame, faces, face_padding)
                else:
                    # No face detected — apply mild whole-frame sharpening
                    frame = self._mild_sharpen(frame)
            else:
                # No detector — apply mild whole-frame sharpening
                frame = self._mild_sharpen(frame)
            restored.append(frame)

        return restored

    def _load_detector(self):
        """Load YOLO anime face detector."""
        if self._detector is not None or self._detector_failed:
            return self._detector

        try:
            from ultralytics import YOLO
            # Try anime face model from ADetailer
            try:
                self._detector = YOLO("face_yolov8n.pt")
                print("  Anime face detector loaded (face_yolov8n)")
                return self._detector
            except Exception:
                pass

            # Fallback: standard YOLOv8 nano (detects "person" class)
            try:
                self._detector = YOLO("yolov8n.pt")
                print("  Face detector loaded (yolov8n fallback)")
                return self._detector
            except Exception:
                pass

            self._detector_failed = True
            print("  Face detector unavailable, using whole-frame sharpening")
            return None

        except ImportError:
            self._detector_failed = True
            print("  ultralytics not installed, using whole-frame sharpening")
            return None

    def _detect_faces(
        self, detector, frame: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """Detect face bounding boxes in a frame."""
        try:
            results = detector(np.array(frame), verbose=False, conf=0.3)
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # Filter very small detections
                    if (x2 - x1) > 20 and (y2 - y1) > 20:
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception:
            return []

    def _enhance_face_regions(
        self,
        frame: Image.Image,
        faces: List[Tuple[int, int, int, int]],
        padding: float = 0.25,
    ) -> Image.Image:
        """Sharpen detected face regions with bilateral filter + unsharp mask."""
        try:
            import cv2

            frame_arr = np.array(frame)
            h, w = frame_arr.shape[:2]

            for (x1, y1, x2, y2) in faces:
                # Add padding
                fw, fh = x2 - x1, y2 - y1
                px, py = int(fw * padding), int(fh * padding)
                x1 = max(0, x1 - px)
                y1 = max(0, y1 - py)
                x2 = min(w, x2 + px)
                y2 = min(h, y2 + py)

                face_region = frame_arr[y1:y2, x1:x2].copy()

                # Bilateral filter — smooths noise while preserving edges
                face_region = cv2.bilateralFilter(face_region, 5, 50, 50)

                # Unsharp mask — sharpen fine details (eyes, hair strands)
                blurred = cv2.GaussianBlur(face_region, (0, 0), 2.0)
                face_region = cv2.addWeighted(face_region, 1.5, blurred, -0.5, 0)

                # Slight contrast boost on face
                lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
                l_channel = lab[:, :, 0].astype(np.float32)
                l_channel = np.clip(l_channel * 1.05, 0, 255).astype(np.uint8)
                lab[:, :, 0] = l_channel
                face_region = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                # Feathered paste-back (avoid hard edges)
                mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
                feather = min(8, (y2 - y1) // 8, (x2 - x1) // 8)
                if feather > 0:
                    mask[:feather, :] *= np.linspace(0, 1, feather)[:, None]
                    mask[-feather:, :] *= np.linspace(1, 0, feather)[:, None]
                    mask[:, :feather] *= np.linspace(0, 1, feather)[None, :]
                    mask[:, -feather:] *= np.linspace(1, 0, feather)[None, :]

                mask_3ch = mask[:, :, None]
                original = frame_arr[y1:y2, x1:x2].astype(np.float32)
                enhanced = face_region.astype(np.float32)
                blended = (mask_3ch * enhanced + (1 - mask_3ch) * original)
                frame_arr[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)

            return Image.fromarray(frame_arr)

        except ImportError:
            return self._mild_sharpen(frame)

    def _mild_sharpen(self, frame: Image.Image) -> Image.Image:
        """Apply mild sharpening to the entire frame (fallback)."""
        return frame.filter(ImageFilter.SHARPEN)
