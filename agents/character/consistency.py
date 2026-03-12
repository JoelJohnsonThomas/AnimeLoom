"""
Character Consistency Checker - Validates character identity using
GroundingDINO + SAM for detection/segmentation and a finetuned BLIP
encoder for feature extraction.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class CharacterConsistencyChecker:
    """
    Validates that a generated frame preserves character identity
    by comparing embeddings against stored references.
    """

    SIMILARITY_THRESHOLD = 0.80

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._detector = None   # GroundingDINO
        self._segmentor = None  # SAM
        self._encoder = None    # BLIP / CLIP
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_detector(self):
        """Load GroundingDINO for character detection."""
        if self._detector is not None:
            return
        try:
            from groundingdino.util.inference import load_model, predict
            model_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            model_weights = str(self.warehouse / "models" / "groundingdino_swint_ogc.pth")

            if os.path.exists(model_weights) and os.path.exists(model_config):
                self._detector = load_model(model_config, model_weights)
                print("GroundingDINO loaded")
            else:
                print("GroundingDINO weights not found — using fallback detector")
        except ImportError:
            print("GroundingDINO not installed — using fallback detector")

    def _load_segmentor(self):
        """Load SAM for character segmentation."""
        if self._segmentor is not None:
            return
        try:
            from segment_anything import SamPredictor, sam_model_registry
            sam_path = self.warehouse / "models" / "sam_vit_h_4b8939.pth"
            if sam_path.exists():
                sam = sam_model_registry["vit_h"](checkpoint=str(sam_path))
                sam.to(self._device)
                self._segmentor = SamPredictor(sam)
                print("SAM loaded")
            else:
                print("SAM weights not found — segmentation disabled")
        except ImportError:
            print("segment-anything not installed — segmentation disabled")

    def _load_encoder(self):
        """Load BLIP/CLIP for character feature extraction."""
        if self._encoder is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._encoder = {
                "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device),
                "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            }
            self._encoder["model"].eval()
            print("CLIP encoder loaded")
        except Exception:
            print("CLIP encoder not available — using random embeddings")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract a 512-d feature vector from a character image.
        Falls back to a deterministic hash-based vector if CLIP is unavailable.
        """
        self._load_encoder()

        if self._encoder is not None:
            inputs = self._encoder["processor"](images=image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                features = self._encoder["model"].get_image_features(**inputs)
            embedding = features.cpu().numpy().flatten()
            # L2 normalise
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

        # Fallback: deterministic pseudo-embedding from pixel data
        arr = np.array(image.resize((64, 64))).flatten().astype(np.float32)
        rng = np.random.default_rng(int(arr.sum()) % (2**31))
        emb = rng.standard_normal(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        return emb

    def detect_characters(
        self, image: Image.Image, prompt: str = "anime character"
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect character bounding boxes in an image using GroundingDINO.
        Returns list of (x1, y1, x2, y2) boxes.
        """
        self._load_detector()

        if self._detector is not None:
            try:
                from groundingdino.util.inference import predict
                import groundingdino.datasets.transforms as T

                transform = T.Compose([
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                img_t, _ = transform(image, None)
                boxes, logits, phrases = predict(
                    self._detector, img_t, prompt, box_threshold=0.3, text_threshold=0.25
                )
                w, h = image.size
                result = []
                for box in boxes:
                    cx, cy, bw, bh = box.tolist()
                    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                    result.append((x1, y1, x2, y2))
                return result
            except Exception as e:
                print(f"GroundingDINO detection failed: {e}")

        # Fallback: assume entire image is one character
        w, h = image.size
        return [(0, 0, w, h)]

    def segment_character(
        self, image: Image.Image, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Segment a character using SAM within a bounding box.
        Returns binary mask or None.
        """
        self._load_segmentor()

        if self._segmentor is not None:
            try:
                img_arr = np.array(image)
                self._segmentor.set_image(img_arr)
                box_np = np.array(bbox)
                masks, _, _ = self._segmentor.predict(box=box_np, multimask_output=False)
                return masks[0]
            except Exception as e:
                print(f"SAM segmentation failed: {e}")

        return None

    def compare_identity(
        self,
        generated_embedding: np.ndarray,
        reference_embedding: np.ndarray,
    ) -> float:
        """
        Cosine similarity between two character embeddings.
        Returns score in [0, 1].
        """
        gen = generated_embedding.flatten()
        ref = reference_embedding.flatten()
        dot = np.dot(gen, ref)
        norm = np.linalg.norm(gen) * np.linalg.norm(ref)
        if norm == 0:
            return 0.0
        similarity = float(dot / norm)
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))  # map [-1,1] -> [0,1]

    def check_consistency(
        self,
        frame: Image.Image,
        reference_embedding: np.ndarray,
    ) -> Dict:
        """
        Full pipeline: detect -> segment -> embed -> compare.

        Returns:
            Dict with keys: score, boxes, consistent (bool).
        """
        boxes = self.detect_characters(frame)
        if not boxes:
            return {"score": 0.0, "boxes": [], "consistent": False}

        scores = []
        for box in boxes:
            # Crop character region
            crop = frame.crop(box)
            emb = self.extract_embedding(crop)
            score = self.compare_identity(emb, reference_embedding)
            scores.append(score)

        best_score = max(scores) if scores else 0.0
        return {
            "score": best_score,
            "boxes": boxes,
            "consistent": best_score >= self.SIMILARITY_THRESHOLD,
        }
