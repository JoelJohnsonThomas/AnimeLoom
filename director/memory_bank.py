"""
Asset Memory Bank - Persistent storage for character assets, LoRAs, and scene data.
This is the core memory system that maintains character identity across shots.
"""

import os
import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
from PIL import Image


class AssetMemoryBank:
    """
    Persistent storage for character assets, LoRAs, and scene data.
    This is the core memory system that maintains character identity.
    """

    def __init__(self, warehouse_path: str = None):
        self.warehouse = Path(warehouse_path or os.getenv("AI_CACHE_ROOT", "./warehouse"))
        self.lora_dir = self.warehouse / "lora"
        self.datasets_dir = self.warehouse / "datasets"
        self.memory_index_path = self.warehouse / "memory_index.pkl"

        # Create directories
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load memory index
        self.db = self._load_or_create_db()

    def _load_or_create_db(self) -> Dict:
        """Load existing memory index or create new one."""
        if self.memory_index_path.exists():
            try:
                with open(self.memory_index_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

        return {
            "characters": {},       # id -> {name, lora_path, embedding, multi_views, ...}
            "scenes": {},           # id -> {background, lighting, style, ...}
            "shots": {},            # id -> {character_ids, scene_id, pose_refs, ...}
            "lora_registry": {},    # hash -> character_id
            "version": "1.0",
        }

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self):
        """Save current memory state to disk."""
        with open(self.memory_index_path, "wb") as f:
            pickle.dump(self.db, f)
        print(f"Memory checkpoint saved to {self.memory_index_path}")

    def load_latest_checkpoint(self) -> bool:
        """Reload memory from latest checkpoint."""
        if self.memory_index_path.exists():
            with open(self.memory_index_path, "rb") as f:
                self.db = pickle.load(f)
            print(f"Memory loaded from {self.memory_index_path}")
            return True
        return False

    # ------------------------------------------------------------------
    # Character management
    # ------------------------------------------------------------------

    def create_character(
        self,
        name: str,
        images: List[str],
        description: str = "",
    ) -> str:
        """
        Register a new character with reference images.

        Args:
            name: Character name.
            images: List of paths to character reference images.
            description: Text description of character.

        Returns:
            character_id
        """
        char_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        self.db["characters"][char_id] = {
            "name": name,
            "description": description,
            "lora_path": None,
            "embedding": None,
            "multi_views": images,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "shot_count": 0,
        }

        # Extract embedding from first valid image
        embedding = self._extract_embedding(images[0]) if images else None
        if embedding is not None:
            self.db["characters"][char_id]["embedding"] = embedding.tolist()

        # Train LoRA placeholder (actual training delegated to trainer.py)
        lora_path = self._prepare_lora_dir(char_id)
        self.db["characters"][char_id]["lora_path"] = str(lora_path)

        # Register in lora_registry for reverse lookup
        lora_hash = hashlib.md5(str(lora_path).encode()).hexdigest()
        self.db["lora_registry"][lora_hash] = char_id

        self.save_checkpoint()
        return char_id

    def get_character(self, char_id_or_name: str) -> Optional[Dict]:
        """Get character by ID or name (case-insensitive)."""
        # Try as ID first
        if char_id_or_name in self.db["characters"]:
            data = dict(self.db["characters"][char_id_or_name])
            data["id"] = char_id_or_name
            return data

        # Try as name
        for char_id, char_data in self.db["characters"].items():
            if char_data["name"].lower() == char_id_or_name.lower():
                data = dict(char_data)
                data["id"] = char_id
                return data

        return None

    def get_characters_for_shot(self, shot: Dict) -> Dict[str, Dict]:
        """Return all character data needed for a shot."""
        characters = {}
        for char_name in shot.get("characters", []):
            char_data = self.get_character(char_name)
            if char_data:
                characters[char_name] = char_data
        return characters

    def update_character_views(
        self, char_name: str, video_path: str, shot_index: int
    ):
        """Update character with new views from a generated video."""
        char_data = self.get_character(char_name)
        if char_data and "id" in char_data:
            char_id = char_data["id"]
            self.db["characters"][char_id]["shot_count"] += 1
            self.db["characters"][char_id]["last_used"] = datetime.now().isoformat()

    def update_character_lora(self, char_id: str, lora_path: str):
        """Update the LoRA path for a character after training."""
        if char_id in self.db["characters"]:
            self.db["characters"][char_id]["lora_path"] = str(lora_path)
            self.save_checkpoint()

    def update_character_embedding(self, char_id: str, embedding: np.ndarray):
        """Store a real embedding vector for a character."""
        if char_id in self.db["characters"]:
            self.db["characters"][char_id]["embedding"] = embedding.tolist()
            self.save_checkpoint()

    def list_characters(self) -> List[Dict]:
        """List all characters in memory."""
        return [
            {
                "id": char_id,
                "name": data["name"],
                "shot_count": data.get("shot_count", 0),
                "has_lora": data.get("lora_path") is not None,
                "created": data["created"],
                "last_used": data["last_used"],
            }
            for char_id, data in self.db["characters"].items()
        ]

    def get_character_lora_path(self, char_id: str) -> Optional[Path]:
        """Get path to character's LoRA weights (only if file exists).
        Checks both diffusers and PEFT weight filenames."""
        if char_id in self.db["characters"]:
            lora_path = self.db["characters"][char_id].get("lora_path")
            if lora_path:
                p = Path(lora_path)
                if p.exists():
                    return p
                # PEFT saves as adapter_model.safetensors
                adapter = p.parent / "adapter_model.safetensors"
                if adapter.exists():
                    return adapter
        return None

    def delete_character(self, char_id: str) -> bool:
        """Remove a character from the memory bank."""
        if char_id in self.db["characters"]:
            del self.db["characters"][char_id]
            # Clean lora_registry
            self.db["lora_registry"] = {
                h: cid
                for h, cid in self.db["lora_registry"].items()
                if cid != char_id
            }
            self.save_checkpoint()
            return True
        return False

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def create_scene(
        self,
        name: str,
        background: str = "",
        lighting: str = "neutral",
        style: str = "anime",
    ) -> str:
        """Create a new scene entry."""
        scene_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        self.db["scenes"][scene_id] = {
            "name": name,
            "background": background,
            "lighting": lighting,
            "style": style,
            "created": datetime.now().isoformat(),
        }
        self.save_checkpoint()
        return scene_id

    def get_scene(self, scene_id: str) -> Optional[Dict]:
        return self.db["scenes"].get(scene_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract a character embedding from an image.
        Returns a 512-d placeholder; real implementation uses BLIP / CLIP.
        """
        try:
            Image.open(image_path).resize((224, 224))
            return np.random.default_rng(42).standard_normal(512).astype(np.float32)
        except Exception:
            return None

    def _prepare_lora_dir(self, char_id: str) -> Path:
        """Create the LoRA directory for a character and return the weight path."""
        lora_dir = self.lora_dir / char_id
        lora_dir.mkdir(exist_ok=True)
        lora_path = lora_dir / "pytorch_lora_weights.safetensors"
        # Placeholder file — real training writes actual weights here
        if not lora_path.exists():
            lora_path.write_text("placeholder")
        return lora_path
