"""
Kaggle Trainer - Notebook-compatible LoRA training for Kaggle's
30h/week free P100 GPU.

Usage in a Kaggle notebook cell:
    from cloud.kaggle_trainer import KaggleTrainer
    trainer = KaggleTrainer()
    trainer.train("character_name", ["/kaggle/input/charsheet/img1.png"])
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional


class KaggleTrainer:
    """
    LoRA training wrapper optimised for the Kaggle kernel environment.

    - Uses /kaggle/working as the default output directory
    - Saves checkpoints to /kaggle/working/warehouse
    - Supports exporting trained LoRA to a Kaggle output dataset
    """

    KAGGLE_WORKING = "/kaggle/working"
    KAGGLE_INPUT = "/kaggle/input"

    def __init__(self, warehouse_path: str = None):
        self.warehouse = Path(
            warehouse_path
            or os.getenv("AI_CACHE_ROOT", f"{self.KAGGLE_WORKING}/warehouse")
        )
        self.warehouse.mkdir(parents=True, exist_ok=True)
        (self.warehouse / "lora").mkdir(exist_ok=True)

        os.environ["AI_CACHE_ROOT"] = str(self.warehouse)

    def train(
        self,
        character_name: str,
        image_paths: List[str],
        rank: int = 16,
        max_steps: int = 500,
    ) -> str:
        """
        Train a LoRA on Kaggle P100 GPU.

        Args:
            character_name: Name of the character.
            image_paths: Paths to reference images.
            rank: LoRA rank (16 recommended for P100 VRAM).
            max_steps: Training steps (500 keeps within session limits).

        Returns:
            Path to trained LoRA weights.
        """
        from director.memory_bank import AssetMemoryBank
        from agents.character.trainer import LoRATrainer

        # Register character
        memory = AssetMemoryBank(str(self.warehouse))
        char_id = memory.create_character(character_name, image_paths)

        # Train
        trainer = LoRATrainer(str(self.warehouse))
        trainer.rank = rank
        trainer.max_train_steps = max_steps

        lora_path = trainer.train_character_lora(
            character_images=image_paths,
            character_id=char_id,
            character_name=character_name,
            rank=rank,
            max_steps=max_steps,
        )

        memory.update_character_lora(char_id, str(lora_path))
        print(f"Training complete! LoRA at {lora_path}")

        return str(lora_path)

    def export_lora(self, character_id: str, output_dir: str = None) -> Optional[str]:
        """
        Copy trained LoRA to an output directory for Kaggle dataset export.
        """
        output_dir = output_dir or f"{self.KAGGLE_WORKING}/output_lora"
        os.makedirs(output_dir, exist_ok=True)

        src = self.warehouse / "lora" / character_id
        if not src.exists():
            print(f"LoRA directory not found: {src}")
            return None

        dst = Path(output_dir) / character_id
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        print(f"LoRA exported to {dst}")
        return str(dst)

    def list_trained_loras(self) -> List[dict]:
        """List all LoRAs in the warehouse."""
        lora_dir = self.warehouse / "lora"
        results = []
        for subdir in lora_dir.iterdir():
            if subdir.is_dir():
                weight_file = subdir / "pytorch_lora_weights.safetensors"
                results.append({
                    "character_id": subdir.name,
                    "path": str(weight_file),
                    "exists": weight_file.exists(),
                    "size_mb": weight_file.stat().st_size / 1e6 if weight_file.exists() else 0,
                })
        return results
