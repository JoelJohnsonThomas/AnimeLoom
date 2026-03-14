"""
LoRA Manager - Manages loading and unloading of character LoRA adapters.
Supports both SD 1.5/2.1 and SDXL-based models (e.g. Animagine XL 3.1).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

_SDXL_KEYWORDS = ["xl", "animagine"]


def _is_sdxl(model_id: str) -> bool:
    lower = model_id.lower()
    return any(kw in lower for kw in _SDXL_KEYWORDS)


class LoRAManager:
    """
    Manages loading and unloading of character LoRA adapters
    into a Stable Diffusion pipeline.
    """

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self.lora_dir = self.warehouse / "lora"
        self.lora_dir.mkdir(exist_ok=True)

        # Cache for loaded adapter names
        self.loaded_loras: Dict[str, Path] = {}

        # Base pipeline (lazy loaded)
        self._pipeline = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def load_base_pipeline(self, model_id: str = "cagliostrolab/animagine-xl-3.1"):
        """Load the base Stable Diffusion pipeline (SD or SDXL)."""
        if self._pipeline is not None:
            return self._pipeline

        from diffusers import DPMSolverMultistepScheduler

        if _is_sdxl(model_id):
            from diffusers import StableDiffusionXLPipeline

            self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )
        else:
            from diffusers import StableDiffusionPipeline

            self._pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

        self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )
        self._pipeline = self._pipeline.to(self._device)
        return self._pipeline

    def load_base_pipeline_for_character(self, character_id: str):
        """Load the correct pipeline based on a character's training metadata."""
        meta_path = self.lora_dir / character_id / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            model_id = meta.get("base_model", "cagliostrolab/animagine-xl-3.1")
        else:
            model_id = "cagliostrolab/animagine-xl-3.1"
        return self.load_base_pipeline(model_id)

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.load_base_pipeline()
        return self._pipeline

    # ------------------------------------------------------------------
    # LoRA lookup
    # ------------------------------------------------------------------

    def find_character_lora(self, character_id: str) -> Optional[Path]:
        """
        Find LoRA weights for a character by ID.
        Searches the warehouse/lora directory tree.
        """
        direct = self.lora_dir / character_id / "pytorch_lora_weights.safetensors"
        if direct.exists():
            return direct

        # Broader search
        for subdir in self.lora_dir.iterdir():
            if not subdir.is_dir():
                continue
            candidate = subdir / "pytorch_lora_weights.safetensors"
            if candidate.exists() and character_id in subdir.name:
                return candidate

        return None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load_lora(self, lora_path: Path, adapter_name: str = None):
        """Load LoRA weights into the pipeline."""
        pipe = self.pipeline
        adapter_name = adapter_name or lora_path.parent.name

        if adapter_name in self.loaded_loras:
            return pipe

        try:
            pipe.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
            )
            self.loaded_loras[adapter_name] = lora_path
            print(f"Loaded LoRA: {adapter_name}")
        except Exception as e:
            print(f"Error loading LoRA {lora_path}: {e}")

        return pipe

    def unload_lora(self, adapter_name: str):
        """Unload a specific LoRA adapter."""
        if adapter_name in self.loaded_loras:
            try:
                self.pipeline.unload_lora_weights()
            except Exception:
                pass
            self.loaded_loras.pop(adapter_name, None)
            print(f"Unloaded LoRA: {adapter_name}")

    def unload_all_loras(self):
        """Unload every LoRA adapter."""
        if self._pipeline is not None:
            try:
                self._pipeline.unload_lora_weights()
            except Exception:
                pass
        self.loaded_loras.clear()
        print("Unloaded all LoRAs")

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def get_available_loras(self) -> List[Dict]:
        """List all LoRA adapters found in the warehouse."""
        loras = []
        if not self.lora_dir.exists():
            return loras

        for subdir in self.lora_dir.iterdir():
            if not subdir.is_dir():
                continue
            weight_file = subdir / "pytorch_lora_weights.safetensors"
            if weight_file.exists():
                loras.append(
                    {
                        "character_id": subdir.name,
                        "path": str(weight_file),
                        "size_bytes": weight_file.stat().st_size,
                        "loaded": subdir.name in self.loaded_loras,
                    }
                )
        return loras
