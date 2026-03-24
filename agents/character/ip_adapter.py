"""
IP-Adapter Character Conditioner — inference-time character consistency
without requiring LoRA training.

Works by injecting reference image features into the diffusion process
via IP-Adapter (Image Prompt Adapter). Compatible with AnimateDiff and
any SD 1.5 / SDXL pipeline that supports `load_ip_adapter()`.

Advantages over LoRA-only approach:
  - Zero training cost (inference-time only)
  - Works with any reference image, no fine-tuning needed
  - Can be combined with LoRA for strongest identity preservation
"""

from pathlib import Path
from typing import Optional

from PIL import Image


class IPAdapterConditioner:
    """
    Manages IP-Adapter loading and character image conditioning.

    Usage:
        conditioner = IPAdapterConditioner(warehouse_path)
        conditioner.inject(pipeline, reference_image, scale=0.6)
        # pipeline is now conditioned — pass ip_adapter_image to __call__
    """

    # SD 1.5 IP-Adapter variants
    _SD15_ADAPTERS = [
        ("h94/IP-Adapter", "models", "ip-adapter_sd15.bin"),
        ("h94/IP-Adapter", "models", "ip-adapter-plus_sd15.bin"),
    ]

    # SDXL IP-Adapter variants
    _SDXL_ADAPTERS = [
        ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin"),
    ]

    # Face-focused variant (strongest identity preservation)
    _FACE_ADAPTER = ("h94/IP-Adapter-FaceID", "models", "ip-adapter-faceid_sd15.bin")

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._loaded_adapter_type: Optional[str] = None

    def inject(
        self,
        pipeline,
        reference_image: Optional[Image.Image] = None,
        scale: float = 0.6,
        adapter_type: str = "sd15",
    ) -> bool:
        """
        Load IP-Adapter into a diffusers pipeline.

        Args:
            pipeline: A diffusers pipeline (AnimateDiff, SD 1.5, SDXL).
            reference_image: Character reference image (optional, can be
                passed later via ip_adapter_image kwarg).
            scale: IP-Adapter conditioning strength (0.0-1.0).
                   0.6 is a good balance between identity and prompt adherence.
            adapter_type: "sd15" or "sdxl".

        Returns:
            True if IP-Adapter was loaded successfully.
        """
        adapters = self._SD15_ADAPTERS if adapter_type == "sd15" else self._SDXL_ADAPTERS
        cache_dir = str(self.warehouse / "models")

        for repo_id, subfolder, weight_name in adapters:
            try:
                pipeline.load_ip_adapter(
                    repo_id,
                    subfolder=subfolder,
                    weight_name=weight_name,
                    cache_dir=cache_dir,
                )
                pipeline.set_ip_adapter_scale(scale)
                self._loaded_adapter_type = adapter_type
                print(f"  IP-Adapter loaded: {weight_name} (scale={scale})")
                return True
            except Exception as e:
                print(f"  IP-Adapter {weight_name} failed: {e}")
                continue

        print("  No IP-Adapter variant available")
        return False

    def get_reference_image(
        self, character_name: str, warehouse_path: str
    ) -> Optional[Image.Image]:
        """
        Retrieve a reference image for a character from the memory bank.

        Args:
            character_name: Name of the character.
            warehouse_path: Path to the warehouse directory.

        Returns:
            PIL Image or None if no reference found.
        """
        try:
            from director.memory_bank import AssetMemoryBank

            memory = AssetMemoryBank(warehouse_path)
            char_data = memory.get_character(character_name)
            if char_data and char_data.get("multi_views"):
                for view_path in char_data["multi_views"]:
                    if Path(view_path).exists():
                        return Image.open(view_path).convert("RGB")
        except Exception:
            pass
        return None

    @staticmethod
    def prepare_image(
        image: Image.Image,
        target_width: int = 512,
        target_height: int = 768,
    ) -> Image.Image:
        """Resize and prepare a reference image for IP-Adapter input."""
        return image.resize((target_width, target_height), Image.LANCZOS)
