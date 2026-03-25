"""
RunPod notebook helpers — replaces Colab survival mode.

Usage in a Jupyter notebook on RunPod:
    from cloud.runpod_notebook import setup_runpod_environment, check_gpu
    warehouse = setup_runpod_environment()
    check_gpu()
"""

import os
from pathlib import Path


def setup_runpod_environment(warehouse_path="/mnt/network-volume/warehouse"):
    """Configure AnimeLoom for a RunPod pod.

    Sets AI_CACHE_ROOT and creates the warehouse directory tree.
    Returns the warehouse Path.
    """
    warehouse = Path(warehouse_path)
    os.environ["AI_CACHE_ROOT"] = str(warehouse)

    for d in [
        "models",
        "lora",
        "datasets/raw",
        "datasets/tagged",
        "outputs",
        "checkpoints",
        "references",
    ]:
        (warehouse / d).mkdir(parents=True, exist_ok=True)

    print(f"Warehouse: {warehouse}")
    return warehouse


def check_gpu():
    """Print GPU name and VRAM. Returns (name, gb) or (None, 0)."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
        return name, mem
    print("WARNING: No GPU detected!")
    return None, 0


def predownload_models(warehouse_path="/mnt/network-volume/warehouse"):
    """Pre-download heavy models to the network volume so generation starts instantly.

    Call this once after creating a new network volume. Takes ~10-15 min.
    """
    cache = Path(warehouse_path) / "models"
    print("Pre-downloading models to network volume (one-time)...")

    # AnimateDiff motion adapter
    try:
        from diffusers import MotionAdapter
        MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3",
            cache_dir=str(cache),
        )
        print("  [ok] AnimateDiff motion adapter")
    except Exception as e:
        print(f"  [skip] AnimateDiff motion adapter: {e}")

    # SD 1.5 anime base model
    try:
        from diffusers import StableDiffusionPipeline
        StableDiffusionPipeline.from_pretrained(
            "Lykon/dreamshaper-8",
            cache_dir=str(cache),
        )
        print("  [ok] DreamShaper 8 (SD 1.5 anime)")
    except Exception as e:
        print(f"  [skip] DreamShaper 8: {e}")

    # IP-Adapter
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            "h94/IP-Adapter",
            filename="models/ip-adapter_sd15.bin",
            cache_dir=str(cache),
        )
        print("  [ok] IP-Adapter SD 1.5")
    except Exception as e:
        print(f"  [skip] IP-Adapter: {e}")

    # SDXL anime
    try:
        from diffusers import StableDiffusionXLPipeline
        StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-3.1",
            cache_dir=str(cache),
        )
        print("  [ok] Animagine XL 3.1 (SDXL)")
    except Exception as e:
        print(f"  [skip] Animagine XL: {e}")

    print("Pre-download complete. Models persist on network volume.")
