#!/usr/bin/env python3
"""
Download base models required by AnimeLoom.
Only downloads if models are not already cached.
"""

import os
import sys
from pathlib import Path

WAREHOUSE = Path(os.getenv("AI_CACHE_ROOT", "./warehouse"))
MODELS_DIR = WAREHOUSE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_sd_model():
    """Download Stable Diffusion 2.1 via diffusers (cached by HF Hub)."""
    print("Checking Stable Diffusion 2.1...")
    try:
        from diffusers import StableDiffusionPipeline

        StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            cache_dir=str(MODELS_DIR),
        )
        print("  Stable Diffusion 2.1: OK")
    except Exception as e:
        print(f"  SD 2.1 download skipped: {e}")


def download_clip():
    """Download CLIP ViT-B/32 for character embeddings."""
    print("Checking CLIP ViT-B/32...")
    try:
        from transformers import CLIPModel, CLIPProcessor

        CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(MODELS_DIR))
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(MODELS_DIR))
        print("  CLIP ViT-B/32: OK")
    except Exception as e:
        print(f"  CLIP download skipped: {e}")


def download_groundingdino():
    """Download GroundingDINO weights."""
    print("Checking GroundingDINO...")
    weight_path = MODELS_DIR / "groundingdino_swint_ogc.pth"
    if weight_path.exists():
        print("  GroundingDINO: already present")
        return

    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="groundingdino_swint_ogc.pth",
            local_dir=str(MODELS_DIR),
        )
        print("  GroundingDINO: OK")
    except Exception as e:
        print(f"  GroundingDINO download skipped: {e}")


def download_sam():
    """Download SAM ViT-H weights."""
    print("Checking SAM ViT-H...")
    weight_path = MODELS_DIR / "sam_vit_h_4b8939.pth"
    if weight_path.exists():
        print("  SAM: already present")
        return

    try:
        import urllib.request

        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        print("  Downloading SAM ViT-H (~2.4 GB)...")
        urllib.request.urlretrieve(url, str(weight_path))
        print("  SAM: OK")
    except Exception as e:
        print(f"  SAM download skipped: {e}")


def main():
    print("=" * 50)
    print("AnimeLoom - Model Download")
    print(f"Warehouse: {WAREHOUSE}")
    print("=" * 50)
    print()

    download_sd_model()
    download_clip()
    download_groundingdino()
    download_sam()

    print()
    print("Model download complete!")
    print(f"Models stored in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
