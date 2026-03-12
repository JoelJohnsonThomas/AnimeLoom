#!/usr/bin/env python3
"""
Dataset preparation script for AnimeLoom.
Downloads and organizes character datasets into the format
AssetMemoryBank expects.

Supported sources:
  1. HuggingFace character_similarity dataset
  2. Local image folders
  3. Auto-captioning with BLIP

Usage:
  python scripts/prepare_dataset.py --source huggingface --split tiny
  python scripts/prepare_dataset.py --source local --input ./my_chars/yuki
  python scripts/prepare_dataset.py --caption ./warehouse/datasets/raw/yuki
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WAREHOUSE = Path(os.getenv("AI_CACHE_ROOT", str(PROJECT_ROOT / "warehouse")))
RAW_DIR = WAREHOUSE / "datasets" / "raw"
TAGGED_DIR = WAREHOUSE / "datasets" / "tagged"


# ------------------------------------------------------------------
# HuggingFace dataset download
# ------------------------------------------------------------------

def download_huggingface(split: str = "v0_tiny", max_characters: int = 50):
    """
    Download the deepghs/character_similarity dataset.

    Splits available:
      v0_tiny   — 514 characters, 10k images   (prototyping)
      v1_pruned — 3,982 characters, 241k images (production)
      v1        — 4,001 characters, 292k images  (full)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install the datasets library first:  pip install datasets")
        sys.exit(1)

    print(f"Downloading deepghs/character_similarity ({split})…")
    ds = load_dataset("deepghs/character_similarity", split)

    # Group images by character ID
    characters: dict = {}
    for item in ds["train"]:
        cid = str(item.get("character_id", item.get("id", "unknown")))
        characters.setdefault(cid, []).append(item)

    print(f"Found {len(characters)} characters")

    count = 0
    for cid, items in characters.items():
        if count >= max_characters:
            break
        if len(items) < 3:
            continue  # need at least 3 images

        char_dir = RAW_DIR / cid
        char_dir.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(items[:20]):  # cap at 20 images
            img = item.get("image")
            if img is None:
                continue
            img.save(str(char_dir / f"{cid}_{i:03d}.png"))

        count += 1
        print(f"  [{count}/{max_characters}] {cid}: {min(len(items), 20)} images")

    print(f"\nSaved {count} characters to {RAW_DIR}")


# ------------------------------------------------------------------
# Local folder import
# ------------------------------------------------------------------

def import_local(input_dir: str, character_name: str = None):
    """Copy a local folder of images into the raw dataset directory."""
    src = Path(input_dir)
    if not src.exists():
        print(f"Directory not found: {src}")
        sys.exit(1)

    name = character_name or src.name
    dst = RAW_DIR / name
    dst.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    copied = 0
    for f in sorted(src.iterdir()):
        if f.suffix.lower() in exts:
            shutil.copy2(f, dst / f.name)
            copied += 1

    print(f"Imported {copied} images for '{name}' → {dst}")


# ------------------------------------------------------------------
# Auto-captioning with BLIP
# ------------------------------------------------------------------

def caption_folder(folder: str):
    """Generate text captions for every image in a character folder."""
    folder = Path(folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
    except ImportError:
        print("Install transformers and torch first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BLIP captioner on {device}…")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    char_name = folder.name
    out_dir = TAGGED_DIR / char_name
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in exts]

    print(f"Captioning {len(images)} images for '{char_name}'…")
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_length=60)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        full_caption = f"{char_name}, {caption}, anime character"

        # Copy image and save caption
        shutil.copy2(img_path, out_dir / img_path.name)
        (out_dir / f"{img_path.stem}.txt").write_text(full_caption)
        print(f"  {img_path.name}: {full_caption}")

    print(f"\nTagged images saved to {out_dir}")


# ------------------------------------------------------------------
# Register tagged characters into AssetMemoryBank
# ------------------------------------------------------------------

def register_all():
    """Register every tagged character folder into the AssetMemoryBank."""
    from director.memory_bank import AssetMemoryBank

    memory = AssetMemoryBank(str(WAREHOUSE))
    tagged = TAGGED_DIR
    if not tagged.exists():
        print("No tagged datasets found. Run --caption first.")
        return

    for char_dir in sorted(tagged.iterdir()):
        if not char_dir.is_dir():
            continue
        images = [
            str(f)
            for f in sorted(char_dir.iterdir())
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not images:
            continue

        # Read description from first caption file
        desc = ""
        first_txt = next(char_dir.glob("*.txt"), None)
        if first_txt:
            desc = first_txt.read_text().strip()

        name = char_dir.name
        existing = memory.get_character(name)
        if existing:
            print(f"  '{name}' already registered — skipping")
            continue

        cid = memory.create_character(name=name, images=images, description=desc)
        print(f"  Registered '{name}' ({len(images)} images) → {cid}")

    print("\nDone. Characters are now available for AnimeLoom scripts.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AnimeLoom dataset preparation")
    sub = parser.add_subparsers(dest="command")

    # huggingface
    hf = sub.add_parser("huggingface", help="Download HuggingFace character dataset")
    hf.add_argument("--split", default="v0_tiny",
                     choices=["v0_tiny", "v1_pruned", "v1"])
    hf.add_argument("--max-chars", type=int, default=50)

    # local
    loc = sub.add_parser("local", help="Import a local image folder")
    loc.add_argument("input_dir", help="Path to folder of character images")
    loc.add_argument("--name", help="Character name (defaults to folder name)")

    # caption
    cap = sub.add_parser("caption", help="Auto-caption images with BLIP")
    cap.add_argument("folder", help="Path to character image folder")

    # register
    sub.add_parser("register", help="Register all tagged characters in AssetMemoryBank")

    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TAGGED_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "huggingface":
        download_huggingface(args.split, args.max_chars)
    elif args.command == "local":
        import_local(args.input_dir, args.name)
    elif args.command == "caption":
        caption_folder(args.folder)
    elif args.command == "register":
        register_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
