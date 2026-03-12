#!/usr/bin/env python3
"""
Standalone LoRA training script for AnimeLoom.
Works on Colab, Kaggle, GCP, or any machine with a GPU.
Supports both SD 1.5/2.1 and SDXL-based models (e.g. Animagine XL 3.1).

Usage:
  python scripts/train_lora.py --name Yuki --images ./my_chars/yuki/
  python scripts/train_lora.py --name Yuki --images ./my_chars/yuki/ --rank 16 --steps 500
  python scripts/train_lora.py --name Yuki --tagged ./warehouse/datasets/tagged/yuki/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WAREHOUSE = Path(os.getenv("AI_CACHE_ROOT", str(PROJECT_ROOT / "warehouse")))

# SDXL model identifiers (checked by substring)
_SDXL_KEYWORDS = ["xl", "animagine"]


def _is_sdxl(model_id: str) -> bool:
    """Detect whether a model ID refers to an SDXL-based model."""
    lower = model_id.lower()
    return any(kw in lower for kw in _SDXL_KEYWORDS)


def collect_images(folder: Path):
    """Gather image paths from a folder."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(str(f) for f in folder.iterdir() if f.suffix.lower() in exts)


def collect_captions(folder: Path):
    """Gather caption files paired with images."""
    captions = {}
    for txt in folder.glob("*.txt"):
        captions[txt.stem] = txt.read_text().strip()
    return captions


def train(
    character_name: str,
    image_dir: Path,
    rank: int = 32,
    lr: float = 1e-4,
    steps: int = 1000,
    batch_size: int = 1,
    resolution: int = 1024,
    base_model: str = "cagliostrolab/animagine-xl-3.1",
    use_captions: bool = False,
):
    """Train a LoRA adapter for a single character."""
    import torch
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    images = collect_images(image_dir)
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    captions = collect_captions(image_dir) if use_captions else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sdxl = _is_sdxl(base_model)

    print(f"Device: {device}")
    print(f"Character: {character_name}")
    print(f"Images: {len(images)}")
    print(f"Model type: {'SDXL' if sdxl else 'SD'}")
    print(f"LoRA rank: {rank}, lr: {lr}, steps: {steps}")

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------
    class CharDS(Dataset):
        def __init__(self, paths, caps, res):
            self.paths = paths
            self.caps = caps
            self.res = res

        def __len__(self):
            return max(len(self.paths) * 10, steps)  # repeat

        def __getitem__(self, idx):
            p = self.paths[idx % len(self.paths)]
            img = Image.open(p).convert("RGB").resize(
                (self.res, self.res), Image.Resampling.LANCZOS
            )
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            stem = Path(p).stem
            cap = self.caps.get(stem, f"{character_name}, 1girl, anime character portrait")
            return {"pixel_values": tensor, "caption": cap}

    ds = CharDS(images, captions, resolution)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # ----------------------------------------------------------------
    # Load base model
    # ----------------------------------------------------------------
    from diffusers import DDPMScheduler

    print(f"\nLoading base model: {base_model} …")

    if sdxl:
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=torch.float16, variant="fp16",
        )
    else:
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=torch.float16, safety_checker=None
        )

    pipe.to(device)

    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    if sdxl:
        tokenizer_2 = pipe.tokenizer_2
        text_encoder = pipe.text_encoder
        text_encoder_2 = pipe.text_encoder_2
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
    else:
        text_encoder = pipe.text_encoder
        text_encoder.requires_grad_(False)

    # ----------------------------------------------------------------
    # Attach LoRA via PEFT
    # ----------------------------------------------------------------
    try:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
        )
        unet = get_peft_model(unet, lora_cfg)
        unet.print_trainable_parameters()
    except ImportError:
        print("peft not installed — falling back to full UNet fine-tune (not recommended)")
        unet.requires_grad_(True)

    # ----------------------------------------------------------------
    # Optimizer + scheduler
    # ----------------------------------------------------------------
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)

    from diffusers.optimization import get_scheduler as get_lr_sched

    lr_sched = get_lr_sched(
        "cosine", optimizer=optimizer, num_warmup_steps=min(100, steps // 10),
        num_training_steps=steps,
    )

    # ----------------------------------------------------------------
    # Text encoding helpers
    # ----------------------------------------------------------------
    def encode_text_sd(caps_batch):
        """Encode text for SD 1.5/2.1 (single text encoder)."""
        tok = tokenizer(
            caps_batch, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            hidden = text_encoder(tok.input_ids.to(device))[0]
        return hidden, {}

    def encode_text_sdxl(caps_batch):
        """Encode text for SDXL (dual text encoders + pooled output)."""
        tok1 = tokenizer(
            caps_batch, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        tok2 = tokenizer_2(
            caps_batch, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            enc1 = text_encoder(tok1.input_ids.to(device))[0]
            enc2_out = text_encoder_2(tok2.input_ids.to(device))
            enc2 = enc2_out[0]
            pooled = enc2_out[1]

        hidden = torch.cat([enc1, enc2], dim=-1)

        # SDXL time IDs: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
        time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            dtype=torch.float16, device=device,
        ).repeat(len(caps_batch), 1)

        added_cond = {"text_embeds": pooled, "time_ids": time_ids}
        return hidden, added_cond

    encode_text = encode_text_sdxl if sdxl else encode_text_sd

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    unet.train()
    global_step = 0
    print(f"\nTraining for {steps} steps …\n")

    while global_step < steps:
        for batch in dl:
            if global_step >= steps:
                break

            pv = batch["pixel_values"].to(device, dtype=torch.float16)
            caps = batch["caption"]

            # Encode image → latent
            with torch.no_grad():
                latents = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor

            # Encode text
            enc_hidden, added_cond = encode_text(caps)

            # Noise
            noise = torch.randn_like(latents)
            ts = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy = noise_scheduler.add_noise(latents, noise, ts)

            # Predict
            unet_kwargs = dict(
                sample=noisy, timestep=ts, encoder_hidden_states=enc_hidden,
            )
            if added_cond:
                unet_kwargs["added_cond_kwargs"] = added_cond

            pred = unet(**unet_kwargs).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())

            loss.backward()
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 50 == 0 or global_step == 1:
                print(f"  step {global_step:>5}/{steps}  loss={loss.item():.5f}")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out_dir = WAREHOUSE / "lora" / character_name.lower().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        unet.save_pretrained(str(out_dir))
    except Exception:
        torch.save(
            {k: v for k, v in unet.state_dict().items() if "lora" in k},
            str(out_dir / "pytorch_lora_weights.safetensors"),
        )

    # Metadata
    meta = {
        "character_name": character_name,
        "character_id": character_name.lower().replace(" ", "_"),
        "lora_path": str(out_dir / "pytorch_lora_weights.safetensors"),
        "base_model": base_model,
        "is_sdxl": sdxl,
        "rank": rank,
        "learning_rate": lr,
        "training_steps": steps,
        "num_images": len(images),
        "resolution": resolution,
        "trained_at": datetime.now().isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nLoRA saved to {out_dir}")
    print(f"Metadata: {out_dir / 'metadata.json'}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Train a character LoRA for AnimeLoom")
    p.add_argument("--name", required=True, help="Character name")
    p.add_argument("--images", help="Folder of raw character images")
    p.add_argument("--tagged", help="Folder of tagged images (with .txt captions)")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--base-model", default="cagliostrolab/animagine-xl-3.1")
    args = p.parse_args()

    if args.tagged:
        image_dir = Path(args.tagged)
        use_captions = True
    elif args.images:
        image_dir = Path(args.images)
        use_captions = False
    else:
        print("Provide --images or --tagged")
        sys.exit(1)

    train(
        character_name=args.name,
        image_dir=image_dir,
        rank=args.rank,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        resolution=args.resolution,
        base_model=args.base_model,
        use_captions=use_captions,
    )


if __name__ == "__main__":
    main()
