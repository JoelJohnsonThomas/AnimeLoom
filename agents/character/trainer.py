"""
LoRA Trainer - Trains LoRA adapters for character consistency.
Uses PEFT + Diffusers for efficient fine-tuning (rank 16-32, fp16).
Supports both SD 1.5/2.1 and SDXL-based models (e.g. Animagine XL 3.1).
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

_SDXL_KEYWORDS = ["xl", "animagine"]


def _is_sdxl(model_id: str) -> bool:
    lower = model_id.lower()
    return any(kw in lower for kw in _SDXL_KEYWORDS)


class CharacterDataset(Dataset):
    """Dataset for character LoRA training from reference images."""

    def __init__(self, images: List[str], size: int = 1024, repeats: int = 10):
        self.images = [p for p in images if os.path.exists(p)]
        self.size = size
        self.repeats = repeats

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, idx):
        img_path = self.images[idx % len(self.images)]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)

        # Normalise to [-1, 1]
        pixel_values = np.array(image, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(pixel_values).permute(2, 0, 1)

        return {
            "pixel_values": pixel_values,
            "caption": "a character portrait, anime style",
        }


class LoRATrainer:
    """
    Trains rank-16/32 fp16 LoRA adapters for character identity preservation.
    """

    DEFAULT_RANK = 32
    DEFAULT_LR = 1e-4
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_GRAD_ACCUM = 4
    DEFAULT_MAX_STEPS = 1000

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self.lora_dir = self.warehouse / "lora"
        self.lora_dir.mkdir(exist_ok=True)

        self.rank = self.DEFAULT_RANK
        self.learning_rate = self.DEFAULT_LR
        self.train_batch_size = self.DEFAULT_BATCH_SIZE
        self.gradient_accumulation_steps = self.DEFAULT_GRAD_ACCUM
        self.max_train_steps = self.DEFAULT_MAX_STEPS

    def train_character_lora(
        self,
        character_images: List[str],
        character_id: str,
        character_name: str = None,
        rank: int = None,
        max_steps: int = None,
    ) -> Path:
        """
        Train a LoRA adapter for a character.

        Args:
            character_images: Paths to reference images.
            character_id: Unique character identifier.
            character_name: Human-readable name (used in prompts).
            rank: LoRA rank override (default 32).
            max_steps: Training step override.

        Returns:
            Path to saved LoRA weights (.safetensors).
        """
        from diffusers import DDPMScheduler
        from peft import LoraConfig, get_peft_model

        rank = rank or self.rank
        max_steps = max_steps or self.max_train_steps
        output_dir = self.lora_dir / character_id
        output_dir.mkdir(exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- Load base model ------------------------------------------------
        model_id = "cagliostrolab/animagine-xl-3.1"
        sdxl = _is_sdxl(model_id)

        if sdxl:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, variant="fp16",
            )
        else:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )

        pipe.to(device)

        # Freeze everything
        pipe.vae.requires_grad_(False)
        pipe.unet.requires_grad_(False)

        if sdxl:
            pipe.text_encoder.requires_grad_(False)
            pipe.text_encoder_2.requires_grad_(False)
        else:
            pipe.text_encoder.requires_grad_(False)

        # ---- Attach LoRA via PEFT -------------------------------------------
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        unet = get_peft_model(pipe.unet, lora_config)
        unet.print_trainable_parameters()

        # ---- Dataset & DataLoader -------------------------------------------
        resolution = 1024 if sdxl else 512
        dataset = CharacterDataset(character_images, size=resolution)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
        )

        # ---- Optimizer & scheduler ------------------------------------------
        trainable = [p for p in unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=1e-2)
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        from diffusers.optimization import get_scheduler as get_lr_scheduler

        lr_scheduler = get_lr_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=min(100, max_steps // 10),
            num_training_steps=max_steps,
        )

        # ---- Text encoding helpers ------------------------------------------
        def encode_text(caps_batch):
            if sdxl:
                tok1 = pipe.tokenizer(
                    caps_batch, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt",
                )
                tok2 = pipe.tokenizer_2(
                    caps_batch, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt",
                )
                with torch.no_grad():
                    enc1 = pipe.text_encoder(tok1.input_ids.to(device))[0]
                    enc2_out = pipe.text_encoder_2(tok2.input_ids.to(device))
                    enc2 = enc2_out[0]
                    pooled = enc2_out[1]
                hidden = torch.cat([enc1, enc2], dim=-1)
                time_ids = torch.tensor(
                    [[resolution, resolution, 0, 0, resolution, resolution]],
                    dtype=torch.float16, device=device,
                ).repeat(len(caps_batch), 1)
                return hidden, {"text_embeds": pooled, "time_ids": time_ids}
            else:
                tok = pipe.tokenizer(
                    caps_batch, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt",
                )
                with torch.no_grad():
                    hidden = pipe.text_encoder(tok.input_ids.to(device))[0]
                return hidden, {}

        # ---- Training loop --------------------------------------------------
        global_step = 0
        unet.train()

        print(f"Training LoRA for '{character_name or character_id}' "
              f"(rank={rank}, steps={max_steps})...")

        for epoch in range(max_steps // max(len(dataloader), 1) + 1):
            for batch in dataloader:
                if global_step >= max_steps:
                    break

                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                caps = batch["caption"]

                # Encode images -> latents
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                # Encode text
                encoder_hidden, added_cond = encode_text(caps)

                # Sample noise & timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict & loss
                unet_kwargs = dict(
                    sample=noisy_latents, timestep=timesteps,
                    encoder_hidden_states=encoder_hidden,
                )
                if added_cond:
                    unet_kwargs["added_cond_kwargs"] = added_cond

                noise_pred = unet(**unet_kwargs).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                if global_step % 100 == 0:
                    print(f"  step {global_step}/{max_steps}  loss={loss.item():.4f}")

        # ---- Save -----------------------------------------------------------
        lora_path = output_dir / "pytorch_lora_weights.safetensors"

        # Save via PEFT
        unet.save_pretrained(str(output_dir))
        print(f"LoRA saved to {output_dir}")

        # Move back to float16 for inference
        unet.to(torch.float16)
        return lora_path
