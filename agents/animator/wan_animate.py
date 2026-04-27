"""
Wan2.2-Animate-14B wrapper — face-locked character animation.

Wan2.2 Animate decouples body motion (skeleton) and facial expression (face image).
The face is literally pasted from the reference image at every frame, giving near-
perfect identity consistency across shots.

Usage (Move Mode):
    animator = Wan22AnimateWrapper(warehouse_path)
    animator.load(offload_mode="model")  # or "sequential"
    out_frames = animator.animate(
        reference_image=keyframe_pil,        # the SDXL identity keyframe
        driving_frames=wan22_i2v_clip_pil,   # frames from a Wan2.2 I2V driving clip
        num_inference_steps=30,
        guidance_scale=4.0,
    )
    animator.unload()
"""

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image


class Wan22AnimateWrapper:
    """Thin wrapper around Wan-AI/Wan2.2-Animate-14B (Move Mode)."""

    MODEL_ID = "Wan-AI/Wan2.2-Animate-14B"

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._pipeline = None
        self._offload_mode: Optional[str] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, offload_mode: str = "model") -> bool:
        """
        Load Wan2.2-Animate-14B with the requested offload strategy.

        Args:
            offload_mode: "model" (faster, more VRAM) or "sequential" (slower, less VRAM).

        Returns:
            True if loaded successfully.
        """
        if self._pipeline is not None:
            return True

        try:
            from diffusers import DiffusionPipeline

            self._pipeline = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
                cache_dir=str(self.warehouse / "models"),
            )

            if offload_mode == "sequential":
                self._pipeline.enable_sequential_cpu_offload()
            else:
                self._pipeline.enable_model_cpu_offload()

            if hasattr(self._pipeline, "vae"):
                try: self._pipeline.vae.enable_slicing()
                except Exception: pass
                try: self._pipeline.vae.enable_tiling()
                except Exception: pass

            self._offload_mode = offload_mode
            print(f"  Wan2.2-Animate-14B loaded ({offload_mode} offload)")
            return True
        except Exception as e:
            print(f"  Wan2.2-Animate-14B load failed: {e}")
            self._pipeline = None
            return False

    def animate(
        self,
        reference_image: Image.Image,
        driving_frames: List[Image.Image],
        width: int = 480,
        height: int = 832,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int = 42,
    ) -> Optional[List[Image.Image]]:
        """
        Run Move Mode: animate `reference_image` using motion from `driving_frames`.

        The output character has the face/identity of `reference_image` and the
        body motion of `driving_frames`. Frame count = len(driving_frames).

        Args:
            reference_image: PIL Image of the character (identity source).
            driving_frames: List of PIL Images = motion source (skeleton extracted
                            internally by Wan2.2-Animate's PoseAndFace pipeline).
            width, height: Output resolution. Should match driving_frames.
            num_inference_steps: Diffusion steps (30 = quality, 20 = faster).
            guidance_scale: Lower = more motion freedom, higher = locks to reference.
            seed: RNG seed.

        Returns:
            List of PIL Image frames or None on failure.
        """
        if self._pipeline is None:
            print("  Wan2.2-Animate not loaded — call load() first")
            return None

        if not driving_frames:
            print("  No driving frames provided")
            return None

        ref = reference_image.convert("RGB").resize((width, height), Image.LANCZOS)
        driving = [f.convert("RGB").resize((width, height), Image.LANCZOS) for f in driving_frames]

        gen = torch.Generator("cpu").manual_seed(seed)

        try:
            result = self._pipeline(
                image=ref,
                video=driving,
                height=height,
                width=width,
                num_frames=len(driving),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            )
            return list(result.frames[0])
        except TypeError:
            try:
                result = self._pipeline(
                    reference_image=ref,
                    driving_video=driving,
                    height=height,
                    width=width,
                    num_frames=len(driving),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )
                return list(result.frames[0])
            except Exception as e:
                print(f"  Wan2.2-Animate generation failed (alt API): {e}")
                return None
        except Exception as e:
            print(f"  Wan2.2-Animate generation failed: {e}")
            return None

    def unload(self):
        """Free VRAM occupied by the Animate pipeline."""
        if self._pipeline is None:
            return
        try:
            del self._pipeline
        except Exception:
            pass
        self._pipeline = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
