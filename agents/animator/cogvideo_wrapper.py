"""
CogVideoX-2B Animator — primary text-to-video backend for T4 GPUs.

Generates 49-frame 480×720 clips at 8fps (~6 seconds per shot).
Supports both text-to-video and image-to-video (reference keyframe conditioning).
"""

import gc
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


class CogVideoXAnimator:
    """
    Text-to-video anime generation via CogVideoX-2B (THUDM).

    Fits on T4 (16GB VRAM) with CPU offloading.  Produces 49-frame clips
    at 480×720 resolution, 8 fps.  Supports image-conditioned generation
    for character consistency (feed an SDXL+LoRA reference keyframe).
    """

    _MODEL_ID = "THUDM/CogVideoX-2b"
    _I2V_MODEL_ID = "THUDM/CogVideoX-5b-I2V"  # image-to-video variant

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self.output_dir = self.warehouse / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._t2v_pipe = None
        self._i2v_pipe = None
        self._t2v_failed = False
        self._i2v_failed = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def _load_t2v_pipeline(self):
        """Lazy-load the CogVideoX text-to-video pipeline."""
        if self._t2v_pipe is not None or self._t2v_failed:
            return

        try:
            from diffusers import CogVideoXPipeline

            # Check local cache first
            local_path = self.warehouse / "models" / "cogvideox-2b"
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = self._MODEL_ID

            self._t2v_pipe = CogVideoXPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                cache_dir=str(self.warehouse / "models"),
            )
            self._t2v_pipe.enable_model_cpu_offload()
            self._t2v_pipe.vae.enable_slicing()
            self._t2v_pipe.vae.enable_tiling()
            print(f"CogVideoX T2V loaded: {model_path}")
        except Exception as e:
            print(f"CogVideoX T2V not available: {e}")
            self._t2v_failed = True

    def _load_i2v_pipeline(self):
        """Lazy-load the CogVideoX image-to-video pipeline."""
        if self._i2v_pipe is not None or self._i2v_failed:
            return

        try:
            from diffusers import CogVideoXImageToVideoPipeline

            local_path = self.warehouse / "models" / "cogvideox-5b-i2v"
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = self._I2V_MODEL_ID

            self._i2v_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                cache_dir=str(self.warehouse / "models"),
            )
            self._i2v_pipe.enable_model_cpu_offload()
            self._i2v_pipe.vae.enable_slicing()
            self._i2v_pipe.vae.enable_tiling()
            print(f"CogVideoX I2V loaded: {model_path}")
        except Exception as e:
            print(f"CogVideoX I2V not available: {e}")
            self._i2v_failed = True

    def _cleanup_gpu(self):
        """Free GPU memory after generation."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        description: str,
        character_loras: Optional[Dict[str, str]] = None,
        reference_image: Optional[str] = None,
        shot_index: int = 0,
        num_frames: int = 49,
        width: int = 720,
        height: int = 480,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
    ) -> Optional[Dict]:
        """
        Generate a video clip via CogVideoX.

        If *reference_image* is provided, uses image-to-video mode for
        character consistency.  Otherwise uses text-to-video.

        Returns:
            Dict with video_path, metadata — or None if generation fails.
        """
        output_path = self.output_dir / f"cogvx_shot_{shot_index:04d}_{int(time.time())}.mp4"

        # Enhance prompt for anime style
        anime_prompt = self._enhance_prompt(description)

        # Try image-to-video first (better character consistency)
        if reference_image and Path(reference_image).exists():
            result = self._generate_i2v(
                anime_prompt, reference_image, str(output_path),
                num_frames, width, height, guidance_scale, num_inference_steps,
            )
            if result:
                return {
                    "video_path": str(output_path),
                    "shot_index": shot_index,
                    "prompt": description,
                    "num_frames": num_frames,
                    "status": "cogvideox_i2v",
                }

        # Fall back to text-to-video
        result = self._generate_t2v(
            anime_prompt, str(output_path),
            num_frames, width, height, guidance_scale, num_inference_steps,
        )
        if result:
            return {
                "video_path": str(output_path),
                "shot_index": shot_index,
                "prompt": description,
                "num_frames": num_frames,
                "status": "cogvideox_t2v",
            }

        return None

    # ------------------------------------------------------------------
    # Text-to-Video
    # ------------------------------------------------------------------

    def _generate_t2v(
        self,
        prompt: str,
        output_path: str,
        num_frames: int = 49,
        width: int = 720,
        height: int = 480,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
    ) -> bool:
        """Generate video from text prompt."""
        try:
            self._load_t2v_pipeline()
            if self._t2v_pipe is None:
                return False

            result = self._t2v_pipe(
                prompt=prompt,
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(self._device).manual_seed(
                    hash(prompt) % (2**31)
                ),
            )

            frames = result.frames[0]
            self._frames_to_video(frames, output_path, fps=8)
            print(f"  CogVideoX T2V generated: {output_path} ({len(frames)} frames)")

            self._cleanup_gpu()
            return True

        except Exception as e:
            print(f"  CogVideoX T2V generation failed: {e}")
            self._cleanup_gpu()
            return False

    # ------------------------------------------------------------------
    # Image-to-Video (reference keyframe conditioning)
    # ------------------------------------------------------------------

    def _generate_i2v(
        self,
        prompt: str,
        reference_image_path: str,
        output_path: str,
        num_frames: int = 49,
        width: int = 720,
        height: int = 480,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
    ) -> bool:
        """Generate video conditioned on a reference image."""
        try:
            self._load_i2v_pipeline()
            if self._i2v_pipe is None:
                return False

            ref_img = Image.open(reference_image_path).convert("RGB")
            ref_img = ref_img.resize((width, height), Image.LANCZOS)

            result = self._i2v_pipe(
                prompt=prompt,
                image=ref_img,
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(self._device).manual_seed(
                    hash(prompt) % (2**31)
                ),
            )

            frames = result.frames[0]
            self._frames_to_video(frames, output_path, fps=8)
            print(f"  CogVideoX I2V generated: {output_path} ({len(frames)} frames)")

            self._cleanup_gpu()
            return True

        except Exception as e:
            print(f"  CogVideoX I2V generation failed: {e}")
            self._cleanup_gpu()
            return False

    # ------------------------------------------------------------------
    # Prompt enhancement
    # ------------------------------------------------------------------

    def _enhance_prompt(self, description: str) -> str:
        """Enhance a user prompt for anime-style video generation."""
        anime_keywords = [
            "anime style", "high quality animation", "detailed anime art",
            "vibrant colors", "smooth motion",
        ]
        # Only add keywords if not already present
        lower = description.lower()
        additions = [kw for kw in anime_keywords if kw not in lower]
        if additions:
            return f"{description}, {', '.join(additions)}"
        return description

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _frames_to_video(self, frames: list, output_path: str, fps: int = 8):
        """Write a list of PIL Images / numpy arrays to an MP4 file."""
        try:
            import cv2

            if not frames:
                return

            first = np.array(frames[0])
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for frame in frames:
                arr = np.array(frame)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                writer.write(arr)

            writer.release()
        except ImportError:
            frame_dir = Path(output_path).with_suffix("")
            frame_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                if isinstance(frame, Image.Image):
                    frame.save(str(frame_dir / f"frame_{i:04d}.png"))

    def unload(self):
        """Explicitly unload all pipelines to free VRAM."""
        self._t2v_pipe = None
        self._i2v_pipe = None
        self._cleanup_gpu()
