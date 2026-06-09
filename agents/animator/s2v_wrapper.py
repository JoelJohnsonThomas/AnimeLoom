"""
Wan2.2-S2V speech-to-video wrapper — audio-driven talking characters.

Takes a character reference image + an audio clip and generates a video
whose lip/face motion follows the speech. Chosen over post-hoc lip-sync
tools (LatentSync/MuseTalk) because their face detectors are trained on
real human faces and routinely fail on anime characters; S2V shares the
Wan ecosystem and handles stylized faces.

Note: the official Wan-AI/Wan2.2-S2V-14B repo is the raw checkpoint
layout. The diffusers-loadable conversion (WanSpeechToVideoPipeline) is
tolgacangoz/Wan2.2-S2V-14B-Diffusers.
"""

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from agents.animator.wan_wrapper import WAN_NEGATIVE_PROMPT


class WanS2VWrapper:
    """Wraps Wan2.2-S2V-14B (speech-to-video) via diffusers."""

    MODEL_ID = "tolgacangoz/Wan2.2-S2V-14B-Diffusers"

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._pipeline = None
        self._load_failed = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self, offload_mode: Optional[str] = None) -> bool:
        if self._pipeline is not None:
            return True
        if self._load_failed:
            return False

        if offload_mode is None:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                offload_mode = "none" if total > 38e9 else "model"
            else:
                offload_mode = "model"

        try:
            import diffusers

            pipeline_cls = getattr(diffusers, "WanSpeechToVideoPipeline", None)
            if pipeline_cls is None:
                print(
                    f"  WanSpeechToVideoPipeline not in diffusers "
                    f"{diffusers.__version__} — upgrade diffusers for lip-sync"
                )
                self._load_failed = True
                return False

            from diffusers import AutoencoderKLWan

            vae = AutoencoderKLWan.from_pretrained(
                self.MODEL_ID,
                subfolder="vae",
                torch_dtype=torch.float32,
                cache_dir=str(self.warehouse / "models"),
            )
            self._pipeline = pipeline_cls.from_pretrained(
                self.MODEL_ID,
                vae=vae,
                torch_dtype=torch.bfloat16,
                cache_dir=str(self.warehouse / "models"),
            )

            if offload_mode == "none":
                self._pipeline.to(self._device)
            elif offload_mode == "sequential":
                self._pipeline.enable_sequential_cpu_offload()
            else:
                self._pipeline.enable_model_cpu_offload()

            if hasattr(self._pipeline, "vae"):
                try:
                    self._pipeline.vae.enable_slicing()
                    self._pipeline.vae.enable_tiling()
                except Exception:
                    pass

            print(f"  Wan2.2-S2V-14B loaded (bfloat16, {offload_mode} offload)")
            return True
        except Exception as e:
            print(f"  Wan2.2-S2V-14B load failed: {e}")
            self._pipeline = None
            self._load_failed = True
            return False

    def unload(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        reference_image: Image.Image,
        audio_path: str,
        prompt: str = "",
        negative_prompt: str = WAN_NEGATIVE_PROMPT,
        width: int = 832,
        height: int = 480,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        fps: int = 16,
        seed: int = 42,
    ) -> Optional[List[Image.Image]]:
        """Generate a talking-character video driven by an audio clip.

        Returns frames (16 fps; duration follows the audio) or None.
        """
        if not self.load():
            return None
        if not Path(audio_path).exists():
            print(f"  S2V: audio file missing: {audio_path}")
            return None

        try:
            import librosa

            audio, sampling_rate = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"  S2V: failed to load audio ({e})")
            return None

        ref = reference_image.convert("RGB").resize((width, height), Image.LANCZOS)
        generator = torch.Generator("cpu").manual_seed(seed)

        try:
            result = self._pipeline(
                image=ref,
                audio=audio,
                sampling_rate=sampling_rate,
                prompt=prompt or None,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return list(result.frames[0])
        except Exception as e:
            print(f"  S2V generation failed: {e}")
            return None
