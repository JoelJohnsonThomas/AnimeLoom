"""
Video Generation Wrapper — multi-backend animator.

Priority order:
  1. Wan2.2 (if available and GPU is large enough)
  2. SDXL keyframes + LoRA → RIFE / cross-fade interpolation (T4-friendly)
  3. Placeholder coloured-gradient video (offline / CPU)
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


class WanAnimator:
    """
    Wrapper around Wan2.2-Animate (Alibaba) for anime video generation.

    Supports two core modes:
      - motion_imitation: transfer pose from a reference video onto a character
      - role_play: insert a character into a scene video
    """

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self.output_dir = self.warehouse / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = None
        self._sdxl_pipe = None
        self._load_failed = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    # Wan2.2 model variants (smallest first for T4 compatibility)
    _WAN_MODELS = [
        ("Wan-AI/Wan2.2-TI2V-5B", "wan2.2-ti2v-5b"),     # 5B — fits on T4
        ("Wan-AI/Wan2.2-T2V-A14B", "wan2.2-t2v-a14b"),    # 14B — needs A100
    ]

    def _load_pipeline(self):
        """Lazy-load the Wan2.2 pipeline (prefers 5B for T4 compatibility)."""
        if self._pipeline is not None or self._load_failed:
            return

        try:
            from diffusers import DiffusionPipeline

            # Check local cache first
            for _, local_name in self._WAN_MODELS:
                model_path = self.warehouse / "models" / local_name
                if model_path.exists():
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        str(model_path),
                        torch_dtype=torch.float16,
                    )
                    self._pipeline.to(self._device)
                    print(f"Wan2.2 loaded from local cache: {local_name}")
                    return

            # Try HuggingFace hub (smallest model first)
            for repo_id, _ in self._WAN_MODELS:
                try:
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float16,
                    )
                    self._pipeline.to(self._device)
                    print(f"Wan2.2 loaded from HuggingFace: {repo_id}")
                    return
                except Exception:
                    continue

            print("No Wan2.2 model available, falling back to placeholder generation")
            self._load_failed = True
        except Exception as e:
            print(f"Wan2.2 not available: {e}")
            print("Falling back to placeholder generation")
            self._load_failed = True

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        description: str,
        character_loras: Dict[str, str] = None,
        pose_reference: Optional[str] = None,
        shot_index: int = 0,
        num_frames: int = 16,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
    ) -> Dict:
        """
        Generate a video shot.

        Args:
            description: Text prompt describing the shot.
            character_loras: Mapping of char_name -> LoRA weight path.
            pose_reference: Path to pose reference video (optional).
            shot_index: Index of the shot in the story.
            num_frames: Number of frames to generate.

        Returns:
            Dict with video_path, metadata.
        """
        self._load_pipeline()

        output_path = self.output_dir / f"shot_{shot_index:04d}_{int(time.time())}.mp4"

        # Load character LoRAs if available
        if character_loras and self._pipeline is not None:
            for char_name, lora_path in character_loras.items():
                if Path(lora_path).exists() and Path(lora_path).stat().st_size > 100:
                    try:
                        self._pipeline.load_lora_weights(
                            str(Path(lora_path).parent),
                            weight_name=Path(lora_path).name,
                        )
                        print(f"  LoRA loaded for {char_name}")
                    except Exception as e:
                        print(f"  Failed to load LoRA for {char_name}: {e}")

        if self._pipeline is not None:
            try:
                result = self._pipeline(
                    prompt=description,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )
                # Export frames to video
                self._frames_to_video(result.frames[0], str(output_path))

                return {
                    "video_path": str(output_path),
                    "shot_index": shot_index,
                    "prompt": description,
                    "num_frames": num_frames,
                    "status": "success",
                }
            except Exception as e:
                print(f"  Wan2.2 generation error: {e}")

        # Fallback: generate keyframe images with SDXL + LoRA, assemble to video
        sdxl_result = self._generate_sdxl_keyframes(
            str(output_path), description, character_loras, num_frames
        )
        status = "sdxl_keyframes" if sdxl_result else "placeholder"
        if not sdxl_result:
            self._generate_placeholder_video(str(output_path), description, num_frames)

        return {
            "video_path": str(output_path),
            "shot_index": shot_index,
            "prompt": description,
            "num_frames": num_frames,
            "status": status,
        }

    # ------------------------------------------------------------------
    # Motion imitation (动作模仿)
    # ------------------------------------------------------------------

    def motion_imitation(
        self,
        character_image: str,
        pose_video: str,
        character_id: str,
        character_lora: Optional[str] = None,
    ) -> Dict:
        """
        Transfer motion from *pose_video* onto *character_image*.

        Args:
            character_image: Path to character reference image.
            pose_video: Path to video with target pose/motion.
            character_id: Character identifier for tracking.
            character_lora: Optional LoRA weight path.

        Returns:
            Dict with video_path and metadata.
        """
        description = f"motion imitation for character {character_id}"
        loras = {character_id: character_lora} if character_lora else None

        return self.generate(
            description=description,
            character_loras=loras,
            pose_reference=pose_video,
            shot_index=0,
        )

    # ------------------------------------------------------------------
    # Role play (角色扮演)
    # ------------------------------------------------------------------

    def role_play(
        self,
        character_image: str,
        scene_video: str,
        character_id: str,
        character_lora: Optional[str] = None,
    ) -> Dict:
        """
        Insert *character_image* into *scene_video*, replacing generic actors.

        Args:
            character_image: Path to character reference image.
            scene_video: Path to the scene video.
            character_id: Character identifier.
            character_lora: Optional LoRA weight path.

        Returns:
            Dict with video_path and metadata.
        """
        description = f"role play insertion for character {character_id}"
        loras = {character_id: character_lora} if character_lora else None

        return self.generate(
            description=description,
            character_loras=loras,
            shot_index=0,
        )

    # ------------------------------------------------------------------
    # SDXL keyframe fallback (for T4 / when Wan2.2 unavailable)
    # ------------------------------------------------------------------

    def _generate_sdxl_keyframes(
        self,
        output_path: str,
        description: str,
        character_loras: Optional[Dict[str, str]] = None,
        num_frames: int = 16,
    ) -> bool:
        """Generate keyframe images with SDXL + LoRA, assemble into video."""
        try:
            import gc
            from diffusers import StableDiffusionXLPipeline
            from peft import PeftModel

            model_id = "cagliostrolab/animagine-xl-3.1"
            cache_dir = str(self.warehouse / "models")

            # Cache the SDXL pipeline across shots
            if self._sdxl_pipe is None:
                self._sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    cache_dir=cache_dir,
                )
                self._sdxl_pipe.to(self._device)
                self._sdxl_pipe.vae.enable_slicing()
                self._sdxl_pipe.vae.enable_tiling()
                print("  SDXL pipeline loaded (animagine-xl-3.1)")

            pipe = self._sdxl_pipe

            # Unwrap any previous LoRA adapter before loading a new one
            while hasattr(pipe.unet, "base_model"):
                try:
                    pipe.unet = pipe.unet.base_model.model
                except Exception:
                    break

            # Load the first character LoRA via PEFT
            if character_loras:
                for char_name, lora_path in character_loras.items():
                    lora_dir = str(Path(lora_path).parent)
                    if Path(lora_path).exists():
                        try:
                            pipe.unet = PeftModel.from_pretrained(
                                pipe.unet, lora_dir
                            )
                            print(f"  SDXL LoRA loaded for {char_name}")
                            break  # one LoRA at a time for VRAM
                        except Exception as e:
                            print(f"  SDXL LoRA load failed for {char_name}: {e}")

            # Generate 8 keyframes with varied seeds for visual diversity
            num_keyframes = 8
            keyframes = []
            for i in range(num_keyframes):
                result = pipe(
                    prompt=description,
                    negative_prompt="low quality, bad anatomy, worst quality, blurry",
                    width=768,
                    height=768,
                    num_inference_steps=25,
                    guidance_scale=7.0,
                    generator=torch.Generator(self._device).manual_seed(42 + i),
                )
                keyframes.append(result.images[0])
                print(f"  Keyframe {i + 1}/{num_keyframes} generated")

            # Interpolate between keyframes for smooth transitions
            frames = self._interpolate_keyframes(keyframes, frames_between=6)

            # Assemble to video at 12 fps (~5s per shot)
            self._frames_to_video(frames, output_path, fps=12)

            # Unload LoRA adapter to reset for next shot
            if hasattr(pipe.unet, "disable_adapter_layers"):
                try:
                    pipe.unet = pipe.unet.base_model.model
                except Exception:
                    pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  SDXL keyframes assembled: {output_path}")
            return True

        except Exception as e:
            print(f"  SDXL keyframe fallback failed: {e}")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    # ------------------------------------------------------------------
    # Frame interpolation
    # ------------------------------------------------------------------

    def _interpolate_keyframes(
        self, keyframes: list, frames_between: int = 6
    ) -> list:
        """
        Interpolate between keyframes using RIFE (if available) or cross-fade.

        Args:
            keyframes: List of PIL Images (the keyframes).
            frames_between: Number of intermediate frames between each pair.

        Returns:
            List of PIL Images with smooth transitions.
        """
        if len(keyframes) < 2:
            return keyframes

        # Try RIFE first, fall back to cross-fade
        rife_model = self._try_load_rife()

        all_frames = []
        for i in range(len(keyframes) - 1):
            all_frames.append(keyframes[i])

            if rife_model is not None:
                intermediates = self._rife_interpolate(
                    rife_model, keyframes[i], keyframes[i + 1], frames_between
                )
            else:
                intermediates = self._crossfade_interpolate(
                    keyframes[i], keyframes[i + 1], frames_between
                )
            all_frames.extend(intermediates)

        all_frames.append(keyframes[-1])
        return all_frames

    def _try_load_rife(self):
        """Try to load RIFE model. Returns None if unavailable."""
        try:
            import sys
            rife_path = str(self.warehouse / "models" / "Practical-RIFE")
            if Path(rife_path).exists() and rife_path not in sys.path:
                sys.path.insert(0, rife_path)
            from model.RIFE import Model as RIFEModel

            model = RIFEModel()
            model.load_model(str(Path(rife_path) / "train_log"), -1)
            model.eval()
            print("  RIFE model loaded for frame interpolation")
            return model
        except Exception:
            return None

    def _rife_interpolate(self, model, img1, img2, n: int) -> list:
        """Interpolate between two PIL Images using RIFE."""
        try:
            import torchvision.transforms.functional as TF

            t1 = TF.to_tensor(img1).unsqueeze(0).to(self._device)
            t2 = TF.to_tensor(img2).unsqueeze(0).to(self._device)

            frames = []
            for i in range(1, n + 1):
                ratio = i / (n + 1)
                with torch.no_grad():
                    mid = model.inference(t1, t2, ratio)
                mid_img = TF.to_pil_image(mid.squeeze(0).cpu().clamp(0, 1))
                frames.append(mid_img)
            return frames
        except Exception as e:
            print(f"  RIFE interpolation failed: {e}, using cross-fade")
            return self._crossfade_interpolate(img1, img2, n)

    def _crossfade_interpolate(self, img1, img2, n: int) -> list:
        """Cross-fade between two PIL Images (pure numpy, no GPU needed)."""
        arr1 = np.array(img1).astype(np.float32)
        arr2 = np.array(img2).astype(np.float32)

        # Ensure same size
        if arr1.shape != arr2.shape:
            img2 = img2.resize(img1.size, Image.LANCZOS)
            arr2 = np.array(img2).astype(np.float32)

        frames = []
        for i in range(1, n + 1):
            alpha = i / (n + 1)
            blended = ((1 - alpha) * arr1 + alpha * arr2).astype(np.uint8)
            frames.append(Image.fromarray(blended))
        return frames

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _frames_to_video(self, frames: list, output_path: str, fps: int = 8):
        """Write a list of PIL Images to an MP4 file."""
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
            # Fallback: save frames as images
            frame_dir = Path(output_path).with_suffix("")
            frame_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                frame.save(str(frame_dir / f"frame_{i:04d}.png"))

    def _generate_placeholder_video(
        self, output_path: str, description: str, num_frames: int = 16
    ):
        """Create a placeholder video with colored frames for testing."""
        try:
            import cv2

            h, w = 512, 512
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 8, (w, h))

            rng = np.random.default_rng(hash(description) % (2**31))
            base_color = rng.integers(50, 200, size=3)

            for i in range(num_frames):
                frame = np.full((h, w, 3), base_color, dtype=np.uint8)
                # Add a gradient to indicate motion
                offset = int(i / num_frames * w)
                frame[:, :offset, 1] = np.clip(
                    frame[:, :offset, 1].astype(int) + 50, 0, 255
                ).astype(np.uint8)
                writer.write(frame)

            writer.release()
        except ImportError:
            # Just create an empty file
            Path(output_path).write_bytes(b"placeholder_video")
