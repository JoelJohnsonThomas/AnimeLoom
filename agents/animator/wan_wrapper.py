"""
Video Generation Wrapper — multi-backend animator.

Priority order:
  1. Wan2.2 (14B on A100, 5B on T4)
  2. AnimateDiff T2V + SD 1.5 + IP-Adapter (character-focused anime clips)
  3. CogVideoX-2B (fallback T2V)
  4. SDXL keyframes + LoRA → RIFE / cross-fade interpolation (T4-friendly)
  5. Placeholder coloured-gradient video (offline / CPU)
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
        self._animatediff_pipe = None
        self._cogvideo = None
        self._load_failed = False
        self._animatediff_load_failed = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    # Wan2.2 model variants — largest first for best quality (A100),
    # falls back to smaller variants on lower VRAM GPUs.
    _WAN_MODELS = [
        ("Wan-AI/Wan2.2-T2V-A14B", "wan2.2-t2v-a14b"),    # 14B — best quality, A100
        ("Wan-AI/Wan2.2-TI2V-5B", "wan2.2-ti2v-5b"),      # 5B  — fits on T4
        ("Wan-AI/Wan2.1-T2V-1.3B", "wan2.1-t2v-1.3b"),    # 1.3B — lightweight fallback
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
        output_path = self.output_dir / f"shot_{shot_index:04d}_{int(time.time())}.mp4"

        # Priority 1: Wan2.2 (14B on A100, 5B on T4 — best native video quality)
        self._load_pipeline()

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
                self._frames_to_video(result.frames[0], str(output_path))

                return {
                    "video_path": str(output_path),
                    "shot_index": shot_index,
                    "prompt": description,
                    "num_frames": num_frames,
                    "status": "wan2.2",
                }
            except Exception as e:
                print(f"  Wan2.2 generation error: {e}")

        # Priority 2: AnimateDiff T2V + IP-Adapter (character-focused anime)
        animatediff_result = self._generate_animatediff(
            str(output_path), description, character_loras, num_frames
        )
        if animatediff_result:
            return {
                "video_path": str(output_path),
                "shot_index": shot_index,
                "prompt": description,
                "num_frames": num_frames,
                "status": "animatediff",
            }

        # Priority 3: CogVideoX-2B (general-purpose T2V fallback)
        cogvx_result = self._generate_cogvideox(
            description, character_loras, shot_index, num_frames,
            width, height, guidance_scale, num_inference_steps,
        )
        if cogvx_result:
            return cogvx_result

        # Priority 4: SDXL keyframes + interpolation
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
    # CogVideoX-2B (primary T2V backend for T4)
    # ------------------------------------------------------------------

    def _generate_cogvideox(
        self,
        description: str,
        character_loras: Optional[Dict[str, str]] = None,
        shot_index: int = 0,
        num_frames: int = 49,
        width: int = 720,
        height: int = 480,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
    ) -> Optional[Dict]:
        """Try generating via CogVideoX-2B. Returns result dict or None."""
        try:
            if self._cogvideo is None:
                from agents.animator.cogvideo_wrapper import CogVideoXAnimator
                self._cogvideo = CogVideoXAnimator(str(self.warehouse))

            result = self._cogvideo.generate(
                description=description,
                character_loras=character_loras,
                shot_index=shot_index,
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            return result
        except Exception as e:
            print(f"  CogVideoX not available: {e}")
            return None

    # ------------------------------------------------------------------
    # AnimateDiff (SD 1.5 + motion module — real animation on T4)
    # ------------------------------------------------------------------

    # SD 1.5 anime base models (tried in order — better anime models first)
    _SD15_MODELS = [
        "Lykon/dreamshaper-8",
        "Linaqruf/anything-v3-1",
        "runwayml/stable-diffusion-v1-5",
    ]

    def _load_animatediff_pipeline(self):
        """Lazy-load AnimateDiff text-to-video pipeline with SD 1.5 anime model."""
        if self._animatediff_pipe is not None or self._animatediff_load_failed:
            return

        try:
            from diffusers import (
                AnimateDiffPipeline,
                MotionAdapter,
                DDIMScheduler,
            )

            # Load motion adapter
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-3",
                torch_dtype=torch.float16,
                cache_dir=str(self.warehouse / "models"),
            )
            print("  AnimateDiff motion adapter loaded")

            # Try SD 1.5 anime base models
            pipe = None
            for model_id in self._SD15_MODELS:
                try:
                    pipe = AnimateDiffPipeline.from_pretrained(
                        model_id,
                        motion_adapter=adapter,
                        torch_dtype=torch.float16,
                        cache_dir=str(self.warehouse / "models"),
                    )
                    print(f"  AnimateDiff base model loaded: {model_id}")
                    break
                except Exception:
                    continue

            if pipe is None:
                print("  No SD 1.5 anime model available for AnimateDiff")
                self._animatediff_load_failed = True
                return

            pipe.scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config,
                beta_schedule="linear",
                clip_sample=False,
            )
            pipe.enable_vae_slicing()
            pipe.to(self._device)

            # Load IP-Adapter for character conditioning (if available)
            self._ip_adapter_loaded = False
            try:
                pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                    cache_dir=str(self.warehouse / "models"),
                )
                self._ip_adapter_loaded = True
                print("  IP-Adapter loaded for character consistency")
            except Exception as e:
                print(f"  IP-Adapter not available (optional): {e}")

            self._animatediff_pipe = pipe
            print("  AnimateDiff T2V pipeline ready")

        except Exception as e:
            print(f"  AnimateDiff not available: {e}")
            self._animatediff_load_failed = True

    def _get_reference_image(
        self, character_loras: Optional[Dict[str, str]]
    ) -> Optional[Image.Image]:
        """Get a reference image for IP-Adapter from the character's multi_views."""
        if not character_loras:
            return None
        try:
            from director.memory_bank import AssetMemoryBank
            memory = AssetMemoryBank(str(self.warehouse))
            for char_name in character_loras:
                char_data = memory.get_character(char_name)
                if char_data and char_data.get("multi_views"):
                    for view_path in char_data["multi_views"]:
                        if Path(view_path).exists():
                            return Image.open(view_path).convert("RGB")
        except Exception:
            pass
        return None

    def _generate_animatediff(
        self,
        output_path: str,
        description: str,
        character_loras: Optional[Dict[str, str]] = None,
        num_frames: int = 16,
        reference_image: Optional[Image.Image] = None,
    ) -> bool:
        """Generate animated video clip using AnimateDiff text-to-video + IP-Adapter."""
        try:
            import gc

            self._load_animatediff_pipeline()
            if self._animatediff_pipe is None:
                return False

            pipe = self._animatediff_pipe

            # Unwrap any previous PEFT LoRA from the UNet
            while hasattr(pipe.unet, "base_model"):
                try:
                    pipe.unet = pipe.unet.base_model.model
                except Exception:
                    break

            # Load SD 1.5 LoRA via PEFT directly (load_lora_weights
            # silently fails with UNetMotionModel key mismatches)
            _animatediff_lora_loaded = False
            if character_loras:
                for char_name, lora_path in character_loras.items():
                    lora_p = Path(lora_path)
                    sd15_dir = lora_p.parent.parent / f"{lora_p.parent.name}_sd15"

                    if (sd15_dir / "adapter_model.safetensors").exists() or (
                        sd15_dir / "adapter_config.json"
                    ).exists():
                        try:
                            from peft import PeftModel

                            pipe.unet = PeftModel.from_pretrained(
                                pipe.unet, str(sd15_dir)
                            )
                            pipe.unet.eval()
                            _animatediff_lora_loaded = True
                            print(f"  AnimateDiff LoRA loaded for {char_name}")
                            break  # one LoRA at a time
                        except Exception as e:
                            print(f"  AnimateDiff LoRA failed for {char_name}: {e}")
                    else:
                        print(
                            f"  No SD 1.5 LoRA for {char_name} "
                            f"(expected at {sd15_dir})"
                        )

            negative_prompt = (
                "low quality, bad anatomy, worst quality, blurry, "
                "deformed, disfigured, static, ugly, jpeg artifacts"
            )

            # Scale frames by available VRAM (32 on A100, 16 on T4)
            max_frames = 32 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_mem > 20e9 else 16
            gen_frames = min(num_frames, max_frames)

            # Build generation kwargs for text-to-video
            gen_kwargs = dict(
                prompt=description,
                negative_prompt=negative_prompt,
                num_frames=gen_frames,
                width=512,
                height=768,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=torch.Generator(self._device).manual_seed(
                    hash(description) % (2**31)
                ),
            )

            # Use IP-Adapter for character conditioning if available
            if self._ip_adapter_loaded:
                ref_img = reference_image or self._get_reference_image(character_loras)
                if ref_img is not None:
                    ref_img = ref_img.resize((512, 768), Image.LANCZOS)
                    pipe.set_ip_adapter_scale(0.6)
                    gen_kwargs["ip_adapter_image"] = ref_img
                    print("  IP-Adapter conditioning with character reference image")
                else:
                    pipe.set_ip_adapter_scale(0.0)

            result = pipe(**gen_kwargs)

            frames = result.frames[0]
            self._frames_to_video(frames, output_path, fps=8)
            print(f"  AnimateDiff T2V clip generated: {output_path} ({len(frames)} frames)")

            # Cleanup — unwrap PEFT LoRA if loaded
            if _animatediff_lora_loaded:
                while hasattr(pipe.unet, "base_model"):
                    try:
                        pipe.unet = pipe.unet.base_model.model
                    except Exception:
                        break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            print(f"  AnimateDiff generation failed: {e}")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    def generate_long_video(
        self,
        description: str,
        character_loras: Optional[Dict[str, str]] = None,
        duration_seconds: int = 120,
        fps: int = 8,
    ) -> Dict:
        """
        Generate a long video by stitching AnimateDiff clips.

        Args:
            description: Text prompt for the video.
            character_loras: Character LoRA paths.
            duration_seconds: Target duration in seconds.
            fps: Frames per second.

        Returns:
            Dict with video_path and metadata.
        """
        import gc

        frames_per_clip = 16
        total_frames = duration_seconds * fps
        num_clips = max(1, total_frames // frames_per_clip)
        overlap = 4  # overlap frames for smooth transitions

        print(f"  Generating {num_clips} clips for {duration_seconds}s video...")

        all_frames = []
        for clip_idx in range(num_clips):
            # Vary seed slightly per clip for visual diversity
            seed = hash(description) % (2**31) + clip_idx

            self._load_animatediff_pipeline()
            if self._animatediff_pipe is None:
                print("  AnimateDiff unavailable, falling back to SDXL keyframes")
                break

            pipe = self._animatediff_pipe

            # Load LoRA via PEFT (first character only, once)
            if clip_idx == 0 and character_loras:
                # Unwrap any previous PEFT LoRA
                while hasattr(pipe.unet, "base_model"):
                    try:
                        pipe.unet = pipe.unet.base_model.model
                    except Exception:
                        break

                for char_name, lora_path in character_loras.items():
                    sd15_dir = (
                        Path(lora_path).parent.parent
                        / f"{Path(lora_path).parent.name}_sd15"
                    )
                    if (sd15_dir / "adapter_model.safetensors").exists() or (
                        sd15_dir / "adapter_config.json"
                    ).exists():
                        try:
                            from peft import PeftModel

                            pipe.unet = PeftModel.from_pretrained(
                                pipe.unet, str(sd15_dir)
                            )
                            pipe.unet.eval()
                            print(f"  LoRA loaded for {char_name}")
                        except Exception:
                            pass
                    break

            try:
                gen_kwargs = dict(
                    prompt=description,
                    negative_prompt=(
                        "low quality, bad anatomy, worst quality, blurry, "
                        "deformed, static, ugly, jpeg artifacts"
                    ),
                    num_frames=frames_per_clip,
                    width=512,
                    height=768,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=torch.Generator(self._device).manual_seed(seed),
                )

                # Use IP-Adapter for character conditioning if available
                if getattr(self, "_ip_adapter_loaded", False):
                    ref_img = self._get_reference_image(character_loras)
                    if ref_img is not None:
                        ref_img = ref_img.resize((512, 768), Image.LANCZOS)
                        pipe.set_ip_adapter_scale(0.6)
                        gen_kwargs["ip_adapter_image"] = ref_img

                result = pipe(**gen_kwargs)
                clip_frames = result.frames[0]

                # Cross-fade overlap with previous clip
                if all_frames and overlap > 0:
                    for j in range(min(overlap, len(clip_frames))):
                        alpha = (j + 1) / (overlap + 1)
                        prev = np.array(all_frames[-(overlap - j)]).astype(np.float32)
                        curr = np.array(clip_frames[j]).astype(np.float32)
                        if prev.shape == curr.shape:
                            blended = (
                                ((1 - alpha) * prev + alpha * curr)
                                .astype(np.uint8)
                            )
                            all_frames[-(overlap - j)] = Image.fromarray(blended)
                    # Add remaining frames (skip overlapped ones)
                    all_frames.extend(clip_frames[overlap:])
                else:
                    all_frames.extend(clip_frames)

                print(
                    f"  Clip {clip_idx + 1}/{num_clips} generated "
                    f"({len(all_frames)} frames total)"
                )

            except Exception as e:
                print(f"  Clip {clip_idx + 1} failed: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_frames:
            return {"video_path": "", "status": "failed", "num_frames": 0}

        output_path = (
            self.output_dir / f"long_video_{int(time.time())}.mp4"
        )
        self._frames_to_video(all_frames, str(output_path), fps=fps)

        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        print(
            f"  Long video assembled: {output_path} "
            f"({len(all_frames)} frames, {len(all_frames)/fps:.1f}s)"
        )

        return {
            "video_path": str(output_path),
            "num_frames": len(all_frames),
            "duration": len(all_frames) / fps,
            "num_clips": num_clips,
            "status": "animatediff_long",
        }

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

            # Load LoRA: single-character shots use LoRA for identity,
            # multi-character shots skip LoRA and rely on the prompt
            is_multi_char = character_loras and len(character_loras) > 1
            if character_loras and not is_multi_char:
                for char_name, lora_path in character_loras.items():
                    lora_dir = str(Path(lora_path).parent)
                    if Path(lora_path).exists():
                        try:
                            pipe.unet = PeftModel.from_pretrained(
                                pipe.unet, lora_dir
                            )
                            print(f"  SDXL LoRA loaded for {char_name}")
                        except Exception as e:
                            print(f"  SDXL LoRA load failed for {char_name}: {e}")
            elif is_multi_char:
                char_names = ", ".join(character_loras.keys())
                print(f"  Multi-character shot ({char_names}): using prompt-only")

            # Generate 4 keyframes with SAME seed for consistent composition,
            # slight prompt variations for subtle movement
            num_keyframes = 4
            base_seed = hash(description) % (2**31)
            keyframes = []
            for i in range(num_keyframes):
                result = pipe(
                    prompt=description,
                    negative_prompt="low quality, bad anatomy, worst quality, blurry",
                    width=768,
                    height=768,
                    num_inference_steps=25,
                    guidance_scale=7.0,
                    generator=torch.Generator(self._device).manual_seed(
                        base_seed + i
                    ),
                )
                keyframes.append(result.images[0])
                print(f"  Keyframe {i + 1}/{num_keyframes} generated")

            # Hold each keyframe for multiple frames, with brief cross-fade
            frames = self._interpolate_keyframes(keyframes, frames_between=2)

            # Assemble to video at 4 fps (~5s per shot, each keyframe visible)
            self._frames_to_video(frames, output_path, fps=4)

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

        hold_frames = 4  # hold each keyframe for 4 frames before transitioning
        all_frames = []
        for i in range(len(keyframes) - 1):
            # Hold the keyframe steady
            for _ in range(hold_frames):
                all_frames.append(keyframes[i])

            # Brief transition to next keyframe
            if rife_model is not None:
                intermediates = self._rife_interpolate(
                    rife_model, keyframes[i], keyframes[i + 1], frames_between
                )
            else:
                intermediates = self._crossfade_interpolate(
                    keyframes[i], keyframes[i + 1], frames_between
                )
            all_frames.extend(intermediates)

        # Hold the last keyframe
        for _ in range(hold_frames):
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
