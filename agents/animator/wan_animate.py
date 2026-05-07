"""
Wan2.2-Animate-14B wrapper — face-locked character animation.

Wan2.2-Animate decouples body skeleton (pose_video) from facial expression
(face_video). The character identity comes from `image=` (the SDXL keyframe);
expressions are driven by per-frame face crops from a driving clip.

Pipeline (per shot):
    driving_frames -> DWPose -> pose_video (skeleton overlay PNGs)
    driving_frames -> RetinaFace bbox -> face_video (face crops)
    WanAnimatePipeline(image=ref, pose_video, face_video) -> output frames

Usage:
    animator = Wan22AnimateWrapper(warehouse_path)
    animator.load(offload_mode="model")
    out_frames = animator.animate(
        reference_image=keyframe_pil,
        driving_frames=wan22_i2v_clip_pil,
        prompt="anime girl walks through forest",
        num_inference_steps=20,
        guidance_scale=1.0,
    )
    animator.unload()
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class Wan22AnimateWrapper:
    """Wraps Wan-AI/Wan2.2-Animate-14B-Diffusers (animate mode)."""

    MODEL_ID = "Wan-AI/Wan2.2-Animate-14B-Diffusers"

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._pipeline = None
        self._pose_detector = None
        self._face_detector = None
        self._offload_mode: Optional[str] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self, offload_mode: str = "model") -> bool:
        if self._pipeline is not None:
            return True

        try:
            from diffusers import WanAnimatePipeline

            self._pipeline = WanAnimatePipeline.from_pretrained(
                self.MODEL_ID,
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
                try: self._pipeline.vae.enable_slicing()
                except Exception: pass
                try: self._pipeline.vae.enable_tiling()
                except Exception: pass

            self._offload_mode = offload_mode
            print(f"  Wan2.2-Animate-14B loaded (bfloat16, {offload_mode} offload)")
            return True
        except Exception as e:
            print(f"  Wan2.2-Animate-14B load failed: {e}")
            self._pipeline = None
            return False

    def unload(self):
        if self._pipeline is not None:
            try: del self._pipeline
            except Exception: pass
            self._pipeline = None
        if self._pose_detector is not None:
            try: del self._pose_detector
            except Exception: pass
            self._pose_detector = None
        if self._face_detector is not None:
            try: del self._face_detector
            except Exception: pass
            self._face_detector = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Pose video extraction (DWPose -> OpenPose fallback)
    # ------------------------------------------------------------------

    def _get_pose_detector(self):
        if self._pose_detector is not None:
            return self._pose_detector
        try:
            from controlnet_aux import DWposeDetector
            self._pose_detector = DWposeDetector.from_pretrained(
                "yzd-v/DWPose", cache_dir=str(self.warehouse / "models")
            )
            print("    Pose: DWPose loaded")
            return self._pose_detector
        except Exception as e:
            print(f"    DWPose unavailable ({e}); trying OpenPose")
        try:
            from controlnet_aux import OpenposeDetector
            self._pose_detector = OpenposeDetector.from_pretrained(
                "lllyasviel/Annotators", cache_dir=str(self.warehouse / "models")
            )
            print("    Pose: OpenPose loaded")
            return self._pose_detector
        except Exception as e:
            print(f"    OpenPose unavailable ({e}); pose extraction disabled")
            return None

    def _extract_pose_video(self, frames: List[Image.Image]) -> List[Image.Image]:
        det = self._get_pose_detector()
        if det is None:
            return [Image.new("RGB", frames[0].size, (0, 0, 0)) for _ in frames]
        out: List[Image.Image] = []
        prev = None
        for f in frames:
            try:
                pose = det(f)
                if not isinstance(pose, Image.Image):
                    pose = Image.fromarray(np.array(pose))
                pose = pose.resize(f.size, Image.LANCZOS)
                prev = pose
            except Exception:
                pose = prev if prev is not None else Image.new("RGB", f.size, (0, 0, 0))
            out.append(pose)
        return out

    # ------------------------------------------------------------------
    # Face video extraction (RetinaFace via facexlib -> bbox crop)
    # ------------------------------------------------------------------

    def _get_face_detector(self):
        if self._face_detector is not None:
            return self._face_detector
        try:
            from facexlib.detection import init_detection_model
            self._face_detector = init_detection_model(
                "retinaface_resnet50",
                half=False,
                device=self._device,
            )
            print("    Face: RetinaFace loaded")
            return self._face_detector
        except Exception as e:
            print(f"    RetinaFace unavailable ({e}); using center-crop fallback")
            return None

    def _extract_face_video(
        self, frames: List[Image.Image], crop_size: int = 512, pad: float = 0.30
    ) -> List[Image.Image]:
        det = self._get_face_detector()
        out: List[Image.Image] = []
        prev_crop = None
        for f in frames:
            crop = None
            if det is not None:
                try:
                    arr = np.array(f.convert("RGB"))
                    bgr = arr[:, :, ::-1].copy()
                    with torch.no_grad():
                        bboxes = det.detect_faces(bgr, 0.5)
                    if bboxes is not None and len(bboxes) > 0:
                        x1, y1, x2, y2 = [int(v) for v in bboxes[0][:4]]
                        w, h = x2 - x1, y2 - y1
                        cx, cy = x1 + w / 2, y1 + h / 2
                        side = max(w, h) * (1 + pad)
                        l = max(0, int(cx - side / 2))
                        t = max(0, int(cy - side / 2))
                        r = min(f.width,  int(cx + side / 2))
                        b = min(f.height, int(cy + side / 2))
                        crop = f.crop((l, t, r, b)).resize((crop_size, crop_size), Image.LANCZOS)
                except Exception:
                    crop = None
            if crop is None:
                if prev_crop is not None:
                    crop = prev_crop
                else:
                    s = min(f.size)
                    l = (f.width  - s) // 2
                    t = (f.height - s) // 2
                    crop = f.crop((l, t, l + s, t + s)).resize((crop_size, crop_size), Image.LANCZOS)
            prev_crop = crop
            out.append(crop)
        return out

    # ------------------------------------------------------------------
    # Animate
    # ------------------------------------------------------------------

    def animate(
        self,
        reference_image: Image.Image,
        driving_frames: List[Image.Image],
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 480,
        height: int = 832,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        seed: int = 42,
    ) -> Optional[List[Image.Image]]:
        if self._pipeline is None:
            print("  Wan2.2-Animate not loaded — call load() first")
            return None
        if not driving_frames:
            print("  No driving frames provided")
            return None

        ref = reference_image.convert("RGB").resize((width, height), Image.LANCZOS)
        driving = [f.convert("RGB").resize((width, height), Image.LANCZOS) for f in driving_frames]

        pose_video = self._extract_pose_video(driving)
        face_video = self._extract_face_video(driving)
        print(f"    pose_video={len(pose_video)}, face_video={len(face_video)}")

        gen = torch.Generator("cpu").manual_seed(seed)
        try:
            result = self._pipeline(
                image=ref,
                pose_video=pose_video,
                face_video=face_video,
                prompt=prompt or None,
                negative_prompt=negative_prompt or None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                mode="animate",
                generator=gen,
            )
            frames = result.frames[0]
            return list(frames)
        except Exception as e:
            print(f"    Wan2.2-Animate generation failed: {e}")
            return None
