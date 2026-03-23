"""
Video Upscaler — temporal (RIFE) and spatial (Real-ESRGAN) upscaling.

Takes raw 480p/8fps clips and outputs 720p+/24fps smooth anime video.
Models are loaded sequentially to stay within T4 VRAM limits.
"""

import gc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class VideoUpscaler:
    """
    Two-stage video upscaler:
      1. Temporal: RIFE 8fps → 24fps  (3× frame interpolation)
      2. Spatial:  Real-ESRGAN-anime 480p → 720p/1080p
    """

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._rife_model = None
        self._realesrgan_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upscale_video(
        self,
        input_path: str,
        output_path: str,
        target_fps: int = 24,
        spatial_scale: int = 2,
        source_fps: int = 8,
    ) -> str:
        """
        Full upscale pipeline: temporal → spatial.

        Args:
            input_path:    Path to raw video (e.g. 480p @ 8fps).
            output_path:   Where to write the upscaled video.
            target_fps:    Desired output FPS (default 24).
            spatial_scale: Upscale factor (2 = 480p→960p, 1 = skip).
            source_fps:    FPS of the input video.

        Returns:
            Path to the upscaled video.
        """
        frames = self._read_video_frames(input_path)
        if not frames:
            print(f"  Upscaler: no frames read from {input_path}")
            return input_path

        print(f"  Upscaler: {len(frames)} frames @ {source_fps}fps → "
              f"target {target_fps}fps, {spatial_scale}× spatial")

        # Step 1: temporal upscaling (RIFE)
        if target_fps > source_fps:
            multiplier = round(target_fps / source_fps)
            frames = self._temporal_upscale(frames, multiplier)
            print(f"  Temporal upscale: {len(frames)} frames @ ~{target_fps}fps")

        # Step 2: spatial upscaling (Real-ESRGAN)
        if spatial_scale > 1:
            frames = self._spatial_upscale(frames, spatial_scale)
            if frames:
                h, w = np.array(frames[0]).shape[:2]
                print(f"  Spatial upscale: {w}×{h}")

        # Write output
        self._write_video(frames, output_path, fps=target_fps)
        print(f"  Upscaled video saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Temporal upscaling (RIFE)
    # ------------------------------------------------------------------

    def _temporal_upscale(
        self, frames: List[Image.Image], multiplier: int
    ) -> List[Image.Image]:
        """Interpolate between consecutive frames to increase FPS."""
        if multiplier <= 1 or len(frames) < 2:
            return frames

        rife = self._load_rife()
        if rife is not None:
            return self._rife_interpolate_sequence(rife, frames, multiplier)

        # Fallback: simple frame duplication + cross-fade
        return self._crossfade_temporal(frames, multiplier)

    def _load_rife(self):
        """Try to load RIFE model for frame interpolation."""
        if self._rife_model is not None:
            return self._rife_model

        try:
            import sys
            rife_path = str(self.warehouse / "models" / "Practical-RIFE")
            if Path(rife_path).exists() and rife_path not in sys.path:
                sys.path.insert(0, rife_path)
            from model.RIFE import Model as RIFEModel

            model = RIFEModel()
            model.load_model(str(Path(rife_path) / "train_log"), -1)
            model.eval()
            self._rife_model = model
            print("  RIFE model loaded for temporal upscaling")
            return model
        except Exception:
            print("  RIFE not available, using cross-fade temporal upscaling")
            return None

    def _rife_interpolate_sequence(
        self, model, frames: List[Image.Image], multiplier: int
    ) -> List[Image.Image]:
        """Interpolate an entire frame sequence with RIFE."""
        import torchvision.transforms.functional as TF

        result = []
        for i in range(len(frames) - 1):
            result.append(frames[i])
            t1 = TF.to_tensor(frames[i]).unsqueeze(0).to(self._device)
            t2 = TF.to_tensor(frames[i + 1]).unsqueeze(0).to(self._device)

            for j in range(1, multiplier):
                ratio = j / multiplier
                try:
                    with torch.no_grad():
                        mid = model.inference(t1, t2, ratio)
                    mid_img = TF.to_pil_image(mid.squeeze(0).cpu().clamp(0, 1))
                    result.append(mid_img)
                except Exception:
                    # Fallback: linear blend
                    alpha = ratio
                    a1 = np.array(frames[i]).astype(np.float32)
                    a2 = np.array(frames[i + 1]).astype(np.float32)
                    blended = ((1 - alpha) * a1 + alpha * a2).astype(np.uint8)
                    result.append(Image.fromarray(blended))

        result.append(frames[-1])

        self._unload_rife()
        return result

    def _crossfade_temporal(
        self, frames: List[Image.Image], multiplier: int
    ) -> List[Image.Image]:
        """Cross-fade between frames (CPU fallback for RIFE)."""
        result = []
        for i in range(len(frames) - 1):
            result.append(frames[i])
            a1 = np.array(frames[i]).astype(np.float32)
            a2 = np.array(frames[i + 1]).astype(np.float32)
            if a1.shape != a2.shape:
                frames[i + 1] = frames[i + 1].resize(frames[i].size, Image.LANCZOS)
                a2 = np.array(frames[i + 1]).astype(np.float32)
            for j in range(1, multiplier):
                alpha = j / multiplier
                blended = ((1 - alpha) * a1 + alpha * a2).astype(np.uint8)
                result.append(Image.fromarray(blended))
        result.append(frames[-1])
        return result

    def _unload_rife(self):
        self._rife_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Spatial upscaling (Real-ESRGAN)
    # ------------------------------------------------------------------

    def _spatial_upscale(
        self, frames: List[Image.Image], scale: int = 2
    ) -> List[Image.Image]:
        """Upscale frames using Real-ESRGAN-anime."""
        model = self._load_realesrgan(scale)
        if model is None:
            print("  Real-ESRGAN not available, skipping spatial upscale")
            return frames

        upscaled = []
        for i, frame in enumerate(frames):
            try:
                img_np = np.array(frame)
                # Real-ESRGAN expects BGR
                import cv2
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                output, _ = model.enhance(img_bgr, outscale=scale)
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                upscaled.append(Image.fromarray(output_rgb))
            except Exception as e:
                # Keep original frame on failure
                upscaled.append(frame)
                if i == 0:
                    print(f"  Real-ESRGAN frame upscale failed: {e}")

        self._unload_realesrgan()
        return upscaled

    def _load_realesrgan(self, scale: int = 2):
        """Load Real-ESRGAN with anime-optimised model."""
        if self._realesrgan_model is not None:
            return self._realesrgan_model

        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # Use anime-specific model (RealESRGAN_x4plus_anime_6B)
            model_name = "RealESRGAN_x4plus_anime_6B"
            model_path = self.warehouse / "models" / f"{model_name}.pth"

            if not model_path.exists():
                # Download from GitHub release
                self._download_realesrgan_model(model_name, str(model_path))

            if not model_path.exists():
                return None

            # 6-block RRDB for anime
            net = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4,
            )
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=net,
                tile=256,       # tile to save VRAM
                tile_pad=10,
                pre_pad=0,
                half=True,      # fp16
                device=self._device,
            )
            self._realesrgan_model = upsampler
            print("  Real-ESRGAN anime model loaded")
            return upsampler
        except Exception as e:
            print(f"  Real-ESRGAN not available: {e}")
            return None

    def _download_realesrgan_model(self, model_name: str, save_path: str):
        """Download Real-ESRGAN model weights."""
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename=f"{model_name}.pth",
                local_dir=str(self.warehouse / "models"),
            )
            print(f"  Downloaded {model_name}")
        except Exception:
            # Fallback: direct URL
            try:
                import urllib.request
                url = (
                    f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
                    f"v0.2.2.4/{model_name}.pth"
                )
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, save_path)
                print(f"  Downloaded {model_name} from GitHub")
            except Exception as e:
                print(f"  Failed to download {model_name}: {e}")

    def _unload_realesrgan(self):
        self._realesrgan_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def _read_video_frames(self, video_path: str) -> List[Image.Image]:
        """Read all frames from a video file."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            cap.release()
            return frames
        except Exception as e:
            print(f"  Failed to read video: {e}")
            return []

    def _write_video(
        self, frames: List[Image.Image], output_path: str, fps: int = 24
    ):
        """Write frames to MP4."""
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
                frame.save(str(frame_dir / f"frame_{i:04d}.png"))
