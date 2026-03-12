"""
ControlNet Pose Conditioner - Extracts pose from reference videos
using OpenPose and conditions generation via ControlNet.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class PoseConditioner:
    """
    Extracts pose skeletons from reference videos/images using OpenPose
    and provides ControlNet conditioning for pose-guided generation.
    """

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self._openpose = None
        self._controlnet_pipe = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # OpenPose
    # ------------------------------------------------------------------

    def _load_openpose(self):
        """Load the OpenPose detector."""
        if self._openpose is not None:
            return

        try:
            from controlnet_aux import OpenposeDetector

            self._openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            print("OpenPose detector loaded")
        except ImportError:
            print("controlnet_aux not installed — pose extraction uses fallback")

    def extract_pose(self, image: Image.Image) -> Image.Image:
        """
        Extract an OpenPose skeleton from an image.
        Returns a pose-map image of the same size.
        """
        self._load_openpose()

        if self._openpose is not None:
            try:
                return self._openpose(image)
            except Exception as e:
                print(f"OpenPose extraction failed: {e}")

        # Fallback: return a blank canvas (indicates no pose info)
        return Image.new("RGB", image.size, (0, 0, 0))

    def extract_poses_from_video(
        self, video_path: str, max_frames: int = 32
    ) -> List[Image.Image]:
        """
        Extract pose maps from each frame of a video.
        """
        frames = self._read_video_frames(video_path, max_frames)
        return [self.extract_pose(frame) for frame in frames]

    # ------------------------------------------------------------------
    # ControlNet conditioning
    # ------------------------------------------------------------------

    def _load_controlnet(self):
        """Load ControlNet pipeline with OpenPose conditioning."""
        if self._controlnet_pipe is not None:
            return

        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetPipeline,
                UniPCMultistepScheduler,
            )

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16,
            )
            self._controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "sd2-community/stable-diffusion-2-1",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            self._controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self._controlnet_pipe.scheduler.config
            )
            self._controlnet_pipe.to(self._device)
            print("ControlNet OpenPose pipeline loaded")
        except Exception as e:
            print(f"ControlNet not available: {e}")

    def generate_with_pose(
        self,
        prompt: str,
        pose_image: Image.Image,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Optional[Image.Image]:
        """
        Generate an image conditioned on a pose skeleton.
        """
        self._load_controlnet()

        if self._controlnet_pipe is not None:
            try:
                result = self._controlnet_pipe(
                    prompt=prompt,
                    image=pose_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                return result.images[0]
            except Exception as e:
                print(f"ControlNet generation failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Video I/O helpers
    # ------------------------------------------------------------------

    def _read_video_frames(
        self, video_path: str, max_frames: int = 32
    ) -> List[Image.Image]:
        """Read frames from a video file."""
        frames: List[Image.Image] = []
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // max_frames)

            idx = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                idx += 1

            cap.release()
        except ImportError:
            print("opencv-python required for video reading")

        return frames

    def save_pose_video(
        self, poses: List[Image.Image], output_path: str, fps: int = 8
    ):
        """Write pose maps to a video file."""
        try:
            import cv2

            if not poses:
                return

            first = np.array(poses[0])
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for pose in poses:
                arr = np.array(pose)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                writer.write(arr)

            writer.release()
        except ImportError:
            pass
