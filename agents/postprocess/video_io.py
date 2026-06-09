"""
Shared video I/O helpers — H.264 writing with quality control.

Every clip in the pipeline passes through 2-3 encode generations
(generation → post-processing → assembly), so intermediates must be
written near-lossless (crf≈10) and only the final assembly at delivery
quality (crf≈16). The old OpenCV mp4v writer had no quality control and
visibly blurred output after repeated re-encodes.

Backend order: imageio-ffmpeg (libx264) → cv2 "avc1" → cv2 "mp4v".
"""

import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

FrameLike = Union[Image.Image, np.ndarray]


def _to_rgb_array(frame: FrameLike) -> np.ndarray:
    arr = np.array(frame) if isinstance(frame, Image.Image) else np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _pad_even(arr: np.ndarray) -> np.ndarray:
    """Pad to even dimensions — yuv420p requires width/height % 2 == 0."""
    h, w = arr.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return arr


def write_video_h264(
    frames: List[FrameLike],
    output_path: str,
    fps: float = 16,
    crf: int = 16,
    preset: str = "medium",
) -> str:
    """Write frames (PIL Images or HxWx3 uint8 arrays) to an H.264 MP4.

    crf 10 ≈ visually lossless (intermediates), crf 16 ≈ delivery quality.
    """
    if not frames:
        return output_path

    arrays = [_pad_even(_to_rgb_array(f)) for f in frames]

    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"],
            macro_block_size=1,
        )
        for arr in arrays:
            writer.append_data(arr)
        writer.close()
        return output_path
    except Exception as e:
        print(f"  imageio/libx264 unavailable ({e}), falling back to OpenCV")

    import cv2

    h, w = arrays[0].shape[:2]
    writer = None
    for fourcc_name in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError(f"No video writer available for {output_path}")

    for arr in arrays:
        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path


def read_video_frames(video_path: str) -> List[Image.Image]:
    """Read all frames of a video as RGB PIL Images."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def get_video_fps(video_path: str, default: float = 16.0) -> float:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


def mux_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """Mux an audio track into a video (video stream copied, audio → AAC)."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_path),
            ],
            capture_output=True,
            timeout=300,
        )
        return result.returncode == 0 and Path(output_path).exists()
    except Exception as e:
        print(f"  Audio mux failed: {e}")
        return False
