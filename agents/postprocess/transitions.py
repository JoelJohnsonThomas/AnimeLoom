"""
Shot Transitions — smooth visual transitions between video clips.

Used during final assembly to replace hard-cut concatenation.
"""

from typing import List

import numpy as np
from PIL import Image


def cross_dissolve(
    clip_a_frames: List[Image.Image],
    clip_b_frames: List[Image.Image],
    overlap_frames: int = 6,
) -> List[Image.Image]:
    """
    Cross-dissolve the tail of clip_a into the head of clip_b.

    Returns:
        Combined frame list with the overlap region blended.
    """
    if overlap_frames <= 0 or not clip_a_frames or not clip_b_frames:
        return list(clip_a_frames) + list(clip_b_frames)

    overlap = min(overlap_frames, len(clip_a_frames), len(clip_b_frames))

    # Non-overlapping head of clip_a
    result = list(clip_a_frames[:-overlap])

    # Blended overlap region
    for i in range(overlap):
        alpha = (i + 1) / (overlap + 1)
        a = np.array(clip_a_frames[-(overlap - i)]).astype(np.float32)
        b = np.array(clip_b_frames[i]).astype(np.float32)
        # Handle size mismatch
        if a.shape != b.shape:
            b_img = clip_b_frames[i].resize(
                clip_a_frames[0].size, Image.LANCZOS
            )
            b = np.array(b_img).astype(np.float32)
        blended = ((1 - alpha) * a + alpha * b).astype(np.uint8)
        result.append(Image.fromarray(blended))

    # Non-overlapping tail of clip_b
    result.extend(clip_b_frames[overlap:])
    return result


def fade_to_black(
    frames: List[Image.Image], duration_frames: int = 8
) -> List[Image.Image]:
    """Fade the last *duration_frames* of a clip to black."""
    if not frames or duration_frames <= 0:
        return list(frames)

    result = list(frames)
    n = min(duration_frames, len(result))

    for i in range(n):
        idx = len(result) - n + i
        alpha = 1.0 - (i + 1) / n  # 1.0 → 0.0
        arr = np.array(result[idx]).astype(np.float32) * alpha
        result[idx] = Image.fromarray(arr.astype(np.uint8))

    return result


def fade_from_black(
    frames: List[Image.Image], duration_frames: int = 8
) -> List[Image.Image]:
    """Fade the first *duration_frames* of a clip from black."""
    if not frames or duration_frames <= 0:
        return list(frames)

    result = list(frames)
    n = min(duration_frames, len(result))

    for i in range(n):
        alpha = (i + 1) / n  # 0.0 → 1.0
        arr = np.array(result[i]).astype(np.float32) * alpha
        result[i] = Image.fromarray(arr.astype(np.uint8))

    return result
