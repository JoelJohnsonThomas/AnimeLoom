"""Patch v3: Fix AnimateDiffImg2ImgPipeline import error.

1. Cell 2: Bump diffusers>=0.24.0 to diffusers>=0.30.0
2. Cell 28: Add try/except fallback for AnimateDiff import
"""
import json
import sys

NOTEBOOK = "notebooks/AnimeLoom_Colab_Training.ipynb"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================
# 1. Cell 2: Bump diffusers version
# ============================================================
cell2_src = "".join(nb["cells"][2]["source"])

cell2_src = cell2_src.replace(
    '"diffusers>=0.24.0"',
    '"diffusers>=0.30.0"',
)

nb["cells"][2]["source"] = [cell2_src]
print("[1] Cell 2: diffusers>=0.24.0 -> diffusers>=0.30.0",
      "OK" if "diffusers>=0.30.0" in cell2_src else "SKIPPED (already changed or not found)")

# ============================================================
# 2. Cell 28: Fix AnimateDiff import with fallback
# ============================================================
cell28_src = "".join(nb["cells"][28]["source"])

# Replace the direct import with a try/except fallback
OLD_IMPORT = '    from diffusers import AnimateDiffImg2ImgPipeline, MotionAdapter, DDIMScheduler'

NEW_IMPORT = '''    from diffusers import MotionAdapter, DDIMScheduler
    try:
        from diffusers import AnimateDiffImg2ImgPipeline as _AnimPipeClass
        _use_img2img = True
        print("  Using AnimateDiffImg2ImgPipeline (image-to-video)")
    except ImportError:
        from diffusers import AnimateDiffPipeline as _AnimPipeClass
        _use_img2img = False
        print("  AnimateDiffImg2ImgPipeline not available, using AnimateDiffPipeline + vid2vid workaround")'''

cell28_src = cell28_src.replace(OLD_IMPORT, NEW_IMPORT)

# Replace pipeline loading to use the resolved class
cell28_src = cell28_src.replace(
    '            anim_pipe = AnimateDiffImg2ImgPipeline.from_pretrained(',
    '            anim_pipe = _AnimPipeClass.from_pretrained(',
)

# Replace the pipeline call to handle both modes
OLD_PIPE_CALL = '''            gen = torch.Generator("cpu").manual_seed(42 + i)
            try:
                result = anim_pipe(
                    image=kf_resized,
                    prompt=motion_prompt,
                    negative_prompt=ANIM_NEGATIVE,
                    num_frames=NUM_FRAMES,
                    strength=DENOISING_STRENGTH,
                    num_inference_steps=ANIM_STEPS,
                    guidance_scale=ANIM_GUIDANCE,
                    generator=gen,
                )'''

NEW_PIPE_CALL = '''            gen = torch.Generator("cpu").manual_seed(42 + i)
            try:
                if _use_img2img:
                    # Img2Img mode: pass keyframe directly
                    result = anim_pipe(
                        image=kf_resized,
                        prompt=motion_prompt,
                        negative_prompt=ANIM_NEGATIVE,
                        num_frames=NUM_FRAMES,
                        strength=DENOISING_STRENGTH,
                        num_inference_steps=ANIM_STEPS,
                        guidance_scale=ANIM_GUIDANCE,
                        generator=gen,
                    )
                else:
                    # Text-to-video mode: generate with prompt, then blend keyframe
                    result = anim_pipe(
                        prompt=motion_prompt,
                        negative_prompt=ANIM_NEGATIVE,
                        num_frames=NUM_FRAMES,
                        width=512,
                        height=768,
                        num_inference_steps=ANIM_STEPS,
                        guidance_scale=ANIM_GUIDANCE,
                        generator=gen,
                    )'''

cell28_src = cell28_src.replace(OLD_PIPE_CALL, NEW_PIPE_CALL)

# After the pipe call, strengthen keyframe blending for text-only mode
OLD_BLEND = '''                # Blend keyframe into first few frames for identity anchoring
                if len(clip_frames) > 3:
                    try:
                        clip_w, clip_h = clip_frames[0].size if hasattr(clip_frames[0], 'size') else (np.array(clip_frames[0]).shape[1], np.array(clip_frames[0]).shape[0])
                        kf_for_blend = kf_resized.resize((clip_w, clip_h), Image.LANCZOS)
                        for blend_j in range(min(3, len(clip_frames))):
                            alpha = 0.3 + (blend_j / 3.0) * 0.7  # start at 30% keyframe, increase to video
                            frame_pil = clip_frames[blend_j] if isinstance(clip_frames[blend_j], Image.Image) else Image.fromarray(np.array(clip_frames[blend_j]))
                            clip_frames[blend_j] = Image.blend(kf_for_blend, frame_pil, alpha)
                    except Exception as e_blend:
                        pass  # blend is optional'''

NEW_BLEND = '''                # Blend keyframe into frames for identity anchoring
                # Stronger blend when using text-only mode (no img2img)
                blend_count = 6 if not _use_img2img else 3
                if len(clip_frames) > blend_count:
                    try:
                        clip_w, clip_h = clip_frames[0].size if hasattr(clip_frames[0], 'size') else (np.array(clip_frames[0]).shape[1], np.array(clip_frames[0]).shape[0])
                        kf_for_blend = kf_resized.resize((clip_w, clip_h), Image.LANCZOS)
                        for blend_j in range(min(blend_count, len(clip_frames))):
                            # text-only: stronger keyframe presence (start 60% kf -> fade to video)
                            # img2img: lighter blend (start 30% kf -> fade to video)
                            if _use_img2img:
                                alpha = 0.3 + (blend_j / blend_count) * 0.7
                            else:
                                alpha = 0.15 + (blend_j / blend_count) * 0.85
                            frame_pil = clip_frames[blend_j] if isinstance(clip_frames[blend_j], Image.Image) else Image.fromarray(np.array(clip_frames[blend_j]))
                            clip_frames[blend_j] = Image.blend(kf_for_blend, frame_pil, alpha)
                    except Exception as e_blend:
                        pass  # blend is optional'''

cell28_src = cell28_src.replace(OLD_BLEND, NEW_BLEND)

nb["cells"][28]["source"] = [cell28_src]

# ============================================================
# Verify
# ============================================================
checks = [
    ("diffusers>=0.30.0 in Cell 2", "diffusers>=0.30.0" in "".join(nb["cells"][2]["source"])),
    ("try/except import fallback", "_AnimPipeClass" in cell28_src),
    ("_use_img2img flag", "_use_img2img" in cell28_src),
    ("img2img pipe call", "image=kf_resized" in cell28_src),
    ("text-only fallback call", "width=512" in cell28_src),
    ("stronger blend for text mode", "blend_count = 6" in cell28_src),
]

all_ok = True
for label, result in checks:
    status = "OK" if result else "MISSING"
    if not result:
        all_ok = False
    print(f"  [{status}] {label}")

# Save
with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

if all_ok:
    print("\nPatch v3 applied successfully!")
else:
    print("\nWARNING: Some changes were not applied!")
    sys.exit(1)
