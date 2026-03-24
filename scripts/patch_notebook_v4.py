"""Patch v4: Fix AnimateDiff — use AnimateDiffVideoToVideoPipeline (vid2vid).

Replaces all references to AnimateDiffImg2ImgPipeline and _AnimPipeClass
with AnimateDiffVideoToVideoPipeline, and changes pipe call to use
video= parameter instead of image=.
"""
import json
import sys

NOTEBOOK = "notebooks/AnimeLoom_Colab_Training.ipynb"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

cell28_src = "".join(nb["cells"][28]["source"])

# ============================================================
# 1. Fix import — replace any broken import with correct one
# ============================================================

# Remove the try/except fallback import block if present
OLD_IMPORT_BLOCK = """    from diffusers import MotionAdapter, DDIMScheduler
    try:
        from diffusers import AnimateDiffImg2ImgPipeline as _AnimPipeClass
        _use_img2img = True
        print("  Using AnimateDiffImg2ImgPipeline (image-to-video)")
    except ImportError:
        from diffusers import AnimateDiffPipeline as _AnimPipeClass
        _use_img2img = False
        print("  AnimateDiffImg2ImgPipeline not available, using AnimateDiffPipeline + vid2vid workaround")"""

NEW_IMPORT = """    from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler"""

if OLD_IMPORT_BLOCK in cell28_src:
    cell28_src = cell28_src.replace(OLD_IMPORT_BLOCK, NEW_IMPORT)
    print("[1] Replaced try/except import block with direct AnimateDiffVideoToVideoPipeline import")
else:
    # Try simpler replacements
    cell28_src = cell28_src.replace(
        "from diffusers import AnimateDiffImg2ImgPipeline, MotionAdapter, DDIMScheduler",
        "from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler",
    )
    print("[1] Replaced direct AnimateDiffImg2ImgPipeline import")

# ============================================================
# 2. Fix pipeline loading — replace _AnimPipeClass or AnimateDiffImg2ImgPipeline
# ============================================================
cell28_src = cell28_src.replace(
    "_AnimPipeClass.from_pretrained(",
    "AnimateDiffVideoToVideoPipeline.from_pretrained(",
)
cell28_src = cell28_src.replace(
    "AnimateDiffImg2ImgPipeline.from_pretrained(",
    "AnimateDiffVideoToVideoPipeline.from_pretrained(",
)
print("[2] Fixed pipeline loading")

# ============================================================
# 3. Fix pipeline call — replace image= with video= pattern
# ============================================================

# Pattern A: the if/else _use_img2img branching
OLD_CALL_BRANCHED = """            gen = torch.Generator("cpu").manual_seed(42 + i)
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
                    )"""

# Pattern B: simple image= call
OLD_CALL_IMG = """            gen = torch.Generator("cpu").manual_seed(42 + i)
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
                )"""

# Pattern C: text-only call (no image at all)
OLD_CALL_TEXT = """            gen = torch.Generator("cpu").manual_seed(42 + i)
            try:
                result = anim_pipe(
                    prompt=motion_prompt,
                    negative_prompt=ANIM_NEGATIVE,
                    num_frames=NUM_FRAMES,
                    width=512,
                    height=768,
                    num_inference_steps=ANIM_STEPS,
                    guidance_scale=ANIM_GUIDANCE,
                    generator=gen,
                )"""

NEW_CALL = """            gen = torch.Generator("cpu").manual_seed(42 + i)
            try:
                # Create static video from keyframe (vid2vid adds motion)
                input_video = [kf_resized] * NUM_FRAMES

                result = anim_pipe(
                    video=input_video,
                    prompt=motion_prompt,
                    negative_prompt=ANIM_NEGATIVE,
                    strength=DENOISING_STRENGTH,
                    num_inference_steps=ANIM_STEPS,
                    guidance_scale=ANIM_GUIDANCE,
                    generator=gen,
                )"""

if OLD_CALL_BRANCHED in cell28_src:
    cell28_src = cell28_src.replace(OLD_CALL_BRANCHED, NEW_CALL)
    print("[3] Replaced branched if/else pipe call with vid2vid call")
elif OLD_CALL_IMG in cell28_src:
    cell28_src = cell28_src.replace(OLD_CALL_IMG, NEW_CALL)
    print("[3] Replaced image= pipe call with vid2vid call")
elif OLD_CALL_TEXT in cell28_src:
    cell28_src = cell28_src.replace(OLD_CALL_TEXT, NEW_CALL)
    print("[3] Replaced text-only pipe call with vid2vid call")
else:
    print("[3] WARNING: Could not find pipe call to replace!")

# ============================================================
# 4. Simplify blending — remove _use_img2img conditionals
# ============================================================
OLD_BLEND_CONDITIONAL = """                # Blend keyframe into frames for identity anchoring
                # Stronger blend when using text-only mode (no img2img)
                blend_count = 6 if not _use_img2img else 3
                if len(clip_frames) > blend_count:
                    try:
                        clip_w, clip_h = clip_frames[0].size if hasattr(clip_frames[0], 'size') else (np.array(clip_frames[0]).shape[1], np.array(clip_frames[0]).shape[0])
                        kf_for_blend = kf_resized.resize((clip_w, clip_h), Image.LANCZOS)
                        for blend_j in range(min(blend_count, len(clip_frames))):
                            if _use_img2img:
                                alpha = 0.3 + (blend_j / blend_count) * 0.7
                            else:
                                alpha = 0.15 + (blend_j / blend_count) * 0.85
                            frame_pil = clip_frames[blend_j] if isinstance(clip_frames[blend_j], Image.Image) else Image.fromarray(np.array(clip_frames[blend_j]))
                            clip_frames[blend_j] = Image.blend(kf_for_blend, frame_pil, alpha)
                    except Exception as e_blend:
                        pass  # blend is optional"""

NEW_BLEND = """                # Blend keyframe into first few frames for identity anchoring
                if len(clip_frames) > 3:
                    try:
                        clip_w, clip_h = clip_frames[0].size if hasattr(clip_frames[0], 'size') else (np.array(clip_frames[0]).shape[1], np.array(clip_frames[0]).shape[0])
                        kf_for_blend = kf_resized.resize((clip_w, clip_h), Image.LANCZOS)
                        for blend_j in range(min(3, len(clip_frames))):
                            alpha = 0.3 + (blend_j / 3.0) * 0.7
                            frame_pil = clip_frames[blend_j] if isinstance(clip_frames[blend_j], Image.Image) else Image.fromarray(np.array(clip_frames[blend_j]))
                            clip_frames[blend_j] = Image.blend(kf_for_blend, frame_pil, alpha)
                    except Exception as e_blend:
                        pass  # blend is optional"""

if OLD_BLEND_CONDITIONAL in cell28_src:
    cell28_src = cell28_src.replace(OLD_BLEND_CONDITIONAL, NEW_BLEND)
    print("[4] Simplified blending (removed _use_img2img conditional)")
else:
    print("[4] Blending already simplified or not found")

# ============================================================
# 5. Also bump diffusers version in Cell 2
# ============================================================
cell2_src = "".join(nb["cells"][2]["source"])
if "diffusers>=0.24.0" in cell2_src:
    cell2_src = cell2_src.replace('"diffusers>=0.24.0"', '"diffusers>=0.30.0"')
    nb["cells"][2]["source"] = [cell2_src]
    print("[5] Cell 2: diffusers>=0.24.0 -> diffusers>=0.30.0")
elif "diffusers>=0.30.0" in cell2_src:
    print("[5] Cell 2: diffusers already >=0.30.0")
else:
    print("[5] Cell 2: diffusers version string not found")

# ============================================================
# Save
# ============================================================
nb["cells"][28]["source"] = [cell28_src]

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# ============================================================
# Verify
# ============================================================
checks = [
    ("AnimateDiffVideoToVideoPipeline import", "AnimateDiffVideoToVideoPipeline" in cell28_src),
    ("VideoToVideo from_pretrained", "AnimateDiffVideoToVideoPipeline.from_pretrained" in cell28_src),
    ("vid2vid call (video=input_video)", "video=input_video" in cell28_src),
    ("strength=DENOISING_STRENGTH", "strength=DENOISING_STRENGTH" in cell28_src),
    ("MotionAdapter", "MotionAdapter" in cell28_src),
    ("SD 1.5 LoRA auto-train", "train_sd15_lora" in cell28_src),
    ("PeftModel LoRA loading", "PeftModel.from_pretrained" in cell28_src),
    ("anime base model", "anything-v3-1" in cell28_src),
    ("CogVideoX fallback", "CogVideoXImageToVideoPipeline" in cell28_src),
    ("No _AnimPipeClass remnants", "_AnimPipeClass" not in cell28_src),
    ("No _use_img2img remnants", "_use_img2img" not in cell28_src),
    ("No AnimateDiffImg2ImgPipeline remnants", "AnimateDiffImg2ImgPipeline" not in cell28_src),
    ("diffusers>=0.30.0 in Cell 2", "diffusers>=0.30.0" in "".join(nb["cells"][2]["source"])),
]

all_ok = True
for label, result in checks:
    status = "OK" if result else "FAIL"
    if not result:
        all_ok = False
    print(f"  [{status}] {label}")

if all_ok:
    print("\nPatch v4 applied successfully! All checks passed.")
else:
    print("\nWARNING: Some checks failed!")
    sys.exit(1)
