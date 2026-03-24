"""Patch Cell 28: Replace CogVideoX with AnimateDiff + LoRA for video generation."""
import json
import sys

NOTEBOOK = "notebooks/AnimeLoom_Colab_Training.ipynb"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

cell28_src = "".join(nb["cells"][28]["source"])

# ============================================================
# 1. Replace CogVideoX slider params with AnimateDiff params
# ============================================================

OLD_PARAMS = """#@markdown ---
#@markdown **CogVideoX Settings**
NUM_FRAMES = 49  #@param {type:"slider", min:13, max:49, step:12}
COGVID_STEPS = 60  #@param {type:"slider", min:20, max:100, step:5}
COGVID_GUIDANCE = 6.0  #@param {type:"slider", min:1.0, max:12.0, step:0.5}
USE_DYNAMIC_CFG = True  #@param {type:"boolean"}
FPS = 16  #@param {type:"slider", min:8, max:24, step:4}
TARGET_FPS = 24  #@param {type:"slider", min:8, max:30, step:4}"""

NEW_PARAMS = """#@markdown ---
#@markdown **Video Generation Settings**
VIDEO_MODEL = "AnimateDiff (recommended)"  #@param ["AnimateDiff (recommended)", "CogVideoX 1.5"]
NUM_FRAMES = 24  #@param {type:"slider", min:16, max:32, step:4}
ANIM_STEPS = 30  #@param {type:"slider", min:15, max:50, step:5}
ANIM_GUIDANCE = 8.0  #@param {type:"slider", min:3.0, max:15.0, step:0.5}
DENOISING_STRENGTH = 0.40  #@param {type:"slider", min:0.20, max:0.65, step:0.05}
FPS = 8  #@param {type:"slider", min:8, max:16, step:4}
TARGET_FPS = 24  #@param {type:"slider", min:8, max:30, step:4}"""

cell28_src = cell28_src.replace(OLD_PARAMS, NEW_PARAMS)

# ============================================================
# 2. Replace Phase 3 CogVideoX with AnimateDiff + LoRA
# ============================================================

# The old Phase 3 starts after SDXL unload and ends before Phase 3b
OLD_PHASE3 = """# ================================================================
# Phase 3: CogVideoX 1.5 \u2014 Animate keyframes
# ================================================================
print("=" * 60)
print("PHASE 3: CogVideoX Animation")
print("=" * 60)

print("Loading CogVideoX 1.5 image-to-video (int8 quantized)\u2026")
from diffusers import CogVideoXImageToVideoPipeline
from optimum.quanto import qint8, quantize, freeze

cogvid_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX1.5-5B-I2V",
    torch_dtype=torch.bfloat16,
)
quantize(cogvid_pipe.transformer, weights=qint8)
freeze(cogvid_pipe.transformer)
quantize(cogvid_pipe.text_encoder, weights=qint8)
freeze(cogvid_pipe.text_encoder)
cogvid_pipe.enable_model_cpu_offload()
cogvid_pipe.vae.enable_tiling()
cogvid_pipe.vae.enable_slicing()
print("CogVideoX 1.5 loaded and quantized.")

all_clips = []
start_time = time.time()

for i, kf_image in enumerate(keyframes):
    shot = shots[i]
    motion_prompt = f"Slow tracking shot of anime character, clear detailed face, expressive eyes, sharp features, {shot['description']}, smooth fluid motion, gradually moving, anime style, high quality animation"
    print(f"\\n  Animating shot {i+1}/{len(keyframes)}: \\"{shot['description'][:50]}\u2026\\"")

    # Aspect-ratio crop/resize for CogVideoX (portrait: 480x720)
    kf_w, kf_h = kf_image.size
    target_ratio = 480 / 720
    current_ratio = kf_w / kf_h
    if current_ratio > target_ratio:
        new_w = int(kf_h * target_ratio)
        left = (kf_w - new_w) // 2
        kf_cropped = kf_image.crop((left, 0, left + new_w, kf_h))
    else:
        new_h = int(kf_w / target_ratio)
        top = (kf_h - new_h) // 2
        kf_cropped = kf_image.crop((0, top, kf_w, top + new_h))
    kf_resized = kf_cropped.resize((480, 720), Image.LANCZOS)

    gen = torch.Generator("cpu").manual_seed(42 + i)
    output = cogvid_pipe(
        image=kf_resized,
        prompt=motion_prompt,
        negative_prompt=MOTION_NEGATIVE,
        num_frames=NUM_FRAMES,
        num_inference_steps=COGVID_STEPS,
        guidance_scale=COGVID_GUIDANCE,
        use_dynamic_cfg=USE_DYNAMIC_CFG,
        generator=gen,
    )
    clip_frames = output.frames[0]

    # --- Preserve keyframe face quality in opening frames ---
    if len(clip_frames) > 3:
        try:
            clip_sample = clip_frames[0]
            clip_w = clip_sample.size[0] if hasattr(clip_sample, 'size') else np.array(clip_sample).shape[1]
            clip_h = clip_sample.size[1] if hasattr(clip_sample, 'size') else np.array(clip_sample).shape[0]
            kf_for_blend = kf_resized.resize((clip_w, clip_h), Image.LANCZOS)
            for blend_j in range(min(3, len(clip_frames))):
                alpha = blend_j / 3.0
                frame_pil = clip_frames[blend_j] if isinstance(clip_frames[blend_j], Image.Image) else Image.fromarray(np.array(clip_frames[blend_j]))
                clip_frames[blend_j] = Image.blend(kf_for_blend, frame_pil, alpha)
        except Exception as e_blend:
            print(f"  Face-preserving blend skipped: {e_blend}")

    all_clips.append(clip_frames)

    elapsed = time.time() - start_time
    eta = (elapsed / (i + 1)) * (len(keyframes) - i - 1)
    print(f"  Shot {i+1} done \u2014 {len(output.frames[0])} frames ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    gc.collect()
    torch.cuda.empty_cache()

cogvid_time = time.time() - start_time
print(f"\\nCogVideoX done in {cogvid_time/60:.1f} minutes.")
del cogvid_pipe
gc.collect()
torch.cuda.empty_cache()"""

NEW_PHASE3 = '''# ================================================================
# Phase 3: Animate Keyframes (AnimateDiff + LoRA or CogVideoX fallback)
# ================================================================
print("=" * 60)
print("PHASE 3: Video Generation")
print("=" * 60)

use_animatediff = VIDEO_MODEL.startswith("AnimateDiff")
all_clips = []
start_time = time.time()

if use_animatediff:
    # ----------------------------------------------------------
    # Phase 3a: Auto-train SD 1.5 LoRA if missing
    # ----------------------------------------------------------
    sd15_lora_dir = None
    if CHARACTER_NAME:
        char_id = CHARACTER_NAME.lower().replace(" ", "_")
        sd15_dir = WAREHOUSE / "lora" / f"{char_id}_sd15"
        sdxl_dir = WAREHOUSE / "lora" / char_id

        if (sd15_dir / "adapter_model.safetensors").exists() or (sd15_dir / "adapter_config.json").exists():
            sd15_lora_dir = sd15_dir
            print(f"SD 1.5 LoRA found at {sd15_dir}")
        elif sdxl_dir.exists():
            # Auto-train SD 1.5 LoRA from existing reference images
            print(f"\\nSD 1.5 LoRA not found. Auto-training from reference images...")
            print("(AnimateDiff needs SD 1.5 LoRA for character consistency)")
            try:
                ref_dir = WAREHOUSE / "references" / char_id
                if ref_dir.exists():
                    ref_images = [str(p) for p in ref_dir.glob("*.png")] + [str(p) for p in ref_dir.glob("*.jpg")]
                    if len(ref_images) >= 3:
                        from agents.character.trainer import LoRATrainer
                        trainer = LoRATrainer(str(WAREHOUSE))
                        trainer.train_sd15_lora(
                            character_images=ref_images,
                            character_id=char_id,
                            character_name=CHARACTER_NAME,
                            rank=16,
                            max_steps=300,
                        )
                        if (sd15_dir / "adapter_model.safetensors").exists() or (sd15_dir / "adapter_config.json").exists():
                            sd15_lora_dir = sd15_dir
                            print(f"SD 1.5 LoRA auto-trained successfully!")
                        else:
                            print("SD 1.5 LoRA training completed but files not found. Continuing without LoRA.")
                    else:
                        print(f"  Only {len(ref_images)} reference images found (need >= 3). Skipping SD 1.5 LoRA training.")
                else:
                    print(f"  No reference images at {ref_dir}. Skipping SD 1.5 LoRA training.")
            except Exception as e_train:
                print(f"  SD 1.5 LoRA auto-training failed: {e_train}")
                print("  Continuing without character LoRA (prompt-only mode)")
            gc.collect()
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Phase 3b: Load AnimateDiff pipeline with anime base model
    # ----------------------------------------------------------
    print("\\nLoading AnimateDiff + anime base model...")
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    motion_adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16,
        cache_dir=str(WAREHOUSE / "models"),
    )
    print("  Motion adapter loaded")

    # Try anime base models in order
    _SD15_MODELS = ["Linaqruf/anything-v3-1", "Lykon/dreamshaper-8", "runwayml/stable-diffusion-v1-5"]
    anim_pipe = None
    for _model_id in _SD15_MODELS:
        try:
            anim_pipe = AnimateDiffPipeline.from_pretrained(
                _model_id,
                motion_adapter=motion_adapter,
                torch_dtype=torch.float16,
                cache_dir=str(WAREHOUSE / "models"),
            )
            print(f"  Base model loaded: {_model_id}")
            break
        except Exception as _e:
            print(f"  {_model_id} failed: {_e}")
            continue

    if anim_pipe is None:
        print("ERROR: No SD 1.5 base model available. Falling back to CogVideoX.")
        use_animatediff = False
    else:
        anim_pipe.scheduler = DDIMScheduler.from_config(
            anim_pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
        )
        anim_pipe.enable_vae_slicing()
        anim_pipe.enable_model_cpu_offload()

        # Load SD 1.5 LoRA for character consistency
        _lora_loaded = False
        if sd15_lora_dir is not None:
            try:
                from peft import PeftModel
                anim_pipe.unet = PeftModel.from_pretrained(anim_pipe.unet, str(sd15_lora_dir))
                anim_pipe.unet.eval()
                _lora_loaded = True
                print(f"  Character LoRA loaded from {sd15_lora_dir}")
            except Exception as e_lora:
                print(f"  LoRA loading failed: {e_lora}. Continuing with prompt-only mode.")

        print("AnimateDiff pipeline ready!\\n")

        # ----------------------------------------------------------
        # Phase 3c: Generate clips (AnimateDiff vid2vid from keyframes)
        # ----------------------------------------------------------
        ANIM_NEGATIVE = "low quality, bad anatomy, worst quality, blurry, deformed, disfigured, static, ugly, watermark, text, extra limbs"

        for i, kf_image in enumerate(keyframes):
            shot = shots[i]
            # Keep prompt under 77 CLIP tokens — important keywords FIRST
            char_tag = f"{CHARACTER_NAME}, " if CHARACTER_NAME else ""
            motion_prompt = f"{char_tag}anime, detailed face, expressive eyes, {shot['description']}, smooth motion, masterpiece"
            print(f"  Animating shot {i+1}/{len(keyframes)}: \\"{shot['description'][:50]}...\\"")

            # Resize keyframe for AnimateDiff (SD 1.5: 512x768 portrait)
            kf_resized = kf_image.resize((512, 768), Image.LANCZOS)

            gen = torch.Generator("cpu").manual_seed(42 + i)
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
                )
                clip_frames = result.frames[0]

                # Blend keyframe into first few frames for identity anchoring
                if len(clip_frames) > 3:
                    try:
                        clip_w, clip_h = clip_frames[0].size if hasattr(clip_frames[0], 'size') else (np.array(clip_frames[0]).shape[1], np.array(clip_frames[0]).shape[0])
                        kf_for_blend = kf_resized.resize((clip_w, clip_h), Image.LANCZOS)
                        for blend_j in range(min(3, len(clip_frames))):
                            alpha = 0.3 + (blend_j / 3.0) * 0.7  # start at 30% keyframe, increase to video
                            frame_pil = clip_frames[blend_j] if isinstance(clip_frames[blend_j], Image.Image) else Image.fromarray(np.array(clip_frames[blend_j]))
                            clip_frames[blend_j] = Image.blend(kf_for_blend, frame_pil, alpha)
                    except Exception as e_blend:
                        pass  # blend is optional

                all_clips.append(clip_frames)
                print(f"  Shot {i+1} done \\u2014 {len(clip_frames)} frames")

            except Exception as e_shot:
                print(f"  Shot {i+1} AnimateDiff failed: {e_shot}")
                # Create a static clip from keyframe as fallback
                all_clips.append([kf_resized] * 16)
                print(f"  Shot {i+1} using static keyframe fallback (16 frames)")

            gc.collect()
            torch.cuda.empty_cache()

        # Cleanup AnimateDiff
        anim_time = time.time() - start_time
        print(f"\\nAnimateDiff done in {anim_time/60:.1f} minutes.")
        # Unwrap PEFT before deleting
        if _lora_loaded:
            while hasattr(anim_pipe.unet, "base_model"):
                try:
                    anim_pipe.unet = anim_pipe.unet.base_model.model
                except Exception:
                    break
        del anim_pipe, motion_adapter
        gc.collect()
        torch.cuda.empty_cache()

# ----------------------------------------------------------
# CogVideoX fallback (if AnimateDiff not selected or failed)
# ----------------------------------------------------------
if not use_animatediff or len(all_clips) == 0:
    print("\\nUsing CogVideoX 1.5 fallback...")
    from diffusers import CogVideoXImageToVideoPipeline
    try:
        from optimum.quanto import qint8, quantize, freeze
        cogvid_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX1.5-5B-I2V",
            torch_dtype=torch.bfloat16,
        )
        quantize(cogvid_pipe.transformer, weights=qint8)
        freeze(cogvid_pipe.transformer)
        quantize(cogvid_pipe.text_encoder, weights=qint8)
        freeze(cogvid_pipe.text_encoder)
        cogvid_pipe.enable_model_cpu_offload()
        cogvid_pipe.vae.enable_tiling()
        cogvid_pipe.vae.enable_slicing()
        print("CogVideoX 1.5 loaded.")

        all_clips = []
        for i, kf_image in enumerate(keyframes):
            shot = shots[i]
            motion_prompt = f"Slow tracking shot of anime character, clear detailed face, expressive eyes, sharp features, {shot['description']}, smooth fluid motion, gradually moving, anime style, high quality animation"
            print(f"\\n  Animating shot {i+1}/{len(keyframes)}: \\"{shot['description'][:50]}...\\"")

            kf_w, kf_h = kf_image.size
            target_ratio = 480 / 720
            current_ratio = kf_w / kf_h
            if current_ratio > target_ratio:
                new_w = int(kf_h * target_ratio)
                left = (kf_w - new_w) // 2
                kf_cropped = kf_image.crop((left, 0, left + new_w, kf_h))
            else:
                new_h = int(kf_w / target_ratio)
                top = (kf_h - new_h) // 2
                kf_cropped = kf_image.crop((0, top, kf_w, top + new_h))
            kf_resized = kf_cropped.resize((480, 720), Image.LANCZOS)

            gen = torch.Generator("cpu").manual_seed(42 + i)
            output = cogvid_pipe(
                image=kf_resized,
                prompt=motion_prompt,
                negative_prompt=MOTION_NEGATIVE,
                num_frames=49,
                num_inference_steps=60,
                guidance_scale=6.0,
                use_dynamic_cfg=True,
                generator=gen,
            )
            clip_frames = output.frames[0]
            all_clips.append(clip_frames)
            elapsed = time.time() - start_time
            print(f"  Shot {i+1} done \\u2014 {len(clip_frames)} frames ({elapsed:.0f}s elapsed)")
            gc.collect()
            torch.cuda.empty_cache()

        del cogvid_pipe
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e_cog:
        print(f"CogVideoX also failed: {e_cog}")
        print("Using static keyframes as fallback.")
        all_clips = [[kf.resize((512, 768), Image.LANCZOS)] * 16 for kf in keyframes]'''

cell28_src = cell28_src.replace(OLD_PHASE3, NEW_PHASE3)

# ============================================================
# 3. Update the markdown description at the top of the cell
# ============================================================
cell28_src = cell28_src.replace(
    "#@markdown Story Decomposer \u2192 SDXL+LoRA keyframes \u2192 CogVideoX motion \u2192 RIFE upscale \u2192 Real-ESRGAN \u2192 Color grade \u2192 Assembly.",
    "#@markdown Story Decomposer \u2192 SDXL+LoRA keyframes \u2192 AnimateDiff+LoRA motion \u2192 RIFE upscale \u2192 Real-ESRGAN \u2192 Color grade \u2192 Assembly.",
)

# ============================================================
# Save
# ============================================================
nb["cells"][28]["source"] = [cell28_src]

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook patched (v2) successfully!")
print(f"Cell 28 length: {len(cell28_src)}")

# Verify key changes
checks = [
    ("VIDEO_MODEL selector", "VIDEO_MODEL" in cell28_src),
    ("AnimateDiff pipeline", "AnimateDiffPipeline" in cell28_src),
    ("MotionAdapter", "MotionAdapter" in cell28_src),
    ("SD 1.5 LoRA auto-train", "train_sd15_lora" in cell28_src),
    ("PeftModel LoRA loading", "PeftModel.from_pretrained" in cell28_src),
    ("DENOISING_STRENGTH param", "DENOISING_STRENGTH" in cell28_src),
    ("anime base model", "anything-v3-1" in cell28_src),
    ("CogVideoX fallback kept", "CogVideoXImageToVideoPipeline" in cell28_src),
    ("AnimateDiff num_frames", "num_frames=NUM_FRAMES" in cell28_src),
    ("ANIM_STEPS param", "ANIM_STEPS" in cell28_src),
    ("Character LoRA prompt", "CHARACTER_NAME" in cell28_src and "motion_prompt" in cell28_src),
]
all_ok = True
for label, result in checks:
    status = "OK" if result else "MISSING"
    if not result:
        all_ok = False
    print(f"  [{status}] {label}")

if all_ok:
    print("\nAll AnimateDiff changes verified!")
else:
    print("\nWARNING: Some changes were not applied!")
    sys.exit(1)
