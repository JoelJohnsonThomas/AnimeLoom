"""Patch the Colab notebook with all 7 quality fixes."""
import json
import sys

NOTEBOOK = "notebooks/AnimeLoom_Colab_Training.ipynb"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================
# FIX CELL 2 (Setup)
# ============================================================
cell2_src = "".join(nb["cells"][2]["source"])

# Replace google-generativeai with google-genai
cell2_src = cell2_src.replace(
    '"google-generativeai",',
    '"google-genai", "google-generativeai",',
)

# Add ultralytics for face detection
cell2_src = cell2_src.replace(
    '"gradio"],',
    '"gradio", "ultralytics"],',
)

# Add torchvision compatibility shim + RIFE setup before `import torch`
SHIM_CODE = r'''
# --- Fix torchvision compatibility for GFPGAN/basicsr ---
import torchvision
if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as _F
    import types as _types
    _ftmod = _types.ModuleType('torchvision.transforms.functional_tensor')
    for _attr in ['rgb_to_grayscale', 'normalize', 'resize', 'pad']:
        if hasattr(_F, _attr):
            setattr(_ftmod, _attr, getattr(_F, _attr))
    sys.modules['torchvision.transforms.functional_tensor'] = _ftmod
    torchvision.transforms.functional_tensor = _ftmod
    print("torchvision compatibility shim applied")

# --- Install RIFE for temporal upscaling ---
RIFE_DIR = WAREHOUSE / "models" / "Practical-RIFE"
if not RIFE_DIR.exists():
    print("Installing RIFE for frame interpolation...")
    subprocess.run(["git", "clone", "https://github.com/hzwer/Practical-RIFE", str(RIFE_DIR)], capture_output=True)
    import urllib.request
    train_log = RIFE_DIR / "train_log"
    train_log.mkdir(parents=True, exist_ok=True)
    flownet_path = train_log / "flownet.pkl"
    if not flownet_path.exists():
        try:
            urllib.request.urlretrieve(
                "https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/flownet.pkl",
                str(flownet_path)
            )
            print("RIFE model weights downloaded")
        except Exception as e:
            print(f"RIFE download failed (temporal upscale will use fallback): {e}")
else:
    print("RIFE already installed")
'''

cell2_src = cell2_src.replace(
    "import torch\n",
    SHIM_CODE + "\nimport torch\n",
)

nb["cells"][2]["source"] = [cell2_src]

# ============================================================
# FIX CELL 28 (Text-to-Anime)
# ============================================================
cell28_src = "".join(nb["cells"][28]["source"])

# --- Change 1: Pass character_name to decomposer ---
cell28_src = cell28_src.replace(
    "decomposer = StoryDecomposer(gemini_api_key=GEMINI_API_KEY or None)",
    "decomposer = StoryDecomposer(gemini_api_key=GEMINI_API_KEY or None, character_name=CHARACTER_NAME or None)",
)

# --- Change 5: Lower CogVideoX guidance default ---
cell28_src = cell28_src.replace(
    'COGVID_GUIDANCE = 7.5  #@param {type:"slider", min:1.0, max:12.0, step:0.5}',
    'COGVID_GUIDANCE = 6.0  #@param {type:"slider", min:1.0, max:12.0, step:0.5}',
)

# --- Change 5: Improve motion prompt ---
cell28_src = cell28_src.replace(
    """motion_prompt = f"anime character, {shot['description']}, smooth motion, detailed face, sharp features, stable camera angle\"""",
    """motion_prompt = f"Slow tracking shot of anime character, clear detailed face, expressive eyes, sharp features, {shot['description']}, smooth fluid motion, gradually moving, anime style, high quality animation\"""",
)

# --- Change 6: Higher keyframe resolution ---
cell28_src = cell28_src.replace(
    'IMAGE_WIDTH = 512  #@param {type:"slider", min:512, max:1024, step:128}',
    'IMAGE_WIDTH = 768  #@param {type:"slider", min:512, max:1024, step:128}',
)
cell28_src = cell28_src.replace(
    'IMAGE_HEIGHT = 768  #@param {type:"slider", min:512, max:1024, step:128}',
    'IMAGE_HEIGHT = 1152  #@param {type:"slider", min:512, max:1152, step:128}',
)

# --- Change 6: Face-preserving first frames after CogVideoX generation ---
OLD_APPEND = """    all_clips.append(output.frames[0])

    elapsed = time.time() - start_time"""

NEW_APPEND = """    clip_frames = output.frames[0]

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

    elapsed = time.time() - start_time"""

cell28_src = cell28_src.replace(OLD_APPEND, NEW_APPEND)

# --- Change 4: Insert motion trimming before Phase 4 ---
OLD_PHASE4 = """# ================================================================
# Phase 4: Post-Processing"""

NEW_PHASE4 = '''# ================================================================
# Phase 3b: Trim Static Tails (fixes frozen ending)
# ================================================================
print("\\nTrimming static tail frames...")
try:
    from agents.postprocess.motion_trim import MotionTrimmer
    trimmer = MotionTrimmer()
    for trim_idx, clip in enumerate(all_clips):
        original_len = len(clip)
        all_clips[trim_idx] = trimmer.trim_static_tail(clip)
        trimmed = original_len - len(all_clips[trim_idx])
        if trimmed > 0:
            print(f"  Clip {trim_idx+1}: {original_len} \\u2192 {len(all_clips[trim_idx])} frames (trimmed {trimmed} static frames)")
        else:
            print(f"  Clip {trim_idx+1}: {original_len} frames (no static tail detected)")
    # Extend last clip with ping-pong to avoid abrupt ending
    if len(all_clips) > 0 and len(all_clips[-1]) < 35:
        all_clips[-1] = trimmer.extend_last_clip_with_pingpong(all_clips[-1], target_frames=40)
        print(f"  Last clip extended to {len(all_clips[-1])} frames via ping-pong")
except Exception as e:
    print(f"  Motion trimming skipped ({e})")

# ================================================================
# Phase 4: Post-Processing'''

cell28_src = cell28_src.replace(OLD_PHASE4, NEW_PHASE4)

# --- Change 7: Replace GFPGAN with anime face restore ---
OLD_GFPGAN = '''if FACE_RESTORE:
    # Face restoration with GFPGAN
    try:
        from gfpgan import GFPGANer
        restorer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
        )
        print("Restoring faces with GFPGAN\u2026")
        for clip_idx, clip in enumerate(all_clips):
            restored = []
            for frame_pil in clip:
                frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                _, _, out_bgr = restorer.enhance(frame_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                restored.append(Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)))
            all_clips[clip_idx] = restored
        del restorer
        gc.collect()
        torch.cuda.empty_cache()
        print("Face restoration done.")
    except Exception as e:
        print(f"Face restoration skipped ({e})")'''

NEW_FACE = '''if FACE_RESTORE:
    # Anime face restoration (replaces GFPGAN which doesn\u2019t work on anime faces)
    try:
        from agents.postprocess.face_restore import AnimeFaceRestorer
        restorer = AnimeFaceRestorer()
        print("Restoring anime faces\u2026")
        for clip_idx, clip in enumerate(all_clips):
            all_clips[clip_idx] = restorer.restore_frames(clip)
        print("Anime face restoration done.")
    except Exception as e:
        print(f"Face restoration skipped ({e})")'''

cell28_src = cell28_src.replace(OLD_GFPGAN, NEW_FACE)

# --- Change 2: Add torchvision shim before Real-ESRGAN ---
OLD_ESRGAN = "if SPATIAL_UPSCALE:"
NEW_ESRGAN = '''if SPATIAL_UPSCALE:
    # Ensure torchvision compatibility shim is active for basicsr
    import torchvision as _tv
    if not hasattr(_tv.transforms, 'functional_tensor'):
        import torchvision.transforms.functional as _F2
        import types as _types2
        _ftmod2 = _types2.ModuleType('torchvision.transforms.functional_tensor')
        for _a in ['rgb_to_grayscale', 'normalize', 'resize', 'pad']:
            if hasattr(_F2, _a):
                setattr(_ftmod2, _a, getattr(_F2, _a))
        sys.modules['torchvision.transforms.functional_tensor'] = _ftmod2
        _tv.transforms.functional_tensor = _ftmod2

if SPATIAL_UPSCALE:'''

# Only replace first occurrence
cell28_src = cell28_src.replace(OLD_ESRGAN, NEW_ESRGAN, 1)

# --- Change 3: Replace linear blend temporal upscale with RIFE ---
OLD_TEMPORAL = '''if TARGET_FPS > FPS:
    print(f"\\nTemporal upscaling: {FPS}fps \u2192 {TARGET_FPS}fps\u2026")
    multiplier = round(TARGET_FPS / FPS)
    for clip_idx, clip in enumerate(all_clips):
        upscaled = []
        for j in range(len(clip) - 1):
            upscaled.append(clip[j])
            a1 = np.array(clip[j]).astype(np.float32)
            a2 = np.array(clip[j + 1]).astype(np.float32)
            for k in range(1, multiplier):
                alpha = k / multiplier
                blended = ((1 - alpha) * a1 + alpha * a2).astype(np.uint8)
                upscaled.append(Image.fromarray(blended))
        upscaled.append(clip[-1])
        all_clips[clip_idx] = upscaled
        print(f"  Clip {clip_idx+1}: {len(clip)} \u2192 {len(upscaled)} frames")
    output_fps = TARGET_FPS
else:
    output_fps = FPS'''

NEW_TEMPORAL = '''if TARGET_FPS > FPS:
    print(f"\\nTemporal upscaling: {FPS}fps \u2192 {TARGET_FPS}fps\u2026")
    multiplier = round(TARGET_FPS / FPS)
    # Try RIFE for proper optical-flow interpolation (no ghosting)
    rife_available = False
    try:
        from agents.postprocess.upscaler import VideoUpscaler
        _upscaler = VideoUpscaler(str(WAREHOUSE))
        _rife_model = _upscaler._load_rife()
        if _rife_model is not None:
            rife_available = True
            print("  Using RIFE for temporal upscaling (optical flow)")
    except Exception:
        pass

    for clip_idx, clip in enumerate(all_clips):
        original_len = len(clip)
        if rife_available:
            try:
                all_clips[clip_idx] = _upscaler._rife_interpolate_sequence(_rife_model, clip, multiplier)
            except Exception as e_rife:
                print(f"  RIFE failed for clip {clip_idx+1}: {e_rife}, using linear blend")
                upscaled = []
                for j in range(len(clip) - 1):
                    upscaled.append(clip[j])
                    a1 = np.array(clip[j]).astype(np.float32)
                    a2 = np.array(clip[j + 1]).astype(np.float32)
                    for k in range(1, multiplier):
                        alpha = k / multiplier
                        blended = ((1 - alpha) * a1 + alpha * a2).astype(np.uint8)
                        upscaled.append(Image.fromarray(blended))
                upscaled.append(clip[-1])
                all_clips[clip_idx] = upscaled
        else:
            upscaled = []
            for j in range(len(clip) - 1):
                upscaled.append(clip[j])
                a1 = np.array(clip[j]).astype(np.float32)
                a2 = np.array(clip[j + 1]).astype(np.float32)
                for k in range(1, multiplier):
                    alpha = k / multiplier
                    blended = ((1 - alpha) * a1 + alpha * a2).astype(np.uint8)
                    upscaled.append(Image.fromarray(blended))
            upscaled.append(clip[-1])
            all_clips[clip_idx] = upscaled
        print(f"  Clip {clip_idx+1}: {original_len} \u2192 {len(all_clips[clip_idx])} frames")

    # Clean up RIFE to free VRAM
    if rife_available:
        try:
            _upscaler._unload_rife()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    output_fps = TARGET_FPS
else:
    output_fps = FPS'''

cell28_src = cell28_src.replace(OLD_TEMPORAL, NEW_TEMPORAL)

nb["cells"][28]["source"] = [cell28_src]

# ============================================================
# Save
# ============================================================
with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook patched successfully!")
print(f"Cell 2 length: {len(cell2_src)}")
print(f"Cell 28 length: {len(cell28_src)}")

# Verify key changes applied
checks = [
    ("google-genai in Cell 2", "google-genai" in cell2_src),
    ("torchvision shim in Cell 2", "functional_tensor" in cell2_src),
    ("RIFE setup in Cell 2", "Practical-RIFE" in cell2_src),
    ("character_name in Cell 28", "character_name=CHARACTER_NAME" in cell28_src),
    ("COGVID_GUIDANCE 6.0 in Cell 28", "COGVID_GUIDANCE = 6.0" in cell28_src),
    ("motion_trim in Cell 28", "MotionTrimmer" in cell28_src),
    ("AnimeFaceRestorer in Cell 28", "AnimeFaceRestorer" in cell28_src),
    ("RIFE upscaler in Cell 28", "VideoUpscaler" in cell28_src),
    ("face-preserving blend in Cell 28", "kf_for_blend" in cell28_src),
    ("tracking shot prompt in Cell 28", "Slow tracking shot" in cell28_src),
    ("768x1152 keyframes", "IMAGE_WIDTH = 768" in cell28_src),
]
all_ok = True
for label, result in checks:
    status = "OK" if result else "MISSING"
    if not result:
        all_ok = False
    print(f"  [{status}] {label}")

if all_ok:
    print("\nAll 7 changes verified!")
else:
    print("\nWARNING: Some changes were not applied!")
    sys.exit(1)
