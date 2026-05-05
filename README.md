# AnimeLoom

**Anime Character Consistency Engine** — generate studio-quality anime video from a text story while keeping the same character face, identity, and style across every shot.

AnimeLoom orchestrates a multi-stage pipeline (story decomposition, identity-locked keyframes, motion synthesis, face restoration, post-processing) to turn a one-paragraph prompt into a smooth anime sequence on a single GPU. Built around the latest 2026 open-source video models with a face-lock pass that pastes the character's face from a reference image into every output frame.

---

## What's New in v2

| Stage | v1 (older Colab path) | v1.5 | v2 (current) |
|-------|----------------------|------|--------------|
| Keyframes | SDXL + LoRA | SDXL + LoRA + IP-Adapter SDXL + img2img chaining | unchanged from v1.5 |
| Story decomposition | Rule-based | Gemini Flash | Two-stage Gemini -> Claude refinement |
| Video model | CogVideoX 1.5 | Wan2.1-I2V-14B | **Wan2.2-I2V-A14B** (MoE) + anime LoRA |
| Face consistency | None | GFPGAN per frame | **Wan2.2-Animate face lock** (face pasted from keyframe at every frame) |
| Post-processing | GFPGAN + Real-ESRGAN | + two-pass temporal smoothing, every-2nd-frame GFPGAN | unchanged from v1.5 |
| Identity consistency | ~70% | ~85% | **~95%+** |

The v2 path runs Wan2.2 in two passes per shot:

1. **Phase 3a** — Wan2.2-I2V-A14B (with anime LoRA) generates a *driving clip* that captures motion only
2. **Phase 3b** — Wan2.2-Animate-14B takes the SDXL keyframe as reference and the driving clip as motion source, producing a face-locked output where the character's face is literally pasted from the keyframe at every frame

This decoupled design comes from the Wan-Animate paper (arXiv 2509.14055) and is the single biggest available jump for "consistent anime faces" in open-source 2026.

---

## Pipeline

```
Story (text)
   |
   v
Phase 1: Decompose (Gemini story planning -> Claude cinematic refinement)
   |
   v
Phase 2: SDXL + LoRA + IP-Adapter -> identity-locked keyframes (img2img chaining,
                                     adaptive strength decay, dynamic anchor refresh,
                                     quality gate with drift detection)
   |
   v
Phase 3a: Wan2.2-I2V-A14B (+ anime LoRA) -> driving clips (motion source)
   |
   v
Phase 3b: Wan2.2-Animate-14B(reference=keyframe, driving=clip) -> face-locked frames
   |
   v
Phase 4: GFPGAN every-2nd-frame face restoration + two-pass temporal smoothing
   |
   v
Phase 5: RIFE temporal upscale (16fps -> 24fps) + Real-ESRGAN spatial upscale
   |
   v
Phase 6: Cross-dissolve assembly -> final mp4
```

## Requirements

- **Recommended GPU**: NVIDIA RTX A6000 (48GB VRAM). Each Wan2.2 14B variant peaks around 42-46GB with model CPU offload.
- **Minimum GPU**: any 24GB+ card with sequential offload (slower; 480x640 max resolution).
- Python 3.10+, PyTorch 2.5.1 + CUDA 12.4, ffmpeg, Redis (optional, for Celery).
- API keys (optional but recommended): Gemini (free, 1500 req/day at aistudio.google.com/apikey) and Anthropic Claude.

## Quick Start (RunPod A6000 — primary path)

1. Spin up an A6000 (48GB) pod on RunPod with the PyTorch image
2. Open Jupyter, then `notebooks/AnimeLoom_RunPod.ipynb`
3. Run cells in order:
   - **Cell 1** — installs pinned deps (torch 2.5.1+cu124, diffusers 0.36, ftfy, gfpgan, facexlib, etc.)
   - **Cell 2** — downloads a character LoRA from HuggingFace (default: `AnimeLoom/sakura-haruno`; also available: `AnimeLoom/denji`, `AnimeLoom/yuki-nagato`)
   - **Cell 2.5** — patches the story decomposer for two-stage Gemini->Claude refinement
   - **Cell 3** — runs the full v2 pipeline (Phase 1 -> 6) and renders the final video

Configure in Cell 3:
- `STORY_TEXT` — your one-paragraph story
- `CHARACTER_NAME` — must match the LoRA from Cell 2
- `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` — both optional; falls back gracefully

## Quick Start (CLI / standalone)

```bash
git clone https://github.com/JoelJohnsonThomas/AnimeLoom.git
cd AnimeLoom
chmod +x setup.sh
./setup.sh
```

```bash
python main.py --text "A girl walks through a cherry blossom forest at sunset"
python main.py --script script.txt --quality high
python main.py --api      # FastAPI server
python main.py --test     # smoke test
```

## Architecture

```
+------------------------------------------------------------+
|                       DirectorAgent                         |
|  +-----------+  +---------------+  +--------------------+   |
|  | Story     |->| WorkflowGraph |->| Shot Executor      |   |
|  | Decomposer|  | (DAG)         |  | + Checkpointing    |   |
|  +-----------+  +---------------+  +---------+----------+   |
+------------------------------------------------+------------+
            |              |               |              |
            v              v               v              v
    +---------------+ +-----------+ +-----------+ +--------------+
    | Character     | | Animator  | | Evaluator | | Asset        |
    | Agent         | | Agent     | | Agent     | | MemoryBank   |
    |               | |           | |           | |              |
    | * LoRA train  | | * Wan2.2  | | * Identity| | * LoRAs      |
    | * IP-Adapter  | | * Animate | | * Motion  | | * Embeddings |
    | * Consistency | | * RIFE    | | * Visual  | | * Scenes     |
    +---------------+ +-----------+ +-----------+ +--------------+
```

## Project Structure

```
animeloom/
├── director/
│   ├── agent.py                   # main orchestrator (script parsing, shot execution)
│   ├── workflow.py                 # shot dependency DAG with topological ordering
│   └── memory_bank.py              # persistent character/scene/shot storage
├── agents/
│   ├── story/
│   │   └── decomposer.py           # two-stage Gemini -> Claude story decomposer
│   ├── character/
│   │   ├── trainer.py              # LoRA fine-tuning (PEFT, rank 16-32)
│   │   ├── lora_manager.py         # adapter load/unload
│   │   ├── ip_adapter.py           # IPAdapterConditioner (SDXL face-image conditioning)
│   │   └── consistency.py          # GroundingDINO + SAM + CLIP identity validation
│   ├── animator/
│   │   ├── wan_wrapper.py          # multi-backend video wrapper
│   │   ├── wan_animate.py          # Wan2.2-Animate-14B face-lock wrapper (NEW in v2)
│   │   ├── cogvideo_wrapper.py     # CogVideoX fallback
│   │   ├── pixverse.py             # PixVerse external fallback
│   │   └── controlnet.py           # OpenPose pose conditioning
│   ├── postprocess/
│   │   ├── upscaler.py             # RIFE temporal + Real-ESRGAN spatial
│   │   ├── face_restore.py         # GFPGAN/CodeFormer face restoration
│   │   ├── color_grade.py          # anime LUT grading
│   │   └── transitions.py          # cross-dissolve assembly
│   └── evaluator/
│       ├── character_score.py       # CLIP-based identity consistency
│       ├── motion_score.py          # optical flow motion fidelity
│       └── visual_score.py          # sharpness, colour, smoothness
├── api/
│   ├── app.py                       # FastAPI application
│   ├── routes/{characters,generation}.py
│   └── schemas/models.py
├── jobs/
│   ├── worker.py                    # Celery async worker
│   └── tasks/{training,generation}.py
├── cloud/
│   ├── colab_survival.py            # 4-min keep-alive + 5-min checkpointing
│   ├── kaggle_trainer.py            # Kaggle P100 trainer
│   └── gcp_setup.sh                 # GCP T4 VM provisioning
├── notebooks/
│   └── AnimeLoom_RunPod.ipynb       # primary v2 pipeline notebook
├── warehouse/                        # runtime asset storage
│   ├── models/                       # base model weights
│   ├── lora/                         # character LoRA adapters
│   ├── outputs/                      # generated videos
│   └── checkpoints/                  # resume checkpoints
├── main.py                           # CLI entry point
├── setup.sh
├── requirements.txt
└── sample_script.txt
```

## Models Used (v2)

| Stage | Model | Purpose | VRAM |
|-------|-------|---------|------|
| Keyframes | `cagliostrolab/animagine-xl-3.1` (SDXL) + character LoRA | identity-locked anime stills | ~12GB |
| Identity conditioning | `h94/IP-Adapter` `ip-adapter_sdxl.bin` | image-to-image face anchoring | shares SDXL UNet |
| Story decomposer | Gemini 2.5 Flash + Claude Sonnet 4.6 | shot list + cinematic refinement | API only |
| Driving clip | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (MoE 14B) | motion source for Phase 3b | ~42GB peak |
| Anime style | Wan 2.2 anime LoRA (Civitai community) | anime aesthetic on Wan output | <1GB |
| Face lock | `Wan-AI/Wan2.2-Animate-14B` | reference face + driving motion -> output | ~46GB peak |
| Face restore | GFPGAN v1.4 | every-2nd-frame face cleanup | ~3GB |
| Temporal upscale | RIFE 4.x | 16fps -> 24fps interpolation | ~6GB |
| Spatial upscale | Real-ESRGAN x4plus_anime_6B | 480p -> 720p+ sharpening | ~6GB |

Each phase fully unloads before the next loads, so peak VRAM stays within A6000 limits.

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | PyTorch 2.5.1+cu124, Diffusers 0.36, PEFT, Transformers, Accelerate |
| Video | Wan2.2-I2V-A14B, Wan2.2-Animate-14B, CogVideoX-2B (fallback), AnimateDiff (fallback) |
| Identity | IP-Adapter SDXL, character LoRA, GroundingDINO + SAM + CLIP |
| NLP | Gemini 2.5 Flash, Claude Sonnet 4.6, rule-based fallback |
| Post | RIFE, Real-ESRGAN, GFPGAN, OpenCV, ffmpeg |
| API | FastAPI, Uvicorn, Pydantic |
| Queue | Celery, Redis |
| Infra | RunPod (primary), Google Colab, Kaggle, GCP |

## Settings (Cell 3 of the notebook)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_WIDTH` x `IMAGE_HEIGHT` | 768 x 1152 | SDXL keyframe resolution (portrait) |
| `SDXL_STEPS` | 35 | SDXL inference steps |
| `SDXL_GUIDANCE` | 7.0 | SDXL guidance scale |
| `LORA_SCALE` | 1.15 | character LoRA scale (early shots; relaxed to 1.0 after shot 2) |
| `WAN_W` x `WAN_H` | 480x832 (auto-detected from VRAM) | Wan2.2 output resolution |
| `NUM_FRAMES` | 33 | frames per Wan2.2 clip |
| `WAN_STEPS` | 30 | Wan2.2 inference steps |
| `WAN_GUIDANCE` | 3.0 | lower = more motion freedom |
| `FPS` -> `TARGET_FPS` | 16 -> 24 | source fps and RIFE-interpolated fps |
| `FACE_RESTORE` | True | GFPGAN every-2nd-frame face restoration |
| `SPATIAL_UPSCALE` | True | Real-ESRGAN x4plus_anime_6B |
| `COLOR_GRADE` | True | anime LUT grading |
| `WAN_ANIME_LORA_REPO` | `Kijai/wan22-anime-style` | Wan2.2 anime style LoRA repo (skip on failure) |

## Story Script Format (CLI)

```
SCENE: Character introduction
CHAR: Sakura
A young woman with pink hair walks through a cherry blossom forest

SCENE: Bridge
CHAR: Sakura
She stops at a wooden bridge and looks at the river below

SCENE: Wind
CHAR: Sakura
The wind gently moves her hair as petals fall around her
```

Directives: `SCENE:` (or `SHOT:`) starts a new shot, `CHAR:` lists character names, `POSE:` references a pose video, free text is the prompt.

## Training a Character LoRA

| Image count | Use case |
|-------------|----------|
| 10-15 | prototyping; identity may drift on extreme angles |
| 20-30 | studio quality; cover front, 3/4, side, expressions, lighting |
| 30+ | diminishing returns |

Best practices: official screencaps over fan art, mix front + 3/4 + side views, mix expressions, include full-body and close-up shots, use 512px+ on the shortest side.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/character/create` | upload character sheet, train LoRA |
| `GET` | `/character/list` | list all characters |
| `GET` | `/character/{id}` | get character details |
| `DELETE` | `/character/{id}` | delete a character |
| `POST` | `/generate/shot` | generate single shot |
| `POST` | `/generate/sequence` | generate multi-shot sequence |
| `POST` | `/generate/text-to-anime` | full text -> anime video |
| `GET` | `/job/{job_id}` | check generation job status |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CACHE_ROOT` | `./warehouse` | root directory for all assets |
| `GEMINI_API_KEY` | — | Gemini Flash API key (free tier sufficient) |
| `ANTHROPIC_API_KEY` | — | Claude Sonnet API key (~$0.003 per story) |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery job queue URL |
| `PIXVERSE_API_KEY` | — | PixVerse fallback API key (optional) |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8080` | FastAPI bind |

## Expected Quality on A6000

| Metric | v1 | v1.5 | v2 (current) |
|--------|----|----- |--------------|
| Identity consistency | ~70% | ~85% | **~95%+** |
| Face stability across shots | low | medium | **near-perfect** (face pasted from keyframe) |
| Motion smoothness | ok | good | **better** (Wan2.2 MoE temporal attention) |
| Anime aesthetic | good | good | **stronger** (Wan2.2 anime LoRA) |
| Visual quality | 6-7/10 | 7.5-8/10 | **8.7-9.2/10** |

To break 9.5/10, the next paradigm shift is HunyuanVideo full fp16 on H100 (80GB), or temporal-conditioning models like Sora / Veo 2 — both outside the A6000 envelope.

## How It Works

1. **Story Decomposition** — Gemini 2.5 Flash plans a structured shot list (SCENE/CHAR/ACTION/CAMERA/MOOD per shot, all sharing one environment). Claude Sonnet 4.6 refines each shot into cinematic anime language with required body movement. Falls back to rule-based on missing keys.
2. **Keyframe Generation** — SDXL + animagine-xl-3.1 + character LoRA generate keyframe 0 with text2img. Keyframes 1+ use img2img chaining (`StableDiffusionXLImg2ImgPipeline`) with adaptive strength decay (0.40 -> 0.25). IP-Adapter SDXL conditions every shot on the identity anchor (refreshed every 3 shots). A pixel-drift quality gate regenerates outliers.
3. **Driving Clip (Phase 3a)** — Wan2.2-I2V-A14B (Mixture-of-Experts) plus an optional Wan 2.2 anime LoRA generates a short clip per shot. Center-crop resize avoids face-proportion distortion. Face quality at this stage is irrelevant - it gets overwritten in Phase 3b.
4. **Face Lock (Phase 3b)** — Wan2.2-Animate-14B decouples skeleton (body motion) from facial expression. The driving clip provides motion; the SDXL keyframe is the face reference. Output has the keyframe's face at every frame with the driving clip's motion. Falls back to Track A (driving clips become final) if Animate is unavailable.
5. **Face Restoration** — Two-pass temporal smoothing wraps a face-region-only GFPGAN pass applied to every 2nd frame (prevents identity drift from over-restoration; preserves anime texture).
6. **Temporal + Spatial Upscale** — RIFE interpolates 16fps -> 24fps; Real-ESRGAN x4plus_anime_6B sharpens each frame.
7. **Color Grading** — anime LUT grading with palette presets (warm, cool, vibrant, muted).
8. **Assembly** — cross-dissolve between adjacent clips, final mp4 written via OpenCV.

## Contributing

Contributions welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch
5. Open a pull request

