# AnimeLoom

**Anime Character Consistency Engine** — maintain character identity across long-form animation using a Director-Orchestrated multi-agent pipeline.

AnimeLoom coordinates specialized AI agents (character training, video generation, quality evaluation) to produce consistent anime sequences where characters look the same across every shot. Built to run on free/cheap GPU resources (Colab, Kaggle, GCP free credits) for **under $20 total**.

---

## Features

- **Character Identity Preservation** — LoRA fine-tuning (rank 16-32, fp16) locks in each character's visual identity via SDXL
- **Studio-Quality Video Pipeline** — SDXL keyframes + CogVideoX 1.5 animation + GFPGAN face restoration + Real-ESRGAN anime sharpening
- **Full Body Anime Output** — Head-to-toe character rendering in portrait orientation (512x768 → 480x720)
- **Interactive Gradio Studio** — Configure settings, preview keyframes, and generate video with a web UI directly in Colab
- **Multi-Agent Pipeline** — Director orchestrates Character, Animator, and Evaluator agents with a dependency-aware workflow graph
- **Quality-Gated Output** — shots scoring below 0.85 consistency are automatically regenerated (up to 3 attempts)
- **Colab Survival Mode** — keepalive every 4 min, auto-checkpoint every 5 min, Google Drive persistence, resume after disconnect
- **Multiple Generation Backends** — Wan2.2-Animate primary, PixVerse fallback, ControlNet pose conditioning
- **Detection + Segmentation** — GroundingDINO + SAM isolate characters; CLIP embeddings measure identity similarity
- **REST API** — FastAPI endpoints for character creation, shot generation, sequence processing, and job tracking
- **Async Job Queue** — Celery + Redis for background LoRA training and batch video generation

## Video Generation Pipeline

The Colab notebook (`notebooks/AnimeLoom_Colab_Training.ipynb`) runs a 4-phase pipeline:

```
Phase 1: SDXL + LoRA        → Character-consistent keyframes (512x768 portrait)
Phase 2: CogVideoX 1.5      → Animate keyframes into motion clips (480x720, int8 quantized)
Phase 3: GFPGAN + Real-ESRGAN → Face restoration + anime frame sharpening
Phase 4: Cross-fade stitch   → Blend clips into final video (mp4)
```

### Key Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_WIDTH` | 512 | SDXL output width (portrait) |
| `IMAGE_HEIGHT` | 768 | SDXL output height (portrait) |
| `COGVID_STEPS` | 60 | CogVideoX inference steps |
| `COGVID_GUIDANCE` | 7.5 | CogVideoX guidance scale |
| `FPS` | 16 | Output video framerate |
| `FACE_RESTORE` | True | Enable GFPGAN + Real-ESRGAN post-processing |
| `DENOISING_STRENGTH` | 0.45 | Img2img strength for keyframe continuity |

### Prompt Engineering for Quality

- **Full body framing**: `"full body, head to toe, facing viewer, front view"`
- **Studio look**: `"anime screencap, studio quality, sharp lineart, vibrant colors"`
- **Face stability**: `"symmetrical eyes, stable eye shape, detailed face"`
- **Motion stability**: `"consistent pose, stable camera angle, slow deliberate blink"`
- **Negative prompts**: `"3d render, cgi, photorealistic, distorted eyes, blurry hair, mouth blur"`

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DirectorAgent                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Script   │→ │ WorkflowGraph│→ │ Shot Executor  │  │
│  │ Parser   │  │ (DAG)        │  │ + Checkpoint   │  │
│  └──────────┘  └──────────────┘  └───────┬───────┘  │
└──────────────────────────────────────────┼──────────┘
            ┌──────────────┬───────────────┼──────────────┐
            ▼              ▼               ▼              ▼
    ┌──────────────┐ ┌───────────┐ ┌────────────┐ ┌────────────┐
    │ Character    │ │ Animator  │ │ Evaluator  │ │ Asset      │
    │ Agent        │ │ Agent     │ │ Agent      │ │ MemoryBank │
    │              │ │           │ │            │ │            │
    │ • LoRA Train │ │ • Wan2.2  │ │ • Identity │ │ • LoRAs    │
    │ • LoRA Mgmt  │ │ • PixVerse│ │ • Motion   │ │ • Embeds   │
    │ • Consistency│ │ • CtrlNet │ │ • Visual   │ │ • Scenes   │
    └──────────────┘ └───────────┘ └────────────┘ └────────────┘
```

## Project Structure

```
animeloom/
├── director/
│   ├── agent.py              # Main orchestrator
│   ├── workflow.py            # Shot dependency graph (DAG)
│   └── memory_bank.py         # Persistent asset storage
├── agents/
│   ├── character/
│   │   ├── trainer.py         # LoRA fine-tuning (PEFT)
│   │   ├── lora_manager.py    # Adapter load/unload
│   │   └── consistency.py     # GroundingDINO + SAM + CLIP
│   ├── animator/
│   │   ├── wan_wrapper.py     # Wan2.2-Animate integration
│   │   ├── pixverse.py        # PixVerse fallback
│   │   └── controlnet.py      # OpenPose pose conditioning
│   └── evaluator/
│       ├── character_score.py  # Identity consistency
│       ├── motion_score.py     # Motion fidelity
│       └── visual_score.py     # Frame quality
├── api/
│   ├── app.py                 # FastAPI application
│   ├── routes/
│   │   ├── characters.py      # Character CRUD
│   │   └── generation.py      # Shot & sequence generation
│   └── schemas/
│       └── models.py          # Pydantic models
├── jobs/
│   ├── worker.py              # Celery async worker
│   └── tasks/
│       ├── training.py        # Background LoRA training
│       └── generation.py      # Background video generation
├── cloud/
│   ├── colab_survival.py      # Keep-alive + checkpointing
│   ├── kaggle_trainer.py      # Kaggle P100 training wrapper
│   └── gcp_setup.sh           # GCP T4 VM provisioning
├── notebooks/
│   └── AnimeLoom_Colab_Training.ipynb  # Full Colab pipeline
├── warehouse/                  # Runtime asset storage
│   ├── models/                 # Base model weights
│   ├── lora/                   # Character LoRA adapters
│   ├── datasets/               # Training data
│   ├── outputs/                # Generated videos
│   └── checkpoints/            # Resume checkpoints
├── scripts/
│   └── download_models.py     # Download required model weights
├── main.py                     # CLI entry point
├── setup.sh                    # One-command setup
├── requirements.txt            # Python dependencies
├── sample_script.txt           # Example story script
└── .env.example                # Environment config template
```

## Quick Start

### Option A: Google Colab (Recommended)

1. Open `notebooks/AnimeLoom_Colab_Training.ipynb` in Google Colab
2. Set runtime to **A100 GPU** (Runtime → Change runtime type → A100)
3. Run cells in order:
   - **Cell 1** — Setup environment, install dependencies, mount Google Drive
   - **Cell 2** — Upload/download character reference images (10-30 images recommended)
   - **Cell 3** — Auto-caption images with BLIP
   - **Cell 4** — Train character LoRA (~15-20 min on A100)
   - **Cell 5** — Test LoRA with sample images
   - **Cell 9** — Generate short anime clip (SDXL + CogVideoX 1.5)
   - **Cell 10** — Generate long anime video (2+ minutes)
   - **Cell 11** — Launch Gradio Interactive Studio (web UI)

#### Gradio Interactive Studio

Cell 11 launches a web UI with:
- Character selection (auto-discovers trained LoRAs)
- SDXL and CogVideoX parameter sliders
- Editable keyframe and motion prompts
- **Preview** — generates 1 test keyframe + estimated stats before full run
- **Generate** — full 4-phase pipeline with progress bar
- Shareable public URL via `share=True`

### Option B: Local / CLI

```bash
git clone https://github.com/JoelJohnsonThomas/AnimeLoom.git
cd AnimeLoom
chmod +x setup.sh
./setup.sh
```

```bash
python main.py --test              # smoke test
python main.py --script script.txt # process a story
python main.py --api               # start FastAPI server
python main.py --colab             # Colab survival mode
```

### Option C: Kaggle (Free 30h/week P100)

```python
from cloud.kaggle_trainer import KaggleTrainer

trainer = KaggleTrainer()
lora_path = trainer.train("Denji", ["/kaggle/input/charsheet/denji.png"], rank=16, max_steps=500)
trainer.export_lora(lora_path)
```

## Training Data Guidelines

For studio-quality character output:

| Image Count | Quality Level | Notes |
|-------------|---------------|-------|
| 10-15 | Good for prototyping | Basic identity, may drift on angles |
| 20-30 | Studio quality | Cover diverse angles, expressions, poses, lighting |
| 30+ | Diminishing returns | Only needed for very complex character designs |

**Best practices:**
- Use official anime screenshots (not fan art)
- Include front, 3/4, and side profile views
- Mix expressions: neutral, happy, serious, surprised
- Include full body and close-up shots
- Avoid heavily compressed or low-resolution images (512px+ on shortest side)

## Script Format

AnimeLoom uses a simple text-based script format:

```
SCENE: Character introduction
CHAR: Denji
A young boy with blonde messy hair stands on a city street

SCENE: Walking scene
CHAR: Denji
POSE: walking_pose.mp4
Denji walks through the city, looking around

SCENE: Rooftop
CHAR: Denji
Denji stands on a rooftop at sunset, hair blowing in the wind
```

**Directives:**
- `SCENE:` or `SHOT:` — starts a new shot
- `CHAR:` — declares a character in the shot (comma-separated for multiple)
- `POSE:` — references a pose video for motion transfer
- Free text — scene description / generation prompt

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/character/create` | Upload character sheet, train LoRA |
| `GET` | `/character/list` | List all characters |
| `GET` | `/character/{id}` | Get character details |
| `DELETE` | `/character/{id}` | Delete a character |
| `POST` | `/generate/shot` | Generate single shot |
| `POST` | `/generate/sequence` | Generate multi-shot sequence |
| `GET` | `/job/{job_id}` | Check generation job status |

### Example: Create a Character

```bash
curl -X POST http://localhost:8080/character/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Denji",
    "description": "Young boy with blonde messy hair and sharp teeth",
    "image_paths": ["./images/denji_front.png", "./images/denji_side.png"]
  }'
```

### Example: Generate a Sequence

```bash
curl -X POST http://localhost:8080/generate/sequence \
  -H "Content-Type: application/json" \
  -d '{
    "script": "SCENE: Intro\nCHAR: Denji\nDenji stands on a city street",
    "story_id": "chainsaw_ep01"
  }'
```

## Tech Stack

| Category | Tools |
|----------|-------|
| **Image Generation** | SDXL + PEFT LoRA |
| **Video Generation** | CogVideoX 1.5 (int8 quantized via optimum-quanto) |
| **Face Restoration** | GFPGAN v1.4 |
| **Frame Sharpening** | Real-ESRGAN (x4plus_anime_6B) |
| **Detection** | GroundingDINO + SAM + CLIP |
| **Video Processing** | OpenCV, ffmpeg |
| **Web UI** | Gradio |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Queue** | Celery, Redis |
| **Infra** | Google Colab, Kaggle, GCP |

## Budget Breakdown

| Resource | Cost | What You Get |
|----------|------|--------------|
| Google Colab Pro | $10/month | A100/V100 GPU, longer runtimes |
| Kaggle | Free | 30h/week P100 GPU |
| Google Cloud | Free $300 credits | ~850 hours T4 GPU |
| **Total** | **< $20** | **Full pipeline capability** |

## How It Works

1. **Script Parsing** — `DirectorAgent` parses your script into individual shots, extracting characters, descriptions, and pose references

2. **Dependency Planning** — `WorkflowGraph` builds a DAG ensuring characters are trained before their shots are generated. Independent shots can run in parallel

3. **Character Training** — For each new character, a LoRA adapter is trained from reference images using PEFT (rank 32, ~1000 steps)

4. **Keyframe Generation** — SDXL with character LoRA generates consistent keyframes in portrait orientation (512x768). Img2img with low denoising strength maintains continuity between frames

5. **Video Animation** — CogVideoX 1.5 animates each keyframe into motion clips (49 frames each). Int8 quantization keeps VRAM under control on A100

6. **Post-Processing** — GFPGAN restores facial details, Real-ESRGAN (anime model) sharpens all frames. Cross-fade stitching blends clips together

7. **Quality Evaluation** — Generated shots are scored on character consistency (CLIP cosine similarity), motion fidelity (optical flow), and visual quality (sharpness, colour stability). Shots below 0.85 are regenerated

8. **Checkpointing** — Every 5 minutes, full state is saved. After a Colab disconnect, resume exactly where you left off

9. **Assembly** — All passing shots are concatenated via ffmpeg into the final video

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CACHE_ROOT` | `./warehouse` | Root directory for all assets |
| `GOOGLE_DRIVE_MOUNT` | `/content/drive/MyDrive/AniLoom/warehouse` | Drive path for Colab persistence |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery job queue |
| `PIXVERSE_API_KEY` | — | PixVerse API key (optional fallback) |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8080` | FastAPI bind port |

## Requirements

- Python 3.9+
- CUDA-capable GPU (A100 recommended for CogVideoX 1.5, T4/P100/V100 for training only)
- ffmpeg (for video assembly)
- Redis (optional, for Celery job queue)
- ~20-25 GB VRAM for full video pipeline (CogVideoX 1.5)

## Contributing

Contributions welcome! Please open an issue or pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
