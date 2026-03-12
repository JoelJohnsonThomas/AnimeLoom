# AnimeLoom

**Anime Character Consistency Engine** — maintain character identity across long-form animation using a Director-Orchestrated multi-agent pipeline.

AnimeLoom coordinates specialized AI agents (character training, video generation, quality evaluation) to produce consistent anime sequences where characters look the same across every shot. Built to run on free/cheap GPU resources (Colab, Kaggle, GCP free credits) for **under $20 total**.

---

## Features

- **Character Identity Preservation** — LoRA fine-tuning (rank 16-32, fp16) locks in each character's visual identity
- **Multi-Agent Pipeline** — Director orchestrates Character, Animator, and Evaluator agents with a dependency-aware workflow graph
- **Quality-Gated Output** — shots scoring below 0.85 consistency are automatically regenerated (up to 3 attempts)
- **Colab Survival Mode** — keepalive every 4 min, auto-checkpoint every 5 min, Google Drive persistence, resume after disconnect
- **Multiple Generation Backends** — Wan2.2-Animate primary, PixVerse fallback, ControlNet pose conditioning
- **Detection + Segmentation** — GroundingDINO + SAM isolate characters; CLIP embeddings measure identity similarity
- **REST API** — FastAPI endpoints for character creation, shot generation, sequence processing, and job tracking
- **Async Job Queue** — Celery + Redis for background LoRA training and batch video generation

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

### 1. Clone & Install

```bash
git clone https://github.com/JoelJohnsonThomas/AnimeLoom.git
cd AnimeLoom
chmod +x setup.sh
./setup.sh
```

### 2. Run Smoke Test

```bash
python main.py --test
```

Expected output:
```
Created character: ffe3da79150c
Characters in memory: [...]
Parsed 1 shot(s) from sample script
Checkpoint save/resume: OK
All tests passed!
```

### 3. Process a Script

```bash
python main.py --script sample_script.txt
```

### 4. Start the API Server

```bash
python main.py --api
```

API available at `http://localhost:8080` — see [API Endpoints](#api-endpoints) below.

## Script Format

AnimeLoom uses a simple text-based script format:

```
SCENE: Character introduction
CHAR: Yuki
A young girl with long blue hair stands in a field of flowers

SCENE: Walking scene
CHAR: Yuki
POSE: walking_pose.mp4
Yuki walks through the forest, looking around curiously

SCENE: Conversation
CHAR: Yuki
CHAR: Kenji
Yuki meets Kenji by the river. They talk about their day.
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
    "name": "Yuki",
    "description": "Young girl with long blue hair",
    "image_paths": ["./test_images/yuki_front.png", "./test_images/yuki_side.png"]
  }'
```

### Example: Generate a Sequence

```bash
curl -X POST http://localhost:8080/generate/sequence \
  -H "Content-Type: application/json" \
  -d '{
    "script": "SCENE: Intro\nCHAR: Yuki\nYuki stands in a flower field",
    "story_id": "my_story_001"
  }'
```

## Cloud Deployment

### Google Colab (Recommended for Getting Started)

```python
# In a Colab notebook cell:
!git clone https://github.com/yourusername/AnimeLoom.git
%cd AnimeLoom
!pip install -r requirements.txt

from main import setup_warehouse
from director.agent import DirectorAgent
from cloud.colab_survival import ColabSurvival

warehouse = setup_warehouse()
director = DirectorAgent(str(warehouse))

# Enable survival mode
survival = ColabSurvival(director)
survival.mount_google_drive()
survival.setup_warehouse_on_drive()
survival.start()

# Process your script
result = director.process_story(open("sample_script.txt").read())
```

### Kaggle (Free 30h/week P100)

```python
# In a Kaggle notebook cell:
from cloud.kaggle_trainer import KaggleTrainer

trainer = KaggleTrainer()
lora_path = trainer.train("Yuki", ["/kaggle/input/charsheet/yuki.png"], rank=16, max_steps=500)
trainer.export_lora(lora_path)
```

### Google Cloud ($300 Free Credits)

```bash
# ~$0.35/hour = 850+ hours from free credits
chmod +x cloud/gcp_setup.sh
./cloud/gcp_setup.sh
```

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

4. **Video Generation** — Each shot is generated via Wan2.2-Animate with character LoRAs loaded. Falls back to PixVerse if primary fails

5. **Quality Evaluation** — Generated shots are scored on character consistency (CLIP cosine similarity), motion fidelity (optical flow), and visual quality (sharpness, colour stability). Shots below 0.85 are regenerated

6. **Checkpointing** — Every 5 minutes, full state is saved. After a Colab disconnect, resume exactly where you left off

7. **Assembly** — All passing shots are concatenated via ffmpeg into the final video

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_CACHE_ROOT` | `./warehouse` | Root directory for all assets |
| `GOOGLE_DRIVE_MOUNT` | `/content/drive/MyDrive/anime_warehouse` | Drive path for Colab persistence |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery job queue |
| `PIXVERSE_API_KEY` | — | PixVerse API key (optional fallback) |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8080` | FastAPI bind port |

## Requirements

- Python 3.9+
- CUDA-capable GPU (T4 / P100 / V100 / A100 recommended)
- ffmpeg (for video assembly)
- Redis (optional, for Celery job queue)

## Contributing

Contributions welcome! Please open an issue or pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
