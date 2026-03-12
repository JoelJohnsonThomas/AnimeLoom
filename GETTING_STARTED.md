# AnimeLoom - Getting Started Guide

A step-by-step guide to using AnimeLoom for anime character consistency.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Verify Setup](#2-verify-setup)
3. [Create Your First Character](#3-create-your-first-character)
4. [Write a Script](#4-write-a-script)
5. [Generate a Video Sequence](#5-generate-a-video-sequence)
6. [Use the API Server](#6-use-the-api-server)
7. [Run on Google Colab](#7-run-on-google-colab)
8. [Run on Kaggle](#8-run-on-kaggle)
9. [Run on Google Cloud](#9-run-on-google-cloud)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Installation

### Option A: Quick Setup (Linux / macOS / Colab)

```bash
cd AnimeLoom
chmod +x setup.sh
./setup.sh
```

### Option B: Manual Setup (Windows / any OS)

```bash
cd AnimeLoom

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create the warehouse (asset storage)
# Windows PowerShell:
mkdir -p warehouse/models, warehouse/lora, warehouse/datasets, warehouse/outputs, warehouse/checkpoints
# Or let the program create it automatically on first run
```

### Option C: Minimal Install (no GPU, for testing)

```bash
pip install pillow numpy python-dotenv pydantic fastapi uvicorn tqdm
```

This lets you run `--test`, the API server, and script parsing without GPU dependencies.

---

## 2. Verify Setup

Run the smoke test to make sure everything works:

```bash
python main.py --test
```

**Expected output:**
```
Warehouse initialised at warehouse
Running system test...

  Created placeholder test image: test_images\char_test.png
  Created character: a1b2c3d4e5f6
  Characters in memory: [{"id": "a1b2c3d4e5f6", "name": "Test Character", ...}]
  Parsed 1 shot(s) from sample script
  Checkpoint save/resume: OK

All tests passed!
```

If you see "All tests passed!" you're good to go.

---

## 3. Create Your First Character

Before generating video, you need to register characters with reference images.

### Step 1: Prepare Reference Images

Create a folder with 3-5 images of your character from different angles:

```
my_characters/
├── yuki_front.png      # Front view
├── yuki_side.png       # Side view
├── yuki_back.png       # Back view (optional)
└── yuki_expression.png # Different expression (optional)
```

**Image tips:**
- Use clear, high-quality anime character art
- 512x512 or larger resolution
- Consistent art style across all reference images
- White or transparent background works best

### Step 2: Register via Python

```python
from director.memory_bank import AssetMemoryBank

memory = AssetMemoryBank("./warehouse")

# Create character from reference images
char_id = memory.create_character(
    name="Yuki",
    images=[
        "my_characters/yuki_front.png",
        "my_characters/yuki_side.png",
        "my_characters/yuki_expression.png"
    ],
    description="Young girl with long blue hair, wearing a white school uniform"
)

print(f"Character registered! ID: {char_id}")
print(f"LoRA saved at: warehouse/lora/{char_id}/")
```

### Step 3: Register via API (alternative)

Start the API server first (`python main.py --api`), then:

```bash
curl -X POST http://localhost:8080/character/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Yuki",
    "description": "Young girl with long blue hair",
    "image_paths": ["my_characters/yuki_front.png", "my_characters/yuki_side.png"]
  }'
```

### Step 4: Verify Character Exists

```python
# List all characters
for char in memory.list_characters():
    print(f"  {char['name']} (ID: {char['id']}, shots: {char['shot_count']})")
```

---

## 4. Write a Script

AnimeLoom uses a simple text-based script format. Create a `.txt` file:

### Script Format

```
SCENE: <scene title>
CHAR: <character name>
CHAR: <another character>       (optional, for multi-character scenes)
POSE: <pose_reference.mp4>      (optional, for motion transfer)
<description of what happens in this scene>
```

### Example: `my_story.txt`

```
SCENE: Morning at school
CHAR: Yuki
Yuki walks through the school gates, her blue hair blowing in the wind

SCENE: Classroom meeting
CHAR: Yuki
CHAR: Kenji
Yuki sits at her desk. Kenji walks over and waves hello.

SCENE: Rooftop lunch
CHAR: Yuki
POSE: eating_pose.mp4
Yuki eats lunch alone on the rooftop, looking at the sky

SCENE: After school
CHAR: Yuki
CHAR: Kenji
They walk home together along the river as the sun sets
```

### Rules

| Directive | Required? | Description |
|-----------|-----------|-------------|
| `SCENE:` or `SHOT:` | Yes | Starts a new shot. Text after `:` is the scene title |
| `CHAR:` | Yes | Character in this shot. Use one `CHAR:` line per character |
| `POSE:` | No | Path to a pose reference video for motion transfer |
| Free text | Yes | Description/prompt for what happens in the scene |

**Important:** Character names in the script must match names you registered in Step 3.

---

## 5. Generate a Video Sequence

### Option A: Command Line

```bash
# Process your script
python main.py --script my_story.txt

# With a custom story ID (for resume support)
python main.py --script my_story.txt --story-id my_first_story
```

### Option B: Python Script

```python
from director.agent import DirectorAgent

director = DirectorAgent("./warehouse")

# Read your script
with open("my_story.txt") as f:
    script = f.read()

# Process the entire story
result = director.process_story(script, story_id="my_first_story")

# Results
print(f"Story ID:    {result['story_id']}")
print(f"Shots:       {len(result['shots'])}")
print(f"Characters:  {result['character_count']}")
print(f"Final video: {result['final_video']}")

# Check individual shot scores
for shot in result['shots']:
    print(f"  Shot {shot['shot_index']}: quality={shot['quality_score']:.2f}")
```

### What happens during processing

1. **Parsing** — Script is split into individual shots
2. **Character lookup** — For each shot, character LoRAs are loaded from the warehouse
3. **Generation** — Animator agent generates each shot using Wan2.2-Animate
4. **Quality check** — Evaluator scores character consistency (target: >0.85)
5. **Regeneration** — Shots below 0.85 are regenerated (up to 3 attempts)
6. **Checkpointing** — State is saved every 5 minutes
7. **Assembly** — All shots are concatenated into a final video via ffmpeg

### Output

Generated files are saved to:
```
warehouse/
├── outputs/
│   └── my_first_story/
│       ├── shot_0.mp4
│       ├── shot_1.mp4
│       ├── shot_2.mp4
│       ├── shot_3.mp4
│       └── final_video.mp4    ← your finished video
└── checkpoints/
    └── my_first_story/
        └── checkpoint_*.pkl   ← resume points
```

---

## 6. Use the API Server

The API lets you control AnimeLoom from any HTTP client or frontend.

### Start the Server

```bash
python main.py --api
```

Server runs at `http://localhost:8080`. Open `http://localhost:8080/docs` in your browser for interactive Swagger docs.

### API Endpoints

#### Create a Character

```bash
curl -X POST http://localhost:8080/character/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Yuki",
    "description": "Blue-haired anime girl",
    "image_paths": ["my_characters/yuki_front.png"]
  }'
```

**Response:**
```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Yuki",
  "shot_count": 0,
  "has_lora": true,
  "created": "2026-03-11T10:30:00",
  "last_used": "2026-03-11T10:30:00"
}
```

#### List All Characters

```bash
curl http://localhost:8080/character/list
```

#### Generate a Single Shot

```bash
curl -X POST http://localhost:8080/generate/shot \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Yuki stands in a flower field at sunset",
    "characters": ["Yuki"],
    "width": 512,
    "height": 512,
    "num_frames": 16
  }'
```

**Response:**
```json
{
  "job_id": "abc-123-def",
  "status": "pending",
  "progress": 0.0
}
```

#### Generate a Full Sequence

```bash
curl -X POST http://localhost:8080/generate/sequence \
  -H "Content-Type: application/json" \
  -d '{
    "script": "SCENE: Intro\nCHAR: Yuki\nYuki stands in a flower field",
    "story_id": "my_story_001"
  }'
```

#### Check Job Status

```bash
curl http://localhost:8080/job/abc-123-def
```

**Response (while running):**
```json
{
  "job_id": "abc-123-def",
  "status": "running",
  "progress": 0.5
}
```

**Response (when done):**
```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "story_id": "my_story_001",
    "final_video": "warehouse/outputs/my_story_001/final_video.mp4"
  }
}
```

#### Delete a Character

```bash
curl -X DELETE http://localhost:8080/character/a1b2c3d4e5f6
```

---

## 7. Run on Google Colab

Colab gives you free/cheap GPU access. AnimeLoom has built-in Colab survival mode.

### Step 1: Open a New Colab Notebook

### Step 2: Setup Cell

```python
# Cell 1: Install
!git clone https://github.com/yourusername/AnimeLoom.git
%cd AnimeLoom
!pip install -r requirements.txt
```

### Step 3: Mount Google Drive (persistence)

```python
# Cell 2: Mount Drive for persistence across sessions
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['AI_CACHE_ROOT'] = '/content/drive/MyDrive/anime_warehouse'
```

### Step 4: Create Characters

```python
# Cell 3: Upload character images
from google.colab import files
uploaded = files.upload()  # Upload your character reference images

# Register character
from director.memory_bank import AssetMemoryBank
memory = AssetMemoryBank(os.environ['AI_CACHE_ROOT'])

# Save uploaded files
for filename, data in uploaded.items():
    with open(filename, 'wb') as f:
        f.write(data)

char_id = memory.create_character(
    name="Yuki",
    images=list(uploaded.keys()),
    description="Blue-haired anime girl"
)
print(f"Character created: {char_id}")
```

### Step 5: Process with Survival Mode

```python
# Cell 4: Process with auto-checkpoint and keepalive
from director.agent import DirectorAgent
from cloud.colab_survival import ColabSurvival

director = DirectorAgent(os.environ['AI_CACHE_ROOT'])

# Enable survival mode (keepalive every 4min, checkpoint every 5min)
survival = ColabSurvival(director)
survival.start()

# Process your script
script = """
SCENE: Intro
CHAR: Yuki
Yuki stands in a flower field at sunset
"""

result = director.process_story(script, story_id="colab_story_001")
print(result)
```

### Step 6: Resume After Disconnect

If Colab disconnects, just re-run setup cells, then:

```python
# Cell: Resume from checkpoint
from director.agent import DirectorAgent
from cloud.colab_survival import ColabSurvival

director = DirectorAgent('/content/drive/MyDrive/anime_warehouse')
survival = ColabSurvival(director)

if survival.resume_from_checkpoint("colab_story_001"):
    print("Resumed! Continuing...")
    result = director.continue_processing()
else:
    print("No checkpoint found, starting fresh")
```

---

## 8. Run on Kaggle

Kaggle gives 30 hours/week of free P100 GPU — great for LoRA training.

### Step 1: Create a New Kaggle Notebook

### Step 2: Setup

```python
!git clone https://github.com/yourusername/AnimeLoom.git
import sys
sys.path.insert(0, '/kaggle/working/AnimeLoom')
!pip install -r /kaggle/working/AnimeLoom/requirements.txt
```

### Step 3: Train LoRA on Kaggle GPU

```python
from cloud.kaggle_trainer import KaggleTrainer

trainer = KaggleTrainer()

# Upload character images to /kaggle/input/your-dataset/
lora_path = trainer.train(
    character_name="Yuki",
    image_paths=["/kaggle/input/your-dataset/yuki_front.png"],
    rank=16,
    max_steps=500
)

# Download the trained LoRA
trainer.export_lora(lora_path)
```

---

## 9. Run on Google Cloud

For long-running jobs, use GCP's free $300 credits (~850 hours of T4 GPU).

```bash
# Provision a T4 VM
chmod +x cloud/gcp_setup.sh
./cloud/gcp_setup.sh

# SSH into the VM
gcloud compute ssh anime-engine --zone=us-central1-a

# On the VM:
cd AnimeLoom
python main.py --script my_story.txt
```

**Cost:** ~$0.35/hour = 850+ hours from $300 free credits.

---

## 10. Troubleshooting

### "Module not found" errors

```bash
# Make sure you're in the project root
cd AnimeLoom
# And your virtual environment is active
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### "CUDA not available"

The system works without GPU using fallback paths. Quality will be lower but everything runs. For real generation, use Colab/Kaggle/GCP.

```python
import torch
print(torch.cuda.is_available())  # Should be True on GPU machines
```

### "Character not found" during generation

Make sure the character name in your script **exactly matches** the name you used during `create_character()`. Names are case-insensitive but must otherwise match.

```python
# Check what characters exist
memory = AssetMemoryBank("./warehouse")
for c in memory.list_characters():
    print(c['name'])  # Use these exact names in your script
```

### Colab keeps disconnecting

Make sure survival mode is running:
```python
survival = ColabSurvival(director)
survival.start()  # Must call this!
```
Your progress is saved to Google Drive every 5 minutes automatically.

### Out of VRAM

AnimeLoom lazy-loads models. If you run out of VRAM:
- Reduce image size to 256x256
- Use `rank=8` instead of `rank=32` for LoRA training
- Close other GPU processes
- Use Kaggle P100 (16GB) or Colab A100 (40GB)

### Want to start fresh

```bash
# Delete all generated data (keeps code intact)
rm -rf warehouse/
python main.py --test  # Re-creates warehouse structure
```

---

## Quick Reference

| What you want to do | Command |
|---------------------|---------|
| Verify installation | `python main.py --test` |
| Process a script | `python main.py --script my_story.txt` |
| Start API server | `python main.py --api` |
| Run on Colab | `python main.py --colab` |
| Resume a story | `python main.py --script my_story.txt --story-id my_story` |
| View API docs | Open `http://localhost:8080/docs` |

---

## Typical Workflow

```
1. Prepare character reference images (3-5 per character)
       ↓
2. Register characters (Python or API)
       ↓
3. Write your script (SCENE/CHAR/POSE format)
       ↓
4. Run generation (CLI or API)
       ↓
5. Check quality scores
       ↓
6. Find your video in warehouse/outputs/
```
