# AnimeLoom - Anime Character Consistency Engine

## Project Goal
Open-source Director-Orchestrated Pipeline for maintaining anime character identity across long-form animation using multiple specialized agents. Budget: under $20 total.

## Architecture
- **DirectorAgent** (`director/agent.py`): Main orchestrator — script parsing, shot execution, quality-gated regeneration, checkpointing, ffmpeg assembly
- **WorkflowGraph** (`director/workflow.py`): DAG-based shot dependency management with topological ordering and parallel group detection
- **AssetMemoryBank** (`director/memory_bank.py`): Persistent character/scene/shot storage, LoRA registry, pickle checkpoint save/load
- **CharacterAgent** (`agents/character/`): LoRA training (PEFT, rank 16-32, fp16), adapter load/unload, GroundingDINO+SAM+CLIP identity validation
- **AnimatorAgent** (`agents/animator/`): Wan2.2-Animate (motion imitation + role play), PixVerse fallback, OpenPose ControlNet conditioning
- **QualityEvaluator** (`agents/evaluator/`): Character consistency (cosine similarity), motion fidelity (optical flow), visual quality (sharpness, colour, smoothness)
- **API** (`api/`): FastAPI with character CRUD, shot/sequence generation, job status endpoints
- **Jobs** (`jobs/`): Celery + Redis async worker for background LoRA training and video generation
- **Cloud** (`cloud/`): Colab survival (keepalive 4min, checkpoint 5min, Drive mount), Kaggle P100 trainer, GCP T4 VM setup script

## Infrastructure
- Google Colab Pro ($10), Kaggle (30h/week free P100), Google Cloud $300 credits (~850h T4)
- Warehouse path configured via `AI_CACHE_ROOT` environment variable
- All models lazy-loaded to minimise VRAM; graceful fallbacks when dependencies unavailable

## Key Tech Stack
- **ML**: PyTorch, Diffusers, PEFT, Transformers, GroundingDINO, SAM, CLIP
- **Video**: OpenCV, ffmpeg, Wan2.2-Animate, PixVerse API
- **API**: FastAPI, Uvicorn, Pydantic
- **Queue**: Celery, Redis
- **Infra**: Google Colab, Kaggle, GCP, Google Drive persistence

## File Map (39 files)
```
director/       agent.py, workflow.py, memory_bank.py
agents/char/    lora_manager.py, trainer.py, consistency.py
agents/anim/    wan_wrapper.py, pixverse.py, controlnet.py
agents/eval/    character_score.py, motion_score.py, visual_score.py
api/            app.py, routes/{characters,generation}.py, schemas/models.py
jobs/           worker.py, tasks/{training,generation}.py
cloud/          colab_survival.py, kaggle_trainer.py, gcp_setup.sh
root/           main.py, setup.sh, requirements.txt, sample_script.txt
```

## Commands
```bash
python main.py --test              # smoke test
python main.py --script script.txt # process a story
python main.py --api               # start FastAPI server
python main.py --colab             # Colab survival mode
```
