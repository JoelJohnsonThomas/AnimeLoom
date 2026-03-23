# AnimeLoom - Anime Character Consistency Engine

## Project Goal
Open-source Director-Orchestrated Pipeline for text-to-anime video generation with character consistency across long-form animation. Sora-like interface: type a story, get smooth anime video. Budget: under $20 total.

## Architecture
- **DirectorAgent** (`director/agent.py`): Main orchestrator — script parsing, text-to-story decomposition, shot execution, post-processing, quality-gated regeneration, checkpointing, cross-dissolve assembly
- **WorkflowGraph** (`director/workflow.py`): DAG-based shot dependency management with topological ordering and parallel group detection
- **AssetMemoryBank** (`director/memory_bank.py`): Persistent character/scene/shot storage, LoRA registry, pickle checkpoint save/load
- **StoryDecomposer** (`agents/story/decomposer.py`): NLP story → shot script (Gemini Flash primary, rule-based fallback)
- **CharacterAgent** (`agents/character/`): LoRA training (PEFT, rank 16-32, fp16), adapter load/unload, GroundingDINO+SAM+CLIP identity validation
- **AnimatorAgent** (`agents/animator/`): CogVideoX-2B (primary T2V), Wan2.2-Animate, AnimateDiff, SDXL keyframes, PixVerse fallback
- **PostProcessor** (`agents/postprocess/`): RIFE temporal upscale (8→24fps), Real-ESRGAN spatial upscale (480p→720p+), anime color grading, cross-dissolve transitions
- **QualityEvaluator** (`agents/evaluator/`): Character consistency (cosine similarity), motion fidelity (optical flow), visual quality (sharpness, colour, smoothness)
- **API** (`api/`): FastAPI with character CRUD, shot/sequence/text-to-anime generation, job status endpoints
- **Jobs** (`jobs/`): Celery + Redis async worker for background LoRA training and video generation
- **Cloud** (`cloud/`): Colab survival (keepalive 4min, checkpoint 5min, Drive mount), Kaggle P100 trainer, GCP T4 VM setup script, model pre-download helpers

## Video Generation Pipeline
```
Text → StoryDecomposer → Script → DAG → Per-shot:
  1. CogVideoX-2B (49 frames, 480×720, 8fps) or fallbacks
  2. RIFE temporal upscale → 24fps
  3. Real-ESRGAN spatial upscale → 720p/1080p
  4. Anime color grading
  5. Quality evaluation (threshold 0.70)
→ Cross-dissolve assembly → Final video
```

## Infrastructure
- Google Colab Pro ($10), Kaggle (30h/week free P100), Google Cloud $300 credits (~850h T4)
- Gemini 1.5 Flash free tier (1500 req/day) for story decomposition
- Warehouse path configured via `AI_CACHE_ROOT` environment variable
- All models lazy-loaded to minimise VRAM; graceful fallbacks when dependencies unavailable

## Key Tech Stack
- **ML**: PyTorch, Diffusers, PEFT, Transformers, CogVideoX, GroundingDINO, SAM, CLIP
- **Video**: OpenCV, ffmpeg, Real-ESRGAN, RIFE, CogVideoX-2B, Wan2.2-Animate, AnimateDiff
- **NLP**: Gemini 1.5 Flash API, rule-based fallback
- **API**: FastAPI, Uvicorn, Pydantic
- **Queue**: Celery, Redis
- **Infra**: Google Colab, Kaggle, GCP, Google Drive persistence

## File Map
```
director/       agent.py, workflow.py, memory_bank.py
agents/story/   decomposer.py
agents/char/    lora_manager.py, trainer.py, consistency.py
agents/anim/    cogvideo_wrapper.py, wan_wrapper.py, pixverse.py, controlnet.py
agents/post/    upscaler.py, color_grade.py, transitions.py
agents/eval/    character_score.py, motion_score.py, visual_score.py
api/            app.py, routes/{characters,generation}.py, schemas/models.py
jobs/           worker.py, tasks/{training,generation}.py
cloud/          colab_survival.py, kaggle_trainer.py, gcp_setup.sh
root/           main.py, setup.sh, requirements.txt, sample_script.txt
```

## Commands
```bash
python main.py --text "A girl walks through a cherry blossom forest"  # text-to-anime
python main.py --text "story..." --quality high                       # high quality preset
python main.py --script script.txt   # process a formatted story script
python main.py --test                # smoke test
python main.py --api                 # start FastAPI server
python main.py --colab               # Colab survival mode
```
