#!/bin/bash
# AnimeLoom — RunPod Pod Setup Script
# Run this once when your pod starts (or add as a startup script in your template).
# Prerequisites: RunPod pod with GPU + network volume mounted at /mnt/network-volume

set -e

echo "============================================"
echo " AnimeLoom — RunPod Setup"
echo "============================================"

# ── 1. Clone or update repo ──────────────────────────────────────
cd /workspace
if [ ! -d "AnimeLoom" ]; then
    echo "Cloning AnimeLoom..."
    git clone https://github.com/JoelJohnsonThomas/AnimeLoom.git
else
    echo "Updating AnimeLoom..."
    cd AnimeLoom && git pull && cd /workspace
fi
cd AnimeLoom

# ── 2. Set warehouse on network volume (persists across restarts) ─
export AI_CACHE_ROOT=/mnt/network-volume/warehouse
echo "export AI_CACHE_ROOT=/mnt/network-volume/warehouse" >> ~/.bashrc

mkdir -p "$AI_CACHE_ROOT"/{models,lora,datasets/raw,datasets/tagged,outputs,checkpoints,references}
echo "Warehouse: $AI_CACHE_ROOT"

# ── 3. Install Python dependencies ──────────────────────────────
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip

# Core ML — DO NOT reinstall torch (base image has the correct version for the pod's CUDA driver)
# Pin diffusers + transformers for torch 2.4.x compatibility
pip install -q diffusers==0.30.3 transformers==4.44.2 accelerate==0.33.0
pip install -q safetensors peft>=0.7.0 sentencepiece protobuf

# Vision / detection
pip install -q opencv-python-headless pillow scikit-image scikit-learn
pip install -q controlnet-aux einops omegaconf ultralytics

# Upscaling / restoration
pip install -q realesrgan basicsr==1.4.2 gfpgan facexlib

# Quantisation (for CogVideoX int8)
pip install -q optimum-quanto 2>/dev/null || echo "optimum-quanto optional — skipping"

# Story decomposition (Gemini free tier)
pip install -q google-genai google-generativeai

# API / job queue (optional — only if running server mode)
pip install -q fastapi uvicorn celery redis pydantic python-dotenv

echo "Dependencies installed."

# ── 4. Install RIFE for temporal upscaling (8fps → 24fps) ───────
if [ ! -d "$AI_CACHE_ROOT/models/Practical-RIFE" ]; then
    echo ""
    echo "Installing RIFE temporal interpolation..."
    git clone https://github.com/hzwer/Practical-RIFE "$AI_CACHE_ROOT/models/Practical-RIFE"
    cd "$AI_CACHE_ROOT/models/Practical-RIFE"
    wget -q https://github.com/hzwer/Practical-RIFE/releases/download/v4.26/rife426.zip
    unzip -q rife426.zip -d train_log && rm rife426.zip
    cd /workspace/AnimeLoom
    echo "RIFE installed."
else
    echo "RIFE already installed."
fi

# ── 5. torchvision compatibility shim (for basicsr/gfpgan) ──────
python3 -c "
import torchvision, types, sys
if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F
    m = types.ModuleType('torchvision.transforms.functional_tensor')
    for a in ['rgb_to_grayscale','normalize','resize','pad']:
        if hasattr(F,a): setattr(m,a,getattr(F,a))
    sys.modules['torchvision.transforms.functional_tensor'] = m
    print('torchvision shim installed')
else:
    print('torchvision shim not needed')
"

# ── 6. Quick GPU check ──────────────────────────────────────────
echo ""
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name} ({mem:.1f} GB)')
else:
    print('WARNING: No GPU detected!')
"

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " AnimeLoom ready on RunPod!"
echo "============================================"
echo ""
echo "Usage:"
echo "  cd /workspace/AnimeLoom"
echo "  python main.py --text 'A girl walks through a cherry blossom forest' --quality standard"
echo "  python main.py --api          # Start FastAPI server"
echo "  python main.py --test         # Smoke test"
echo ""
echo "Warehouse: $AI_CACHE_ROOT"
echo "Models will download on first use and persist on your network volume."
