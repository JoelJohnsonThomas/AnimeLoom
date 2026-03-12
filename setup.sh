#!/bin/bash
set -e

echo "=== AnimeLoom - Anime Character Consistency Engine ==="
echo "Setting up environment..."

# Create virtual environment if not in one
if [ -z "$VIRTUAL_ENV" ] && [ -z "$COLAB_RELEASE_TAG" ]; then
    python -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated"
fi

# Install dependencies
pip install -r requirements.txt

# Set up warehouse
export AI_CACHE_ROOT=${AI_CACHE_ROOT:-$(pwd)/warehouse}
mkdir -p "$AI_CACHE_ROOT"/{models,lora,datasets,outputs,checkpoints}

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "AI_CACHE_ROOT=$AI_CACHE_ROOT" > .env
    echo "GOOGLE_DRIVE_MOUNT=/content/drive/MyDrive/anime_warehouse" >> .env
    echo "REDIS_URL=redis://localhost:6379/0" >> .env
    echo ".env file created"
fi

echo ""
echo "Setup complete!"
echo "Run 'python main.py --test' to verify installation."
echo "Run 'python main.py --script sample_script.txt' to process a script."
