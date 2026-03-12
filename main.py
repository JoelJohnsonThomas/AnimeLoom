#!/usr/bin/env python3
"""
AnimeLoom - Anime Character Consistency Engine
Main entry point.
"""

import os
import sys
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from director.agent import DirectorAgent
from director.memory_bank import AssetMemoryBank
from cloud.colab_survival import ColabSurvival


def setup_warehouse() -> Path:
    """Initialise the warehouse directory structure."""
    warehouse = Path(os.getenv("AI_CACHE_ROOT", "./warehouse"))
    for subdir in ("models", "lora", "datasets", "outputs", "checkpoints"):
        (warehouse / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Warehouse initialised at {warehouse}")
    return warehouse


def run_test(warehouse: Path):
    """Smoke test: create a character, list it, verify memory round-trip."""
    print("Running system test...\n")

    memory = AssetMemoryBank(str(warehouse))

    # Create a dummy test image
    test_img_dir = PROJECT_ROOT / "test_images"
    test_img_dir.mkdir(exist_ok=True)
    test_img = test_img_dir / "char_test.png"

    if not test_img.exists():
        from PIL import Image
        img = Image.new("RGB", (512, 512), (120, 180, 255))
        img.save(str(test_img))
        print(f"  Created placeholder test image: {test_img}")

    # Create character
    char_id = memory.create_character(
        name="Test Character",
        images=[str(test_img)],
        description="A test anime character with blue hair",
    )
    print(f"  Created character: {char_id}")

    # List
    chars = memory.list_characters()
    print(f"  Characters in memory: {json.dumps(chars, indent=2)}")

    # Parse a sample script
    director = DirectorAgent(str(warehouse))
    sample_script = (
        "SCENE: Test intro\n"
        "CHAR: Test Character\n"
        "A character stands in a field\n"
    )
    shots = director.parse_script(sample_script)
    print(f"  Parsed {len(shots)} shot(s) from sample script")

    # Checkpoint round-trip
    director.current_job_id = "test_job"
    director.save_checkpoint()
    assert director.resume_story("test_job"), "Checkpoint resume failed"
    print("  Checkpoint save/resume: OK")

    print("\nAll tests passed!")


def run_api():
    """Start the FastAPI server."""
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    print(f"Starting API server on {host}:{port}")
    uvicorn.run("api.app:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="AnimeLoom - Anime Character Consistency Engine"
    )
    parser.add_argument("--script", type=str, help="Path to script file")
    parser.add_argument("--story-id", type=str, help="Story ID to resume")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--colab", action="store_true", help="Run in Colab survival mode")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")

    args = parser.parse_args()

    warehouse = setup_warehouse()

    # --- API mode ---
    if args.api:
        run_api()
        return

    # --- Test mode ---
    if args.test:
        run_test(warehouse)
        return

    # --- Director mode ---
    director = DirectorAgent(str(warehouse))

    # Colab survival
    if args.colab:
        survival = ColabSurvival(director)
        survival.mount_google_drive()
        survival.setup_warehouse_on_drive()
        survival.start()

        if args.story_id:
            if survival.resume_from_checkpoint(args.story_id):
                print(f"Resumed story {args.story_id}")
                result = director.continue_processing()
                print(json.dumps(result, indent=2, default=str))
                return

    # Process script
    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Script file not found: {script_path}")
            sys.exit(1)

        script = script_path.read_text()
        result = director.process_story(script, args.story_id)
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
