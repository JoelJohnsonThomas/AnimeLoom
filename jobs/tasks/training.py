"""
Celery tasks for LoRA training.
"""

import os
from jobs.worker import celery_app


@celery_app.task(bind=True, name="jobs.tasks.training.train_lora")
def train_lora(self, character_id: str, image_paths: list, character_name: str = None):
    """
    Background LoRA training task.

    Args:
        character_id: Unique character identifier.
        image_paths: List of paths to reference images.
        character_name: Human-readable name.

    Returns:
        Dict with lora_path and status.
    """
    warehouse = os.getenv("AI_CACHE_ROOT", "./warehouse")

    self.update_state(state="TRAINING", meta={"character_id": character_id, "progress": 0})

    try:
        from agents.character.trainer import LoRATrainer

        trainer = LoRATrainer(warehouse)
        lora_path = trainer.train_character_lora(
            character_images=image_paths,
            character_id=character_id,
            character_name=character_name,
        )

        # Update memory bank
        from director.memory_bank import AssetMemoryBank
        memory = AssetMemoryBank(warehouse)
        memory.update_character_lora(character_id, str(lora_path))

        return {"status": "completed", "lora_path": str(lora_path), "character_id": character_id}

    except Exception as e:
        return {"status": "failed", "error": str(e), "character_id": character_id}
