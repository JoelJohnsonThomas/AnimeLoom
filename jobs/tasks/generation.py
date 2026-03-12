"""
Celery tasks for video generation.
"""

import os
from jobs.worker import celery_app


@celery_app.task(bind=True, name="jobs.tasks.generation.generate_shot")
def generate_shot_task(
    self,
    description: str,
    characters: list,
    pose_ref: str = None,
    shot_index: int = 0,
):
    """
    Background single-shot generation task.
    """
    warehouse = os.getenv("AI_CACHE_ROOT", "./warehouse")

    self.update_state(state="GENERATING", meta={"shot_index": shot_index, "progress": 0})

    try:
        from director.agent import DirectorAgent

        director = DirectorAgent(warehouse)
        shot = {
            "description": description,
            "characters": characters,
            "pose_ref": pose_ref,
        }

        result = director._execute_shot(shot, shot_index)

        return {
            "status": "completed",
            "video_path": result.get("video_path"),
            "quality_score": result.get("quality_score", 0),
            "shot_index": shot_index,
        }

    except Exception as e:
        return {"status": "failed", "error": str(e), "shot_index": shot_index}


@celery_app.task(bind=True, name="jobs.tasks.generation.generate_sequence")
def generate_sequence_task(self, script: str, story_id: str = None):
    """
    Background full-sequence generation task.
    """
    warehouse = os.getenv("AI_CACHE_ROOT", "./warehouse")

    self.update_state(state="GENERATING", meta={"story_id": story_id, "progress": 0})

    try:
        from director.agent import DirectorAgent

        director = DirectorAgent(warehouse)
        result = director.process_story(script, story_id=story_id)

        return {
            "status": "completed",
            "story_id": result.get("story_id"),
            "shot_count": len(result.get("shots", [])),
            "final_video": result.get("final_video"),
        }

    except Exception as e:
        return {"status": "failed", "error": str(e), "story_id": story_id}
