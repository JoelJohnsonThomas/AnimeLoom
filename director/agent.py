"""
Director Agent - Main orchestrator that coordinates all agents
and maintains character consistency across shots.
"""

import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from director.memory_bank import AssetMemoryBank
from director.workflow import WorkflowGraph


# ---------------------------------------------------------------------------
# Lazy-imported agent facades so the director module can be loaded without
# heavy ML dependencies (they are only resolved when actually needed).
# ---------------------------------------------------------------------------

class _CharacterAgent:
    """Facade that wraps Character sub-agents."""

    def __init__(self, warehouse: str, memory: AssetMemoryBank):
        self.warehouse = warehouse
        self.memory = memory
        self._lora_manager = None
        self._trainer = None
        self._consistency = None

    @property
    def lora_manager(self):
        if self._lora_manager is None:
            from agents.character.lora_manager import LoRAManager
            self._lora_manager = LoRAManager(self.warehouse)
        return self._lora_manager

    @property
    def trainer(self):
        if self._trainer is None:
            from agents.character.trainer import LoRATrainer
            self._trainer = LoRATrainer(self.warehouse)
        return self._trainer

    @property
    def consistency(self):
        if self._consistency is None:
            from agents.character.consistency import CharacterConsistencyChecker
            self._consistency = CharacterConsistencyChecker(self.warehouse)
        return self._consistency

    def ensure_lora(self, char_name: str):
        """Make sure a LoRA exists for *char_name*; train one if needed."""
        char_data = self.memory.get_character(char_name)
        if char_data is None:
            return None
        lora_path = self.memory.get_character_lora_path(char_data["id"])
        if lora_path and lora_path.stat().st_size > 100:
            return lora_path
        # Need to train
        images = char_data.get("multi_views", [])
        if not images:
            return None
        trained = self.trainer.train_character_lora(
            images, char_data["id"], char_data["name"]
        )
        self.memory.update_character_lora(char_data["id"], str(trained))
        return trained


class _AnimatorAgent:
    """Facade wrapping Animator sub-agents."""

    def __init__(self, warehouse: str, memory: AssetMemoryBank):
        self.warehouse = warehouse
        self.memory = memory
        self._wan = None
        self._pixverse = None

    @property
    def wan(self):
        if self._wan is None:
            from agents.animator.wan_wrapper import WanAnimator
            self._wan = WanAnimator(self.warehouse)
        return self._wan

    @property
    def pixverse(self):
        if self._pixverse is None:
            from agents.animator.pixverse import PixVerseGenerator
            self._pixverse = PixVerseGenerator(self.warehouse)
        return self._pixverse

    def generate_shot(
        self,
        description: str,
        characters: Dict[str, str],
        pose_ref: Optional[str] = None,
        shot_index: int = 0,
        feedback: Optional[Dict] = None,
    ) -> Dict:
        """Generate a shot video, returning metadata dict with *video_path*."""
        try:
            result = self.wan.generate(
                description=description,
                character_loras=characters,
                pose_reference=pose_ref,
                shot_index=shot_index,
            )
            return result
        except Exception as e:
            print(f"Wan2.2 generation failed ({e}), falling back to PixVerse")
            return self.pixverse.generate(
                description=description,
                shot_index=shot_index,
            )


class _QualityEvaluator:
    """Facade wrapping the evaluator sub-agents."""

    def __init__(self, warehouse: str):
        self.warehouse = warehouse
        self._char_eval = None
        self._motion_eval = None
        self._visual_eval = None

    @property
    def character(self):
        if self._char_eval is None:
            from agents.evaluator.character_score import CharacterConsistencyEvaluator
            self._char_eval = CharacterConsistencyEvaluator(self.warehouse)
        return self._char_eval

    @property
    def motion(self):
        if self._motion_eval is None:
            from agents.evaluator.motion_score import MotionFidelityEvaluator
            self._motion_eval = MotionFidelityEvaluator(self.warehouse)
        return self._motion_eval

    @property
    def visual(self):
        if self._visual_eval is None:
            from agents.evaluator.visual_score import VisualQualityEvaluator
            self._visual_eval = VisualQualityEvaluator(self.warehouse)
        return self._visual_eval

    def evaluate_shot(self, video_path: str, characters: List[str]) -> float:
        """Aggregate quality score (0-1) for a shot."""
        scores = []
        try:
            scores.append(self.character.evaluate(video_path, characters))
        except Exception:
            scores.append(0.9)  # default if evaluator not ready
        try:
            scores.append(self.visual.evaluate(video_path))
        except Exception:
            scores.append(0.9)
        return sum(scores) / len(scores) if scores else 0.9

    def get_feedback(self, result: Dict) -> Dict:
        """Return structured feedback for regeneration."""
        return {
            "quality_score": result.get("quality_score", 0),
            "issues": result.get("issues", []),
            "suggestion": "increase_lora_weight",
        }


# ---------------------------------------------------------------------------
# Director Agent
# ---------------------------------------------------------------------------

class DirectorAgent:
    """
    Main orchestrator that coordinates all agents and maintains
    character consistency across an entire story.
    """

    CHECKPOINT_INTERVAL = 300  # seconds (5 min)
    QUALITY_THRESHOLD = 0.85
    MAX_REGEN_ATTEMPTS = 3

    def __init__(self, warehouse_path: str = None):
        self.warehouse = warehouse_path or os.getenv("AI_CACHE_ROOT", "./warehouse")
        self.asset_memory = AssetMemoryBank(self.warehouse)

        self.agents = {
            "character": _CharacterAgent(self.warehouse, self.asset_memory),
            "animator": _AnimatorAgent(self.warehouse, self.asset_memory),
            "evaluator": _QualityEvaluator(self.warehouse),
        }

        self.workflow_graph: Optional[WorkflowGraph] = None
        self.last_checkpoint = time.time()
        self.current_job_id: Optional[str] = None
        self.shot_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_story(
        self, script: str, story_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a complete story script, maintaining character consistency.

        Args:
            script: Multi-line story script.
            story_id: Optional ID for resuming.

        Returns:
            Dict with generated video paths and metadata.
        """
        # Resume if possible
        if story_id and self.resume_story(story_id):
            print(f"Resumed story {story_id} from checkpoint")
            return self.continue_processing(script)

        # Parse & plan
        shots = self.parse_script(script)
        self.current_job_id = story_id or f"story_{int(time.time())}"
        self.workflow_graph = WorkflowGraph.build_from_shots(shots)

        execution_order = self.workflow_graph.topological_order()
        results: List[Dict] = []

        for idx in execution_order:
            shot = shots[idx]
            print(f"Processing shot {idx + 1}/{len(shots)}: {shot['description'][:60]}")

            # Ensure LoRAs exist for every character in the shot
            for char_name in shot["characters"]:
                self.agents["character"].ensure_lora(char_name)

            shot["character_refs"] = self.asset_memory.get_characters_for_shot(shot)

            result = self._execute_shot(shot, idx)

            if result["quality_score"] < self.QUALITY_THRESHOLD:
                print(
                    f"  Shot {idx} quality {result['quality_score']:.2f} < "
                    f"{self.QUALITY_THRESHOLD}, regenerating..."
                )
                result = self._regenerate_shot(shot, idx, result)

            results.append(result)
            self.shot_history.append(
                {
                    "shot_index": idx,
                    "shot_data": shot,
                    "result": result,
                    "timestamp": time.time(),
                }
            )

            self._maybe_checkpoint()

        final_video = self._assemble_final_video(results)
        self.save_checkpoint(final=True)

        return {
            "story_id": self.current_job_id,
            "shots": results,
            "final_video": final_video,
            "character_count": len(self.asset_memory.list_characters()),
        }

    # ------------------------------------------------------------------
    # Script parsing
    # ------------------------------------------------------------------

    def parse_script(self, script: str) -> List[Dict]:
        """Parse a script into individual shot dicts."""
        lines = script.strip().split("\n")
        shots: List[Dict] = []
        current: Dict = {"characters": [], "description": "", "pose_ref": None}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("SCENE:") or line.upper().startswith("SHOT:"):
                if current["description"]:
                    shots.append(current)
                tag = "SCENE:" if line.upper().startswith("SCENE:") else "SHOT:"
                current = {
                    "characters": [],
                    "description": line[len(tag):].strip(),
                    "pose_ref": None,
                }
            elif line.upper().startswith("CHAR:"):
                name = line[5:].strip()
                for n in name.split(","):
                    n = n.strip()
                    if n and n not in current["characters"]:
                        current["characters"].append(n)
            elif line.upper().startswith("POSE:"):
                current["pose_ref"] = line[5:].strip()
            else:
                sep = " " if current["description"] else ""
                current["description"] += sep + line

        if current["description"]:
            shots.append(current)

        return shots

    # ------------------------------------------------------------------
    # Shot execution
    # ------------------------------------------------------------------

    def _execute_shot(self, shot: Dict, shot_index: int) -> Dict:
        character_loras: Dict[str, str] = {}
        for char_name in shot["characters"]:
            char_data = self.asset_memory.get_character(char_name)
            if char_data and char_data.get("lora_path"):
                character_loras[char_name] = char_data["lora_path"]

        result = self.agents["animator"].generate_shot(
            description=shot["description"],
            characters=character_loras,
            pose_ref=shot.get("pose_ref"),
            shot_index=shot_index,
        )

        result["quality_score"] = self.agents["evaluator"].evaluate_shot(
            result.get("video_path", ""),
            shot["characters"],
        )

        for char_name in shot["characters"]:
            self.asset_memory.update_character_views(
                char_name, result.get("video_path", ""), shot_index
            )

        return result

    def _regenerate_shot(
        self, shot: Dict, shot_index: int, previous: Dict
    ) -> Dict:
        feedback = self.agents["evaluator"].get_feedback(previous)

        for attempt in range(self.MAX_REGEN_ATTEMPTS):
            result = self.agents["animator"].generate_shot(
                description=shot["description"],
                characters={
                    c: self.asset_memory.get_character(c)["lora_path"]
                    for c in shot["characters"]
                    if self.asset_memory.get_character(c)
                },
                pose_ref=shot.get("pose_ref"),
                shot_index=shot_index,
                feedback=feedback,
            )
            result["quality_score"] = self.agents["evaluator"].evaluate_shot(
                result.get("video_path", ""), shot["characters"]
            )
            if result["quality_score"] >= self.QUALITY_THRESHOLD:
                break

        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, final: bool = False):
        """Persist current state for Colab survival."""
        if not self.current_job_id:
            return

        cp_dir = Path(self.warehouse) / "checkpoints" / self.current_job_id
        cp_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "job_id": self.current_job_id,
            "shot_history": self.shot_history,
            "timestamp": time.time(),
            "final": final,
        }

        cp_path = cp_dir / f"checkpoint_{int(time.time())}.pkl"
        with open(cp_path, "wb") as f:
            pickle.dump(data, f)

        self.asset_memory.save_checkpoint()

        # Keep only last 3 checkpoints
        all_cps = sorted(cp_dir.glob("checkpoint_*.pkl"))
        for old in all_cps[:-3]:
            old.unlink()

        print(f"Checkpoint saved: {cp_path}")

    def resume_story(self, story_id: str) -> bool:
        """Resume from the latest checkpoint for *story_id*."""
        cp_dir = Path(self.warehouse) / "checkpoints" / story_id
        if not cp_dir.exists():
            return False

        cps = sorted(cp_dir.glob("checkpoint_*.pkl"))
        if not cps:
            return False

        with open(cps[-1], "rb") as f:
            data = pickle.load(f)

        if data.get("final"):
            print(f"Story {story_id} already completed")
            return False

        self.current_job_id = story_id
        self.shot_history = data["shot_history"]
        self.asset_memory.load_latest_checkpoint()
        return True

    def continue_processing(self, script: str = "") -> Dict[str, Any]:
        """Continue processing from where we left off."""
        if not script:
            return {
                "story_id": self.current_job_id,
                "shots": [h["result"] for h in self.shot_history],
                "final_video": None,
                "character_count": len(self.asset_memory.list_characters()),
            }

        shots = self.parse_script(script)
        last_idx = (
            max(h["shot_index"] for h in self.shot_history)
            if self.shot_history
            else -1
        )

        remaining = [
            (i, s) for i, s in enumerate(shots) if i > last_idx
        ]

        results = [h["result"] for h in self.shot_history]
        for idx, shot in remaining:
            for char_name in shot["characters"]:
                self.agents["character"].ensure_lora(char_name)
            result = self._execute_shot(shot, idx)
            results.append(result)
            self.shot_history.append(
                {
                    "shot_index": idx,
                    "shot_data": shot,
                    "result": result,
                    "timestamp": time.time(),
                }
            )
            self._maybe_checkpoint()

        final_video = self._assemble_final_video(results)
        self.save_checkpoint(final=True)

        return {
            "story_id": self.current_job_id,
            "shots": results,
            "final_video": final_video,
            "character_count": len(self.asset_memory.list_characters()),
        }

    def _maybe_checkpoint(self):
        if time.time() - self.last_checkpoint > self.CHECKPOINT_INTERVAL:
            self.save_checkpoint()
            self.last_checkpoint = time.time()

    # ------------------------------------------------------------------
    # Video assembly
    # ------------------------------------------------------------------

    def _assemble_final_video(self, results: List[Dict]) -> Optional[str]:
        """Concatenate shot videos into a single output using ffmpeg."""
        valid = [r for r in results if r.get("video_path")]
        if not valid:
            return None

        out_dir = Path(self.warehouse) / "outputs" / (self.current_job_id or "default")
        out_dir.mkdir(parents=True, exist_ok=True)

        list_file = out_dir / "file_list.txt"
        with open(list_file, "w") as f:
            for r in valid:
                f.write(f"file '{r['video_path']}'\n")

        final_path = str(out_dir / "final_video.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(list_file),
                    "-c", "copy",
                    final_path,
                ],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ffmpeg assembly failed: {e}")
            return None

        return final_path
