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
            print(f"  [ensure_lora] Character '{char_name}' not found in memory bank", flush=True)
            return None

        lora_path = self.memory.get_character_lora_path(char_data["id"])
        if lora_path and lora_path.exists() and lora_path.stat().st_size > 100:
            print(f"  [ensure_lora] Found LoRA for '{char_name}': {lora_path}", flush=True)
            return lora_path

        # Check lora dir by hash-based ID
        lora_dir = Path(self.warehouse) / "lora" / char_data["id"]
        for fname in ["adapter_model.safetensors", "pytorch_lora_weights.safetensors"]:
            candidate = lora_dir / fname
            if candidate.exists() and candidate.stat().st_size > 100:
                print(f"  [ensure_lora] Found LoRA on disk (by ID) for '{char_name}': {candidate}", flush=True)
                self.memory.update_character_lora(char_data["id"], str(candidate))
                return candidate

        # Check lora dir by character name (training script saves here)
        name_key = char_data["name"].lower().replace(" ", "_")
        lora_dir_by_name = Path(self.warehouse) / "lora" / name_key
        for fname in ["adapter_model.safetensors", "pytorch_lora_weights.safetensors"]:
            candidate = lora_dir_by_name / fname
            if candidate.exists() and candidate.stat().st_size > 100:
                print(f"  [ensure_lora] Found LoRA on disk (by name) for '{char_name}': {candidate}", flush=True)
                self.memory.update_character_lora(char_data["id"], str(candidate))
                return candidate

        # Need to train
        print(f"  [ensure_lora] No LoRA found for '{char_name}', training...", flush=True)
        images = char_data.get("multi_views", [])
        if not images:
            print(f"  [ensure_lora] No training images for '{char_name}', skipping", flush=True)
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
        self._s2v = None

    @property
    def wan(self):
        if self._wan is None:
            from agents.animator.wan_wrapper import WanAnimator
            self._wan = WanAnimator(self.warehouse)
        return self._wan

    @property
    def s2v(self):
        if self._s2v is None:
            from agents.animator.s2v_wrapper import WanS2VWrapper
            self._s2v = WanS2VWrapper(self.warehouse)
        return self._s2v

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
        audio_ref: Optional[str] = None,
    ) -> Dict:
        """Generate a shot video, returning metadata dict with *video_path*.

        Shots with an audio reference route to Wan2.2-S2V (audio-driven
        lip-sync); any S2V failure falls through to the standard T2V path.
        """
        if audio_ref and Path(audio_ref).exists():
            result = self._generate_s2v_shot(
                description, characters, shot_index, audio_ref
            )
            if result:
                return result
            print("  S2V failed, falling back to standard generation")
        elif audio_ref:
            print(f"  Audio ref not found ({audio_ref}), using standard generation")

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

    def _generate_s2v_shot(
        self,
        description: str,
        characters: Dict[str, str],
        shot_index: int,
        audio_ref: str,
    ) -> Optional[Dict]:
        """Audio-driven shot via Wan2.2-S2V. Needs a character reference image."""
        import time as _time

        ref_image = self.wan._get_reference_image(characters)
        if ref_image is None:
            print("  S2V: no character reference image available")
            return None

        frames = self.s2v.generate(
            reference_image=ref_image,
            audio_path=audio_ref,
            prompt=description,
        )
        if not frames:
            return None

        out_dir = Path(self.warehouse) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(out_dir / f"shot_{shot_index:04d}_{int(_time.time())}_s2v.mp4")
        self.wan._frames_to_video(frames, video_path, fps=16)

        return {
            "video_path": video_path,
            "shot_index": shot_index,
            "prompt": description,
            "num_frames": len(frames),
            "fps": 16,
            "audio_ref": audio_ref,
            "status": "s2v",
        }

    def generate_long_video(
        self,
        description: str,
        characters: Dict[str, str],
        duration_seconds: int = 120,
        fps: int = 8,
    ) -> Dict:
        """Generate a long AnimateDiff video by stitching 16-frame clips."""
        return self.wan.generate_long_video(
            description=description,
            character_loras=characters,
            duration_seconds=duration_seconds,
            fps=fps,
        )


class _StoryDecomposer:
    """Facade wrapping the story decomposition agent."""

    def __init__(self):
        self._decomposer = None

    @property
    def decomposer(self):
        if self._decomposer is None:
            from agents.story.decomposer import StoryDecomposer
            self._decomposer = StoryDecomposer()
        return self._decomposer

    def decompose(self, text: str) -> str:
        """Convert natural text to SCENE/CHAR script."""
        return self.decomposer.decompose(text)

    def decompose_to_shots(self, text: str):
        return self.decomposer.decompose_to_shots(text)


class _PostProcessor:
    """Facade wrapping the video post-processing pipeline."""

    def __init__(self, warehouse: str):
        self.warehouse = warehouse
        self._upscaler = None
        self._color_grader = None
        self._motion_trimmer = None
        self._face_restorer = None

    @property
    def upscaler(self):
        if self._upscaler is None:
            from agents.postprocess.upscaler import VideoUpscaler
            self._upscaler = VideoUpscaler(self.warehouse)
        return self._upscaler

    @property
    def color_grader(self):
        if self._color_grader is None:
            from agents.postprocess.color_grade import AnimeColorGrader
            self._color_grader = AnimeColorGrader()
        return self._color_grader

    @property
    def motion_trimmer(self):
        if self._motion_trimmer is None:
            from agents.postprocess.motion_trim import MotionTrimmer
            self._motion_trimmer = MotionTrimmer()
        return self._motion_trimmer

    @property
    def face_restorer(self):
        if self._face_restorer is None:
            from agents.postprocess.face_restore import AnimeFaceRestorer
            # Off by default — smears anime line art on Wan2.2 output.
            self._face_restorer = AnimeFaceRestorer(
                enabled=os.getenv("ANIMELOOM_FACE_RESTORE", "0") == "1"
            )
        return self._face_restorer

    def postprocess_video(
        self,
        input_path: str,
        output_path: str,
        target_fps: int = 32,
        spatial_scale: int = 2,
        source_fps: int = 16,
        skip_motion_trim: bool = False,
    ) -> str:
        """
        Run the full post-processing pipeline on a shot video:
          1. Motion trim (cut static trailing frames)
          2. Temporal upscale (RIFE: 8fps → 24fps)
          3. Spatial upscale (Real-ESRGAN: 480p → 720p+)
          4. Face restore (anime face sharpening)
          5. Color grade (anime-style colour enhancement)
        """
        frames = self.upscaler._read_video_frames(input_path)
        if not frames:
            print(f"  PostProcessor: no frames read from {input_path}")
            return input_path

        print(f"  PostProcessor: {len(frames)} frames @ {source_fps}fps")

        # Step 1: Trim static trailing frames (skipped for audio-synced shots)
        if skip_motion_trim:
            print("  Motion trim: skipped (audio-synced shot)")
        else:
            try:
                trimmed = self.motion_trimmer.trim_static_tail(frames)
                if len(trimmed) < len(frames):
                    print(f"  Motion trim: {len(frames)} → {len(trimmed)} frames "
                          f"(cut {len(frames) - len(trimmed)} static frames)")
                frames = trimmed
            except Exception as e:
                print(f"  Motion trim skipped: {e}")

        # Step 2: Temporal upscaling (RIFE)
        if target_fps > source_fps:
            multiplier = round(target_fps / source_fps)
            frames = self.upscaler._temporal_upscale(frames, multiplier)
            print(f"  Temporal upscale: {len(frames)} frames @ ~{target_fps}fps")

        # Step 3: Spatial upscaling (Real-ESRGAN)
        if spatial_scale > 1:
            frames = self.upscaler._spatial_upscale(frames, spatial_scale)
            if frames:
                import numpy as np
                h, w = np.array(frames[0]).shape[:2]
                print(f"  Spatial upscale: {w}×{h}")

        # Step 4: Face restoration (opt-in via ANIMELOOM_FACE_RESTORE=1)
        try:
            if self.face_restorer.enabled:
                frames = self.face_restorer.restore_frames(frames)
                print(f"  Face restore: applied to {len(frames)} frames")
            else:
                print("  Face restore: skipped (disabled)")
        except Exception as e:
            print(f"  Face restore skipped: {e}")

        # Step 5: Color grading
        try:
            frames = self.color_grader.grade_frames(frames)
            print(f"  Color grade: applied")
        except Exception as e:
            print(f"  Color grade skipped: {e}")

        # Write output
        self.upscaler._write_video(frames, output_path, fps=target_fps)
        print(f"  PostProcessor complete: {output_path}")
        return output_path


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

    def evaluate_shot(
        self, video_path: str, characters: List[str],
        pose_ref: Optional[str] = None,
    ) -> float:
        """Aggregate quality score (0-1) for a shot."""
        scores = []
        try:
            scores.append(self.character.evaluate(video_path, characters))
        except Exception:
            scores.append(0.5)  # uncertain — don't assume good
        try:
            scores.append(self.visual.evaluate(video_path))
        except Exception:
            scores.append(0.5)
        # Include motion evaluation (detects static/wobbling output)
        if pose_ref:
            try:
                scores.append(self.motion.evaluate(video_path, pose_ref))
            except Exception:
                scores.append(0.5)
        return sum(scores) / len(scores) if scores else 0.5

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
    QUALITY_THRESHOLD = 0.70  # raised from 0.65 with better generation backends
    MAX_REGEN_ATTEMPTS = 2

    # Quality presets: (target_fps, spatial_scale, source_fps)
    # Wan2.2 generates natively at 16fps; RIFE doubles to 32fps
    # (integral multiplier), mirroring the reference Space's 16fps + RIFE.
    QUALITY_PRESETS = {
        "draft":    (16, 1, 16),  # raw output, no upscaling
        "standard": (32, 2, 16),  # RIFE 2×, 2× spatial (480p → 960p)
        "high":     (32, 2, 16),  # same as standard + stricter quality gate
    }

    def __init__(self, warehouse_path: str = None, quality: str = "standard"):
        self.warehouse = warehouse_path or os.getenv("AI_CACHE_ROOT", "./warehouse")
        self.asset_memory = AssetMemoryBank(self.warehouse)
        self.quality = quality

        self.agents = {
            "character": _CharacterAgent(self.warehouse, self.asset_memory),
            "animator": _AnimatorAgent(self.warehouse, self.asset_memory),
            "evaluator": _QualityEvaluator(self.warehouse),
        }

        self._story_decomposer = _StoryDecomposer()
        self._post_processor = _PostProcessor(self.warehouse)

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

    def process_text_story(
        self, text: str, story_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a natural-language story description (Sora-like interface).

        Decomposes the text into a SCENE/CHAR script, then runs the
        standard pipeline with post-processing.

        Args:
            text: Natural-language story or scene description.
            story_id: Optional ID for resuming.

        Returns:
            Dict with generated video paths and metadata.
        """
        print(f"Decomposing story text ({len(text)} chars)...")
        script = self._story_decomposer.decompose(text)
        print(f"Generated script:\n{script[:500]}{'...' if len(script) > 500 else ''}\n")
        return self.process_story(script, story_id)

    # ------------------------------------------------------------------
    # Script parsing
    # ------------------------------------------------------------------

    def parse_script(self, script: str) -> List[Dict]:
        """Parse a script into individual shot dicts."""
        lines = script.strip().split("\n")
        shots: List[Dict] = []
        current: Dict = {
            "characters": [], "description": "", "pose_ref": None, "audio_ref": None,
        }

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
                    "audio_ref": None,
                }
            elif line.upper().startswith("CHAR:"):
                name = line[5:].strip()
                for n in name.split(","):
                    n = n.strip()
                    if n and n not in current["characters"]:
                        current["characters"].append(n)
            elif line.upper().startswith("POSE:"):
                current["pose_ref"] = line[5:].strip()
            elif line.upper().startswith("AUDIO:"):
                # Speech clip driving lip-sync for this shot (Wan2.2-S2V)
                current["audio_ref"] = line[6:].strip()
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
            # Use the resolved LoRA path (ensure_lora already ran)
            lora_path = self.asset_memory.get_character_lora_path(
                self.asset_memory.get_character(char_name).get("id", "")
            ) if self.asset_memory.get_character(char_name) else None
            if lora_path:
                character_loras[char_name] = str(lora_path)

        result = self.agents["animator"].generate_shot(
            description=shot["description"],
            characters=character_loras,
            pose_ref=shot.get("pose_ref"),
            shot_index=shot_index,
            audio_ref=shot.get("audio_ref"),
        )

        # Post-processing: upscale + color grade (skip for placeholders)
        video_path = result.get("video_path", "")
        is_s2v = result.get("status") == "s2v"
        if video_path and result.get("status") != "placeholder" and self.quality != "draft":
            target_fps, spatial_scale, source_fps = self.QUALITY_PRESETS.get(
                self.quality, self.QUALITY_PRESETS["standard"]
            )
            # Backends report their actual output fps (Wan 16, AnimateDiff 8)
            source_fps = result.get("fps", source_fps)
            pp_path = video_path.replace(".mp4", "_pp.mp4")
            try:
                result["video_path"] = self._post_processor.postprocess_video(
                    video_path, pp_path,
                    target_fps=target_fps,
                    spatial_scale=spatial_scale,
                    source_fps=source_fps,
                    # Trimming frames would desync lips from the audio track
                    skip_motion_trim=is_s2v,
                )
                result["postprocessed"] = True
            except Exception as e:
                print(f"  Post-processing failed: {e}, using raw video")
                result["postprocessed"] = False

        # Re-attach the speech track (post-processing is frames-only)
        if is_s2v and result.get("audio_ref") and result.get("video_path"):
            from agents.postprocess.video_io import mux_audio

            muxed = result["video_path"].replace(".mp4", "_audio.mp4")
            if mux_audio(result["video_path"], result["audio_ref"], muxed):
                result["video_path"] = muxed

        result["quality_score"] = self.agents["evaluator"].evaluate_shot(
            result.get("video_path", ""),
            shot["characters"],
            pose_ref=shot.get("pose_ref"),
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
                    c: str(self.asset_memory.get_character_lora_path(
                        self.asset_memory.get_character(c)["id"]
                    ))
                    for c in shot["characters"]
                    if self.asset_memory.get_character(c)
                    and self.asset_memory.get_character_lora_path(
                        self.asset_memory.get_character(c)["id"]
                    )
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
        """Assemble shot videos with cross-dissolve transitions via ffmpeg."""
        valid = [r for r in results if r.get("video_path")]
        if not valid:
            return None

        out_dir = Path(self.warehouse) / "outputs" / (self.current_job_id or "default")
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = str(out_dir / "final_video.mp4")

        if len(valid) == 1:
            # Single shot — just copy
            import shutil
            shutil.copy2(valid[0]["video_path"], final_path)
            return final_path

        # Try frame-level cross-dissolve assembly
        try:
            from agents.postprocess.transitions import cross_dissolve
            import cv2

            from agents.postprocess.video_io import get_video_fps, write_video_h264

            all_clips_frames = []
            # Clips arrive already retimed by post-processing — read the
            # real fps from the first clip instead of assuming.
            preset_fps = self.QUALITY_PRESETS.get(
                self.quality, self.QUALITY_PRESETS["standard"]
            )[0]
            target_fps = get_video_fps(valid[0]["video_path"], default=preset_fps)

            for r in valid:
                cap = cv2.VideoCapture(r["video_path"])
                clip_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    from PIL import Image as _PILImage
                    clip_frames.append(_PILImage.fromarray(rgb))
                cap.release()
                if clip_frames:
                    all_clips_frames.append(clip_frames)

            if all_clips_frames:
                # Cross-dissolve between consecutive clips
                overlap = min(6, min(len(c) for c in all_clips_frames) // 2)
                merged = all_clips_frames[0]
                for i in range(1, len(all_clips_frames)):
                    merged = cross_dissolve(merged, all_clips_frames[i], overlap)

                # Final delivery encode — crf 16 H.264
                write_video_h264(merged, final_path, fps=target_fps, crf=16)
                print(f"Final video assembled with transitions: {final_path}")
                return final_path

        except Exception as e:
            print(f"  Transition assembly failed ({e}), falling back to ffmpeg concat")

        # Fallback: plain ffmpeg concat
        list_file = out_dir / "file_list.txt"
        with open(list_file, "w") as f:
            for r in valid:
                f.write(f"file '{r['video_path']}'\n")

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
