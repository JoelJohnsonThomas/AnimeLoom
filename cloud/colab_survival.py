"""
Colab Survival Mode - Keeps Google Colab session alive and manages
checkpoint persistence across disconnections.
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional


class ColabSurvival:
    """
    Keeps Colab session alive and manages checkpoints for the Director.

    Features:
    - Keepalive prints every 4 minutes to prevent timeout
    - Automatic checkpointing every 5 minutes
    - Google Drive mounting for persistence
    - Resume from latest checkpoint on reconnect
    """

    def __init__(self, director=None):
        self.director = director
        self.keepalive_interval = 240   # 4 minutes
        self.checkpoint_interval = 300  # 5 minutes
        self.last_keepalive = time.time()
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the survival daemon thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("Colab survival mode activated")

    def stop(self):
        """Stop the survival thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("Colab survival mode stopped")

    def _run(self):
        """Main survival loop."""
        while self.running:
            now = time.time()

            # Keepalive
            if now - self.last_keepalive > self.keepalive_interval:
                print(f"[keepalive] {time.strftime('%H:%M:%S')}")
                self.last_keepalive = now

            # Checkpoint
            if (
                self.director
                and now - self.director.last_checkpoint > self.checkpoint_interval
            ):
                try:
                    self.director.save_checkpoint()
                    self.director.last_checkpoint = now
                except Exception as e:
                    print(f"Checkpoint error: {e}")

            time.sleep(60)

    # ------------------------------------------------------------------
    # Google Drive
    # ------------------------------------------------------------------

    def mount_google_drive(self, mount_path: str = "/content/drive") -> bool:
        """Mount Google Drive for persistent storage."""
        try:
            from google.colab import drive  # type: ignore
            drive.mount(mount_path)
            print(f"Google Drive mounted at {mount_path}")
            return True
        except Exception:
            print("Google Drive mount not available (not in Colab?)")
            return False

    def setup_warehouse_on_drive(
        self, drive_path: str = "/content/drive/MyDrive/anime_warehouse"
    ) -> str:
        """
        Set up the warehouse on Google Drive so assets survive
        across Colab sessions.
        """
        os.makedirs(drive_path, exist_ok=True)

        subdirs = ["models", "lora", "datasets", "outputs", "checkpoints"]
        for d in subdirs:
            os.makedirs(os.path.join(drive_path, d), exist_ok=True)

        os.environ["AI_CACHE_ROOT"] = drive_path
        print(f"Warehouse set to {drive_path}")
        return drive_path

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, story_id: str = None) -> bool:
        """
        Resume from the latest checkpoint.

        Args:
            story_id: Specific story to resume. If None, resumes the
                      most recent story.
        """
        if self.director is None:
            print("No director attached")
            return False

        if story_id:
            return self.director.resume_story(story_id)

        # Find the most recent story checkpoint
        warehouse = Path(os.getenv("AI_CACHE_ROOT", "./warehouse"))
        cp_dir = warehouse / "checkpoints"
        if not cp_dir.exists():
            return False

        try:
            story_dirs = sorted(
                [d for d in cp_dir.iterdir() if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            return False

        if not story_dirs:
            return False

        latest_story = story_dirs[0].name
        return self.director.resume_story(latest_story)
