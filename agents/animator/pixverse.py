"""
PixVerse Generator - Fallback video generator using PixVerse free tier.
Communicates via a lightweight Discord bot integration or direct API.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional


class PixVerseGenerator:
    """
    Fallback video generator that uses PixVerse's free tier.
    Can integrate through Discord bot commands or HTTP API.
    """

    API_BASE = "https://api.pixverse.ai/v1"

    def __init__(self, warehouse_path: str):
        self.warehouse = Path(warehouse_path)
        self.output_dir = self.warehouse / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.getenv("PIXVERSE_API_KEY", "")
        self.discord_webhook = os.getenv("PIXVERSE_DISCORD_WEBHOOK", "")

    def generate(
        self,
        description: str,
        shot_index: int = 0,
        style: str = "anime",
        duration: float = 4.0,
        aspect_ratio: str = "1:1",
    ) -> Dict:
        """
        Generate a video using PixVerse.

        Tries HTTP API first; falls back to Discord webhook;
        finally generates a placeholder if both fail.
        """
        output_path = str(
            self.output_dir / f"pixverse_shot_{shot_index:04d}_{int(time.time())}.mp4"
        )

        # Try API
        if self.api_key:
            result = self._generate_via_api(description, style, duration, aspect_ratio)
            if result and result.get("video_url"):
                self._download_video(result["video_url"], output_path)
                return {
                    "video_path": output_path,
                    "shot_index": shot_index,
                    "prompt": description,
                    "source": "pixverse_api",
                    "status": "success",
                }

        # Try Discord webhook
        if self.discord_webhook:
            result = self._generate_via_discord(description, style)
            if result:
                return {
                    "video_path": result,
                    "shot_index": shot_index,
                    "prompt": description,
                    "source": "pixverse_discord",
                    "status": "success",
                }

        # Placeholder
        self._create_placeholder(output_path, description)
        return {
            "video_path": output_path,
            "shot_index": shot_index,
            "prompt": description,
            "source": "placeholder",
            "status": "placeholder",
        }

    # ------------------------------------------------------------------
    # API integration
    # ------------------------------------------------------------------

    def _generate_via_api(
        self, prompt: str, style: str, duration: float, aspect_ratio: str
    ) -> Optional[Dict]:
        """Submit a generation request to the PixVerse API."""
        try:
            import requests

            resp = requests.post(
                f"{self.API_BASE}/generate",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": prompt,
                    "style": style,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                task_id = data.get("task_id")
                if task_id:
                    return self._poll_task(task_id)
            return None
        except Exception as e:
            print(f"PixVerse API error: {e}")
            return None

    def _poll_task(self, task_id: str, max_wait: int = 300) -> Optional[Dict]:
        """Poll PixVerse API until generation completes."""
        import requests

        start = time.time()
        while time.time() - start < max_wait:
            try:
                resp = requests.get(
                    f"{self.API_BASE}/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "completed":
                        return data
                    if data.get("status") == "failed":
                        return None
            except Exception:
                pass
            time.sleep(10)
        return None

    # ------------------------------------------------------------------
    # Discord integration
    # ------------------------------------------------------------------

    def _generate_via_discord(self, prompt: str, style: str) -> Optional[str]:
        """
        Send a generation command via Discord webhook.
        The Discord bot would need to be set up separately.
        """
        try:
            import requests

            payload = {
                "content": f"/generate prompt:{prompt} style:{style}",
                "username": "AnimeLoom",
            }
            resp = requests.post(self.discord_webhook, json=payload, timeout=10)
            if resp.status_code in (200, 204):
                print("PixVerse Discord command sent — poll manually for result")
            return None
        except Exception as e:
            print(f"Discord webhook error: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _download_video(self, url: str, output_path: str):
        """Download a video from URL to local file."""
        try:
            import requests

            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Download failed: {e}")

    def _create_placeholder(self, output_path: str, description: str):
        """Create a placeholder video file for testing."""
        try:
            import cv2
            import numpy as np

            h, w = 512, 512
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 8, (w, h))

            rng = np.random.default_rng(hash(description) % (2**31))
            color = rng.integers(30, 180, size=3)

            for i in range(16):
                frame = np.full((h, w, 3), color, dtype=np.uint8)
                writer.write(frame)

            writer.release()
        except ImportError:
            Path(output_path).write_bytes(b"placeholder")
