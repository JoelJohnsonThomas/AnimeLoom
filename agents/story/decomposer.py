"""
Story Decomposer — converts natural language text into anime shot scripts.

Two-stage LLM pipeline:
  Stage 1 — Gemini 2.5 Flash: story outline, world-building, shot structure
  Stage 2 — Claude Sonnet:    cinematic refinement, camera directions, motion cues

Single-stage fallbacks:
  Gemini only (if no Anthropic key)
  Rule-based (if no LLM keys at all)
"""

import os
import re
from typing import Dict, List, Optional


class StoryDecomposer:
    """
    Converts a natural-language story into the SCENE/CHAR script format
    that DirectorAgent.parse_script() expects.

    Two-stage pipeline (Gemini → Claude) produces the best results for anime.
    Gemini handles story structure and world consistency; Claude refines each
    shot into cinematic, sakuga-style descriptions with camera directions.

    Usage::

        decomposer = StoryDecomposer(
            gemini_api_key="...",
            anthropic_api_key="...",
            character_name="sakura_haruno",
        )
        shots = decomposer.decompose_to_shots("Sakura walks through...")
    """

    # ------------------------------------------------------------------
    # Stage 1: Gemini prompt — story structure + shot list
    # ------------------------------------------------------------------
    _GEMINI_SYSTEM = """\
You are an anime series director and story planner.

Given a story description, produce a STRUCTURED SHOT LIST.
For each shot output EXACTLY this format (no markdown, no extra text):

SCENE: <one consistent background/environment for ALL shots — do not change between shots>
CHAR: <character name>
ACTION: <what the character physically does — MUST include clear body movement like walking, reaching, turning, leaning. 2-3 sentences.>
CAMERA: <camera movement and framing: wide/medium/close-up + dolly/pan/tracking/static>
MOOD: <emotional tone + lighting keyword>

Rules:
- ALL shots share the SAME scene/environment (single continuous location)
- Each shot MUST have clear visible character movement — NOT static poses
- Good actions: walking forward, slight head turns, reaching out, looking down, leaning on railing
- Avoid: full body spins, jumping, running, extreme head rotations
- 3-5 shots total
- Output ONLY the shot list
"""

    # ------------------------------------------------------------------
    # Stage 2: Claude prompt — cinematic refinement per shot
    # ------------------------------------------------------------------
    _CLAUDE_REFINE_SYSTEM = """\
You are a sakuga animator and cinematographer for a top anime studio (ufotable / Makoto Shinkai level).

You will receive a rough shot description and refine it into a detailed, cinematic anime prompt.

CRITICAL RULES:
1. The character MUST have visible body movement — walking, arm gestures, head tilts, leaning, reaching. NEVER describe a static pose.
2. Describe both CHARACTER motion AND environmental motion (hair flowing, petals, fabric, light shifts)
3. Use anime visual language: volumetric light, depth of field, film grain, sakuga, chromatic aberration
4. Keep the same background/scene as other shots
5. Avoid: full body rotations, jumping, sudden teleportation, extreme head turns
6. Good: "walks slowly forward, arms swaying gently, hair caught by wind" / "turns head left and gazes downward, one hand reaching toward falling petals"

Output ONLY the refined prompt (2-4 sentences). No labels, no markdown.
"""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        character_name: Optional[str] = None,
    ):
        self._gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self._character_name = character_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, text: str) -> str:
        """Convert natural text to a SCENE/CHAR script string."""
        if not text.strip():
            return ""
        if self._is_already_script(text):
            return text

        # Two-stage: Gemini outline → Claude refine
        if self._gemini_key and self._anthropic_key:
            result = self._decompose_two_stage(text)
            if result:
                return result

        # Single-stage Gemini
        if self._gemini_key:
            result = self._decompose_via_gemini(text)
            if result:
                return result

        # Rule-based fallback
        return self._decompose_local(text)

    def decompose_to_shots(self, text: str) -> List[Dict]:
        """Decompose text directly into shot dicts."""
        script = self.decompose(text)
        return self._parse_script(script)

    # ------------------------------------------------------------------
    # Two-stage pipeline: Gemini → Claude
    # ------------------------------------------------------------------

    def _decompose_two_stage(self, text: str) -> Optional[str]:
        """Stage 1: Gemini produces shot list. Stage 2: Claude refines each shot."""
        print("  [Two-stage] Stage 1: Gemini story planning...")
        gemini_script = self._decompose_via_gemini(text)
        if not gemini_script:
            return None

        shots = self._parse_script(gemini_script)
        if not shots:
            return None

        print(f"  [Two-stage] Gemini produced {len(shots)} shots. Stage 2: Claude refinement...")

        # Refine each shot with Claude
        refined_lines = []
        for i, shot in enumerate(shots):
            raw_desc = shot.get("description", "")
            action = shot.get("action", raw_desc)
            camera = shot.get("camera", "")
            mood = shot.get("mood", "")
            chars = shot.get("characters", [self._character_name or "Character"])

            refined = self._refine_shot_via_claude(
                shot_index=i + 1,
                total_shots=len(shots),
                action=action,
                camera=camera,
                mood=mood,
                character=chars[0] if chars else (self._character_name or "Character"),
            )

            refined_lines.append(f"SCENE: {raw_desc}")
            refined_lines.append(f"CHAR: {', '.join(chars)}")
            if camera:
                refined_lines.append(f"CAMERA: {camera}")
            refined_lines.append(refined)
            refined_lines.append("")

        print(f"  [Two-stage] Done — {len(shots)} shots refined by Claude.")
        return "\n".join(refined_lines)

    def _refine_shot_via_claude(
        self,
        shot_index: int,
        total_shots: int,
        action: str,
        camera: str,
        mood: str,
        character: str,
    ) -> str:
        """Use Claude Sonnet to refine a single shot into cinematic anime prose."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._anthropic_key)

            user_msg = (
                f"Shot {shot_index} of {total_shots}.\n"
                f"Character: {character}\n"
                f"Action: {action}\n"
                f"Camera: {camera}\n"
                f"Mood: {mood}\n\n"
                f"Refine this into a cinematic anime shot description."
            )

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=self._CLAUDE_REFINE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            refined = response.content[0].text.strip()
            print(f"    Shot {shot_index}: refined ({len(refined)} chars)")
            return refined

        except ImportError:
            print("    Claude SDK not installed — using raw action text")
            return action
        except Exception as e:
            print(f"    Claude refinement failed for shot {shot_index}: {e}")
            return action

    # ------------------------------------------------------------------
    # Stage 1: Gemini Flash — story structure
    # ------------------------------------------------------------------

    def _decompose_via_gemini(self, text: str) -> Optional[str]:
        result = self._try_genai_new(text)
        if result:
            return result
        result = self._try_genai_legacy(text)
        return result

    def _try_genai_new(self, text: str) -> Optional[str]:
        try:
            from google import genai

            client = genai.Client(api_key=self._gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{self._GEMINI_SYSTEM}\n\nStory:\n{text}",
                config={"temperature": 0.6, "max_output_tokens": 2048},
            )
            result = response.text.strip()
            if "SCENE:" in result and "CHAR:" in result:
                print("  Gemini 2.5 Flash story planning done (new SDK)")
                return result
            return None
        except ImportError:
            return None
        except Exception as e:
            print(f"  Gemini (new SDK) failed: {e}")
            return None

    def _try_genai_legacy(self, text: str) -> Optional[str]:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._gemini_key)
            for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        [{"role": "user", "parts": [f"{self._GEMINI_SYSTEM}\n\nStory:\n{text}"]}],
                        generation_config=genai.GenerationConfig(temperature=0.6, max_output_tokens=2048),
                    )
                    result = response.text.strip()
                    if "SCENE:" in result and "CHAR:" in result:
                        print(f"  Gemini story planning done ({model_name}, legacy SDK)")
                        return result
                except Exception:
                    continue
            return None
        except ImportError:
            return None
        except Exception as e:
            print(f"  Gemini (legacy SDK) failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _decompose_local(self, text: str) -> str:
        """Rule-based decomposition when no LLM API is available."""
        sentences = self._split_sentences(text)
        characters = self._extract_characters(text)

        if self._character_name and self._character_name not in characters:
            characters.insert(0, self._character_name)

        lines = []
        for i, sentence in enumerate(sentences):
            setting = self._infer_setting(sentence)
            chars = self._find_characters_in_text(sentence, characters)
            if not chars:
                chars = [self._character_name] if self._character_name else ["Character"]

            lines.append(f"SCENE: {setting}")
            lines.append(f"CHAR: {', '.join(chars)}")
            lines.append(self._build_anime_prompt(sentence))
            lines.append("")

        return "\n".join(lines)

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_characters(self, text: str) -> List[str]:
        words = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
        stop_words = {
            "The", "This", "That", "There", "They", "Then", "Their",
            "She", "Her", "His", "Him", "But", "And", "For", "With",
            "From", "Into", "Through", "After", "Before", "When", "Where",
            "While", "During", "About", "Each", "Every", "Some", "Many",
            "Scene", "Shot", "Camera", "Petals", "Wind", "River", "Bridge",
            "Forest", "Cherry", "Blossom", "Sunset", "Night", "Morning",
            "Light", "Dark", "Rain", "Snow", "Fire", "Water", "Sky",
        }
        from collections import Counter
        word_counts = Counter(words)
        characters = [
            w for w, count in word_counts.items()
            if w not in stop_words and (count >= 2 or len(words) <= 5)
        ]
        return characters[:5]

    def _infer_setting(self, text: str) -> str:
        lower = text.lower()
        location_patterns = [
            (r'(?:at|in|inside|outside)\s+(?:the|a)\s+(\w+(?:\s+\w+)?)', None),
            (r'(?:forest|garden|school|temple|city|street|room|house|beach|mountain|castle|village)', None),
        ]
        for pattern, _ in location_patterns:
            match = re.search(pattern, lower)
            if match:
                location = match.group(0)
                return f"{location.title()}, anime environment, atmospheric lighting"
        return "Dramatic anime scene, detailed background, atmospheric lighting"

    def _find_characters_in_text(self, text: str, all_characters: List[str]) -> List[str]:
        found = [c for c in all_characters if c.lower() in text.lower()]
        return found[:2]

    def _build_anime_prompt(self, text: str) -> str:
        clean = text.strip().rstrip(".")
        return (
            f"{clean}, anime style, high quality animation, "
            f"detailed character art, vibrant colors, smooth motion"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_already_script(self, text: str) -> bool:
        upper = text.upper()
        return "SCENE:" in upper and "CHAR:" in upper

    def _parse_script(self, script: str) -> List[Dict]:
        """Parse SCENE/CHAR/ACTION/CAMERA/MOOD script into shot dicts."""
        lines = script.strip().split("\n")
        shots: List[Dict] = []
        current: Dict = {
            "characters": [], "description": "",
            "action": "", "camera": "", "mood": "", "pose_ref": None,
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("SCENE:") or upper.startswith("SHOT:"):
                if current["description"]:
                    shots.append(current)
                tag_len = 6  # len("SCENE:") == len("SHOT:") + 1, handle both
                tag_len = 6 if upper.startswith("SCENE:") else 5
                current = {
                    "characters": [], "description": line[tag_len:].strip(),
                    "action": "", "camera": "", "mood": "", "pose_ref": None,
                }
            elif upper.startswith("CHAR:"):
                for n in line[5:].strip().split(","):
                    n = n.strip()
                    if n and n not in current["characters"]:
                        current["characters"].append(n)
            elif upper.startswith("ACTION:"):
                current["action"] = line[7:].strip()
            elif upper.startswith("CAMERA:"):
                current["camera"] = line[7:].strip()
            elif upper.startswith("MOOD:"):
                current["mood"] = line[5:].strip()
            elif upper.startswith("POSE:"):
                current["pose_ref"] = line[5:].strip()
            else:
                # Free-form prose line → append to description
                sep = " " if current["description"] else ""
                current["description"] += sep + line

        if current["description"]:
            shots.append(current)

        return shots
