"""
Story Decomposer — converts natural language text into anime shot scripts.

Primary:  Gemini 1.5 Flash free tier (1500 req/day)
Fallback: Rule-based sentence segmentation + keyword-based scene detection
"""

import os
import re
from typing import Dict, List, Optional


class StoryDecomposer:
    """
    Converts a natural-language story into the SCENE/CHAR script format
    that DirectorAgent.parse_script() expects.

    Usage::

        decomposer = StoryDecomposer()
        script = decomposer.decompose("Sakura runs through a cherry blossom forest...")
        # Returns formatted script string ready for parse_script()
    """

    # Gemini system prompt for anime storyboarding
    _SYSTEM_PROMPT = """\
You are an anime storyboard director.  Given a story description, decompose it
into individual shots.  For each shot, output EXACTLY this format (no markdown):

SCENE: <visual setting description with lighting and mood>
CHAR: <character name(s), comma-separated>
<detailed anime-style visual prompt for the shot, 4-6 seconds of action>

Rules:
- Each shot should be 4-6 seconds of action
- Include camera direction (close-up, wide shot, pan, etc.)
- Include lighting and mood keywords
- Max 2 characters per shot for quality
- Use anime-specific visual language (sakuga, cel shading, etc.)
- If no character names are mentioned, invent suitable anime names
- Output ONLY the script, no commentary or markdown fences
"""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self._api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, text: str) -> str:
        """
        Convert natural text to a SCENE/CHAR script string.

        Tries Gemini Flash first, falls back to local rule-based decomposition.
        """
        if not text.strip():
            return ""

        # If input already looks like a script, return as-is
        if self._is_already_script(text):
            return text

        # Try Gemini
        if self._api_key:
            result = self._decompose_via_gemini(text)
            if result:
                return result

        # Fallback: rule-based
        return self._decompose_local(text)

    def decompose_to_shots(self, text: str) -> List[Dict]:
        """
        Decompose text directly into shot dicts (same format as
        DirectorAgent.parse_script output).
        """
        script = self.decompose(text)
        return self._parse_script(script)

    # ------------------------------------------------------------------
    # Gemini Flash
    # ------------------------------------------------------------------

    def _decompose_via_gemini(self, text: str) -> Optional[str]:
        """Use Gemini 1.5 Flash to decompose a story."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            response = model.generate_content(
                [
                    {"role": "user", "parts": [
                        f"{self._SYSTEM_PROMPT}\n\nStory:\n{text}"
                    ]},
                ],
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )

            result = response.text.strip()
            # Validate output looks like a script
            if "SCENE:" in result and "CHAR:" in result:
                print("  Story decomposed via Gemini Flash")
                return result
            return None

        except Exception as e:
            print(f"  Gemini decomposition failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Local rule-based fallback
    # ------------------------------------------------------------------

    def _decompose_local(self, text: str) -> str:
        """Rule-based decomposition when no LLM API is available."""
        sentences = self._split_sentences(text)
        characters = self._extract_characters(text)

        # Group sentences into scenes (by location keywords or every 2-3 sentences)
        scenes = self._group_into_scenes(sentences)

        lines = []
        for i, scene_sentences in enumerate(scenes):
            scene_text = " ".join(scene_sentences)
            setting = self._infer_setting(scene_text)
            chars = self._find_characters_in_text(scene_text, characters)

            lines.append(f"SCENE: {setting}")
            if chars:
                lines.append(f"CHAR: {', '.join(chars)}")
            else:
                lines.append("CHAR: Character")

            # Build anime prompt
            prompt = self._build_anime_prompt(scene_text)
            lines.append(prompt)
            lines.append("")  # blank line between scenes

        return "\n".join(lines)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_characters(self, text: str) -> List[str]:
        """Extract likely character names from text using capitalization heuristics."""
        # Find capitalised words that appear multiple times or in dialogue context
        words = re.findall(r'\b([A-Z][a-z]{2,})\b', text)

        # Filter out common non-name words
        stop_words = {
            "The", "This", "That", "There", "They", "Then", "Their",
            "She", "Her", "His", "Him", "But", "And", "For", "With",
            "From", "Into", "Through", "After", "Before", "When", "Where",
            "While", "During", "About", "Each", "Every", "Some", "Many",
            "Scene", "Shot", "Camera",
        }

        from collections import Counter
        word_counts = Counter(words)
        characters = [
            w for w, count in word_counts.items()
            if w not in stop_words and (count >= 2 or len(words) <= 5)
        ]

        return characters[:5]  # cap at 5 characters

    def _group_into_scenes(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences into scenes based on location changes or batching."""
        if not sentences:
            return []

        scene_break_keywords = [
            "meanwhile", "later", "suddenly", "elsewhere",
            "at the", "in the", "inside", "outside", "back at",
            "the next", "that night", "that morning",
        ]

        scenes = [[]]
        for sent in sentences:
            lower = sent.lower()
            is_break = any(kw in lower for kw in scene_break_keywords)

            if is_break and scenes[-1]:
                scenes.append([])

            scenes[-1].append(sent)

            # Also break if scene is getting long (3+ sentences)
            if len(scenes[-1]) >= 3 and sent != sentences[-1]:
                scenes.append([])

        return [s for s in scenes if s]

    def _infer_setting(self, text: str) -> str:
        """Infer a visual setting description from the scene text."""
        lower = text.lower()

        # Try to find location keywords
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

    def _find_characters_in_text(
        self, text: str, all_characters: List[str]
    ) -> List[str]:
        """Find which known characters appear in this scene text."""
        found = [c for c in all_characters if c.lower() in text.lower()]
        return found[:2]  # max 2 per shot

    def _build_anime_prompt(self, text: str) -> str:
        """Enhance a scene description into an anime-style prompt."""
        # Clean up and append anime keywords
        clean = text.strip().rstrip(".")
        return (
            f"{clean}, anime style, high quality animation, "
            f"detailed character art, vibrant colors, smooth motion"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_already_script(self, text: str) -> bool:
        """Check if the input is already in SCENE/CHAR format."""
        upper = text.upper()
        return "SCENE:" in upper and "CHAR:" in upper

    def _parse_script(self, script: str) -> List[Dict]:
        """Minimal script parser (matches DirectorAgent.parse_script)."""
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
