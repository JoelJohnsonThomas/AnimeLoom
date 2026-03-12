"""
Pydantic models for the AnimeLoom API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Character
# ------------------------------------------------------------------

class CharacterCreateRequest(BaseModel):
    name: str = Field(..., description="Character name")
    description: str = Field("", description="Text description of the character")
    image_paths: List[str] = Field(..., description="Paths to character reference images")


class CharacterResponse(BaseModel):
    id: str
    name: str
    shot_count: int = 0
    has_lora: bool = False
    created: str
    last_used: str


class CharacterListResponse(BaseModel):
    characters: List[CharacterResponse]
    total: int


# ------------------------------------------------------------------
# Shot generation
# ------------------------------------------------------------------

class ShotGenerateRequest(BaseModel):
    description: str = Field(..., description="Shot description / prompt")
    characters: List[str] = Field(default_factory=list, description="Character names in this shot")
    pose_ref: Optional[str] = Field(None, description="Path to pose reference video")
    width: int = Field(512, ge=128, le=1024)
    height: int = Field(512, ge=128, le=1024)
    num_frames: int = Field(16, ge=4, le=128)


class ShotResult(BaseModel):
    video_path: Optional[str] = None
    shot_index: int = 0
    prompt: str = ""
    quality_score: float = 0.0
    status: str = "pending"


# ------------------------------------------------------------------
# Sequence generation
# ------------------------------------------------------------------

class SequenceGenerateRequest(BaseModel):
    script: str = Field(..., description="Multi-line script with SCENE/CHAR/POSE directives")
    story_id: Optional[str] = Field(None, description="Story ID for resume support")


class SequenceResult(BaseModel):
    story_id: str
    shots: List[ShotResult]
    final_video: Optional[str] = None
    character_count: int = 0


# ------------------------------------------------------------------
# Job status
# ------------------------------------------------------------------

class JobStatus(BaseModel):
    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    progress: float = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None
