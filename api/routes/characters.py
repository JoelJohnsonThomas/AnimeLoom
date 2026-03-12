"""
Character API routes.
"""

import os
from fastapi import APIRouter, HTTPException

from api.schemas.models import (
    CharacterCreateRequest,
    CharacterListResponse,
    CharacterResponse,
)
from director.memory_bank import AssetMemoryBank

router = APIRouter(prefix="/character", tags=["character"])

_warehouse = os.getenv("AI_CACHE_ROOT", "./warehouse")


def _memory() -> AssetMemoryBank:
    return AssetMemoryBank(_warehouse)


@router.post("/create", response_model=CharacterResponse)
async def create_character(req: CharacterCreateRequest):
    """Upload character sheet and train LoRA."""
    memory = _memory()

    # Validate image paths
    for p in req.image_paths:
        if not os.path.exists(p):
            raise HTTPException(status_code=400, detail=f"Image not found: {p}")

    char_id = memory.create_character(
        name=req.name,
        images=req.image_paths,
        description=req.description,
    )

    char_data = memory.get_character(char_id)
    return CharacterResponse(
        id=char_id,
        name=char_data["name"],
        shot_count=char_data.get("shot_count", 0),
        has_lora=char_data.get("lora_path") is not None,
        created=char_data["created"],
        last_used=char_data["last_used"],
    )


@router.get("/list", response_model=CharacterListResponse)
async def list_characters():
    """List all characters in memory."""
    memory = _memory()
    chars = memory.list_characters()
    return CharacterListResponse(
        characters=[CharacterResponse(**c) for c in chars],
        total=len(chars),
    )


@router.get("/{char_id}", response_model=CharacterResponse)
async def get_character(char_id: str):
    """Get a character by ID."""
    memory = _memory()
    char_data = memory.get_character(char_id)
    if char_data is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return CharacterResponse(
        id=char_data.get("id", char_id),
        name=char_data["name"],
        shot_count=char_data.get("shot_count", 0),
        has_lora=char_data.get("lora_path") is not None,
        created=char_data["created"],
        last_used=char_data["last_used"],
    )


@router.delete("/{char_id}")
async def delete_character(char_id: str):
    """Delete a character."""
    memory = _memory()
    if not memory.delete_character(char_id):
        raise HTTPException(status_code=404, detail="Character not found")
    return {"status": "deleted", "id": char_id}
