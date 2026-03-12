"""
Generation API routes — shot and sequence endpoints.
"""

import os
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas.models import (
    JobStatus,
    SequenceGenerateRequest,
    SequenceResult,
    ShotGenerateRequest,
    ShotResult,
)
from director.agent import DirectorAgent

router = APIRouter(tags=["generation"])

_warehouse = os.getenv("AI_CACHE_ROOT", "./warehouse")

# Simple in-memory job store (swap for Redis in production)
_jobs: Dict[str, JobStatus] = {}


def _director() -> DirectorAgent:
    return DirectorAgent(_warehouse)


# ------------------------------------------------------------------
# Single shot
# ------------------------------------------------------------------

@router.post("/generate/shot", response_model=JobStatus)
async def generate_shot(req: ShotGenerateRequest, bg: BackgroundTasks):
    """Generate a single shot with character consistency."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="pending")

    bg.add_task(_run_shot, job_id, req)
    return _jobs[job_id]


def _run_shot(job_id: str, req: ShotGenerateRequest):
    _jobs[job_id].status = "running"
    try:
        director = _director()
        shot = {
            "description": req.description,
            "characters": req.characters,
            "pose_ref": req.pose_ref,
        }
        result = director._execute_shot(shot, 0)
        _jobs[job_id].status = "completed"
        _jobs[job_id].progress = 1.0
        _jobs[job_id].result = result
    except Exception as e:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)


# ------------------------------------------------------------------
# Sequence
# ------------------------------------------------------------------

@router.post("/generate/sequence", response_model=JobStatus)
async def generate_sequence(req: SequenceGenerateRequest, bg: BackgroundTasks):
    """Generate multiple shots maintaining character consistency."""
    job_id = req.story_id or str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="pending")

    bg.add_task(_run_sequence, job_id, req)
    return _jobs[job_id]


def _run_sequence(job_id: str, req: SequenceGenerateRequest):
    _jobs[job_id].status = "running"
    try:
        director = _director()
        result = director.process_story(req.script, story_id=job_id)
        _jobs[job_id].status = "completed"
        _jobs[job_id].progress = 1.0
        _jobs[job_id].result = result
    except Exception as e:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)


# ------------------------------------------------------------------
# Job status
# ------------------------------------------------------------------

@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Check generation job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]
