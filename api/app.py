"""
FastAPI application entry point for the AnimeLoom API.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.characters import router as character_router
from api.routes.generation import router as generation_router

app = FastAPI(
    title="AnimeLoom",
    description="Anime Character Consistency Engine API",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(character_router)
app.include_router(generation_router)


@app.get("/")
async def root():
    return {
        "name": "AnimeLoom",
        "version": "0.1.0",
        "warehouse": os.getenv("AI_CACHE_ROOT", "./warehouse"),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# Opt-in static mount for the built Vite frontend.
# Run `cd frontend && npm run build:embedded` then start FastAPI;
# the Web Studio is available at http://localhost:8080/ui/.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="ui")
