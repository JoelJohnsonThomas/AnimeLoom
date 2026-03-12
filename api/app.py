"""
FastAPI application entry point for the AnimeLoom API.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
