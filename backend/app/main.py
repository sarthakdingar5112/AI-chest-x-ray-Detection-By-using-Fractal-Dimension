"""FastAPI entrypoint for FractalScan."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from app.core.console import ensure_utf8_output

# Must run before any TorchXRayVision weight download (its progress bar is UTF-8).
ensure_utf8_output()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_analyze, routes_health
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fractalscan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-load the CNN once at startup so the first request isn't slow.
    if settings.warm_load_model:
        from app.services.cnn_model import model
        if model.available:
            log.info("CNN model loaded: %s (%d pathologies)", model.name, len(model.pathologies))
        else:
            log.warning("CNN model unavailable — serving fractal-only analysis.")
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_analyze.router)


@app.get("/")
def root():
    return {"name": settings.app_name, "docs": "/docs", "health": "/api/health"}
