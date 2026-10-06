"""Application settings (pydantic-settings)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRACTALSCAN_", env_file=".env", extra="ignore")

    app_name: str = "FractalScan API"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = 25
    cnn_weights: str = "densenet121-res224-all"
    warm_load_model: bool = True


settings = Settings()
