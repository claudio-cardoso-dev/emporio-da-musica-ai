"""Application settings and environment configuration using Pydantic Settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables or .env file."""

    groq_api_key: str = ""
    primary_model: str = "qwen/qwen3-32b"
    fallback_model_1: str = "llama-3.3-70b-versatile"
    fallback_model_2: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1

    openai_api_key: str = ""
    gemini_api_key: str = ""

    app_env: str = "development"
    port: int = 8000
    db_path: str = "src/db/emporio_musica.db"
    data_dir: str = "data"
    confidence_threshold: float = 0.65

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_db_path(self) -> Path:
        """Return absolute path to the SQLite database file."""
        path = Path(self.db_path)
        if not path.is_absolute():
            return BASE_DIR / path
        return path

    @property
    def resolved_data_dir(self) -> Path:
        """Return absolute path to the data directory containing CSVs and PDF."""
        path = Path(self.data_dir)
        if not path.is_absolute():
            return BASE_DIR / path
        return path


settings = Settings()
