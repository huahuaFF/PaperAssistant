"""Centralized application configuration and local storage paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Paper Assistant"
    environment: str = "development"
    api_prefix: str = "/api"

    minimax_api_key: SecretStr | None = None
    minimax_base_url: str = "https://api.minimaxi.com/v1"
    minimax_model: str = "MiniMax-M2.7-highspeed"
    minimax_timeout_seconds: float = 45.0
    minimax_max_retries: int = 1

    dashscope_api_key: SecretStr | None = None
    dashscope_embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024

    sqlite_path: Path = Path("data/app.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    chroma_path: Path = Path("data/chroma")
    paper_storage_path: Path = Path("data/papers")
    parsed_storage_path: Path = Path("data/parsed")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path

    @property
    def database_path(self) -> Path:
        return self.resolve_path(self.sqlite_path)

    @property
    def checkpoint_path(self) -> Path:
        return self.resolve_path(self.checkpoint_db_path)

    @property
    def chroma_directory(self) -> Path:
        return self.resolve_path(self.chroma_path)

    @property
    def paper_directory(self) -> Path:
        return self.resolve_path(self.paper_storage_path)

    @property
    def parsed_directory(self) -> Path:
        return self.resolve_path(self.parsed_storage_path)

    def ensure_local_directories(self) -> None:
        """Create local persistence directories before any service starts."""
        for directory in (
            self.database_path.parent,
            self.checkpoint_path.parent,
            self.chroma_directory,
            self.paper_directory,
            self.parsed_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
