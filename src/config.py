import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_file_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_file_path)


class QdrantConfig(BaseSettings):
    url: str = Field(default="http://localhost:6333")
    vector_size: int = Field(default=768)

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    qdrant: QdrantConfig = Field(default=QdrantConfig())

    model_config = SettingsConfigDict(
        env_file=env_file_path, case_sensitive=False, env_file_encoding="utf-8"
    )


settings = Settings()
