import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
env_file_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_file_path)


class QdrantConfig(BaseSettings):
    url: str = Field(default="http://localhost:6333")

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_", case_sensitive=False, extra="ignore"
    )


class FileLoaderConfig(BaseSettings):
    chunk_overlap: int = Field(default=64)
    max_chunk_length: int = Field(default=500)
    merge_peers: bool = Field(default=True)

    images_scale: float = Field(default=2.0)
    generate_page_images: bool = Field(default=True)
    generate_picture_images: bool = Field(default=True)
    do_ocr: bool = Field(default=True)
    do_table_structure: bool = Field(default=True)
    accelerator_device: str = Field(default="AUTO")
    num_threads: int = Field(default=4)

    model_config = SettingsConfigDict(
        env_prefix="FILE_LOADER_", case_sensitive=False, extra="ignore"
    )


class RAGConfig(BaseSettings):
    llm_model: str = Field(default="meta-llama/llama-3.1-8b-instruct")
    embedding_model: str = Field(
        default="sentence-transformers/distiluse-base-multilingual-cased-v2"
    )
    embedding_max_length: int = Field(default=512)
    collection_name: str = Field(default="uploaded_documents")
    openrouter_key: str = Field(default="")

    model_config = SettingsConfigDict(
        env_prefix="RAG_", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    qdrant: QdrantConfig = Field(default=QdrantConfig())
    file_loader: FileLoaderConfig = Field(default=FileLoaderConfig())
    rag: RAGConfig = Field(default=RAGConfig())

    model_config = SettingsConfigDict(
        env_file=env_file_path,
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
