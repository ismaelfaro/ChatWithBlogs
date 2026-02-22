"""
ChatMyIdeas configuration — driven by environment variables or .env file.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Vector store ──────────────────────────────────────────────────────────
    vector_store: Literal["chroma", "redis"] = "chroma"
    chroma_db_path: str = "./data/chroma"
    redis_url: str = "redis://localhost:6379"
    redis_index_name: str = "chatmyideas_idx"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    # Use any model available in your Ollama installation.
    # Small options: lfm2.5-thinking:latest, llama3.2, phi3:mini, gemma2:2b
    ollama_model: str = "lfm2.5-thinking:latest"
    ollama_timeout: int = 120

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Local sentence-transformers model — no API key required.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── RAG ───────────────────────────────────────────────────────────────────
    top_k_chunks: int = 5

    # ── App server ────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False


settings = Settings()
