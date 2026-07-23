"""Central, environment-driven configuration. No model names or thresholds
are hardcoded elsewhere in the codebase — they're all read from here."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All tunable knobs for the RAG pipeline, overridable via .env or env vars."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    # Any OpenAI-compatible endpoint. Empty = OpenAI. A local Ollama
    # (http://localhost:11434/v1) is the fully-free path: it serves BOTH the
    # /v1/embeddings and /v1/chat/completions this pipeline needs, so retrieval
    # and generation both run offline with no paid key. Set the model names to
    # local ones too, e.g. embedding_model=nomic-embed-text, generation_model=llama3.2.
    openai_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    judge_model: str = "gpt-4o"

    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    semantic_similarity_threshold: float = 0.7
    dedup_similarity_threshold: float = 0.95
    default_chunking_strategy: str = "structure_aware"

    chroma_db_path: str = "data/chroma_db"
    bm25_index_path: str = "data/bm25_index"
    docs_path: str = "docs"

    slack_webhook_url: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
