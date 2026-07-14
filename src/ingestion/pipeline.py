"""Glue: load documents, chunk with a given strategy, embed and store.
Used by the API's /v1/ingest endpoint and by the chunking-strategy comparison runner."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import AsyncOpenAI

from src.config import settings
from src.ingestion.chunker import ChunkingStrategy, chunk_document, local_sentence_similarity_fn
from src.ingestion.embedder import IngestionStats, embed_and_store
from src.ingestion.loader import load_documents

logger = logging.getLogger(__name__)


async def ingest_corpus(
    client: AsyncOpenAI,
    strategy: ChunkingStrategy,
    docs_path: str | Path = settings.docs_path,
    chroma_db_path: str | Path = settings.chroma_db_path,
    bm25_index_path: str | Path = settings.bm25_index_path,
) -> IngestionStats:
    """Load every document under docs_path, chunk it with the given strategy, and index it.

    Note: each call creates/reuses the same ChromaDB collection and BM25 index
    at the given paths — re-ingesting with a different strategy will mix
    chunks from multiple strategies in one index unless you point at
    separate chroma_db_path/bm25_index_path per strategy (which is what
    chunking_compare.py does).
    """
    documents = load_documents(docs_path)
    similarity_fn = local_sentence_similarity_fn() if strategy == "semantic" else None

    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy, similarity_fn))

    logger.info("Chunked %d documents into %d chunks using strategy=%s", len(documents), len(chunks), strategy)
    return await embed_and_store(chunks, client, chroma_db_path, bm25_index_path)
