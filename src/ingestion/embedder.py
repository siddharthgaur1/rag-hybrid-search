"""Embed chunks with text-embedding-3-small, store in ChromaDB, and build a
parallel BM25 index over the same chunk corpus."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import chromadb
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
# USD per 1K tokens for text-embedding-3-small.
EMBEDDING_PRICE_PER_1K_TOKENS = 0.00002


class IngestionStats(BaseModel):
    """Summary of one embed-and-store run."""

    total_chunks: int = Field(..., description="Chunks considered for ingestion.")
    chunks_stored: int = Field(..., description="Chunks actually written to the index.")
    duplicates_skipped: int = Field(..., description="Chunks skipped due to near-duplicate cosine similarity.")
    embedding_cost_usd: float = Field(..., description="Estimated cost of the embedding API calls.")


def _tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


async def _embed_batch(client: AsyncOpenAI, texts: list[str], model: str) -> tuple[list[list[float]], int]:
    response = await client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    total_tokens = response.usage.total_tokens if response.usage else 0
    return vectors, total_tokens


async def embed_and_store(
    chunks: list[Chunk],
    client: AsyncOpenAI,
    chroma_db_path: str | Path = settings.chroma_db_path,
    bm25_index_path: str | Path = settings.bm25_index_path,
    model: str = settings.embedding_model,
    batch_size: int = BATCH_SIZE,
    dedup_threshold: float = settings.dedup_similarity_threshold,
) -> IngestionStats:
    """Embed chunks, deduplicate near-identical ones, and persist to ChromaDB + BM25.

    Deduplication is a linear scan against already-accepted embeddings in this
    run (O(n^2) in chunk count). Fine at corpus sizes in the hundreds of
    chunks; swap in an ANN index (e.g. Chroma's own similarity query) if the
    corpus grows into the tens of thousands.

    Args:
        chunks: Chunks to embed and store.
        client: An initialized AsyncOpenAI client.
        chroma_db_path: Directory for the persistent Chroma collection.
        bm25_index_path: Directory to write the pickled BM25 index into.
        model: Embedding model name.
        batch_size: Chunks per embedding API call.
        dedup_threshold: Cosine similarity above which a chunk is treated as
            a duplicate of one already stored and skipped.

    Returns:
        IngestionStats with counts and estimated cost.
    """
    if not chunks:
        return IngestionStats(total_chunks=0, chunks_stored=0, duplicates_skipped=0, embedding_cost_usd=0.0)

    accepted_chunks: list[Chunk] = []
    accepted_embeddings: list[np.ndarray] = []
    total_tokens = 0
    duplicates_skipped = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        vectors, batch_tokens = await _embed_batch(client, [c.content for c in batch], model)
        total_tokens += batch_tokens

        for chunk, vector in zip(batch, vectors):
            vec = np.array(vector, dtype=np.float32)
            is_duplicate = any(_cosine_similarity(vec, existing) > dedup_threshold for existing in accepted_embeddings)
            if is_duplicate:
                duplicates_skipped += 1
                continue
            accepted_chunks.append(chunk)
            accepted_embeddings.append(vec)

    chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = chroma_client.get_or_create_collection("chunks")
    if accepted_chunks:
        collection.upsert(
            ids=[c.chunk_id for c in accepted_chunks],
            embeddings=[e.tolist() for e in accepted_embeddings],
            documents=[c.content for c in accepted_chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "strategy_used": c.strategy_used,
                    "token_count": c.token_count,
                    "source_file": c.metadata.source_file,
                    "file_path": c.metadata.file_path,
                }
                for c in accepted_chunks
            ],
        )

    bm25_path = Path(bm25_index_path)
    bm25_path.mkdir(parents=True, exist_ok=True)
    corpus_tokens = [_tokenize_for_bm25(c.content) for c in accepted_chunks]
    bm25_index = BM25Okapi(corpus_tokens) if corpus_tokens else None
    with open(bm25_path / "bm25_index.pkl", "wb") as f:
        pickle.dump(
            {
                "index": bm25_index,
                "chunk_ids": [c.chunk_id for c in accepted_chunks],
                "corpus_tokens": corpus_tokens,
                "chunk_contents": [c.content for c in accepted_chunks],
            },
            f,
        )

    cost = (total_tokens / 1000) * EMBEDDING_PRICE_PER_1K_TOKENS
    logger.info(
        "Ingested %d/%d chunks (%d duplicates skipped), est. cost $%.4f",
        len(accepted_chunks), len(chunks), duplicates_skipped, cost,
    )
    return IngestionStats(
        total_chunks=len(chunks),
        chunks_stored=len(accepted_chunks),
        duplicates_skipped=duplicates_skipped,
        embedding_cost_usd=cost,
    )
