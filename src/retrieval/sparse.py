"""Sparse BM25 keyword retrieval — catches exact function names, error codes,
config keys, and version numbers that dense embeddings tend to blur."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)


class SparseResult(BaseModel):
    """A single BM25 hit."""

    chunk_id: str = Field(..., description="ID of the matched chunk.")
    content: str = Field(..., description="Chunk text (from the BM25 corpus).")
    bm25_score: float = Field(..., description="Raw BM25 score (unbounded, higher is better).")


def bm25_index_exists(bm25_index_path: str | Path) -> bool:
    """Whether a BM25 index has been persisted at this path yet."""
    return (Path(bm25_index_path) / "bm25_index.pkl").exists()


def _load_index(bm25_index_path: str | Path) -> dict:
    index_file = Path(bm25_index_path) / "bm25_index.pkl"
    if not index_file.exists():
        raise FileNotFoundError(f"BM25 index not found at {index_file}. Run ingestion first.")
    with open(index_file, "rb") as f:
        return pickle.load(f)


def sparse_search(
    query: str,
    bm25_index_path: str | Path = settings.bm25_index_path,
    top_k: int = 10,
) -> list[SparseResult]:
    """Search the persisted BM25 index for the top-k chunks matching query terms.

    Raises:
        FileNotFoundError: if no BM25 index has been built yet.
        ValueError: if query is empty.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    data = _load_index(bm25_index_path)
    bm25_index = data["index"]
    chunk_ids = data["chunk_ids"]
    chunk_contents = data["chunk_contents"]

    if bm25_index is None or not chunk_ids:
        logger.warning("Sparse search against an empty BM25 index")
        return []

    query_tokens = query.lower().split()
    scores = bm25_index.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        SparseResult(chunk_id=chunk_ids[i], content=chunk_contents[i], bm25_score=float(scores[i]))
        for i in ranked
        if scores[i] > 0
    ]
