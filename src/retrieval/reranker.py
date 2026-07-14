"""Cross-encoder reranking of fused candidates — the precision layer on top of
recall-oriented hybrid retrieval."""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.fusion import FusedResult

logger = logging.getLogger(__name__)


class RerankedResult(BaseModel):
    """A chunk after cross-encoder reranking."""

    chunk_id: str = Field(..., description="ID of the chunk.")
    content: str = Field(..., description="Chunk text.")
    rerank_score: float = Field(..., description="Cross-encoder relevance score (higher is better).")
    pre_rerank_rank: int = Field(..., description="1-indexed rank before reranking (from fusion).")
    post_rerank_rank: int = Field(..., description="1-indexed rank after reranking.")
    dense_similarity: float | None = Field(None, description="Carried through from fusion, for retrieval-confidence scoring.")


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> CrossEncoder:
    """Cross-encoders are expensive to load; cache the single configured model."""
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[FusedResult],
    model_name: str = settings.reranker_model,
    top_k: int = settings.rerank_top_k,
) -> list[RerankedResult]:
    """Score fusion candidates with a cross-encoder and keep the top-k.

    Args:
        query: The original user query.
        candidates: Fused candidates, ordered best-first (pre-rerank order).
        model_name: HuggingFace cross-encoder model name.
        top_k: Number of results to keep after reranking.

    Returns:
        Top-k results ordered by rerank_score, descending.
    """
    if not candidates:
        return []

    model = _get_model(model_name)
    pairs = [(query, c.content) for c in candidates]
    scores = model.predict(pairs)

    pre_rank_by_id = {c.chunk_id: i + 1 for i, c in enumerate(candidates)}
    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

    results = [
        RerankedResult(
            chunk_id=candidate.chunk_id,
            content=candidate.content,
            rerank_score=float(score),
            pre_rerank_rank=pre_rank_by_id[candidate.chunk_id],
            post_rerank_rank=post_rank,
            dense_similarity=candidate.dense_similarity,
        )
        for post_rank, (candidate, score) in enumerate(scored, start=1)
    ]

    positions_changed = sum(1 for r in results if r.pre_rerank_rank != r.post_rerank_rank)
    logger.info(
        "Reranked %d candidates -> top %d; %d/%d positions changed vs. pre-rerank order",
        len(candidates), len(results), positions_changed, len(results),
    )
    return results
