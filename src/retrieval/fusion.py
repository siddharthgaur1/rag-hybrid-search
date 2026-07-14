"""Reciprocal Rank Fusion: combine dense and sparse ranked lists into one."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import settings
from src.retrieval.dense import DenseResult
from src.retrieval.sparse import SparseResult


class FusedResult(BaseModel):
    """A chunk ranked by combined dense + sparse evidence."""

    chunk_id: str = Field(..., description="ID of the chunk.")
    content: str = Field(..., description="Chunk text.")
    fused_score: float = Field(..., description="Weighted RRF score (higher is better).")
    dense_similarity: float | None = Field(None, description="Cosine similarity from dense search, if it was a hit.")
    sparse_score: float | None = Field(None, description="Raw BM25 score from sparse search, if it was a hit.")
    dense_rank: int | None = Field(None, description="1-indexed rank in the dense results, if present.")
    sparse_rank: int | None = Field(None, description="1-indexed rank in the sparse results, if present.")


def reciprocal_rank_fusion(
    dense_results: list[DenseResult],
    sparse_results: list[SparseResult],
    dense_weight: float = settings.dense_weight,
    sparse_weight: float = settings.sparse_weight,
    k: int = settings.rrf_k,
    top_k: int = settings.fusion_top_k,
) -> list[FusedResult]:
    """Combine dense and sparse rankings via weighted Reciprocal Rank Fusion.

    score(d) = dense_weight * 1/(k + dense_rank(d)) + sparse_weight * 1/(k + sparse_rank(d))
    A chunk found by only one retriever still scores via that retriever's term alone.

    Args:
        dense_results: Ranked dense hits (best first).
        sparse_results: Ranked sparse hits (best first).
        dense_weight: Weight applied to the dense RRF term.
        sparse_weight: Weight applied to the sparse RRF term.
        k: RRF's smoothing constant — larger k flattens the influence of rank.
        top_k: Number of fused candidates to return.

    Returns:
        Chunks ranked by fused_score, descending, capped at top_k.
    """
    content_by_id: dict[str, str] = {}
    dense_rank_by_id: dict[str, int] = {}
    dense_sim_by_id: dict[str, float] = {}
    for rank, hit in enumerate(dense_results, start=1):
        dense_rank_by_id[hit.chunk_id] = rank
        dense_sim_by_id[hit.chunk_id] = hit.similarity
        content_by_id[hit.chunk_id] = hit.content

    sparse_rank_by_id: dict[str, int] = {}
    sparse_score_by_id: dict[str, float] = {}
    for rank, hit in enumerate(sparse_results, start=1):
        sparse_rank_by_id[hit.chunk_id] = rank
        sparse_score_by_id[hit.chunk_id] = hit.bm25_score
        content_by_id.setdefault(hit.chunk_id, hit.content)

    all_ids = set(dense_rank_by_id) | set(sparse_rank_by_id)
    fused: list[FusedResult] = []
    for chunk_id in all_ids:
        d_rank = dense_rank_by_id.get(chunk_id)
        s_rank = sparse_rank_by_id.get(chunk_id)
        score = 0.0
        if d_rank is not None:
            score += dense_weight * (1.0 / (k + d_rank))
        if s_rank is not None:
            score += sparse_weight * (1.0 / (k + s_rank))
        fused.append(
            FusedResult(
                chunk_id=chunk_id,
                content=content_by_id[chunk_id],
                fused_score=score,
                dense_similarity=dense_sim_by_id.get(chunk_id),
                sparse_score=sparse_score_by_id.get(chunk_id),
                dense_rank=d_rank,
                sparse_rank=s_rank,
            )
        )

    fused.sort(key=lambda r: r.fused_score, reverse=True)
    return fused[:top_k]
