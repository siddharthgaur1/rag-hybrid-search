"""Shared end-user query pipeline: retrieve, rerank, generate, verify, score
confidence. Used by both the FastAPI service and the Streamlit dashboard so
the two surfaces can't drift out of sync with each other.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.generation.citation import verify_citations
from src.generation.confidence import ConfidenceScores, score_confidence
from src.generation.generator import generate_answer
from src.ingestion.chunker import count_tokens
from src.pricing import estimate_chat_cost, estimate_embedding_cost
from src.retrieval.dense import dense_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import RerankedResult, rerank
from src.retrieval.sparse import sparse_search

logger = logging.getLogger(__name__)


class AskResult(BaseModel):
    """Full result of answering a single user question."""

    answer: str
    verified_citations: list[int]
    unsupported_citations: list[int]
    confidence: ConfidenceScores
    chunks_used: list[RerankedResult]
    latency_ms: float
    cost_estimate_usd: float = Field(..., description="Token-count-based cost estimate across embedding, generation, and judge calls.")


def _estimate_cost(
    question: str,
    context_chunks: list[RerankedResult],
    answer: str,
    num_citation_judge_calls: int,
    generation_model: str,
    judge_model: str,
    embedding_model: str,
) -> float:
    """Approximate cost from token counts of the pieces we already have in hand,
    rather than threading raw API usage objects through every pipeline stage.
    Close enough for relative cost tracking; not exact billing reconciliation.
    """
    context_text = "\n\n".join(c.content for c in context_chunks)
    generation_input_tokens = count_tokens(context_text) + count_tokens(question) + 150  # system prompt overhead
    generation_output_tokens = count_tokens(answer)
    cost = estimate_chat_cost(generation_model, generation_input_tokens, generation_output_tokens)
    cost += estimate_embedding_cost(embedding_model, count_tokens(question))
    # Each citation judge call and the completeness judge call are short (~150 in / ~5 out tokens).
    cost += (num_citation_judge_calls + 1) * estimate_chat_cost(judge_model, 150, 5)
    return cost


async def ask(
    question: str,
    client: AsyncOpenAI,
    chroma_db_path: str | Path = settings.chroma_db_path,
    bm25_index_path: str | Path = settings.bm25_index_path,
    top_k: int = 5,
    use_sparse: bool = True,
    generation_model: str = settings.generation_model,
    judge_model: str = settings.judge_model,
) -> AskResult:
    """Answer a question via hybrid (or dense-only) retrieval + grounded generation.

    Args:
        question: The user's question.
        client: An initialized AsyncOpenAI client.
        chroma_db_path: Path to the persisted Chroma collection.
        bm25_index_path: Path to the persisted BM25 index.
        top_k: Number of chunks to keep after reranking.
        use_sparse: If False, skips BM25 and fusion entirely — dense-only
            retrieval, for the dashboard's hybrid-vs-dense-only comparison.
        generation_model: Model for answer generation.
        judge_model: Model for citation and completeness judging.

    Raises:
        ValueError: if question is empty.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    start = time.perf_counter()

    dense_hits = await dense_search(question, client, chroma_db_path, top_k=10)
    if use_sparse:
        sparse_hits = sparse_search(question, bm25_index_path, top_k=10)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=20)
    else:
        # Dense-only: treat the dense ranking as already "fused" with sparse_weight=0.
        fused = reciprocal_rank_fusion(dense_hits, [], dense_weight=1.0, sparse_weight=0.0, top_k=20)

    reranked = rerank(question, fused, top_k=top_k)
    generated = await generate_answer(question, reranked, client, model=generation_model)
    citation_result = await verify_citations(generated.answer, reranked, client, judge_model)
    confidence = await score_confidence(question, generated.answer, reranked, citation_result, client, judge_model)

    latency_ms = (time.perf_counter() - start) * 1000
    cost = _estimate_cost(
        question, reranked, generated.answer, len(citation_result.verification_scores),
        generation_model, judge_model, settings.embedding_model,
    )

    return AskResult(
        answer=generated.answer,
        verified_citations=citation_result.verified_citations,
        unsupported_citations=citation_result.unsupported_citations,
        confidence=confidence,
        chunks_used=reranked,
        latency_ms=latency_ms,
        cost_estimate_usd=cost,
    )
