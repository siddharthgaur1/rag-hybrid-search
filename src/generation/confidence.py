"""Composite answer confidence from retrieval quality, citation coverage,
and answer completeness."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.generation.citation import CitationVerificationResult
from src.retrieval.reranker import RerankedResult

logger = logging.getLogger(__name__)

RETRIEVAL_WEIGHT = 0.3
CITATION_WEIGHT = 0.4
COMPLETENESS_WEIGHT = 0.3

JUDGE_SYSTEM_PROMPT = """You are grading whether an answer addresses every part of
the question it was asked, given the question and answer. Score 1-5:
1 = ignores the question or addresses none of it, 3 = addresses the main part
but misses secondary aspects, 5 = fully addresses every part of the question.
Return only the integer score."""


class JudgeCompletenessScore(BaseModel):
    score: int = Field(..., ge=1, le=5, description="1-5 completeness score.")


class ConfidenceScores(BaseModel):
    """All four confidence signals for a single generated answer."""

    retrieval_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean dense similarity of the retrieved (post-rerank) chunks.")
    citation_coverage: float = Field(..., ge=0.0, le=1.0, description="Fraction of cited claims that were verified as supported.")
    answer_completeness: float = Field(..., ge=0.0, le=1.0, description="LLM-judge score for whether all parts of the question were addressed, normalized 1-5 -> 0-1.")
    composite: float = Field(..., ge=0.0, le=1.0, description=f"{RETRIEVAL_WEIGHT}*retrieval + {CITATION_WEIGHT}*citation + {COMPLETENESS_WEIGHT}*completeness.")


def retrieval_confidence(chunks: list[RerankedResult]) -> float:
    """Mean dense similarity across the retrieved chunks. 0.0 if none carried a similarity score."""
    similarities = [c.dense_similarity for c in chunks if c.dense_similarity is not None]
    if not similarities:
        return 0.0
    return sum(similarities) / len(similarities)


def citation_coverage(citation_result: CitationVerificationResult) -> float:
    """Fraction of all scored citations that were verified as supported."""
    total = len(citation_result.verification_scores)
    if total == 0:
        return 1.0  # no citations to fail — an answer that made no unsupported claims
    return len(citation_result.verified_citations) / total


async def answer_completeness(question: str, answer: str, client: AsyncOpenAI, judge_model: str = settings.judge_model) -> float:
    """LLM-judge score for whether the answer addresses every part of the question, normalized to 0-1."""
    response = await client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
        ],
        response_format=JudgeCompletenessScore,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Judge model {judge_model} returned no parsable completeness score")
    return (parsed.score - 1) / 4


async def score_confidence(
    question: str,
    answer: str,
    chunks: list[RerankedResult],
    citation_result: CitationVerificationResult,
    client: AsyncOpenAI,
    judge_model: str = settings.judge_model,
) -> ConfidenceScores:
    """Compute all four confidence signals for a generated answer."""
    retrieval = retrieval_confidence(chunks)
    citation = citation_coverage(citation_result)
    completeness = await answer_completeness(question, answer, client, judge_model)
    composite = RETRIEVAL_WEIGHT * retrieval + CITATION_WEIGHT * citation + COMPLETENESS_WEIGHT * completeness

    return ConfidenceScores(
        retrieval_confidence=retrieval,
        citation_coverage=citation,
        answer_completeness=completeness,
        composite=composite,
    )
