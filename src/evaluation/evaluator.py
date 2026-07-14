"""End-to-end RAG evaluation: retrieval relevance, faithfulness, citation
accuracy, answer correctness, and unanswerable-question detection.

Output field names (test_case_id, passed) intentionally match the
llm-regression-detector eval format — see README's "Connection to Project 1"
section for how this plugs into that project's comparator/alerting.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.generation.citation import UNSUPPORTED_THRESHOLD, extract_claim_citations, verify_citations
from src.generation.confidence import score_confidence
from src.generation.generator import generate_answer
from src.retrieval.dense import dense_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.sparse import sparse_search

logger = logging.getLogger(__name__)

CORRECTNESS_PASS_THRESHOLD = 0.6

JUDGE_SYSTEM_PROMPT = """You are grading whether a generated answer covers the
expected key facts for a question. Given the answer and a list of expected
facts, score 0-1 for what fraction of those facts are present in the answer,
in substance (paraphrases count). Return only the decimal fraction."""


class JudgeCorrectnessScore(BaseModel):
    fraction_covered: float = Field(..., ge=0.0, le=1.0, description="Fraction of expected facts present in substance.")


class RagEvalScore(BaseModel):
    """Per-question RAG evaluation result."""

    test_case_id: str = Field(..., description="Golden QA case ID.")
    answer_correctness: float = Field(..., ge=0.0, le=1.0, description="LLM-judge fraction of expected facts covered.")
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Fraction of claim sentences whose citations were all verified.")
    retrieval_relevance: float = Field(..., ge=0.0, le=1.0, description="Fraction of expected source_documents present in top-5 retrieved chunks.")
    citation_accuracy: float = Field(..., ge=0.0, le=1.0, description="Fraction of citations that passed verification.")
    unanswerable_detection: bool = Field(..., description="True if the system's answer/no-answer decision matched has_answer_in_corpus.")
    passed: bool = Field(..., description="answer_correctness >= threshold, for compatibility with comparator-style diffing.")


class RagEvalSummary(BaseModel):
    """Aggregate results across the full golden QA suite."""

    run_id: str
    chunking_strategy: str
    total_cases: int
    aggregate: dict[str, float] = Field(..., description="Mean of each metric across all cases.")
    by_category: dict[str, dict[str, float]] = Field(..., description="Mean of each metric, grouped by test case category.")
    by_difficulty: dict[str, dict[str, float]] = Field(..., description="Mean of each metric, grouped by difficulty.")
    total_cost_usd: float
    results: list[RagEvalScore]


async def _judge_correctness(client: AsyncOpenAI, answer: str, expected_facts: list[str], judge_model: str) -> float:
    if not expected_facts:
        return 1.0  # nothing expected to be present (e.g. unanswerable cases)
    response = await client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Answer: {answer}\n\nExpected facts: {expected_facts}"},
        ],
        response_format=JudgeCorrectnessScore,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Judge model {judge_model} returned no parsable correctness score")
    return parsed.fraction_covered


def _faithfulness_from_claims(answer_text: str, verification_scores: dict[int, int]) -> float:
    claims = extract_claim_citations(answer_text)
    if not claims:
        return 1.0  # no citation-bearing claims to fail (covers correct no-answer responses)
    grounded = sum(
        1 for c in claims if all(verification_scores.get(n, 0) >= UNSUPPORTED_THRESHOLD for n in c.citation_numbers)
    )
    return grounded / len(claims)


async def evaluate_case(
    test_case: dict,
    client: AsyncOpenAI,
    chroma_db_path: str | Path,
    bm25_index_path: str | Path,
    generation_model: str = settings.generation_model,
    judge_model: str = settings.judge_model,
) -> RagEvalScore:
    """Run one golden QA case through the full retrieval + generation + verification pipeline.

    Raises:
        KeyError: if test_case is missing required fields.
    """
    for required_key in ("id", "question", "expected_answer_contains", "source_documents", "has_answer_in_corpus"):
        if required_key not in test_case:
            raise KeyError(f"Test case is missing required key '{required_key}': {test_case}")

    question = test_case["question"]
    dense_hits = await dense_search(question, client, chroma_db_path, top_k=10)
    sparse_hits = sparse_search(question, bm25_index_path, top_k=10)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
    reranked = rerank(question, fused)

    dense_metadata_by_id = {h.chunk_id: h.metadata for h in dense_hits}
    retrieved_file_paths = {dense_metadata_by_id[r.chunk_id]["file_path"] for r in reranked if r.chunk_id in dense_metadata_by_id}
    expected_docs = test_case["source_documents"]
    retrieval_relevance = (
        1.0
        if not expected_docs
        else len([d for d in expected_docs if d in retrieved_file_paths]) / len(expected_docs)
    )

    generated = await generate_answer(question, reranked, client, model=generation_model)
    citation_result = await verify_citations(generated.answer, reranked, client, judge_model)

    correctness = await _judge_correctness(client, generated.answer, test_case["expected_answer_contains"], judge_model)
    faithfulness = _faithfulness_from_claims(generated.answer, citation_result.verification_scores)
    citation_accuracy = (
        len(citation_result.verified_citations) / len(citation_result.verification_scores)
        if citation_result.verification_scores
        else 1.0
    )
    unanswerable_correct = generated.is_no_answer != test_case["has_answer_in_corpus"]

    return RagEvalScore(
        test_case_id=test_case["id"],
        answer_correctness=correctness,
        faithfulness=faithfulness,
        retrieval_relevance=retrieval_relevance,
        citation_accuracy=citation_accuracy,
        unanswerable_detection=unanswerable_correct,
        passed=correctness >= CORRECTNESS_PASS_THRESHOLD,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_metrics(scores: list[RagEvalScore]) -> dict[str, float]:
    return {
        "answer_correctness": _mean([s.answer_correctness for s in scores]),
        "faithfulness": _mean([s.faithfulness for s in scores]),
        "retrieval_relevance": _mean([s.retrieval_relevance for s in scores]),
        "citation_accuracy": _mean([s.citation_accuracy for s in scores]),
        "unanswerable_detection": _mean([1.0 if s.unanswerable_detection else 0.0 for s in scores]),
        "pass_rate": _mean([1.0 if s.passed else 0.0 for s in scores]),
    }


async def run_eval_suite(
    golden_qa: list[dict],
    client: AsyncOpenAI,
    chunking_strategy: str,
    run_id: str,
    chroma_db_path: str | Path = settings.chroma_db_path,
    bm25_index_path: str | Path = settings.bm25_index_path,
) -> RagEvalSummary:
    """Run the full golden QA suite and aggregate results overall, by category, and by difficulty."""
    results = [
        await evaluate_case(tc, client, chroma_db_path, bm25_index_path) for tc in golden_qa
    ]

    by_category: dict[str, list[RagEvalScore]] = {}
    by_difficulty: dict[str, list[RagEvalScore]] = {}
    tc_by_id = {tc["id"]: tc for tc in golden_qa}
    for result in results:
        tc = tc_by_id[result.test_case_id]
        by_category.setdefault(tc["category"], []).append(result)
        by_difficulty.setdefault(tc["difficulty"], []).append(result)

    return RagEvalSummary(
        run_id=run_id,
        chunking_strategy=chunking_strategy,
        total_cases=len(results),
        aggregate=aggregate_metrics(results),
        by_category={cat: aggregate_metrics(scores) for cat, scores in by_category.items()},
        by_difficulty={diff: aggregate_metrics(scores) for diff, scores in by_difficulty.items()},
        total_cost_usd=0.0,  # populated by the caller from token-usage tracking; not measured within this function
        results=results,
    )
