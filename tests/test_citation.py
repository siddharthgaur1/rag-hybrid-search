"""Tests for citation verification and confidence scoring."""

from __future__ import annotations

import pytest

from src.generation.citation import (
    CitationVerificationResult,
    JudgeCitationScore,
    extract_claim_citations,
    verify_citations,
)
from src.generation.confidence import (
    JudgeCompletenessScore,
    citation_coverage,
    retrieval_confidence,
    score_confidence,
)
from src.retrieval.reranker import RerankedResult
from tests.conftest import make_parse_response


def make_reranked(chunk_id: str, content: str, dense_similarity: float | None = 0.8) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id, content=content, rerank_score=0.9, pre_rerank_rank=1, post_rerank_rank=1,
        dense_similarity=dense_similarity,
    )


class TestExtractClaimCitations:
    def test_pairs_sentences_with_their_citation_numbers(self):
        claims = extract_claim_citations("The rate limit is 100/min [1]. It resets hourly [2].")
        assert len(claims) == 2
        assert claims[0].claim == "The rate limit is 100/min ."
        assert claims[0].citation_numbers == [1]
        assert claims[1].citation_numbers == [2]

    def test_sentence_with_multiple_citations(self):
        claims = extract_claim_citations("This is supported by two sources [1][2].")
        assert claims[0].citation_numbers == [1, 2]

    def test_sentences_without_citations_are_skipped(self):
        claims = extract_claim_citations("No citation here. This one has one [1].")
        assert len(claims) == 1


@pytest.mark.asyncio
class TestVerifyCitations:
    async def test_flags_low_scoring_citation_as_unsupported(self, mock_openai_client):
        chunks = [make_reranked("c1", "the rate limit is 100 requests per minute")]
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(JudgeCitationScore(score=2))

        result = await verify_citations("The system supports 1000 users [1].", chunks, mock_openai_client)

        assert result.unsupported_citations == [1]
        assert result.verified_citations == []
        assert result.verification_scores[1] == 2

    async def test_verifies_well_supported_citation(self, mock_openai_client):
        chunks = [make_reranked("c1", "the rate limit is 100 requests per minute")]
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(JudgeCitationScore(score=5))

        result = await verify_citations("The rate limit is 100 req/min [1].", chunks, mock_openai_client)

        assert result.verified_citations == [1]
        assert result.unsupported_citations == []

    async def test_citation_referencing_nonexistent_block_is_unsupported_without_judge_call(self, mock_openai_client):
        chunks = [make_reranked("c1", "some content")]

        result = await verify_citations("A claim citing a block that doesn't exist [5].", chunks, mock_openai_client)

        assert result.unsupported_citations == [5]
        assert result.verification_scores[5] == 1
        mock_openai_client.chat.completions.parse.assert_not_awaited()

    async def test_no_citations_returns_empty_result(self, mock_openai_client):
        result = await verify_citations("Plain answer with no citations.", [], mock_openai_client)
        assert result == CitationVerificationResult(verified_citations=[], unsupported_citations=[], verification_scores={})


class TestConfidenceScoring:
    def test_retrieval_confidence_averages_dense_similarity(self):
        chunks = [make_reranked("c1", "x", dense_similarity=0.9), make_reranked("c2", "y", dense_similarity=0.7)]
        assert retrieval_confidence(chunks) == pytest.approx(0.8)

    def test_retrieval_confidence_zero_when_no_similarities(self):
        chunks = [make_reranked("c1", "x", dense_similarity=None)]
        assert retrieval_confidence(chunks) == 0.0

    def test_citation_coverage_fraction_verified(self):
        result = CitationVerificationResult(verified_citations=[1], unsupported_citations=[2], verification_scores={1: 5, 2: 1})
        assert citation_coverage(result) == pytest.approx(0.5)

    def test_citation_coverage_full_when_no_citations(self):
        result = CitationVerificationResult(verified_citations=[], unsupported_citations=[], verification_scores={})
        assert citation_coverage(result) == 1.0

    @pytest.mark.asyncio
    async def test_composite_score_calculation(self, mock_openai_client):
        chunks = [make_reranked("c1", "x", dense_similarity=1.0)]
        citation_result = CitationVerificationResult(verified_citations=[1], unsupported_citations=[], verification_scores={1: 5})
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(JudgeCompletenessScore(score=5))

        scores = await score_confidence("question?", "answer [1].", chunks, citation_result, mock_openai_client)

        # retrieval=1.0, citation=1.0, completeness=1.0 -> composite = 0.3+0.4+0.3 = 1.0
        assert scores.retrieval_confidence == pytest.approx(1.0)
        assert scores.citation_coverage == pytest.approx(1.0)
        assert scores.answer_completeness == pytest.approx(1.0)
        assert scores.composite == pytest.approx(1.0)
