"""Tests for grounded answer generation: citation parsing, no-context handling, prompt construction."""

from __future__ import annotations

import pytest

from src.generation.generator import (
    NO_ANSWER_PHRASE,
    _build_context_blocks,
    generate_answer,
    parse_citations,
)
from src.retrieval.reranker import RerankedResult
from tests.conftest import make_chat_response


def make_reranked(chunk_id: str, content: str) -> RerankedResult:
    return RerankedResult(chunk_id=chunk_id, content=content, rerank_score=0.9, pre_rerank_rank=1, post_rerank_rank=1)


class TestParseCitations:
    def test_extracts_citations_in_order(self):
        assert parse_citations("The rate limit is 100/min [1]. It resets hourly [2].") == [1, 2]

    def test_deduplicates_repeated_citations(self):
        assert parse_citations("Fact one [1]. Related fact [1]. Another [3].") == [1, 3]

    def test_no_citations_returns_empty_list(self):
        assert parse_citations("Plain text with no citations.") == []


class TestBuildContextBlocks:
    def test_numbers_blocks_starting_at_one(self):
        chunks = [make_reranked("c1", "first chunk"), make_reranked("c2", "second chunk")]
        blocks = _build_context_blocks(chunks)
        assert "[1] first chunk" in blocks
        assert "[2] second chunk" in blocks


@pytest.mark.asyncio
class TestGenerateAnswer:
    async def test_empty_chunks_returns_no_answer_without_api_call(self, mock_openai_client):
        result = await generate_answer("What is the rate limit?", [], mock_openai_client)

        assert result.answer == NO_ANSWER_PHRASE
        assert result.is_no_answer is True
        assert result.raw_citations == []
        mock_openai_client.chat.completions.create.assert_not_awaited()

    async def test_returns_parsed_citations_and_chunk_ids(self, mock_openai_client):
        chunks = [make_reranked("c1", "rate limit is 100/min"), make_reranked("c2", "resets hourly")]
        mock_openai_client.chat.completions.create.return_value = make_chat_response(
            "The rate limit is 100 requests/min [1] and resets hourly [2]."
        )

        result = await generate_answer("What is the rate limit?", chunks, mock_openai_client)

        assert result.raw_citations == [1, 2]
        assert result.context_chunk_ids == ["c1", "c2"]
        assert result.is_no_answer is False

    async def test_detects_no_answer_phrase_in_model_response(self, mock_openai_client):
        chunks = [make_reranked("c1", "unrelated content")]
        mock_openai_client.chat.completions.create.return_value = make_chat_response(NO_ANSWER_PHRASE)

        result = await generate_answer("Something not in the corpus?", chunks, mock_openai_client)

        assert result.is_no_answer is True

    async def test_empty_question_raises(self, mock_openai_client):
        with pytest.raises(ValueError):
            await generate_answer("", [make_reranked("c1", "x")], mock_openai_client)
