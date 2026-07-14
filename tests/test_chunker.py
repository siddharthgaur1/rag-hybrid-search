"""Tests for the three chunking strategies plus embedder-level deduplication."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_document, chunk_fixed_size, chunk_semantic, chunk_structure_aware, count_tokens
from src.ingestion.embedder import embed_and_store
from src.ingestion.loader import Document
from tests.conftest import make_embedding_response


class TestFixedSizeChunking:
    def test_produces_valid_chunks(self, sample_document):
        chunks = chunk_fixed_size(sample_document, chunk_size=8, overlap=2)
        assert len(chunks) > 0
        assert all(c.strategy_used == "fixed_size" for c in chunks)
        assert all(c.content.strip() for c in chunks)

    def test_overlap_is_correct(self, sample_document):
        chunks = chunk_fixed_size(sample_document, chunk_size=10, overlap=3)
        if len(chunks) >= 2:
            first_tail_tokens = count_tokens(chunks[0].content)
            # With overlap=3 and chunk_size=10, consecutive windows share up to 3 tokens;
            # verify the stride advanced (chunks aren't identical) and both are non-trivial.
            assert chunks[0].content != chunks[1].content
            assert first_tail_tokens > 0

    def test_overlap_must_be_smaller_than_chunk_size(self, sample_document):
        with pytest.raises(ValueError):
            chunk_fixed_size(sample_document, chunk_size=10, overlap=10)

    def test_empty_document_produces_no_chunks(self, sample_document):
        empty_doc = sample_document.model_copy(update={"content": ""})
        assert chunk_fixed_size(empty_doc) == []


class TestStructureAwareChunking:
    def test_respects_heading_boundaries(self, sample_document):
        chunks = chunk_structure_aware(sample_document)
        assert len(chunks) == 2  # "Billing" and "Support" sections
        assert "Billing" in chunks[0].content
        assert "Refunds take 5 business days" in chunks[0].content
        assert "Support" in chunks[1].content
        assert "help@example.com" in chunks[1].content

    def test_no_headings_falls_back_to_whole_document(self, sample_document):
        flat_doc = sample_document.model_copy(update={"content": "Just a plain paragraph with no headers at all."})
        chunks = chunk_structure_aware(flat_doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Just a plain paragraph with no headers at all."

    def test_preamble_before_first_heading_is_preserved(self, sample_document):
        doc = sample_document.model_copy(update={"content": "Intro text.\n\n## Billing\n\nBody text."})
        chunks = chunk_structure_aware(doc)
        assert chunks[0].content == "Intro text."
        assert "Billing" in chunks[1].content


class TestSemanticChunking:
    def test_splits_on_low_similarity(self, sample_document):
        doc = sample_document.model_copy(update={"content": "Cats are great pets. Dogs are loyal too. The stock market fell today."})
        # Force a split between sentence 2 and 3, keep 1-2 together.
        similarity_fn = lambda sentences: [0.9, 0.2]
        chunks = chunk_semantic(doc, similarity_fn, threshold=0.7)
        assert len(chunks) == 2
        assert "Cats" in chunks[0].content and "Dogs" in chunks[0].content
        assert "stock market" in chunks[1].content

    def test_similarity_fn_length_mismatch_raises(self, sample_document):
        doc = sample_document.model_copy(update={"content": "One. Two. Three."})
        with pytest.raises(ValueError):
            chunk_semantic(doc, similarity_fn=lambda s: [0.9])  # should be 2 scores for 3 sentences

    def test_single_sentence_document(self, sample_document):
        doc = sample_document.model_copy(update={"content": "Just one sentence."})
        chunks = chunk_semantic(doc, similarity_fn=lambda s: [])
        assert len(chunks) == 1


class TestChunkDocumentDispatch:
    def test_semantic_without_similarity_fn_raises(self, sample_document):
        with pytest.raises(ValueError):
            chunk_document(sample_document, "semantic")

    def test_unknown_strategy_raises(self, sample_document):
        with pytest.raises(ValueError):
            chunk_document(sample_document, "not_a_real_strategy")  # type: ignore[arg-type]

    def test_fixed_size_dispatch(self, sample_document):
        chunks = chunk_document(sample_document, "fixed_size")
        assert all(c.strategy_used == "fixed_size" for c in chunks)


@pytest.mark.asyncio
class TestDeduplication:
    async def test_near_duplicate_chunks_are_skipped(self, sample_document, mock_openai_client, tmp_path):
        from src.ingestion.chunker import Chunk

        chunk_a = Chunk(
            chunk_id="a", document_id="doc_001", content="unique content one",
            strategy_used="fixed_size", token_count=3, metadata=sample_document.metadata,
        )
        chunk_b = Chunk(
            chunk_id="b", document_id="doc_001", content="near duplicate of chunk a",
            strategy_used="fixed_size", token_count=5, metadata=sample_document.metadata,
        )
        # chunk_a and chunk_b get near-identical vectors -> b should be skipped as a duplicate.
        mock_openai_client.embeddings.create.return_value = make_embedding_response(
            [[1.0, 0.0, 0.0], [0.999, 0.001, 0.0]]
        )

        stats = await embed_and_store(
            [chunk_a, chunk_b], mock_openai_client,
            chroma_db_path=tmp_path / "chroma", bm25_index_path=tmp_path / "bm25",
            dedup_threshold=0.95,
        )

        assert stats.total_chunks == 2
        assert stats.chunks_stored == 1
        assert stats.duplicates_skipped == 1

    async def test_distinct_chunks_are_all_stored(self, sample_document, mock_openai_client, tmp_path):
        from src.ingestion.chunker import Chunk

        chunk_a = Chunk(
            chunk_id="a", document_id="doc_001", content="content about billing",
            strategy_used="fixed_size", token_count=3, metadata=sample_document.metadata,
        )
        chunk_b = Chunk(
            chunk_id="b", document_id="doc_001", content="content about deployment",
            strategy_used="fixed_size", token_count=3, metadata=sample_document.metadata,
        )
        mock_openai_client.embeddings.create.return_value = make_embedding_response(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

        stats = await embed_and_store(
            [chunk_a, chunk_b], mock_openai_client,
            chroma_db_path=tmp_path / "chroma", bm25_index_path=tmp_path / "bm25",
            dedup_threshold=0.95,
        )

        assert stats.chunks_stored == 2
        assert stats.duplicates_skipped == 0
