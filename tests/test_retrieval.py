"""Tests for dense search, BM25 sparse search, RRF fusion, and cross-encoder reranking."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
import pytest
from rank_bm25 import BM25Okapi

from src.retrieval.dense import DenseResult, dense_search
from src.retrieval.fusion import SparseResult, reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.sparse import sparse_search
from tests.conftest import make_embedding_response


@pytest.mark.asyncio
class TestDenseSearch:
    async def test_returns_ranked_results(self, mock_openai_client, tmp_path):
        chroma_path = tmp_path / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection("chunks")
        collection.upsert(
            ids=["c1", "c2"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["about billing", "about deployment"],
            metadatas=[{"file_path": "a.md"}, {"file_path": "b.md"}],
        )
        mock_openai_client.embeddings.create.return_value = make_embedding_response([[1.0, 0.0]])

        results = await dense_search("billing question", mock_openai_client, chroma_db_path=chroma_path, top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"  # exact match to query vector should rank first
        assert results[0].similarity >= results[1].similarity

    async def test_empty_collection_returns_empty_list(self, mock_openai_client, tmp_path):
        mock_openai_client.embeddings.create.return_value = make_embedding_response([[1.0, 0.0]])
        results = await dense_search("anything", mock_openai_client, chroma_db_path=tmp_path / "empty_chroma")
        assert results == []

    async def test_empty_query_raises(self, mock_openai_client, tmp_path):
        with pytest.raises(ValueError):
            await dense_search("   ", mock_openai_client, chroma_db_path=tmp_path / "chroma")


class TestSparseSearch:
    def _write_index(self, tmp_path, corpus: list[str]):
        bm25_path = tmp_path / "bm25"
        bm25_path.mkdir()
        corpus_tokens = [doc.lower().split() for doc in corpus]
        index = BM25Okapi(corpus_tokens)
        with open(bm25_path / "bm25_index.pkl", "wb") as f:
            pickle.dump(
                {"index": index, "chunk_ids": [f"c{i}" for i in range(len(corpus))], "corpus_tokens": corpus_tokens, "chunk_contents": corpus},
                f,
            )
        return bm25_path

    def test_finds_exact_keyword_match(self, tmp_path):
        # 3+ docs so the matched term's BM25 IDF is strictly positive — with only
        # 2 docs, a term present in exactly 1 of them sits at the IDF formula's
        # zero boundary (log((N-n+0.5)/(n+0.5)) == log(1) == 0), which is a real
        # property of classic BM25 on tiny corpora, not something to work around
        # in sparse_search itself.
        bm25_path = self._write_index(
            tmp_path,
            [
                "the API returns error code AUTH_INVALID_KEY",
                "general onboarding docs about workflows",
                "deployment guide for the worker fleet",
            ],
        )
        results = sparse_search("AUTH_INVALID_KEY", bm25_index_path=bm25_path, top_k=5)
        assert len(results) >= 1
        assert results[0].chunk_id == "c0"

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sparse_search("anything", bm25_index_path=tmp_path / "nonexistent")

    def test_empty_query_raises(self, tmp_path):
        bm25_path = self._write_index(tmp_path, ["some text"])
        with pytest.raises(ValueError):
            sparse_search("", bm25_index_path=bm25_path)


class TestReciprocalRankFusion:
    def test_combines_dense_and_sparse_rankings(self):
        dense = [DenseResult(chunk_id="a", content="a", metadata={}, similarity=0.9), DenseResult(chunk_id="b", content="b", metadata={}, similarity=0.5)]
        sparse = [SparseResult(chunk_id="b", content="b", bm25_score=10.0), SparseResult(chunk_id="c", content="c", bm25_score=5.0)]

        fused = reciprocal_rank_fusion(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60, top_k=10)

        fused_ids = [r.chunk_id for r in fused]
        assert set(fused_ids) == {"a", "b", "c"}
        # "b" appears in both lists (rank 2 dense, rank 1 sparse) so it should score higher than "c" (sparse-only, rank 2).
        b_score = next(r.fused_score for r in fused if r.chunk_id == "b")
        c_score = next(r.fused_score for r in fused if r.chunk_id == "c")
        assert b_score > c_score

    def test_dense_only_hit_still_included(self):
        dense = [DenseResult(chunk_id="only_dense", content="x", metadata={}, similarity=0.8)]
        fused = reciprocal_rank_fusion(dense, [])
        assert len(fused) == 1
        assert fused[0].sparse_rank is None
        assert fused[0].dense_rank == 1

    def test_top_k_caps_results(self):
        dense = [DenseResult(chunk_id=f"d{i}", content="x", metadata={}, similarity=0.5) for i in range(30)]
        fused = reciprocal_rank_fusion(dense, [], top_k=5)
        assert len(fused) == 5


class TestReranker:
    def test_changes_order_when_scores_disagree(self):
        from src.retrieval.fusion import FusedResult

        candidates = [
            FusedResult(chunk_id="low_fused_high_relevance", content="exact match text", fused_score=0.1),
            FusedResult(chunk_id="high_fused_low_relevance", content="unrelated text", fused_score=0.9),
        ]
        # Reverse the fusion order: the cross-encoder disagrees with fusion ranking.
        fake_model = SimpleNamespace(predict=lambda pairs: [0.95, 0.05])

        with patch("src.retrieval.reranker._get_model", return_value=fake_model):
            results = rerank("query", candidates, top_k=2)

        assert results[0].chunk_id == "low_fused_high_relevance"
        assert results[0].pre_rerank_rank == 1  # it was first in the fused list already in this setup
        assert results[1].chunk_id == "high_fused_low_relevance"

    def test_empty_candidates_returns_empty(self):
        assert rerank("query", []) == []

    def test_top_k_limits_results(self):
        from src.retrieval.fusion import FusedResult

        candidates = [FusedResult(chunk_id=f"c{i}", content=f"text {i}", fused_score=1.0) for i in range(10)]
        fake_model = SimpleNamespace(predict=lambda pairs: list(range(len(pairs))))

        with patch("src.retrieval.reranker._get_model", return_value=fake_model):
            results = rerank("query", candidates, top_k=3)

        assert len(results) == 3
