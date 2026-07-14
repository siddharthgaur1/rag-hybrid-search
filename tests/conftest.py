"""Shared pytest fixtures: sample documents/chunks, a mock OpenAI client, and
an in-memory ChromaDB instance."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import chromadb
import pytest

from src.ingestion.chunker import Chunk
from src.ingestion.loader import Document, DocumentMetadata


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        source_file="faq.md", file_path="support/faq.md", section_heading="FAQ", last_modified="2026-01-01T00:00:00Z"
    )


@pytest.fixture
def sample_document(sample_metadata) -> Document:
    return Document(
        id="doc_001",
        content="## Billing\n\nRefunds take 5 business days.\n\n## Support\n\nEmail us at help@example.com.",
        raw_content="## Billing\n\nRefunds take 5 business days.\n\n## Support\n\nEmail us at help@example.com.",
        metadata=sample_metadata,
    )


@pytest.fixture
def sample_chunk(sample_metadata) -> Chunk:
    return Chunk(
        chunk_id="doc_001_structure_aware_0",
        document_id="doc_001",
        content="Billing\n\nRefunds take 5 business days.",
        strategy_used="structure_aware",
        token_count=10,
        metadata=sample_metadata,
    )


def make_parse_response(parsed, input_tokens: int = 100, output_tokens: int = 20):
    """Build a fake object shaped like an OpenAI ``.chat.completions.parse()`` response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=None))],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


def make_chat_response(content: str, input_tokens: int = 100, output_tokens: int = 20):
    """Build a fake object shaped like an OpenAI ``.chat.completions.create()`` response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, parsed=None))],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


def make_embedding_response(vectors: list[list[float]], total_tokens: int = 50):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v) for v in vectors],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


@pytest.fixture
def mock_openai_client():
    """An AsyncOpenAI-shaped mock whose calls can be configured per-test."""
    client = AsyncMock()
    client.chat.completions.parse = AsyncMock()
    client.chat.completions.create = AsyncMock()
    client.embeddings.create = AsyncMock()
    return client


@pytest.fixture
def in_memory_chroma_collection():
    """A fresh, non-persistent Chroma collection for testing dense search against."""
    test_client = chromadb.EphemeralClient()
    return test_client.get_or_create_collection("test_chunks")
