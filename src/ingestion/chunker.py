"""Three switchable chunking strategies: fixed_size, structure_aware, semantic."""

from __future__ import annotations

import logging
import re
from typing import Callable, Literal

import tiktoken
from pydantic import BaseModel, Field

from src.config import settings
from src.ingestion.loader import Document, DocumentMetadata

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["fixed_size", "structure_aware", "semantic"]
_ENCODING = tiktoken.get_encoding("cl100k_base")

SimilarityFn = Callable[[list[str]], list[float]]
"""Given N sentences, returns N-1 cosine similarities between adjacent sentences."""


class Chunk(BaseModel):
    """A retrievable unit of text produced by a chunking strategy."""

    chunk_id: str = Field(..., description="Unique ID: '{document_id}_{strategy}_{index}'.")
    document_id: str = Field(..., description="ID of the parent Document.")
    content: str = Field(..., min_length=1, description="Chunk text.")
    strategy_used: ChunkingStrategy = Field(..., description="Which chunking strategy produced this chunk.")
    token_count: int = Field(..., ge=0, description="Token count under the cl100k_base encoding.")
    metadata: DocumentMetadata = Field(..., description="Inherited from the parent document.")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _make_chunk(document: Document, content: str, strategy: ChunkingStrategy, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document.id}_{strategy}_{index}",
        document_id=document.id,
        content=content,
        strategy_used=strategy,
        token_count=count_tokens(content),
        metadata=document.metadata,
    )


def chunk_fixed_size(
    document: Document, chunk_size: int = settings.chunk_size_tokens, overlap: int = settings.chunk_overlap_tokens
) -> list[Chunk]:
    """Split into fixed-size token windows with overlap. The baseline strategy.

    Raises:
        ValueError: if overlap >= chunk_size (would never advance).
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    tokens = _ENCODING.encode(document.content)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    stride = chunk_size - overlap
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        content = _ENCODING.decode(window).strip()
        if content:
            chunks.append(_make_chunk(document, content, "fixed_size", len(chunks)))
        start += stride
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_structure_aware(document: Document) -> list[Chunk]:
    """Split on markdown headers, keeping heading context with each chunk.

    Falls back to treating the whole document as one chunk if no headers exist,
    so non-markdown or flat text still gets a valid (if coarse) chunk.
    """
    text = document.content
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return [_make_chunk(document, stripped, "structure_aware", 0)] if stripped else []

    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        heading_stack = heading_stack[: level - 1] + [heading_text]

        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[section_start:section_end].strip()
        if not body:
            continue

        breadcrumb = " > ".join(heading_stack)
        content = f"{breadcrumb}\n\n{body}"
        chunks.append(_make_chunk(document, content, "structure_aware", len(chunks)))

    # Content before the first heading (if any) is preserved as its own chunk.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.insert(0, _make_chunk(document, preamble, "structure_aware", -1))
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{document.id}_structure_aware_{i}"

    return chunks


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_semantic(
    document: Document,
    similarity_fn: SimilarityFn,
    threshold: float = settings.semantic_similarity_threshold,
) -> list[Chunk]:
    """Split where adjacent-sentence similarity drops below threshold.

    Groups semantically coherent sentences together; splits are inserted at
    topic boundaries rather than at fixed intervals. Most expensive strategy
    (requires embedding every sentence) but tends to produce the most
    coherent chunks for downstream retrieval.

    Args:
        document: The document to chunk.
        similarity_fn: Computes cosine similarity between each pair of
            adjacent sentences. Injected so this stays testable without a
            real embedding model.
        threshold: Similarity below this triggers a split.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(document.content) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [_make_chunk(document, sentences[0], "semantic", 0)]

    similarities = similarity_fn(sentences)
    if len(similarities) != len(sentences) - 1:
        raise ValueError(
            f"similarity_fn must return len(sentences)-1 = {len(sentences) - 1} scores, got {len(similarities)}"
        )

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, sim in zip(sentences[1:], similarities):
        if sim < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [_make_chunk(document, " ".join(group), "semantic", i) for i, group in enumerate(groups)]


_LOCAL_SIMILARITY_MODEL_NAME = "all-MiniLM-L6-v2"


def local_sentence_similarity_fn() -> SimilarityFn:
    """Build a SimilarityFn backed by a small local sentence-transformer.

    Semantic chunking needs an embedding for every sentence in the corpus,
    which would be a lot of paid OpenAI calls purely to decide *where* to
    cut chunks. A local model avoids that cost — cheaper and faster for a
    decision that only needs relative similarity, not the exact embedding
    space used for retrieval.
    """
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(_LOCAL_SIMILARITY_MODEL_NAME)

    def _similarity_fn(sentences: list[str]) -> list[float]:
        embeddings = model.encode(sentences, convert_to_tensor=True)
        return [float(util.cos_sim(embeddings[i], embeddings[i + 1])) for i in range(len(sentences) - 1)]

    return _similarity_fn


def chunk_document(document: Document, strategy: ChunkingStrategy, similarity_fn: SimilarityFn | None = None) -> list[Chunk]:
    """Dispatch to the requested chunking strategy.

    Raises:
        ValueError: if strategy is "semantic" and similarity_fn isn't provided,
            or if strategy is not a recognized value.
    """
    if strategy == "fixed_size":
        return chunk_fixed_size(document)
    if strategy == "structure_aware":
        return chunk_structure_aware(document)
    if strategy == "semantic":
        if similarity_fn is None:
            raise ValueError("chunk_document(strategy='semantic') requires a similarity_fn")
        return chunk_semantic(document, similarity_fn)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
