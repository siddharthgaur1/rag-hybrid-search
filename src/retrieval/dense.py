"""Dense vector retrieval over the ChromaDB chunk collection."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)


class DenseResult(BaseModel):
    """A single dense-search hit."""

    chunk_id: str = Field(..., description="ID of the matched chunk.")
    content: str = Field(..., description="Chunk text.")
    metadata: dict = Field(..., description="Chunk metadata as stored in ChromaDB.")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity to the query, 0-1 (higher is better).")


async def dense_search(
    query: str,
    client: AsyncOpenAI,
    chroma_db_path: str | Path = settings.chroma_db_path,
    top_k: int = 10,
    embedding_model: str = settings.embedding_model,
    where: dict | None = None,
) -> list[DenseResult]:
    """Embed the query and return the top-k most similar chunks.

    Args:
        query: Natural-language query text.
        client: An initialized AsyncOpenAI client.
        chroma_db_path: Path to the persisted Chroma collection.
        top_k: Number of results to return.
        embedding_model: Must match the model used to embed the corpus.
        where: Optional Chroma metadata filter, e.g. {"source_file": "faq.md"}
            to restrict by document, or {"strategy_used": "structure_aware"}.

    Returns:
        Results ordered by descending similarity. Empty list if the
        collection has no chunks yet.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    response = await client.embeddings.create(model=embedding_model, input=[query])
    query_vector = response.data[0].embedding

    chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = chroma_client.get_or_create_collection("chunks")
    if collection.count() == 0:
        logger.warning("Dense search against an empty collection at %s", chroma_db_path)
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        where=where,
    )

    hits: list[DenseResult] = []
    for chunk_id, content, metadata, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Chroma's default space is squared L2 on normalized OpenAI embeddings,
        # where distance == 2 * (1 - cosine_similarity); convert back to a 0-1 score.
        similarity = max(0.0, 1 - distance / 2)
        hits.append(DenseResult(chunk_id=chunk_id, content=content, metadata=metadata, similarity=similarity))
    return hits
