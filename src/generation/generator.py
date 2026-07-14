"""Grounded answer generation with inline [N] citations over retrieved chunks."""

from __future__ import annotations

import logging
import re

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.retrieval.reranker import RerankedResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a support assistant answering questions using only the
numbered context blocks provided below. Rules:

1. Answer only using information present in the context blocks.
2. Cite every factual claim with the block number(s) it came from, in the
   form [1], [2], etc., immediately after the claim.
3. If the context does not contain enough information to answer, respond
   exactly: "I don't have enough information to answer that."
4. Do not fabricate citations to blocks that don't support the claim next to them."""

NO_ANSWER_PHRASE = "I don't have enough information to answer that."

_CITATION_RE = re.compile(r"\[(\d+)\]")


class GeneratedAnswer(BaseModel):
    """Result of a single grounded-generation call."""

    answer: str = Field(..., description="Generated answer text, with inline [N] citations.")
    raw_citations: list[int] = Field(..., description="All [N] block numbers referenced in the answer, in order of first appearance.")
    context_chunk_ids: list[str] = Field(..., description="chunk_id for each numbered context block, index 0 = block [1].")
    is_no_answer: bool = Field(..., description="True if the model declined to answer due to insufficient context.")


def _build_context_blocks(chunks: list[RerankedResult]) -> str:
    return "\n\n".join(f"[{i + 1}] {chunk.content}" for i, chunk in enumerate(chunks))


def parse_citations(answer_text: str) -> list[int]:
    """Extract all [N] citation numbers from generated text, in first-appearance order."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer_text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


async def generate_answer(
    question: str,
    chunks: list[RerankedResult],
    client: AsyncOpenAI,
    model: str = settings.generation_model,
) -> GeneratedAnswer:
    """Generate a grounded, cited answer from retrieved chunks.

    Args:
        question: The user's question.
        chunks: Reranked chunks to use as context, best-first. Chunk i becomes
            context block [i+1].
        client: An initialized AsyncOpenAI client.
        model: Chat completion model name.

    Raises:
        ValueError: if question is empty.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    if not chunks:
        logger.info("generate_answer called with no context chunks; returning no-answer without an API call")
        return GeneratedAnswer(answer=NO_ANSWER_PHRASE, raw_citations=[], context_chunk_ids=[], is_no_answer=True)

    context_blocks = _build_context_blocks(chunks)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_blocks}\n\nQuestion: {question}"},
        ],
    )
    answer_text = response.choices[0].message.content or ""
    citations = parse_citations(answer_text)

    return GeneratedAnswer(
        answer=answer_text,
        raw_citations=citations,
        context_chunk_ids=[c.chunk_id for c in chunks],
        is_no_answer=NO_ANSWER_PHRASE in answer_text,
    )
