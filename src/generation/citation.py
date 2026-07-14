"""Citation parsing and LLM-as-judge verification: does the cited chunk
actually support the claim next to it?"""

from __future__ import annotations

import logging
import re

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.retrieval.reranker import RerankedResult

logger = logging.getLogger(__name__)

UNSUPPORTED_THRESHOLD = 3  # judge scores below this (on a 1-5 scale) are flagged

JUDGE_SYSTEM_PROMPT = """You are verifying whether a cited source chunk supports a
claim made in a generated answer. Score 1-5:
1 = the chunk contradicts or has nothing to do with the claim,
3 = the chunk is topically related but doesn't clearly support the specific claim,
5 = the chunk directly and clearly supports the claim.
Return only the integer score."""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[(\d+)\]")


class JudgeCitationScore(BaseModel):
    score: int = Field(..., ge=1, le=5, description="1-5 support score for a single claim/citation pair.")


class ClaimCitation(BaseModel):
    """One claim (a sentence) paired with the citation number(s) it references."""

    claim: str = Field(..., description="The sentence, with citation markers stripped.")
    citation_numbers: list[int] = Field(..., description="[N] numbers referenced within this sentence.")


class CitationVerificationResult(BaseModel):
    """Outcome of verifying every citation in a generated answer."""

    verified_citations: list[int] = Field(..., description="Citation numbers that scored >= UNSUPPORTED_THRESHOLD.")
    unsupported_citations: list[int] = Field(..., description="Citation numbers that scored < UNSUPPORTED_THRESHOLD.")
    verification_scores: dict[int, int] = Field(..., description="citation_number -> 1-5 judge score.")


def extract_claim_citations(answer_text: str) -> list[ClaimCitation]:
    """Split an answer into sentences and pair each with its [N] citation numbers."""
    claims: list[ClaimCitation] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        citation_numbers = [int(n) for n in _CITATION_RE.findall(sentence)]
        if not citation_numbers:
            continue
        claim = _CITATION_RE.sub("", sentence).strip()
        claims.append(ClaimCitation(claim=claim, citation_numbers=citation_numbers))
    return claims


async def _judge_claim_support(client: AsyncOpenAI, judge_model: str, claim: str, chunk_content: str) -> int:
    response = await client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim: {claim}\n\nCited source:\n{chunk_content}"},
        ],
        response_format=JudgeCitationScore,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Judge model {judge_model} returned no parsable citation score")
    return parsed.score


async def verify_citations(
    answer_text: str,
    context_chunks: list[RerankedResult],
    client: AsyncOpenAI,
    judge_model: str = settings.judge_model,
) -> CitationVerificationResult:
    """Verify every [N] citation in an answer against its source chunk.

    A citation number N refers to context_chunks[N-1]. Citations referencing
    a block number outside the context range are treated as unsupported
    (score 1) without a judge call, since there's no chunk to check against.

    Args:
        answer_text: Generated answer containing [N] citations.
        context_chunks: The context blocks the answer was generated from,
            in the same order used at generation time (block [1] = index 0).
        client: An initialized AsyncOpenAI client.
        judge_model: Model used for the support-scoring judge call.
    """
    claims = extract_claim_citations(answer_text)
    scores: dict[int, int] = {}

    for claim_citation in claims:
        for citation_number in claim_citation.citation_numbers:
            if citation_number in scores:
                continue
            if citation_number < 1 or citation_number > len(context_chunks):
                logger.warning("Citation [%d] references a nonexistent context block", citation_number)
                scores[citation_number] = 1
                continue
            chunk_content = context_chunks[citation_number - 1].content
            scores[citation_number] = await _judge_claim_support(
                client, judge_model, claim_citation.claim, chunk_content
            )

    verified = [n for n, s in scores.items() if s >= UNSUPPORTED_THRESHOLD]
    unsupported = [n for n, s in scores.items() if s < UNSUPPORTED_THRESHOLD]
    return CitationVerificationResult(
        verified_citations=sorted(verified), unsupported_citations=sorted(unsupported), verification_scores=scores
    )
