"""FastAPI service exposing the RAG pipeline: ask questions, manage the index, check health."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import Counter
from contextvars import ContextVar
from pathlib import Path

import chromadb
from fastapi import FastAPI, Request
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.generation.confidence import ConfidenceScores
from src.ingestion.chunker import ChunkingStrategy, chunk_document, local_sentence_similarity_fn
from src.ingestion.embedder import IngestionStats, embed_and_store
from src.ingestion.loader import load_document
from src.ingestion.pipeline import ingest_corpus
from src.rag_pipeline import ask as run_ask
from src.retrieval.reranker import RerankedResult
from src.retrieval.sparse import bm25_index_exists

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(request_id)s] %(message)s")
logging.getLogger().addFilter(_RequestIdFilter())
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Hybrid Search", version="1.0.0")
client = AsyncOpenAI()
app.state.indexes_ready = False


def _indexes_exist() -> bool:
    chroma_has_chunks = chromadb.PersistentClient(path=settings.chroma_db_path).get_or_create_collection("chunks").count() > 0
    return chroma_has_chunks and bm25_index_exists(settings.bm25_index_path)


async def _seed_indexes() -> None:
    logger.info("First boot detected, ingesting docs corpus...")
    await ingest_corpus(client, settings.default_chunking_strategy)
    app.state.indexes_ready = True
    logger.info("Corpus ingestion complete, indexes ready.")


@app.on_event("startup")
async def seed_indexes_on_first_boot() -> None:
    if _indexes_exist():
        app.state.indexes_ready = True
        return
    # Run in the background so /health stays responsive (indexes_ready=false) while this ingests.
    asyncio.create_task(_seed_indexes())


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    token = _request_id_ctx.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        _request_id_ctx.reset(token)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.0fms)", request.method, request.url.path, getattr(response, "status_code", "?"), duration_ms)
    response.headers["X-Request-ID"] = request_id
    return response


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    use_sparse: bool = Field(True, description="False for dense-only retrieval (dashboard's hybrid-vs-dense-only comparison).")
    chunking_strategy: str = Field(default_factory=lambda: settings.default_chunking_strategy)


class AskResponse(BaseModel):
    answer: str
    citations: list[int]
    unsupported_citations: list[int]
    confidence_scores: ConfidenceScores
    chunks_used: list[RerankedResult]
    latency_ms: float
    cost_estimate: float


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path relative to the docs root, e.g. 'engineering/faq.md'.")
    chunking_strategy: ChunkingStrategy = Field(default_factory=lambda: settings.default_chunking_strategy)


class IngestResponse(BaseModel):
    chunks_created: int
    duplicates_skipped: int
    cost: float


class DocumentSummary(BaseModel):
    file_path: str
    chunk_count: int


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    index_size: int
    strategy_distribution: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    chroma_connected: bool
    bm25_loaded: bool
    indexes_ready: bool


@app.post("/v1/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest) -> AskResponse:
    result = await run_ask(body.question, client, top_k=body.top_k, use_sparse=body.use_sparse)
    return AskResponse(
        answer=result.answer,
        citations=result.verified_citations,
        unsupported_citations=result.unsupported_citations,
        confidence_scores=result.confidence,
        chunks_used=result.chunks_used,
        latency_ms=result.latency_ms,
        cost_estimate=result.cost_estimate_usd,
    )


@app.get("/v1/documents", response_model=list[DocumentSummary])
async def documents_endpoint() -> list[DocumentSummary]:
    chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
    collection = chroma_client.get_or_create_collection("chunks")
    if collection.count() == 0:
        return []
    all_metadata = collection.get()["metadatas"]
    counts = Counter(m["file_path"] for m in all_metadata)
    return [DocumentSummary(file_path=path, chunk_count=count) for path, count in sorted(counts.items())]


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_endpoint(body: IngestRequest) -> IngestResponse:
    docs_root = Path(settings.docs_path)
    document = load_document(docs_root / body.file_path, docs_root)
    similarity_fn = local_sentence_similarity_fn() if body.chunking_strategy == "semantic" else None
    chunks = chunk_document(document, body.chunking_strategy, similarity_fn)
    stats: IngestionStats = await embed_and_store(chunks, client)
    return IngestResponse(chunks_created=stats.chunks_stored, duplicates_skipped=stats.duplicates_skipped, cost=stats.embedding_cost_usd)


@app.get("/v1/stats", response_model=StatsResponse)
async def stats_endpoint() -> StatsResponse:
    chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
    collection = chroma_client.get_or_create_collection("chunks")
    total_chunks = collection.count()
    if total_chunks == 0:
        return StatsResponse(total_chunks=0, total_documents=0, index_size=0, strategy_distribution={})
    all_metadata = collection.get()["metadatas"]
    total_documents = len({m["file_path"] for m in all_metadata})
    strategy_distribution = dict(Counter(m["strategy_used"] for m in all_metadata))
    return StatsResponse(
        total_chunks=total_chunks, total_documents=total_documents, index_size=total_chunks, strategy_distribution=strategy_distribution
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    chroma_connected = True
    try:
        chromadb.PersistentClient(path=settings.chroma_db_path).heartbeat()
    except Exception:
        chroma_connected = False

    bm25_loaded = bm25_index_exists(settings.bm25_index_path)

    status = "ok" if chroma_connected else "degraded"
    return HealthResponse(
        status=status, chroma_connected=chroma_connected, bm25_loaded=bm25_loaded, indexes_ready=app.state.indexes_ready
    )
