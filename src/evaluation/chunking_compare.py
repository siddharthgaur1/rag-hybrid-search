"""Ingest the corpus with all 3 chunking strategies, run the golden QA suite
against each, and produce a side-by-side comparison.

Usage: python -m src.evaluation.chunking_compare
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.config import ROOT, settings
from src.ingestion.chunker import ChunkingStrategy
from src.ingestion.pipeline import ingest_corpus
from src.evaluation.evaluator import run_eval_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES: list[ChunkingStrategy] = ["fixed_size", "structure_aware", "semantic"]


def _index_paths_for(strategy: str) -> tuple[Path, Path]:
    base = ROOT / "data" / "chunking_compare" / strategy
    return base / "chroma_db", base / "bm25_index"


async def compare_chunking_strategies() -> dict:
    """Ingest + eval each strategy in isolated indexes, and write chunking_comparison.json."""
    load_dotenv()
    client = AsyncOpenAI()
    golden_qa = json.loads((ROOT / "src" / "evaluation" / "golden_qa.json").read_text(encoding="utf-8"))

    report_rows = []
    for strategy in STRATEGIES:
        chroma_path, bm25_path = _index_paths_for(strategy)
        logger.info("Ingesting with strategy=%s", strategy)
        ingestion_stats = await ingest_corpus(client, strategy, chroma_db_path=chroma_path, bm25_index_path=bm25_path)

        run_id = f"{strategy}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        summary = await run_eval_suite(golden_qa, client, strategy, run_id, chroma_db_path=chroma_path, bm25_index_path=bm25_path)
        total_cost = ingestion_stats.embedding_cost_usd + summary.total_cost_usd

        report_rows.append(
            {
                "strategy": strategy,
                "chunks_stored": ingestion_stats.chunks_stored,
                "answer_correctness": summary.aggregate["answer_correctness"],
                "faithfulness": summary.aggregate["faithfulness"],
                "retrieval_relevance": summary.aggregate["retrieval_relevance"],
                "citation_accuracy": summary.aggregate["citation_accuracy"],
                "unanswerable_detection": summary.aggregate["unanswerable_detection"],
                "pass_rate": summary.aggregate["pass_rate"],
                "cost_usd": total_cost,
            }
        )

    output = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "strategies": report_rows}
    output_path = ROOT / "data" / "chunking_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Chunking comparison written to %s", output_path)
    return output


def main() -> None:
    asyncio.run(compare_chunking_strategies())


if __name__ == "__main__":
    main()
