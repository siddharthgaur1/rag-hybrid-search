"""Streamlit dashboard: ask questions, browse the index, review eval results, watch system stats."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import chromadb
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv()

from src.config import settings  # noqa: E402
from src.ingestion.chunker import ChunkingStrategy, chunk_document, local_sentence_similarity_fn  # noqa: E402
from src.ingestion.embedder import embed_and_store  # noqa: E402
from src.ingestion.loader import load_document, load_documents  # noqa: E402
from src.rag_pipeline import ask  # noqa: E402

st.set_page_config(page_title="RAG Hybrid Search", layout="wide")
client = AsyncOpenAI()


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


tab_ask, tab_docs, tab_eval, tab_stats = st.tabs(["Ask", "Documents", "Eval Results", "System Stats"])

with tab_ask:
    st.header("Ask a question")
    question = st.text_input("Question")
    col1, col2 = st.columns(2)
    top_k = col1.slider("top_k", 3, 10, 5)
    strategy = col2.selectbox("Chunking strategy (for display only — index must already use it)", ["structure_aware", "fixed_size", "semantic"])
    compare_toggle = st.toggle("Compare hybrid vs dense-only")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving and generating..."):
            hybrid_result = run_async(ask(question, client, top_k=top_k, use_sparse=True))
            dense_result = run_async(ask(question, client, top_k=top_k, use_sparse=False)) if compare_toggle else None

        cols = st.columns(2) if compare_toggle else [st.container()]
        for col, result, label in zip(cols, [hybrid_result, dense_result], ["Hybrid", "Dense-only"]):
            if result is None:
                continue
            with col:
                st.subheader(label)
                st.write(result.answer)
                st.caption(f"Verified citations: {result.verified_citations} | Unsupported: {result.unsupported_citations}")
                st.caption(f"Latency: {result.latency_ms:.0f}ms | Est. cost: ${result.cost_estimate_usd:.5f}")

                conf = result.confidence
                conf_df = pd.DataFrame(
                    {
                        "signal": ["retrieval", "citation", "completeness", "composite"],
                        "score": [conf.retrieval_confidence, conf.citation_coverage, conf.answer_completeness, conf.composite],
                    }
                )
                st.plotly_chart(px.bar(conf_df, x="signal", y="score", range_y=[0, 1]), use_container_width=True)

                st.write("Retrieved chunks:")
                for i, chunk in enumerate(result.chunks_used, start=1):
                    with st.expander(f"[{i}] {chunk.chunk_id} (rerank score {chunk.rerank_score:.3f})"):
                        st.write(chunk.content)

with tab_docs:
    st.header("Indexed documents")
    try:
        chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = chroma_client.get_or_create_collection("chunks")
        if collection.count() > 0:
            metadatas = collection.get()["metadatas"]
            doc_df = pd.DataFrame(metadatas)["file_path"].value_counts().reset_index()
            doc_df.columns = ["file_path", "chunk_count"]
            st.dataframe(doc_df, use_container_width=True)
        else:
            st.info("No documents indexed yet.")
    except Exception as e:
        st.warning(f"Could not read index: {e}")

    st.subheader("Re-ingest corpus")
    reingest_strategy: ChunkingStrategy = st.selectbox("Strategy", ["structure_aware", "fixed_size", "semantic"], key="reingest")
    if st.button("Re-ingest full corpus with this strategy"):
        with st.spinner("Ingesting..."):
            documents = load_documents(settings.docs_path)
            similarity_fn = local_sentence_similarity_fn() if reingest_strategy == "semantic" else None
            chunks = [c for doc in documents for c in chunk_document(doc, reingest_strategy, similarity_fn)]
            stats = run_async(embed_and_store(chunks, client))
        st.success(f"Stored {stats.chunks_stored} chunks, skipped {stats.duplicates_skipped} duplicates, cost ${stats.embedding_cost_usd:.4f}")

with tab_eval:
    st.header("Evaluation results")
    comparison_path = ROOT / "data" / "chunking_comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        strategy_df = pd.DataFrame(comparison["strategies"])
        st.subheader("Chunking strategy comparison")
        st.dataframe(strategy_df, use_container_width=True)
        metric_cols = ["answer_correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]
        melted = strategy_df.melt(id_vars="strategy", value_vars=metric_cols, var_name="metric", value_name="score")
        st.plotly_chart(px.bar(melted, x="metric", y="score", color="strategy", barmode="group"), use_container_width=True)
    else:
        st.info("Run `python -m src.evaluation.chunking_compare` to generate a comparison.")

    runs_dir = ROOT / "data" / "eval_runs"
    if runs_dir.exists():
        run_files = sorted(runs_dir.glob("*.json"))
        if run_files:
            selected = st.selectbox("Eval run", [f.name for f in run_files])
            run_data = json.loads((runs_dir / selected).read_text(encoding="utf-8"))
            results_df = pd.DataFrame(run_data["results"])
            f1, f2 = st.columns(2)
            passed_filter = f1.selectbox("Status", ["all", "passed", "failed"])
            filtered = results_df
            if passed_filter != "all":
                filtered = filtered[filtered["passed"] == (passed_filter == "passed")]
            st.dataframe(filtered, use_container_width=True)

with tab_stats:
    st.header("System stats")
    try:
        chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = chroma_client.get_or_create_collection("chunks")
        total_chunks = collection.count()
        st.metric("Total chunks", total_chunks)
        if total_chunks > 0:
            metadatas = collection.get()["metadatas"]
            st.metric("Total documents", len({m["file_path"] for m in metadatas}))
            strategy_counts = pd.Series([m["strategy_used"] for m in metadatas]).value_counts()
            st.plotly_chart(px.pie(values=strategy_counts.values, names=strategy_counts.index, title="Strategy distribution"), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not read index: {e}")

    st.caption("Cost tracking and latency percentiles populate once eval runs or /v1/ask calls have been logged to data/eval_runs/.")
