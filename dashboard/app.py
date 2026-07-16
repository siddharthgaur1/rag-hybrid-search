"""Streamlit dashboard: ask questions, browse the index, review eval results, watch system stats.

Talks to the FastAPI service over HTTP (API_BASE_URL) rather than importing
the pipeline/Chroma directly — on Railway the API and dashboard are separate
services with separate filesystems, so this is the only surface that can see
the API's live index.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RAG Hybrid Search", layout="wide")


def api_get(path: str, **kwargs) -> httpx.Response | None:
    try:
        return httpx.get(f"{API_BASE_URL}{path}", timeout=10, **kwargs)
    except httpx.HTTPError:
        return None


def api_post(path: str, **kwargs) -> httpx.Response | None:
    try:
        return httpx.post(f"{API_BASE_URL}{path}", timeout=60, **kwargs)
    except httpx.HTTPError:
        return None


health_response = api_get("/health")
health = health_response.json() if health_response is not None and health_response.status_code == 200 else None

status_col, url_col = st.columns([1, 3])
if health is None:
    status_col.error("API unreachable")
elif not health["indexes_ready"]:
    status_col.warning("Setting up indexes, please wait...")
else:
    status_col.success("API connected")
url_col.caption(f"API: {API_BASE_URL}")

if health is None:
    st.error(f"Could not reach the API at {API_BASE_URL}. Check API_BASE_URL and that the API service is running.")
    st.stop()
if not health["indexes_ready"]:
    st.info("The API is still ingesting the docs corpus on first boot. This page will work once indexing finishes — refresh in a bit.")
    st.stop()

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
            hybrid_response = api_post("/v1/ask", json={"question": question, "top_k": top_k, "use_sparse": True})
            dense_response = api_post("/v1/ask", json={"question": question, "top_k": top_k, "use_sparse": False}) if compare_toggle else None

        if hybrid_response is None or hybrid_response.status_code != 200:
            st.error("The API request failed. Is the API service up?")
        else:
            results = [hybrid_response.json()]
            labels = ["Hybrid"]
            if dense_response is not None and dense_response.status_code == 200:
                results.append(dense_response.json())
                labels.append("Dense-only")

            cols = st.columns(2) if len(results) == 2 else [st.container()]
            for col, result, label in zip(cols, results, labels):
                with col:
                    st.subheader(label)
                    st.write(result["answer"])
                    st.caption(f"Verified citations: {result['citations']} | Unsupported: {result['unsupported_citations']}")
                    st.caption(f"Latency: {result['latency_ms']:.0f}ms | Est. cost: ${result['cost_estimate']:.5f}")

                    conf = result["confidence_scores"]
                    conf_df = pd.DataFrame(
                        {
                            "signal": ["retrieval", "citation", "completeness", "composite"],
                            "score": [conf["retrieval_confidence"], conf["citation_coverage"], conf["answer_completeness"], conf["composite"]],
                        }
                    )
                    st.plotly_chart(px.bar(conf_df, x="signal", y="score", range_y=[0, 1]), use_container_width=True)

                    st.write("Retrieved chunks:")
                    for i, chunk in enumerate(result["chunks_used"], start=1):
                        with st.expander(f"[{i}] {chunk['chunk_id']} (rerank score {chunk['rerank_score']:.3f})"):
                            st.write(chunk["content"])

with tab_docs:
    st.header("Indexed documents")
    docs_response = api_get("/v1/documents")
    if docs_response is not None and docs_response.status_code == 200:
        docs = docs_response.json()
        if docs:
            st.dataframe(pd.DataFrame(docs), use_container_width=True)
        else:
            st.info("No documents indexed yet.")
    else:
        st.warning("Could not reach the API to list documents.")

    st.subheader("Ingest a document")
    st.caption("The live demo is pre-loaded; use this to add more files from the API's docs/ corpus.")
    ingest_file_path = st.text_input("File path (relative to docs/, e.g. 'engineering/faq.md')")
    ingest_strategy = st.selectbox("Strategy", ["structure_aware", "fixed_size", "semantic"], key="ingest_strategy")
    if st.button("Ingest") and ingest_file_path:
        with st.spinner("Ingesting..."):
            ingest_response = api_post("/v1/ingest", json={"file_path": ingest_file_path, "chunking_strategy": ingest_strategy})
        if ingest_response is not None and ingest_response.status_code == 200:
            stats = ingest_response.json()
            st.success(f"Stored {stats['chunks_created']} chunks, skipped {stats['duplicates_skipped']} duplicates, cost ${stats['cost']:.4f}")
        else:
            st.error("Ingestion failed.")

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
    stats_response = api_get("/v1/stats")
    if stats_response is not None and stats_response.status_code == 200:
        stats = stats_response.json()
        st.metric("Total chunks", stats["total_chunks"])
        st.metric("Total documents", stats["total_documents"])
        if stats["strategy_distribution"]:
            st.plotly_chart(
                px.pie(values=list(stats["strategy_distribution"].values()), names=list(stats["strategy_distribution"].keys()), title="Strategy distribution"),
                use_container_width=True,
            )
    else:
        st.warning("Could not reach the API for system stats.")

    st.caption("Cost tracking and latency percentiles populate once eval runs or /v1/ask calls have been logged to data/eval_runs/.")
