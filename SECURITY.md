# Security

## Threat model

A two-service hybrid-RAG system: a FastAPI backend that ingests documents, embeds
them, and answers questions, plus a Streamlit dashboard that talks to it over HTTP.
Untrusted inputs are the documents ingested and the questions asked; the answering
LLM reads retrieved chunks. Operator-run, no authentication.

## What is mitigated

| Risk | Status | Where |
|---|---|---|
| Secrets in git history | **Clean** — `gitleaks`: 0 findings; no `.env` ever tracked |
| Dependency CVEs (excl. Chroma, below) | **Clean** — `pip-audit`: no other known vulnerabilities |
| Container running as root | **Mitigated** — both images run as `appuser` | `Dockerfile:12`, `dashboard/Dockerfile:10` |
| Dashboard → API calls hanging | **Mitigated** — `httpx` calls set explicit timeouts | `dashboard/app.py` |
| Untrusted `pickle.load` | **Mitigated by construction** — the only unpickled file is `bm25_index.pkl`, which this service **writes itself** during ingestion (`src/ingestion/embedder.py`). It is never downloaded or user-supplied. Do not point `bm25_index_path` at an untrusted file. | `src/retrieval/sparse.py:35` |
| ChromaDB server RCE (PYSEC-2026-311) | **Not applicable** — the advisory targets Chroma's HTTP **server** API; this uses an embedded `PersistentClient` and runs no server. No upstream fix exists yet. | `src/api/main.py`, `src/retrieval/dense.py` |

## What is NOT mitigated / notes

- **No authentication or CORS restriction** on the FastAPI service. Bind it to
  localhost or add auth before exposing it.
- **No rate limiting.** The ingest and query endpoints will happily consume CPU and,
  with a paid provider, API budget under load.
- **Prompt injection via ingested documents** — the standard RAG risk. A document can
  carry instructions the answering model reads as context. Confidence scoring and
  citation coverage surface *ungrounded* answers but do not prevent a document from
  steering the model. Treat answers over untrusted corpora as advisory.

## Reporting

Open an issue. Portfolio/demo project, no production deployment, no security SLA.
