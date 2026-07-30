"""Multi-format document loader: .md, .txt, .html -> normalized Document objects."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".html"}


class DocumentMetadata(BaseModel):
    """Provenance metadata attached to every loaded document and inherited by its chunks."""

    source_file: str = Field(..., description="Filename, e.g. 'api_reference.md'.")
    file_path: str = Field(..., description="Path relative to the docs root, e.g. 'engineering/api_reference.md'.")
    section_heading: str | None = Field(None, description="Top-level heading, if the format has one (markdown/html).")
    last_modified: str = Field(..., description="ISO 8601 timestamp of the file's mtime.")


class Document(BaseModel):
    """A single loaded, normalized document."""

    id: str = Field(..., description="Stable ID derived from the file's relative path.")
    content: str = Field(..., description="Normalized plaintext content, structure preserved.")
    raw_content: str = Field(..., description="Unmodified file contents as read from disk.")
    metadata: DocumentMetadata


def _document_id(relative_path: Path) -> str:
    return hashlib.sha1(str(relative_path).encode("utf-8")).hexdigest()[:16]


def _html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator="\n").strip()


def _first_heading(raw: str, suffix: str) -> str | None:
    if suffix == ".md":
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return None
    if suffix == ".html":
        soup = BeautifulSoup(raw, "html.parser")
        heading = soup.find(["h1", "h2", "h3"])
        return heading.get_text().strip() if heading else None
    return None


def load_document(path: Path, docs_root: Path) -> Document:
    """Load and normalize a single supported file into a Document.

    Raises:
        ValueError: if the path escapes docs_root or the extension isn't supported.
        FileNotFoundError: if the file doesn't exist.
    """
    docs_root = docs_root.resolve()
    path = path.resolve()
    if not path.is_relative_to(docs_root):
        # .resolve() collapses ".." before this check, so a body.file_path of
        # "../../etc/passwd" or an absolute path can't escape docs_root.
        raise ValueError(f"Path escapes docs root: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{path.suffix}' for {path}. Supported: {SUPPORTED_EXTENSIONS}")

    raw = path.read_text(encoding="utf-8")
    content = _html_to_text(raw) if path.suffix == ".html" else raw
    relative_path = path.relative_to(docs_root)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    return Document(
        id=_document_id(relative_path),
        content=content,
        raw_content=raw,
        metadata=DocumentMetadata(
            source_file=path.name,
            file_path=str(relative_path).replace("\\", "/"),
            section_heading=_first_heading(raw, path.suffix),
            last_modified=mtime,
        ),
    )


def load_documents(docs_root: str | Path) -> list[Document]:
    """Recursively load every supported file under docs_root.

    Raises:
        FileNotFoundError: if docs_root doesn't exist.
    """
    docs_root = Path(docs_root)
    if not docs_root.exists():
        raise FileNotFoundError(f"Docs root not found: {docs_root}")

    paths = sorted(
        p for p in docs_root.rglob("*") if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    )
    documents = [load_document(p, docs_root) for p in paths]
    logger.info("Loaded %d documents from %s", len(documents), docs_root)
    return documents
