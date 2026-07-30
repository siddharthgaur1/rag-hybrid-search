"""Regression test for the path-traversal fix in load_document."""

from pathlib import Path

import pytest

from src.ingestion.loader import load_document


def test_load_document_rejects_path_outside_docs_root(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("top secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes docs root"):
        load_document(docs_root / ".." / "secret.md", docs_root)


def test_load_document_allows_path_inside_docs_root(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    doc = docs_root / "note.md"
    doc.write_text("# Title\nbody", encoding="utf-8")

    result = load_document(doc, docs_root)
    assert result.metadata.file_path == "note.md"
