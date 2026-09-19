"""Tests pour src/chunking.py (aucune dépendance lourde requise)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import smart_chunk  # noqa: E402


def test_short_text_returns_single_chunk():
    text = "Ceci est un texte court."
    assert smart_chunk(text, chunk_size=800, overlap=150) == [text]


def test_empty_text_returns_empty_list():
    assert smart_chunk("", chunk_size=800, overlap=150) == []


def test_respects_chunk_size_for_long_single_paragraph():
    text = "a" * 2000
    chunks = smart_chunk(text, chunk_size=800, overlap=150)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) > 1


def test_keeps_short_paragraphs_together():
    text = "Premier paragraphe court.\n\nDeuxième paragraphe court."
    chunks = smart_chunk(text, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert "Premier paragraphe" in chunks[0]
    assert "Deuxième paragraphe" in chunks[0]


def test_splits_when_paragraphs_exceed_chunk_size():
    para_a = "A" * 500
    para_b = "B" * 500
    text = f"{para_a}\n\n{para_b}"
    chunks = smart_chunk(text, chunk_size=800, overlap=150)
    assert len(chunks) >= 2


def test_overlap_preserves_boundary_content():
    text = "x" * 795 + " MOTCLE " + "y" * 795
    chunks = smart_chunk(text, chunk_size=800, overlap=150)
    assert any("MOTCLE" in c for c in chunks)