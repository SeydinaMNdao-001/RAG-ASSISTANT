"""
viewer.py — Ouverture et affichage d'une page précise d'un document.

Pour un PDF ou une image : rend la page sous forme d'image (PIL.Image).
Pour un DOCX/TXT/MD (pas de notion de page réelle) : découpe le texte en
"pages virtuelles" de VIRTUAL_PAGE_CHARS caractères — la même taille que
celle utilisée dans ingest.py pour numéroter les passages, pour rester
cohérent avec les numéros de page cités dans les réponses.

Utilisé par l'interface pour la visionneuse intégrée : ouvrir le document
d'origine à la page exacte citée dans une réponse, et feuilleter les pages
voisines.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.config import DATA_DIR, OCR_RENDER_DPI, VIRTUAL_PAGE_CHARS
from src.extract import extract_docx


class PageOutOfRangeError(ValueError):
    """Levée quand le numéro de page demandé n'existe pas dans le document."""


def resolve_path(relative_path: str) -> Path:
    """Reconstruit le chemin absolu d'un document à partir de sa clé relative
    (celle stockée sur chaque Chunk, ex: 'contrats/bail.pdf')."""
    return DATA_DIR / relative_path


def _full_text_for_pagination(path: Path) -> str:
    """Texte complet d'un document sans page réelle (DOCX/TXT/MD), utilisé
    uniquement pour découper/afficher des pages virtuelles."""
    if path.suffix.lower() == ".docx":
        pages = extract_docx(path)
        return pages[0][0] if pages else ""
    return path.read_text(encoding="utf-8", errors="ignore")


def count_pages(path: Path) -> int:
    """Nombre de pages d'un document : réel pour un PDF, 1 pour une image,
    virtuel (basé sur la longueur du texte) pour un DOCX/TXT/MD."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        try:
            return doc.page_count
        finally:
            doc.close()

    if suffix in (".png", ".jpg", ".jpeg"):
        return 1

    text = _full_text_for_pagination(path)
    return max(1, math.ceil(len(text) / VIRTUAL_PAGE_CHARS))


def get_page(path: Path, page_number: int) -> tuple[str, Image.Image | str]:
    """Retourne le contenu d'une page précise.

    Le premier élément du tuple indique le type de contenu retourné :
    "image" (à afficher avec st.image) ou "text" (à afficher avec st.markdown).
    """
    suffix = path.suffix.lower()
    total = count_pages(path)

    if page_number < 1 or page_number > total:
        raise PageOutOfRangeError(
            f"Page {page_number} demandée, mais ce document en a {total}."
        )

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        try:
            page = doc[page_number - 1]
            zoom = OCR_RENDER_DPI / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            return "image", image
        finally:
            doc.close()

    if suffix in (".png", ".jpg", ".jpeg"):
        return "image", Image.open(path)

    text = _full_text_for_pagination(path)
    start = (page_number - 1) * VIRTUAL_PAGE_CHARS
    end = page_number * VIRTUAL_PAGE_CHARS
    return "text", text[start:end]