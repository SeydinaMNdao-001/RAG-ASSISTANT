"""
chunking.py — Découpage de texte en passages ("chunks").

Pourquoi ne pas juste couper tous les 800 caractères, point ?
Parce qu'on risquerait de couper une clause de contrat ou une idée en deux
morceaux distincts, dans deux passages différents. La recherche vectorielle
chercherait alors sur des bouts de phrases incomplets.

La stratégie ici : découper d'abord par PARAGRAPHE (séparés par une ligne
vide), puis regrouper les paragraphes tant que ça tient dans la taille
maximale. Un paragraphe qui dépasse à lui seul la taille maximale est alors
découpé avec une fenêtre glissante (chevauchement).
"""

from __future__ import annotations

import re

# Une "coupure de paragraphe" = une ou plusieurs lignes vides d'affilée.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Découpe un texte trop long (un seul paragraphe géant, par exemple) en
    tranches de taille fixe, avec un chevauchement entre chaque tranche.

    Le chevauchement évite qu'une information pile à la frontière entre deux
    tranches soit coupée et perde son contexte.
    """
    normalized = " ".join(text.split())
    pieces = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        pieces.append(normalized[start:end])
        start += chunk_size - overlap
    return pieces


def smart_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Découpe `text` en une liste de passages.

    Règle : on regroupe les paragraphes successifs tant que leur taille
    cumulée reste sous `chunk_size`. Dès que ça dépasserait, on "ferme" le
    passage en cours et on en commence un nouveau.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]

    if not paragraphs:
        stripped = text.strip()
        paragraphs = [stripped] if stripped else []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) > chunk_size:
            chunks.extend(_split_long_text(para, chunk_size, overlap))
        else:
            current = para

    if current:
        chunks.append(current)

    return chunks