"""
models.py — Structures de données partagées entre les modules.

Isolé dans son propre fichier (jamais exécuté comme script) pour que son nom
de module reste stable, quelle que soit la façon dont ingest.py est lancé.
Sans ça, pickle ne retrouve plus la classe Chunk au moment de la relire.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str          # nom du fichier, pour l'affichage
    category: str         # sous-dossier de data/ (contrats, factures, ...)
    path: str              # chemin relatif à data/, clé stable du fichier d'origine
    page: int | None = None  # réel pour un PDF, virtuel pour un DOCX/TXT