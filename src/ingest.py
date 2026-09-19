"""
ingest.py — Indexation incrémentale des documents.

Un "manifeste" (index/manifest.json) garde une empreinte (hash) de chaque
fichier déjà indexé. À chaque appel de sync() :
- fichier nouveau ou modifié (hash différent) -> extrait, découpé, embeddé
- fichier supprimé -> ses passages sont retirés de l'index
- fichier inchangé -> ignoré, aucun recalcul

Les embeddings et métadonnées sont stockés sur disque (embeddings.npy +
metadata.pkl). L'index FAISS lui-même n'est PAS stocké ici : il est
reconstruit à la volée dans rag.py, à partir de ces fichiers.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import smart_chunk
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    MANIFEST_PATH,
    METADATA_PATH,
    SUPPORTED_EXTENSIONS,
    VIRTUAL_PAGE_CHARS,
)
from src.extract import extract_document

# Empêche deux sync() de tourner en même temps (ex : ajout + suppression
# rapprochés déclenchés par le watcher dans deux threads différents).
_sync_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Charge le modèle d'embeddings une seule fois, puis le réutilise."""
    return SentenceTransformer(EMBEDDING_MODEL)


@dataclass
class Chunk:
    text: str
    source: str          # nom du fichier, pour l'affichage
    category: str         # sous-dossier de data/ (contrats, factures, ...)
    path: str              # chemin relatif à data/, clé stable du fichier d'origine
    page: int | None = None  # réel pour un PDF, virtuel pour un DOCX/TXT


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def file_hash(path: Path) -> str:
    """Empreinte SHA-256 d'un fichier, pour détecter s'il a changé."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def relative_key(path: Path) -> str:
    """Clé stable utilisée dans le manifeste : chemin relatif à data/."""
    return str(path.relative_to(DATA_DIR))


def category_of(path: Path) -> str:
    """La catégorie d'un document = son sous-dossier direct dans data/."""
    rel = path.relative_to(DATA_DIR)
    return rel.parts[0] if len(rel.parts) > 1 else "autres"


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_manifest(manifest: dict) -> None:
    _atomic_write_json(MANIFEST_PATH, manifest)


def load_store() -> tuple[np.ndarray, list[Chunk]]:
    """Charge les embeddings + métadonnées existants (vide si rien encore indexé)."""
    if EMBEDDINGS_PATH.exists() and METADATA_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(METADATA_PATH, "rb") as f:
            chunks: list[Chunk] = pickle.load(f)
        return embeddings, chunks
    return np.zeros((0, 0), dtype=np.float32), []


def save_store(embeddings: np.ndarray, chunks: list[Chunk]) -> None:
    # np.save() ajoute automatiquement ".npy" si le nom ne s'y termine pas déjà.
    # On nomme donc le fichier temporaire pour qu'il se termine déjà par ".npy".
    tmp_emb = EMBEDDINGS_PATH.parent / f"{EMBEDDINGS_PATH.stem}.tmp.npy"
    np.save(tmp_emb, embeddings)
    tmp_emb.replace(EMBEDDINGS_PATH)

    tmp_meta = METADATA_PATH.with_suffix(".pkl.tmp")
    with open(tmp_meta, "wb") as f:
        pickle.dump(chunks, f)
    tmp_meta.replace(METADATA_PATH)


def _scan_files() -> list[Path]:
    return sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _remove_chunks_for(key: str, embeddings: np.ndarray, chunks: list[Chunk]) -> tuple[np.ndarray, list[Chunk]]:
    keep_mask = [c.path != key for c in chunks]
    new_chunks = [c for c, keep in zip(chunks, keep_mask) if keep]
    if embeddings.shape[0] == 0:
        return embeddings, new_chunks
    return embeddings[keep_mask], new_chunks


def sync(data_dir: Path = DATA_DIR, verbose: bool = True) -> SyncResult:
    """Synchronise l'index avec le contenu actuel de data/ (incrémental).

    Protégé par un verrou : si deux déclenchements arrivent presque en même
    temps, le second attend que le premier ait terminé et sauvegardé, au
    lieu de tourner en parallèle sur un état pas encore à jour.
    """
    with _sync_lock:
        return _sync_locked(data_dir=data_dir, verbose=verbose)


def _sync_locked(data_dir: Path, verbose: bool) -> SyncResult:
    manifest = load_manifest()
    embeddings, chunks = load_store()

    current_files = _scan_files()
    current_keys = {relative_key(p) for p in current_files}
    known_keys = set(manifest.keys())

    result = SyncResult()
    to_process: list[Path] = []

    for path in current_files:
        key = relative_key(path)
        new_hash = file_hash(path)
        old_entry = manifest.get(key)
        if old_entry is None:
            result.added.append(key)
            to_process.append(path)
        elif old_entry["hash"] != new_hash:
            result.updated.append(key)
            to_process.append(path)

    removed_keys = known_keys - current_keys
    for key in removed_keys:
        result.removed.append(key)
        embeddings, chunks = _remove_chunks_for(key, embeddings, chunks)
        manifest.pop(key, None)

    for path in to_process:
        embeddings, chunks = _remove_chunks_for(relative_key(path), embeddings, chunks)

    if not to_process and not removed_keys:
        if verbose:
            print("Aucun changement détecté. Index déjà à jour.")
        return result

    new_chunks: list[Chunk] = []

    for path in to_process:
        key = relative_key(path)
        category = category_of(path)
        if verbose:
            print(f"Indexation : {key} (catégorie : {category})")

        try:
            pages = extract_document(path)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  Échec d'extraction pour {key} : {e}")
            continue

        virtual_offset = 0
        for text, page in pages:
            for piece in smart_chunk(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
                if page is not None:
                    resolved_page = page
                else:
                    resolved_page = virtual_offset // VIRTUAL_PAGE_CHARS + 1
                    virtual_offset += len(piece)
                new_chunks.append(
                    Chunk(text=piece, source=path.name, category=category, path=key, page=resolved_page)
                )

        manifest[key] = {"hash": file_hash(path), "category": category}

    if new_chunks:
        if verbose:
            print(f"Chargement du modèle d'embeddings ({EMBEDDING_MODEL}) ...")
        model = _get_embedding_model()

        texts = [c.text for c in new_chunks]
        new_embeddings = model.encode(texts, convert_to_numpy=True).astype(np.float32)

        embeddings = new_embeddings if embeddings.shape[0] == 0 else np.vstack([embeddings, new_embeddings])
        chunks = chunks + new_chunks

    save_store(embeddings, chunks)
    save_manifest(manifest)

    if verbose:
        print(
            f"Synchronisation terminée : {len(result.added)} ajouté(s), "
            f"{len(result.updated)} modifié(s), {len(result.removed)} supprimé(s). "
            f"Total : {len(chunks)} chunks."
        )

    return result


if __name__ == "__main__":
    sync()