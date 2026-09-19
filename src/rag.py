"""
rag.py — Pipeline de réponse aux questions.

1. Charge l'index (reconstruit en mémoire depuis embeddings.npy + metadata.pkl)
2. Recherche vectorielle FAISS -> top candidats (large filet)
3. Reranking par cross-encoder -> passages vraiment pertinents (précision fine)
4. Si aucun passage n'est assez pertinent -> réponse honnête "je ne sais pas"
5. Sinon -> génération de la réponse par Claude, avec citation des sources
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from anthropic import Anthropic
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    CONFIDENCE_THRESHOLD,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
    MAX_TOKENS,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    SYSTEM_PROMPT,
)
from src.ingest import load_store
from src.models import Chunk

NO_ANSWER_MESSAGE = (
    "Je ne trouve pas cette information dans les documents indexés. "
    "Essaie de reformuler ta question, ou vérifie que le document concerné a bien été ajouté."
)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk]
    confident: bool


class IndexNotFoundError(RuntimeError):
    """Levée quand aucun document n'a encore été indexé."""


_store_lock = threading.Lock()
_cache: dict = {"mtime": None, "index": None, "chunks": None}


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _reranker_model() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def _current_version() -> float | None:
    """Date de modification du manifeste = signal de fraîcheur : si le
    watcher a mis à jour l'index, on le détecte ici et on recharge."""
    if MANIFEST_PATH.exists():
        return MANIFEST_PATH.stat().st_mtime
    return None


def _load_index_cached():
    with _store_lock:
        version = _current_version()
        if version is None:
            raise IndexNotFoundError(
                "Aucun document indexé pour l'instant. Ajoute un fichier dans data/."
            )

        if _cache["mtime"] == version and _cache["index"] is not None:
            return _cache["index"], _cache["chunks"]

        embeddings, chunks = load_store()
        if embeddings.shape[0] == 0:
            raise IndexNotFoundError("L'index est vide : aucun chunk n'a pu être extrait des documents.")

        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        _cache.update({"mtime": version, "index": index, "chunks": chunks})
        return index, chunks


def retrieve(question: str, retrieval_k: int = RETRIEVAL_TOP_K, rerank_k: int = RERANK_TOP_K) -> list[RetrievedChunk]:
    """Recherche vectorielle (large filet) suivie d'un reranking (précision fine)."""
    index, chunks = _load_index_cached()

    query_vec = _embedding_model().encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    k = min(retrieval_k, len(chunks))
    _, indices = index.search(query_vec, k)
    candidates = [chunks[i] for i in indices[0] if i != -1]

    if not candidates:
        return []

    pairs = [(question, c.text) for c in candidates]
    rerank_scores = _reranker_model().predict(pairs)

    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    top = ranked[:rerank_k]

    return [RetrievedChunk(chunk=c, score=float(s)) for c, s in top]


def _build_context(retrieved: list[RetrievedChunk]) -> str:
    parts = []
    for i, r in enumerate(retrieved, start=1):
        loc = f"{r.chunk.source} (catégorie : {r.chunk.category})"
        if r.chunk.page:
            loc += f", page {r.chunk.page}"
        parts.append(f"[Extrait {i} — source : {loc}]\n{r.chunk.text}")
    return "\n\n".join(parts)


def answer_question(question: str, rerank_k: int = RERANK_TOP_K) -> RagAnswer:
    """Pipeline complet : retrieval + reranking + seuil de confiance + génération."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY n'est pas définie. Ajoute-la dans un fichier .env "
            "(voir .env.example)."
        )

    retrieved = retrieve(question, rerank_k=rerank_k)

    if not retrieved or retrieved[0].score < CONFIDENCE_THRESHOLD:
        return RagAnswer(answer=NO_ANSWER_MESSAGE, sources=retrieved, confident=False)

    context = _build_context(retrieved)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Extraits de documents :\n\n{context}\n\nQuestion : {question}"}
        ],
    )

    answer_text = "".join(block.text for block in message.content if block.type == "text")

    return RagAnswer(answer=answer_text, sources=retrieved, confident=True)