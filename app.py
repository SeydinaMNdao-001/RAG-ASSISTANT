"""
app.py — Interface : chat + visionneuse de documents.

Lancer avec :
    streamlit run app.py

Le dossier data/ est surveillé en continu : dépose un fichier dans une des
catégories et il est indexé automatiquement, sans rien redémarrer.
"""

from dotenv import load_dotenv
load_dotenv()  # DOIT être avant les imports de src.*, qui lisent les variables d'environnement

from collections import defaultdict

import streamlit as st

from src.config import DATA_DIR
from src.ingest import load_manifest, sync
from src.rag import IndexNotFoundError, answer_question
from src.viewer import PageOutOfRangeError, count_pages, get_page, resolve_path
from src.watcher import start_watcher

st.set_page_config(page_title="Assistant Documents", page_icon="📄", layout="centered")


@st.cache_resource
def _ensure_watcher_running():
    """Démarre la surveillance automatique une seule fois par session serveur,
    et fait une première synchronisation pour les fichiers déjà présents."""
    sync(verbose=False)
    return start_watcher()


_ensure_watcher_running()

# --- État partagé entre les deux vues (chat / visionneuse) ---
st.session_state.setdefault("vue", "💬 Assistant")
st.session_state.setdefault("viewer_doc", None)   # chemin relatif du document ouvert
st.session_state.setdefault("viewer_page", 1)
st.session_state.setdefault("messages", [])


def _open_in_viewer(relative_path: str, page: int | None) -> None:
    """Bascule vers la visionneuse, ouverte sur le document et la page donnés."""
    st.session_state["viewer_doc"] = relative_path
    st.session_state["viewer_page"] = page or 1
    st.session_state["vue"] = "📁 Parcourir les documents"
    st.rerun()


# ============================================================
# BARRE LATÉRALE
# ============================================================
with st.sidebar:
    st.header("Navigation")
    st.radio("Vue", ["💬 Assistant", "📁 Parcourir les documents"], key="vue")

    st.divider()
    st.header("Documents indexés")
    manifest = load_manifest()
    if manifest:
        by_category_count = defaultdict(int)
        for entry in manifest.values():
            by_category_count[entry["category"]] += 1
        for category, count in sorted(by_category_count.items()):
            st.write(f"**{category}** — {count} document(s)")
    else:
        st.info("Aucun document indexé pour l'instant.")

    if st.button("🔄 Forcer une resynchronisation"):
        with st.spinner("Synchronisation..."):
            result = sync(verbose=False)
        st.success(
            f"{len(result.added)} ajouté(s), {len(result.updated)} modifié(s), "
            f"{len(result.removed)} supprimé(s)."
        )
        st.rerun()

    st.divider()
    st.caption(f"Dossier surveillé : `{DATA_DIR}`")


# ============================================================
# VUE 1 : ASSISTANT (CHAT)
# ============================================================
def _render_sources(sources, key_prefix: str) -> None:
    with st.expander("Sources utilisées"):
        for i, s in enumerate(sources):
            loc = f"{s.chunk.source} — {s.chunk.category}"
            if s.chunk.page:
                loc += f", page {s.chunk.page}"
            st.markdown(f"**{loc}** — score {s.score:.2f}")
            st.caption(s.chunk.text[:300] + "…")
            if st.button("📖 Voir cette page", key=f"{key_prefix}_{i}"):
                _open_in_viewer(s.chunk.path, s.chunk.page)


def render_chat_view() -> None:
    st.title("📄 Assistant documentaire")
    st.caption(
        "Pose une question en français. L'assistant répond uniquement à partir des "
        "documents indexés, avec leurs sources — dépose un nouveau fichier dans `data/`, "
        "il est pris en compte automatiquement."
    )

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"], key_prefix=f"hist_{i}")

    question = st.chat_input("Pose ta question sur les documents indexés…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents et génération de la réponse…"):
                try:
                    result = answer_question(question)
                    st.markdown(result.answer)
                    if result.sources:
                        _render_sources(result.sources, key_prefix="new")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result.answer, "sources": result.sources}
                    )
                except IndexNotFoundError as e:
                    st.error(str(e))
                except RuntimeError as e:
                    st.error(str(e))


# ============================================================
# VUE 2 : PARCOURIR LES DOCUMENTS
# ============================================================
def render_browse_view() -> None:
    st.title("📁 Parcourir les documents")

    manifest = load_manifest()
    if not manifest:
        st.info("Aucun document indexé pour l'instant. Ajoute un fichier dans `data/`.")
        return

    by_category = defaultdict(list)
    for relative_path, entry in manifest.items():
        by_category[entry["category"]].append(relative_path)

    categories = sorted(by_category.keys())

    # Pré-sélection si on arrive depuis un clic "Voir cette page"
    preselected_path = st.session_state.get("viewer_doc")
    preselected_category = None
    for cat, paths in by_category.items():
        if preselected_path in paths:
            preselected_category = cat
            break

    category = st.selectbox(
        "Catégorie",
        categories,
        index=categories.index(preselected_category) if preselected_category in categories else 0,
    )

    docs_in_category = sorted(by_category[category])
    default_index = docs_in_category.index(preselected_path) if preselected_path in docs_in_category else 0
    relative_path = st.selectbox("Document", docs_in_category, index=default_index)

    path = resolve_path(relative_path)
    if not path.exists():
        st.error("Ce fichier n'existe plus sur le disque (il a peut-être été supprimé).")
        return

    total_pages = count_pages(path)
    default_page = st.session_state.get("viewer_page", 1)
    default_page = min(max(default_page, 1), total_pages)

    page_number = st.number_input(
        f"Page (1 à {total_pages})", min_value=1, max_value=total_pages, value=default_page
    )

    st.session_state["viewer_doc"] = relative_path
    st.session_state["viewer_page"] = page_number

    try:
        kind, content = get_page(path, page_number)
        if kind == "image":
            st.image(content, use_container_width=True)
        else:
            st.markdown(f"```\n{content}\n```")
    except PageOutOfRangeError as e:
        st.error(str(e))


# ============================================================
# ROUTAGE ENTRE LES DEUX VUES
# ============================================================
if st.session_state["vue"] == "💬 Assistant":
    render_chat_view()
else:
    render_browse_view()