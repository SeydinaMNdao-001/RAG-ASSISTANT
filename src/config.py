"""
config.py — Réglages centraux du projet.

Toutes les autres briques piochent leurs paramètres ici. Rien d'autre que des
constantes dans ce fichier : aucune logique.
"""

import os
from pathlib import Path

# ============================================================
# 1. CHEMINS DU PROJET
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# ============================================================
# 2. ORGANISATION DES DOCUMENTS
# ============================================================
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}

# ============================================================
# 3. DÉCOUPAGE DU TEXTE (chunking)
# ============================================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
VIRTUAL_PAGE_CHARS = 3000

# ============================================================
# 4. OCR (documents scannés / images)
# ============================================================
OCR_ENABLED = True
OCR_LANGUAGES = "fra+eng"
OCR_MIN_TEXT_LENGTH = 20
OCR_RENDER_DPI = 200

# ============================================================
# 5. RECHERCHE (embeddings + reranking)
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CONFIDENCE_THRESHOLD = -3.0

# ============================================================
# 6. SURVEILLANCE AUTOMATIQUE
# ============================================================
WATCH_DEBOUNCE_SECONDS = 2.0

# ============================================================
# 7. GÉNÉRATION (Claude) + lecture de la clé secrète
# ============================================================
ANTHROPIC_MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """Tu es un assistant documentaire qui répond UNIQUEMENT à partir des extraits fournis.

Règles strictes :
- Si la réponse ne se trouve pas dans les extraits, dis clairement que l'information n'est pas disponible dans les documents indexés. N'invente rien, ne suppose rien.
- Cite systématiquement la source de chaque affirmation (nom du fichier, catégorie, et page si disponible).
- Réponds en français, de façon claire, précise et structurée.
- Pour des documents comme des contrats, cite le numéro de clause/article si le texte le mentionne.
- Reste factuel et neutre : tu résumes et cites ce que disent les documents, tu ne donnes pas d'avis juridique ou financier.
"""