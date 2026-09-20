# 📄 Assistant documentaire RAG — universel & auto-actualisé

Un assistant conversationnel qui répond à des questions **en français**, uniquement à
partir des documents fournis — contrats, factures, rapports, PDF scannés, photos de
documents — avec **citation précise des sources**. L'index se met à jour **tout seul**
dès qu'un document est ajouté, modifié ou supprimé.

Projet pensé pour être présenté à une entreprise : une installation = un espace de
documents. Le multi-entreprises est une évolution possible, pas encore construite.

## État d'avancement

- [x] Structure du projet + configuration centrale (`config.py`)
- [x] Découpage du texte en passages (`chunking.py`)
- [x] Extraction PDF / DOCX / images + bascule OCR automatique (`extract.py`)
- [x] Indexation incrémentale avec manifeste (`ingest.py`)
- [x] Surveillance automatique du dossier `data/` (`watcher.py`)
- [x] Recherche + reranking + génération de réponse (`rag.py`) — génération via Groq (gratuit) pour l'instant, facilement remplaçable par Claude
- [ ] Visionneuse de documents page par page (`viewer.py`)
- [ ] Interface Streamlit complète (`app.py`)
- [ ] Déploiement (Hugging Face Spaces)

## Ce que le projet démontre

- Un pipeline **RAG (Retrieval-Augmented Generation)** complet.
- Une **indexation incrémentale** : ajouter un document ne relance pas tout le pipeline.
- Une **surveillance de dossier en temps réel** (`watchdog`), avec verrou anti-concurrence.
- De l'**OCR** pour les documents scannés ou photographiés, bascule automatique.
- Un **reranking** après la recherche vectorielle et un **seuil de confiance** qui fait
  dire "je ne sais pas" à l'assistant plutôt que d'inventer une réponse.

## Organisation des documents

```
data/
├── contrats/    → contrats, accords, conditions générales
├── factures/    → factures, devis, bons de commande
├── rapports/    → rapports d'activité, comptes-rendus
└── autres/      → tout ce qui ne rentre pas ailleurs
```

Chaque sous-dossier = une catégorie. Ajoute un nouveau sous-dossier pour créer une
nouvelle catégorie — aucune configuration requise.

## Stack technique

| Composant | Choix |
|---|---|
| Extraction | `pypdf`, `python-docx`, `PyMuPDF` |
| OCR | `pytesseract` |
| Embeddings | `sentence-transformers` (multilingue) |
| Recherche vectorielle | FAISS |
| Reranking | Cross-encoder multilingue |
| Surveillance | `watchdog` |
| Génération | Claude API (`claude-sonnet-5`) |
| Interface | Streamlit |

## Installation

```bash
git clone https://github.com/<ton-utilisateur>/rag-assistant.git
cd rag-assistant
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

L'OCR nécessite aussi le binaire **Tesseract** :

```bash
# macOS
brew install tesseract tesseract-lang
# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

Copie `.env.example` en `.env` et renseigne ta clé API Anthropic :

```bash
cp .env.example .env
```

## Utilisation

```bash
python -m src.ingest             # indexe les documents présents dans data/
python -m src.ingest --watch     # + surveille data/ en continu
```

L'interface Streamlit (`streamlit run app.py`) arrive dans une prochaine étape.

## Tests

```bash
pytest tests/
```

## Licence

MIT — voir [LICENSE](LICENSE).