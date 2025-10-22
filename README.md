# fake_news_rag
Système RAG pour la détection de fake news 

## Structure projet (Modèle C4)
<details>
    <summary> Modèle C4 </summary>
### Visualisation figma
[Schéma projet Fake News Rag](https://www.figma.com/board/Cv7FSdZAQXQ49bazw3m81t/Sans-titre?node-id=0-1&t=elai7Kr7pmQxHEUO-1)

### C1 - Contexte système
```mermaid
graph TD
    U[Utilisateur] -->|Pose une question| RAG[RAG Fake News App]
    RAG -->|Recherche contextuelle| LLM[Ollama LLM local]
    RAG -->|Interroge| DB[ChromaDB - base vectorielle locale]
    DB --> RAG
    LLM --> RAG
    RAG -->|Renvoie une reponse| U

```
### C2 - Containers

```mermaid
graph TD
    U[Utilisateur] --> CLI[Interface CLI]
    CLI --> PRE[Preprocessing - Nettoyage et Tokenisation]
    PRE --> EMB[Embedding - Vectorisation des textes]
    EMB --> CHROMA[ChromaDB - Stockage vectoriel]
    CHROMA --> RETRIEVAL[Retrieval - Recherche de contexte]
    RETRIEVAL --> LLM[Ollama LLM local]
    LLM --> REP[Réponse générée]
    REP --> CLI
```

### C3 - Composants internes

```mermaid
graph TD
    subgraph src/
        CLI[cli.py\nInterface utilisateur]
        PIPE[rage_pipeline.py\nCoordination du flux RAG]
        PRE[preprocessing.py\nNettoyage et tokenisation]
        EMB[embedding.py\nGénération des embeddings]
        CH[storage_chroma.py\nGestion de la base vectorielle]
        RETR[retrieval.py\nRecherche des documents pertinents]
    end

    CLI --> PIPE
    PIPE --> PRE
    PIPE --> EMB
    PIPE --> CH
    PIPE --> RETR
    PIPE --> CLI
```

### C4 - Vue Code

```mermaid
graph TD
    CLI_PY[cli.py] --> RAG_PY[rag_pipeline.py]
    RAG_PY --> RETRIEVE_PY[retrieval.py]
    RAG_PY --> EMBED_PY[embedding.py]
    RAG_PY --> STORE_PY[storage_chroma.py]
    RAG_PY --> PREPROCESS_PY[preprocessing.py]

```
</details>
## Diagramme de séquence

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant CLI as Interface CLI / App
    participant PRE as Preprocessing
    participant CH as ChromaDB
    participant EMB as Embeddings (Ollama)
    participant LLM as LLM (Ollama)

    U->>CLI: Saisit une question
    CLI->>PRE: Nettoie et tokenise la requete
    PRE-->>CLI: Requete pretraitee

    CLI->>EMB: Genere un vecteur pour la requete
    EMB-->>CLI: Retourne le vecteur d embedding

    CLI->>CH: Recherche les documents similaires
    CH-->>CLI: Retourne les passages les plus pertinents

    CLI->>LLM: Envoie la question + contexte pertinent
    LLM-->>CLI: Genere une reponse contextuelle

    CLI-->>U: Affiche la reponse finale

```

## Installation

### Créer un environnement virtuel

```bash
cd ~/fake_news_rag
python3 -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```
## Installer Ollama

### Installation

```bash
sudo snap install ollama

```

### Vérification de l'installation

```bash
ollama list

```

### Installer un LLM

```bash
# Modèle pour l'embedding pour ChromaDB
ollama pull all-minilm
# Modèle plus 'gros'
ollama pull llama3.2
```

### Vérifier que le modèle réponde

```bash
cd ~/tests
python test_ollama.py
```

## Architecture du projet

```
rag-fake-news/
├─ data/
│  ├─ raw/                 # Données brutes
│  │   ├─ true.csv
│  │   └─ fake.csv
│  └─ processed/           # Données nettoyées pour l'embedding
├─ src/
│  ├─ preprocessing.py
│  ├─ embedding.py
│  ├─ storage_chroma.py
│  ├─ retrieval.py
│  ├─ rag_pipeline.py
│  └─ cli.py
├─ tests/                    # Destiné aux tests
│  ├─ test_preprocessing.py
│  ├─ test_embedding.py
│  ├─ test_retrieval.py
│  └─ ...
├─ notebooks/
├─ requirements.txt
└─ README.md

```
