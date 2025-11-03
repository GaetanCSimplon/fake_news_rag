# app.py

import streamlit as st
import re
from src.rag_pipeline import RAGPipeline

# NOTE: Les constantes sont généralement importées depuis main ou un fichier de config.
# Pour l'autonomie de l'application, nous les redéfinissons ici (assurez-vous qu'elles correspondent à main.py)
CHROMA_PATH = "data/vector_db"
COLLECTION_NAME = "articles"
EMBEDDING_MODEL = "all-minilm"
GENERATION_MODEL = "llama3.2"  # Ou 'phi3:mini'

# ===============================================
# 1. INITIALISATION DU PIPELINE (mis en cache)
# ===============================================


@st.cache_resource
def get_rag_pipeline():
    """Initialise et met en cache l'objet RAGPipeline."""
    try:
        rag_pipe = RAGPipeline(
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL,
        )
        return rag_pipe
    except Exception as e:
        st.error(
            f"Erreur d'initialisation : Vérifiez la connexion à ChromaDB et Ollama. Détail : {e}"
        )
        return None


rag_pipeline = get_rag_pipeline()

# ===============================================
# 2. CONFIGURATION DE L'INTERFACE
# ===============================================

st.set_page_config(
    page_title="RAG Fake News Detector", layout="wide", initial_sidebar_state="expanded"
)
st.title("📰 Système RAG de Détection de Fake News")
st.markdown(
    "Collez l'article à analyser pour obtenir un verdict et sa justification basés sur des articles de référence."
)


# BARRE LATÉRALE ET PARAMÈTRES
st.sidebar.header("Statut & Paramètres")

# if rag_pipeline:
#     try:
#         # 🟢 Appel la collection via la méthode get_collection()
#         count = rag_pipeline.retriever.get_collection().count()
#         st.sidebar.success(f"✅ DB Chroma en Ligne : {count} chunks stockés")
#     except Exception:
#         st.sidebar.error("❌ DB Chroma : Erreur de connexion/collection.")
# else:
#     st.sidebar.warning("Le RAG Pipeline n'a pas pu être initialisé.")

# Statut de la DB
if rag_pipeline and hasattr(rag_pipeline.retriever, "collection"):
    try:
        count = rag_pipeline.retriever.collection.count()
        st.sidebar.success(f"✅ DB Chroma en Ligne : {count} chunks stockés")
    except Exception:
        st.sidebar.error("❌ DB Chroma : Erreur de connexion/collection.")
else:
    st.sidebar.warning("Le RAG Pipeline n'a pas pu être initialisé.")


N_RESULTS = st.sidebar.slider(
    "Nombre de Chunks de Référence (k) :", min_value=1, max_value=10, value=3
)
LLM_MODEL = st.sidebar.text_input(
    "Modèle de Génération (Ollama) :", value=GENERATION_MODEL
)
st.sidebar.caption(f"Modèle d'Embedding: {EMBEDDING_MODEL}")


# ===============================================
# 3. SAISIE UTILISATEUR ET LOGIQUE D'ANALYSE
# ===============================================

user_article = st.text_area(
    "Article à Analyser :",
    height=300,
    placeholder="Collez ici le texte complet de l'article pour vérifier sa véracité...",
)

if st.button("Lancer l'Analyse 🔍") and rag_pipeline:
    if not user_article:
        st.warning("Veuillez coller un article dans la zone de texte pour commencer.")
    else:
        st.subheader("Résultats de l'Analyse RAG")

        with st.spinner(
            f"⏳ Recherche de k={N_RESULTS} chunks similaires et demande au modèle {LLM_MODEL}..."
        ):
            try:
                # La fonction doit retourner (réponse_LLM, docs, metas)
                response, docs, metas = rag_pipeline.analyze_article(
                    text=user_article, model_name=LLM_MODEL, n_results=N_RESULTS
                )

                # Extraction du Verdict et de la Raison (selon le format 'Verdict: X\nReason: Y' défini dans retrieval.py)
                verdict_match = re.search(
                    r"Verdict: (TRUE|FAKE)", response, re.IGNORECASE
                )
                reason_match = re.search(r"Reason: (.*)", response, re.DOTALL)

                verdict = verdict_match.group(1).upper() if verdict_match else "INCONNU"
                reason = (
                    reason_match.group(1).strip() if reason_match else response.strip()
                )

                # --- AFFICHAGE DU VERDICT ---
                if verdict == "TRUE":
                    st.success(f"## ✅ VERDICT : **ARTICLE FIABLE**")
                elif verdict == "FAKE":
                    st.error(f"## 🚨 VERDICT : **FAUSSE NOUVELLE POTENTIELLE**")
                else:
                    st.warning(f"## ❓ VERDICT : **RÉPONSE DU LLM INDÉTERMINÉE**")

                # --- AFFICHAGE DE LA JUSTIFICATION ---
                st.markdown("---")
                st.markdown(f"**Justification du Modèle ({LLM_MODEL}) :**")
                st.info(reason)

                # --- AFFICHAGE DES CHUNKS DE RÉFÉRENCE ---
                st.subheader(f"Articles de Référence Récupérés (Top {len(docs)})")
                st.caption(
                    "Ce sont les passages que le modèle a utilisés comme contexte."
                )

                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    # Assurez-vous que meta['label'] est soit 0 (Fake) soit 1 (True)
                    label_value = meta.get("label", -1)
                    label_text = (
                        "Vrai (Label 1)"
                        if label_value == 1
                        else "Faux (Label 0)"
                        if label_value == 0
                        else "N/A"
                    )

                    col1, col2 = st.columns([1, 4])
                    col1.markdown(f"**Chunk #{i + 1}**")
                    col1.markdown(f"**Label :** `{label_text}`")
                    col2.code(doc, language="text")

            except Exception as e:
                st.error(
                    f"Une erreur s'est produite lors de l'exécution du RAG. Détails : {e}"
                )
