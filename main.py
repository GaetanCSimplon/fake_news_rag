from src.rag_pipeline import RAGPipeline
from src.storage_chroma import ChromaStorage
from datetime import datetime

# === CONFIGURATION ===
CHROMA_PATH = "data/vector_db"
COLLECTION_NAME = "articles"
EMBEDDING_MODEL = "all-minilm"
GENERATION_MODEL = "llama3.2"


def ask_user_article():
    """
    Demande à l'utilisateur d'entrer un article à analyser.
    """
    print("\n=== Analyse d'article ===")
    print("Colle ton article ci-dessous (finis par une ligne vide) :")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    print("=== Système RAG Fake News ===")

    # Initialisation du pipeline RAG
    rag = RAGPipeline(
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL
    )

    # Initialisation du stockage Chroma
    storage = ChromaStorage(
        persist_dir=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    # Étape 1 : Récupération du texte utilisateur
    article_text = ask_user_article()

    # Étape 2 : Analyse via le pipeline RAG
    print("\n[INFO] Lancement de l'analyse RAG...")
    result = rag.analyze_article(article_text, model_name=GENERATION_MODEL, n_results=3)
    print("\n====== RÉPONSE DU MODÈLE ======")
    print(result.strip())


if __name__ == "__main__":
    main()
