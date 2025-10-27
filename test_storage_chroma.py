from src.embedding import OllamaEmbedder
from src.storage_chroma import ChromaStorage

if __name__ == "__main__":
    # --- Exemple d'utilisation ---
    storage = ChromaStorage(
        persist_dir="data/vector_db",
        collection_name="news_articles"
    )

    # 1️⃣ Charger le CSV final
    df = storage.load_embedded_data("data/processed/embedded_chunks_normalized.csv")

    # 2️⃣ Insérer dans Chroma
    storage.insert_into_chroma(df)

    # 3️⃣ Tester une requête sémantique
    storage.query("government announces new policy on climate change", n_results=3)