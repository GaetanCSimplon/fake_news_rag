from src.preprocessing import CSVLoader, DataCleaner, DatasetMerger
from src.embedding import OllamaEmbedder
from src.storage_chroma import ChromaStorage
import os

if __name__ == "__main__":
    
    
    # --- CHARGEMENT ---
    loader = CSVLoader()
    df_true = loader.load_csv("data/raw/True.csv")
    df_fake = loader.load_csv("data/raw/Fake.csv")

    # --- NETTOYAGE ---
    cleaner_true = DataCleaner(df_true)
    cleaned_df_true = (
        cleaner_true
        .add_label(1)
        .drop_empty_rows_and_duplicated()
        .remove_spaces()
        .lower_case()
        .date_format()
        .clean_all_text_columns()
        .get_df()
    )
    cleaner_true.save_csv("data/processed/cleaned_df_true.csv")

    cleaner_fake = DataCleaner(df_fake)
    cleaned_df_fake = (
        cleaner_fake
        .add_label(0)
        .drop_empty_rows_and_duplicated()
        .remove_spaces()
        .lower_case()
        .date_format()
        .clean_all_text_columns()
        .get_df()
    )
    cleaner_fake.save_csv("data/processed/cleaned_df_fake.csv")

    # --- FUSION ---
    merger = DatasetMerger()
    combined_df = merger.merge([cleaned_df_true, cleaned_df_fake])
    combined_df.to_csv("data/processed/cleaned_df_all.csv", index=False)
    print(f"[INFO] Fusion terminée : {combined_df.shape[0]} articles combinés.")

    # --- EMBEDDING ---
    output_path = "data/processed/embedded_chunks_normalized.csv"
    if not os.path.exists(output_path):
        print("\n[INFO] Démarrage de la vectorisation avec Ollama...")
        try:
            embedder = OllamaEmbedder(model_name="all-minilm", chunk_size=300, overlap=30)
            embedded_df = embedder.embed_dataframe(combined_df, text_col="text", output_path=output_path)
        except Exception as e:
            print(f"[ERREUR] Échec de la vectorisation : {e}")
            exit(1)
    else:
        print(f"[INFO] Embeddings déjà existants : {output_path}")
            
    # --- CREATION & STOCKAGE ---
    
    storage = ChromaStorage(persist_dir="data/vector_db", collection_name="articles")
    df_loaded = storage.load_embedded_data(csv_path=output_path)
    storage.insert_into_chroma(df_loaded)
    
    print("\n [SUCCESS] Terminé !")
    
