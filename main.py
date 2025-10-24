from src.preprocessing import CSVLoader, DataCleaner, DatasetMerger
from src.embedding import OllamaEmbedder
import pandas as pd

if __name__ == "__main__":
    # --- LOAD ---
    loader = CSVLoader()
    df_true = loader.load_csv("data/raw/True.csv")
    df_fake = loader.load_csv("data/raw/Fake.csv")

    # --- CLEAN TRUE ---
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

    # --- CLEAN FAKE ---
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

    # --- MERGE CLEANED DATASETS ---
    merger = DatasetMerger()
    combined_df = merger.merge([cleaned_df_true, cleaned_df_fake])
    combined_df.to_csv("data/processed/cleaned_df_all.csv", index=False)
    print(f"[INFO] Fusion terminée : {combined_df.shape[0]} articles combinés.")

    # --- EMBEDDING (Ollama) ---
    print("\n[INFO] Démarrage de la vectorisation avec Ollama...")
    embedder = OllamaEmbedder(model_name="all-minilm")
    combined_df["embedding"] = embedder.embed_texts(combined_df["text"].tolist())

    # --- SAVE FINAL ---
    combined_df.to_csv("data/processed/cleaned_df_all_embeddings.csv", index=False)
    print("\n[SUCCÈS] Pipeline complet exécuté avec succès")
    print(f"Total : {len(combined_df)} articles vectorisés et sauvegardés.")
