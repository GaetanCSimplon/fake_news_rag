from src.preprocessing import CSVLoader, DataCleaner, DatasetMerger
from src.embedding import OllamaEmbedder
import pandas as pd
import os

if __name__ == "__main__":
    # --- 1️⃣ CHARGEMENT ---
    loader = CSVLoader()
    df_true = loader.load_csv("data/raw/True.csv")
    df_fake = loader.load_csv("data/raw/Fake.csv")

    # --- 2️⃣ NETTOYAGE ---
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

    # --- 3️⃣ FUSION ---
    merger = DatasetMerger()
    combined_df = merger.merge([cleaned_df_true, cleaned_df_fake])
    combined_df.to_csv("data/processed/cleaned_df_all.csv", index=False)
    print(f"[INFO] Fusion terminée : {combined_df.shape[0]} articles combinés.")

    # --- 4️⃣ EMBEDDING ---
    output_path = "data/processed/embedded_chunks_normalized.csv"
    if os.path.exists(output_path):
        print(f"[INFO] Embeddings déjà générés : {output_path}")
    else:
        print("\n[INFO] Démarrage de la vectorisation avec Ollama...")
        embedder = OllamaEmbedder(model_name="all-minilm", chunk_size=300, overlap=30)
        embedded_df = embedder.embed_dataframe(combined_df, text_col="text", output_path=output_path)

        if not embedded_df.empty:
            print(f"\n[SUCCÈS] {len(embedded_df)} chunks vectorisés et sauvegardés dans : {output_path}")
        else:
            print("[ERREUR] Aucun embedding généré. Vérifie ton modèle Ollama ou ton DataFrame.")
