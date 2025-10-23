from src.preprocessing import CSVLoader, DataCleaner, TextChunker, DatasetMerger
import pandas as pd

if __name__ == "__main__":
    # --- LOAD ---
    loader = CSVLoader()
    df_true = loader.load_csv("data/raw/True.csv")
    df_fake = loader.load_csv("data/raw/Fake.csv")

    # --- TRUE ---
    cleaner_true = DataCleaner(df_true)
    cleaner_true.add_label(1)
    cleaner_true.drop_empty_rows_and_duplicated() \
        .remove_spaces() \
        .lower_case() \
        .date_format() \
        .clean_all_text_columns()
    cleaned_df_true = cleaner_true.get_df()
    cleaner_true.save_csv("data/processed/cleaned_df_true.csv")

    # --- FAKE ---
    cleaner_fake = DataCleaner(df_fake)
    cleaner_fake.add_label(0)
    cleaner_fake.drop_empty_rows_and_duplicated() \
        .remove_spaces() \
        .lower_case() \
        .date_format() \
        .clean_all_text_columns()
    cleaned_df_fake = cleaner_fake.get_df()
    cleaner_fake.save_csv("data/processed/cleaned_df_fake.csv")

    # --- CHUNKING ---
    chunker = TextChunker()
    cleaned_df_true["chunks"] = cleaned_df_true["text"].apply(chunker.split_text)
    cleaned_df_fake["chunks"] = cleaned_df_fake["text"].apply(chunker.split_text)
    cleaned_df_true.to_csv("data/processed/cleaned_df_true_chunks.csv", index=False)
    cleaned_df_fake.to_csv("data/processed/cleaned_df_fake_chunks.csv", index=False)

    # --- MERGE ---
    merger = DatasetMerger()
    combined_df = merger.merge([cleaned_df_true, cleaned_df_fake])
    combined_df.to_csv("data/processed/cleaned_df_all_chunks.csv", index=False)

    print("\nPipeline terminé avec succès.")
    print(f"Articles vrais traités : {len(cleaned_df_true)}")
    print(f"Articles faux traités : {len(cleaned_df_fake)}")
    print(f"Total : {len(combined_df)} articles traités.")
