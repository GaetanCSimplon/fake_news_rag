from src.preprocessing import CSVLoader, DataCleaner, TextChunker



if __name__ == "__main__":
    loader = CSVLoader()
    df_true = loader.load_csv("~/fake_news_rag/data/raw/True.csv")
    df_fake = loader.load_csv("~/fake_news_rag/data/raw/Fake.csv")
    # data cleaning
    # Initialize cleaner with a DataFrame
    cleaner = DataCleaner(df_true)

    # Apply all cleaning steps
    cleaner.drop_empty_rows_and_duplicated() \
        .remove_spaces() \
        .lower_case() \
        .date_format() \
        .clean_all_text_columns()

    # Get cleaned DataFrame
    cleaned_df_true = cleaner.get_df()

    # Save cleaned DataFrame
    cleaner.save_csv("data/processed/cleaned_df_true.csv")
    # data fake process
    cleaner = DataCleaner(df_fake)

    # Apply all cleaning steps
    cleaner.drop_empty_rows_and_duplicated() \
        .remove_spaces() \
        .lower_case() \
        .date_format() \
        .clean_all_text_columns()

    # Get cleaned DataFrame
    cleaned_df_fake = cleaner.get_df()

    # Save cleaned DataFrame
    cleaner.save_csv("data/processed/cleaned_df_fake.csv")
    
    # chunking
    chunker = TextChunker() # Initialisation du chunker avec les paramètres par défaut
    cleaned_df_true["chunks"] = cleaned_df_true["text"].apply(chunker.split_text) # Application de la méthode split_text à la colonne text
    
    cleaned_df_fake["chunks"] = cleaned_df_fake["text"].apply(chunker.split_text) # Application de la méthode split_text à la colonne text
    # Save chunked DataFrame
    cleaned_df_true.to_csv("data/processed/cleaned_df_true_chunks.csv", index=False)
    cleaned_df_fake.to_csv("data/processed/cleaned_df_fake_chunks.csv", index=False)