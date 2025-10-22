from src.preprocessing import CSVLoader, DataCleaner



if __name__ == "__main__":
    loader = CSVLoader()
    df_true = loader.load_csv("data/row/True.csv")
    df_fake = loader.load_csv("data/row/Fake.csv")
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
    cleaned_df_true = cleaner.get_df()

    # Save cleaned DataFrame
    cleaner.save_csv("data/processed/cleaned_df_fake.csv")
