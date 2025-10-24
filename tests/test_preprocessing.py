import os
import pandas as pd
import pytest
from src.preprocessing import CSVLoader, DataCleaner


# -------------------------------
# TESTS CSVLoader
# -------------------------------
def test_csv_loader(tmp_path):
    # Créer un CSV temporaire
    csv_path = tmp_path / "sample.csv"
    df_original = pd.DataFrame({"text": ["A", "B"], "subject": ["X", "Y"], "date": ["2021-01-01", "2021-01-02"]})
    df_original.to_csv(csv_path, index=False)

    # Charger le CSV avec CSVLoader
    loader = CSVLoader()
    df_loaded = loader.load_csv(csv_path)

    # Vérifications
    assert not df_loaded.empty
    assert df_loaded.shape == (2, 3)
    assert list(df_loaded.columns) == ["text", "subject", "date"]


# -------------------------------
# TESTS DataCleaner
# -------------------------------
@pytest.fixture # Jeu de données pour les tests
def sample_df():
    """Crée un DataFrame de test avec des anomalies connues."""
    data = {
        "text": ["Hello!!!", "   Bye  ", "Visit http://test.com now", None, "Duplicate text", "Duplicate text"],
        "subject": ["News", "Fake", "Real", "News", "Misc", "Misc"],
        "date": ["2021-01-01", "bad-date", "2021-03-03", None, "2021-05-05", "2021-05-05"]
    }
    return pd.DataFrame(data)


def test_drop_empty_and_duplicates(sample_df):
    cleaner = DataCleaner(sample_df.copy())
    cleaned = cleaner.drop_empty_rows_and_duplicated().get_df()
    # Suppression des None dans text/date et doublons sur text
    assert cleaned.shape[0] == 4  # 6 lignes initiales - 2 supprimées
    assert cleaned["text"].isnull().sum() == 0


def test_remove_spaces(sample_df):
    cleaner = DataCleaner(sample_df.copy())
    cleaner.remove_spaces()
    # Vérifie que les espaces en trop ont été supprimés
    assert cleaner.df.loc[1, "text"] == "Bye"


def test_lower_case(sample_df):
    cleaner = DataCleaner(sample_df.copy())
    cleaner.lower_case()
    assert all(cleaner.df["text"].dropna().apply(lambda x: x == x.lower())) # Vérifie si l'ensemble des caractères sont en minuscules


def test_date_format(sample_df):
    cleaner = DataCleaner(sample_df.copy())
    cleaned = cleaner.date_format().get_df()
    # Les lignes avec dates invalides doivent être supprimées
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"]) # Vérifie si la colonne date est au format datetime (renvoie True)
    assert cleaned.shape[0] < sample_df.shape[0] # Vérifie qu'après nettoyage, la quantité de données diminue (car nettoyage de données fait)


def test_clean_text():
    raw_text = "Hello!! <b>World</b> visit https://example.com"
    cleaned = DataCleaner.clean_text(raw_text)
    # Doit retirer les symboles et URL
    assert "http" not in cleaned 
    assert "!" not in cleaned
    assert cleaned.isalnum() or " " in cleaned


def test_clean_all_text_columns(sample_df):
    cleaner = DataCleaner(sample_df.copy())
    cleaner.clean_all_text_columns()
    # Vérifie que toutes les colonnes de type str ont été nettoyées
    assert all(isinstance(val, str) for val in cleaner.df["text"] if pd.notnull(val)) 
    assert "http" not in cleaner.df["text"].iloc[2]


def test_save_csv(tmp_path, sample_df):
    cleaner = DataCleaner(sample_df.copy())
    output_file = tmp_path / "cleaned.csv"
    cleaner.save_csv(output_file)
    assert output_file.exists()


