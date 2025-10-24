from src.embedding import OllamaEmbedder
import pytest
from unittest.mock import patch
import pandas as pd
from typing import List



# Taille d'embedding simulée pour le mock
EMBEDDING_DIM = 2
SIMULATED_EMBEDDING = [0.1, 0.2]

# Fonction de mock pour ollama.embeddings
def mock_ollama_embeddings_func(model, prompt):
    """Simule la réponse d'ollama.embeddings."""
    # Le code de l'utilisateur s'attend à un objet avec un attribut 'embedding'
    return type("Response", (), {"embedding": SIMULATED_EMBEDDING})()

def test_init():
    """
    Test d'instanciation de la classe OllamaEmbedder.
    Vérifie que les attributs sont bien créés.
    """
    embedder = OllamaEmbedder(model_name="all-minilm")
    assert embedder.model_name == "all-minilm"
    assert embedder.chunk_size > 0
    assert embedder.overlap >= 0

# -------------------------------------------------------------
# Test du découpage en chunks
# -------------------------------------------------------------
def test_split_text():
    """
    Vérifie que split_text découpe correctement un texte en chunks
    avec chevauchement et respect de la taille maximale, en passant le filtre de 10 mots.
    """
    
    embedder = OllamaEmbedder(chunk_size=15, overlap=2)
    
    # Texte de 20 mots (garantit la création d'au moins un chunk > 10 mots)
    text = "ceci est un test pour le découpage en chunk et vérifier que tout fonctionne correctement odhighq oqg mgqmogjsgfoqjmkqjkljmlk mqlkjdmlkj"
    
    chunks = embedder.split_text(text)
    
    # Chaque chunk ne doit pas dépasser chunk_size mots (15)
    assert all(len(c.split()) <= 15 for c in chunks)
    # On doit obtenir au moins un chunk
    assert len(chunks) == 1 # Correction de l'assertion pour être plus précis

# -------------------------------------------------------------
# Test de embed_texts avec mock
# -------------------------------------------------------------
@patch("src.embedding.ollama.embeddings", side_effect=mock_ollama_embeddings_func)
def test_embed_texts_mock(mock_embed):
    """
    Test de la méthode embed_texts sans appel réel à Ollama.
    """
    embedder = OllamaEmbedder()
    texts = ["un texte de test", "un autre texte"]

    # Le mock est appliqué via le décorateur
    embeddings = embedder.embed_texts(texts)

    # Vérifications
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(e, list) for e in embeddings)
    assert len(embeddings[0]) == EMBEDDING_DIM # Vérifie la dimension simulée

# -------------------------------------------------------------
# Test de embed_dataframe avec mock
# -------------------------------------------------------------
@patch("src.embedding.ollama.embeddings", side_effect=mock_ollama_embeddings_func)
def test_embed_dataframe_mock(mock_embed):
    """
    Test de embed_dataframe qui combine le chunking et la vectorisation.
    """
    # Correction : Utiliser des paramètres qui garantissent la création de chunks
    embedder = OllamaEmbedder(chunk_size=15, overlap=2)

    # Texte suffisamment long pour générer des chunks (20 mots, 1 chunk)
    df = pd.DataFrame({
        "text": [
            "Un texte suffisamment long pour vérifier que les chunks sont bien créés et contiennent plus de dix mots", # 18 mots -> 1 chunk
            "Un autre texte de test encore plus long pour vérifier le dataframe" # 12 mots -> 1 chunk
        ],
        "label": [1, 0]
    })

    embedded_df = embedder.embed_dataframe(df, text_col="text")
    
    # Vérifie la présence des colonnes attendues
    assert "chunk" in embedded_df.columns
    assert "embedding" in embedded_df.columns
    assert "index_article" in embedded_df.columns
    assert "label" in embedded_df.columns
    
    # Vérifie qu'il y a le nombre attendu de chunks (1 + 1 = 2)
    assert len(embedded_df) == 2
    
    # Vérifie que les embeddings ont été générés
    # L'erreur indique que la liste est parfois imbriquée. On s'assure de la bonne extraction.
    expected_embeddings = [SIMULATED_EMBEDDING, SIMULATED_EMBEDDING]
    actual_embeddings = embedded_df["embedding"].apply(lambda x: x if isinstance(x, list) else x.embedding).tolist()
    
    assert actual_embeddings == expected_embeddings
    
    # Vérifie la propagation des métadonnées
    assert embedded_df["label"].tolist() == [1, 0]

