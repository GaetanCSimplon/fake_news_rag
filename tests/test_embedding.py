from src.embedding import OllamaEmbedder
import pytest

# Vérification - instance de classe
def test_init():
    embedder = OllamaEmbedder(model_name="all-minilm")
    assert embedder.model_name == "all-minilm"
    assert embedder.chunk_size > 0
    assert embedder.overlap >= 0

# Test de découpage

def test_chunking():
    embedder = OllamaEmbedder(chunk_size= 5, overlap= 2)
    text = "ceci est un test pour le découpage en chunk"
    chunks = embedder.split_text(text)
    # Vérifie que la longueur du chunk ne soit pas supérieur au chunk_size
    assert all(len(c.split()) <= 5 for c in chunks)
    # Vérifie qu'il y a au moins un chunk
    assert len(chunks) > 0
    