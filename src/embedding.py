import pandas as pd
from tqdm import tqdm
from typing import List
import ollama


class OllamaEmbedder:
    """
    Classe responsable de la génération d'embeddings à partir de textes 
    via le modèle 'all_minilm' d'Ollama.
    """

    def __init__(self, model_name: str = "all_minilm", chunk_size: int = 200, overlap: int = 50):
        """
        Initialise l'embedder Ollama.
        
        Args:
            model_name (str): Nom du modèle d'embedding disponible via Ollama.
            chunk_size (int): Taille des chunks pour découper les textes longs (en mots).
            overlap (int): Chevauchement entre chunks (en mots).
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        print(f"[INIT] OllamaEmbedder initialisé avec modèle='{model_name}', chunk_size={chunk_size}, overlap={overlap}")

    # Pour découper en chunks
    def split_text(self, text: str) -> List[str]:
        """
        Découpe un texte en plusieurs chunks avec chevauchement.
        """
        if not isinstance(text, str) or not text.strip(): # Si c'est pas des chaines de caractères et avec des whitespaces
            return []
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if len(chunk.split()) > 10: # Si le chunk contient moins de 10 mots
                chunks.append(chunk)
            if end >= len(words): # Quand ça atteint la fin du texte, on s'arrête
                break
            start += self.chunk_size - self.overlap
        return chunks

    # Pour vectoriser le texte
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in tqdm(texts, desc="Vectorisation"):
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response.embedding)
        return embeddings

    # Pour appliquer à l'ensemble du dataframe le découpage et la vectorisation
    def embed_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Applique le chunking et la vectorisation sur tout un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame contenant les textes à vectoriser.
            text_col (str): Nom de la colonne contenant le texte.

        Returns:
            pd.DataFrame: DataFrame enrichi avec colonnes 'chunks' et 'embeddings'.
        """
        if text_col not in df.columns:
            raise ValueError(f"La colonne '{text_col}' est absente du DataFrame.")

        print(f"[INFO] Démarrage de la génération d'embeddings sur {len(df)} articles...")
        tqdm.pandas()

        # Découpage en chunks
        df["chunks"] = df[text_col].progress_apply(self.split_text) 

        # Pour chaque chunk, créer un embedding
        embeddings_data = []
        for i, chunks in enumerate(tqdm(df["chunks"], desc="Génération des embeddings", ncols=80)):
            for chunk in chunks:
                emb = self.embed_texts(chunk)
                embeddings_data.append({
                    "index_article": i,
                    "chunk": chunk,
                    "embedding": emb,
                    "label": df.loc[i, "label"]
                })

        embedded_df = pd.DataFrame(embeddings_data)
        print(f"[OK] {len(embedded_df)} embeddings générés à partir de {len(df)} articles.")
        return embedded_df
