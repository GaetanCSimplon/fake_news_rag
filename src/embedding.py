import pandas as pd
from tqdm import tqdm
from typing import List
import numpy as np
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class OllamaEmbedder:
    """
    Classe responsable de la génération d'embeddings à partir de textes
    via un modèle Ollama (par défaut 'all-minilm').
    """

    def __init__(self, model_name: str = "all-minilm", chunk_size: int = 200, overlap: int = 50, batch_size: int = 8):
        """
        Initialise l'embedder Ollama.

        Args:
            model_name (str): Nom du modèle d'embedding disponible via Ollama.
            chunk_size (int): Taille des chunks pour découper les textes longs (en mots).
            overlap (int): Chevauchement entre chunks (en mots).
            batch_size (int): Nombre de textes traités en parallèle.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size
        print(f"[INIT] OllamaEmbedder initialisé avec modèle='{model_name}', chunk_size={chunk_size}, overlap={overlap}")

    # -----------------------------
    #  Découpage du texte en chunks
    # -----------------------------
    def split_text(self, text: str) -> List[str]:
        """Découpe un texte en plusieurs chunks avec chevauchement."""
        if not isinstance(text, str) or not text.strip(): # Vérifie que le texte est une chaine non vide
            return []

        words = text.split() # Transforme le texte en liste de mots
        chunks, start = [], 0 

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if len(chunk.split()) > 10: # Ignore les segments de moins de 10 mots
                chunks.append(chunk)
            if end >= len(words): # Fin de la boucle si on arrive à la fin du texte à chunker
                break
            start += self.chunk_size - self.overlap  # Pointeur start avance en prenant en compte l'overlap

        return chunks

    # -----------------------------
    # Fonction utilitaire : normalisation L2
    # -----------------------------
    def normalize_vector(self, vec: List[float]) -> List[float]:
        """Normalise un vecteur (L2)."""
        arr = np.array(vec) # Transforme les vecteurs en objet numpy array 
        norm = np.linalg.norm(arr) # Application de la fonction de normalisation L2 (distance euclidienne)
        return (arr / norm).tolist() if norm > 0 else arr.tolist()

    # -----------------------------
    # Vectorisation avec parallélisation
    # -----------------------------
    def embed_texts(self, texts: List[str], max_workers: int = 4) -> List[List[float]]:
        """Crée des embeddings normalisés pour une liste de textes (en parallèle)."""

        def embed_one(text):
            """
            Appelle l'embedder et retourne le vecteur normalisé
            """
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return self.normalize_vector(response.embedding)

        embeddings = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor: # permet d'executer plusieurs vectorisation (embed_one) en parallèle
            futures = {executor.submit(embed_one, text): text for text in texts} # Crée un dict (clé = tâche (objet Future), valeur = texte correpsondant)
            # executor.submit() lance la fonction embed_one dans un thread parallèle
            for f in tqdm(as_completed(futures), total=len(futures), desc="Vectorisation parallèle"): 
                embeddings.append(f.result()) # f.result() = vecteur normalisé renvoyé par la méthode embed_one()

        return embeddings

    # -----------------------------
    # Application à un DataFrame complet
    # -----------------------------
    def embed_dataframe(self, df: pd.DataFrame, text_col: str = "text", output_path: str = None) -> pd.DataFrame:
        """
        Applique le chunking + embedding à un DataFrame entier.
        Sauvegarde partielle automatique si output_path est précisé.
        """
        if text_col not in df.columns:
            raise ValueError(f"La colonne '{text_col}' est absente du DataFrame.")

        tqdm.pandas()
        print(f"[INFO] Démarrage de la génération d'embeddings sur {len(df)} articles...")

        # Étape 1 : découpage en chunks
        df["chunks"] = df[text_col].progress_apply(self.split_text) # utilisation d'apply pour faire appel à la méthode split_text et stockage dans la nouvelle colonne 'chunks'

        # Étape 2 : création du DataFrame de chunks
        all_chunks = [] # Liste de dictionnaire qui contient les données de chaque article
        for i, row in df.iterrows():
            for chunk in row["chunks"]:
                all_chunks.append({
                    "index_article": i,
                    "chunk": chunk,
                    "label": row.get("label", None), # Si aucun label, renvoie None
                    "subject": row.get("subject", None),
                    "date": row.get("date", None),
                })

        chunks_df = pd.DataFrame(all_chunks) # Conversion de la liste d'objets structurés en dataframe
        if chunks_df.empty:
            print("[WARNING] Aucun chunk généré. Vérifie chunk_size / overlap.")
            return pd.DataFrame()

        # Étape 3 : vectorisation
        chunks_df["embedding"] = self.embed_texts(chunks_df["chunk"].tolist()) # Création de la colonne avec les vecteurs (embedding + normalisation)

        # Étape 4 : sauvegarde progressive
        if output_path: 
            os.makedirs(os.path.dirname(output_path), exist_ok=True) 
            chunks_df.to_csv(output_path, index=False)
            print(f"[SAVE] Fichier partiel sauvegardé → {output_path}")

        print(f"[OK] {len(chunks_df)} embeddings générés à partir de {len(df)} articles.")
        return chunks_df
