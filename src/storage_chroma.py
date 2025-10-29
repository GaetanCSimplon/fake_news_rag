import pandas as pd
import numpy as np
import chromadb
# from chromadb.config import Settings
from tqdm import tqdm

class ChromaStorage:
    """
    Classe responsable de la création et de l'insertion des embeddings normalisés
    dans une base vectorielle ChromaDB.
    """

    def __init__(self, persist_dir="data/vector_db", collection_name="articles"):
        """
        Initialise la base Chroma.

        Args:
            persist_dir (str): Chemin de sauvegarde de la base vectorielle.
            collection_name (str): Nom de la collection.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name

        existing = [c.name for c in self.client.list_collections()]
        if collection_name in existing:
            self.collection = self.client.get_collection(collection_name)
            print(f"[INFO] Collection '{collection_name}' chargée depuis {persist_dir}")
        else:
            self.collection = self.client.create_collection(collection_name)
            print(f"[INFO] Nouvelle collection '{collection_name}' créée dans {persist_dir}")


    def load_embedded_data(self, csv_path: str) -> pd.DataFrame:
        """
        Charge le CSV contenant les chunks et embeddings.

        Args:
            csv_path (str): Chemin vers le CSV contenant les embeddings.
        """
        df = pd.read_csv(csv_path)
        # Convertit les chaînes "[0.1, -0.2, ...]" en vraies listes de floats
        df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
        print(f"[INFO] {len(df)} lignes chargées depuis {csv_path}")
        return df


    def insert_into_chroma(self, df: pd.DataFrame, batch_size: int = 100):
        """
        Insère les données vectorielles dans ChromaDB par batchs.

        Args:
            df (pd.DataFrame): DataFrame contenant les chunks + embeddings.
            batch_size (int): Taille des batchs d'insertion.
        """
        total = len(df)
        print(f"[INFO] Insertion de {total} documents dans ChromaDB...")

        for i in tqdm(range(0, total, batch_size), desc="Insertion dans ChromaDB"):
            batch = df.iloc[i:i + batch_size]

            ids = [f"doc_{idx}" for idx in batch.index]
            documents = batch["chunk"].tolist()
            embeddings = batch["embedding"].tolist()

            metadatas = batch[["index_article", "label", "subject", "date"]].to_dict(orient="records")

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

        print(f"[SUCCÈS] {total} documents insérés dans la collection '{self.collection_name}'.")


