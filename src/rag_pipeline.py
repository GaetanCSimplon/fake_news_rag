from src.embedding import OllamaEmbedder
from src.retrieval import RAGAnalyzer
from src.storage_chroma import ChromaStorage
from typing import List, Dict, Tuple


class RAGPipeline:
    """
    Classe d'orchestration du pipeline RAG:
    - Embedding de la requête utilisateur
    - Recherche des documents similaires dans la base vectorielle
    - Génération d'un prompt avec contexte
    - Envoi au modèle sélectionné pour génération d'un verdict
    """

    def __init__(
        self,
        chroma_path: str,
        collection_name: str,
        embedding_model: str = "all-minilm",
    ):
        """
        Initialise le pipeline avec les composants nécessaires
        Args:
            chroma_path (str): Chemin de la base vectorielle ChromaDB ("vector_db").
            collection_name (str): Nom de la collection à interroger ("news_articles")
            embedding_model (str): Nom du modèle d'embedding ("all-minilm).
        """
        print(
            f"[INIT] Initialisation du pipeline RAG avec modèle '{embedding_model}'..."
        )
        self.embedder = OllamaEmbedder(model_name=embedding_model)
        self.retriever = RAGAnalyzer(chroma_path, collection_name, embedding_model)

    # Analyse complète d'un article utilisateur

    def analyze_article(
        self, text: str, model_name: str = "llama3.2", n_results: int = 5
    ) -> Tuple[str, List[str], List[Dict]]:
        """
        Analyse un texte utilisateur en le comparant à la base vectorielle

        Args:
            text (str): Texte de l'article à analyser.
            model_name (str): Modèle de génération textuelle ("llama3.2 ou phi3:mini").
            n_results (int): Nombre de chunks similaires à récupérer

        Return:
            str: Réponse générée par le modèle
        """

        print("\n[INFO] Etape 1 - Vectorisation du texte utilisateur...")
        query_vector = self.retriever.vectorize_query(text)

        print(
            "[INFO] Étape 2 - Recherche des articles similaires dans la base vectorielle..."
        )
        docs, metas = self.retriever.retrieve_similar_docs(
            query_vector, n_results=n_results
        )

        print("[INFO] Étape 3 - Construction du contexte à partir des résultats...")
        context = self.retriever.build_context(docs, metas)

        print(f"[INFO] Etape 4 - Génération du prompt pour le modèle...")
        prompt = self.retriever.build_prompt(text, context)

        print(f"[INFO] Etape 5 - Envoi du prompt au modèle...")
        response = self.retriever.generate_response(prompt, model_name)

        print("\n [SUCCESS] Réponse générée : \n")

        # return response
        return response, docs, metas
