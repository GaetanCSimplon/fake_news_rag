import numpy as np  
import ollama
import chromadb
# from chromadb.utils import embedding_functions
from tqdm import tqdm
from src import embedding
from src.embedding import OllamaEmbedder

class RAGAnalyzer:
    """
    Analyse d'un article en se basant sur les données de la base vectorielle.
    """
    def __init__(self, chroma_path="data/vector_db", 
                 collection_name="news_articles", 
                 embedding_model="all-minilm" ):
        # Connexion à la base vectorielle
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)
        print(f"[INFO] Collection '{collection_name}' chargée depuis '{chroma_path}'")
        # Initialisation de l'embeddeur
        self.embedder = OllamaEmbedder(model_name=embedding_model)
    
    # Vectorisation et normalisation du texte utilisateur
    def vectorize_query(self, text: str) -> list:
        """
        Vectorise et normalise le texte utilisateur (sans chunking)
        """
        if not text.strip():
            raise ValueError("Texte utilisateur vide")
        embeddings = self.embedder.embed_texts([text])
        return embeddings[0] if embeddings else []

    
    # Recherche dans la base vectorielle de documents similaires
    def retrieve_similar_docs(self, query_vector, n_results=5):
        """
        Recherche les documents les plus similaires à un vecteur
        """
        results = self.collection.query(query_embeddings=[query_vector], n_results=n_results) # Envoie une requête pour trouver les vecteurs similaires dans la base vectorielle 
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        
        print(f"\n[INFO] {len(docs)} documents similaires retrouvés :")
        for d, dist in zip(docs, distances):
            print(f" - distance={dist:.4f}")
            # print(f" - extrait={d}")
            
        return docs, metas
    
    # Création du contexte
    def build_context(self, docs, metas):
        """
        Assemble les chunks retrouvés en un texte de contexte.
        """
        context = "\n\n".join([f"[{m.get('date', 'unknown')}] ({m.get('label', '?')}): {doc}" for doc, m in zip(docs, metas)])
        print(f"[CONTEXT] {context}")
        return context
    # Version sous forme de boucle for
    # liste_temp = [] # Liste de stockage de la chaine de caractères finale d'un article pertinent trouvé
        # for doc, m in zip(docs, metas): # on boucle sur les textes et les métadonnées données associées
        #     texte = f"[{m.get('date', 'unknown')}] ({m.get('label', '?')}): {doc}" # on construit la chaine de caractères en sortie
        #     liste_temp.append(texte) # stockage de la chaine de caractères dans la list_temp
        # Au final, ce qui sera retourné est le contexte qui viendra alimenté le prompt pour le LLM
    
    # Création du prompt
    def build_prompt(self, user_text: str, context: str) -> str:
        """
        Crée un prompt en anglais pour le modèle LLM
        """
        prompt = f"""
        You are a fact-checking assistant.
        You are given an article written by an user and several similar news articles.
        
        Your task:
        1. Use provided context to analyze the user's article.
        2. Determine if the article is TRUE (label = 1) or FAKE (label = 0).
        3. Explain briefly why you think so (based only on the retrieved context).
        
        ### CONTEXT
        
        {context}
        
        ### ARTICLE TO ANALYZE
        
        {user_text}
        
        ### RESPONSE FORMAT
        
        Verdict: TRUE or FAKE
        Reason: <your explanation>       
        """
        return prompt.strip()
    
    # Génération du verdict avec le modèle LLM choisi
    
    def generate_response(self, prompt: str, model_name="llama3.2") -> str:
        """
        Envoie le prompt au modèle Ollama et récupère la réponse complète.
        Compatible avec les versions récentes d’Ollama (stream ou non-stream).
        """
        print(f"\n[INFO] Génération de la réponse avec le modèle {model_name}...")

        response = ollama.generate(model=model_name, prompt=prompt, stream=False)

        # Certaines versions de Ollama renvoient directement une clé "response"
        if isinstance(response, dict):
            return response.get("response", "Aucune réponse générée.")
        elif hasattr(response, "response"):
            return response.response
        else:
            return str(response)


    

        

