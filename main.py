from src.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    # Initialisation du pipeline complet
    pipeline = RAGPipeline(
        chroma_path="data/vector_db",
        collection_name="articles",
        embedding_model="all-minilm"
    )

    # Exemple de texte utilisateur
    user_article = """
   Title: "Scientists Discover Plant That Turns Air Into Gasoline: Fossil Fuels to Be Obsolete by 2026!"
    Subtitle: "A secret team of Swiss researchers claims to have unlocked the ultimate green energy solution."
    Geneva, October 28, 2025 — A team of researchers at the Swiss Institute of Green Technologies (SIGT) announced yesterday a groundbreaking discovery: a genetically modified plant, dubbed "Petrolia Mirabilis," capable of converting carbon dioxide (CO₂) from the air into high-quality synthetic gasoline. According to Dr. Hans Müller, the project lead, "This plant can produce up to 500 liters of gasoline per year, with zero negative environmental impact."
    Initial tests, conducted in secret in the Swiss Alps, reportedly showed that "Petrolia Mirabilis" grows at an unprecedented rate and requires minimal water. "In less than six months, we achieved results beyond our wildest expectations," Müller stated during an impromptu press conference. "By 2026, we could replace 80% of the world's fossil fuels."
    The announcement sparked immediate excitement among governments and investors. French President Emmanuel Macron allegedly pledged €10 billion to cultivate the plant in France. "This is the miracle solution we’ve been waiting for to combat climate change," he told reporters.
    However, skeptics abound. "No scientific publication supports these claims," noted Prof. Sophie Laurent, a biologist at the University of Paris. "Moreover, converting CO₂ into gasoline would violate fundamental laws of thermodynamics." Rumors also suggest the plant may produce undetectable toxic gases as a byproduct.
    The SIGT refused to provide samples or concrete evidence, citing "national security concerns." Despite this, countries like China and the U.S. have reportedly ordered millions of plants.
    Stay tuned for updates.
    """

    # Lancement de l’analyse
    result = pipeline.analyze_article(user_article, model_name="llama3.2", n_results=3)
    
    print("====== Réponse =======")
    print(result.strip())
