import os
import re
import chromadb
import ollama
from src.preprocessing import CSVLoader, DataCleaner, DatasetMerger
from src.embedding import OllamaEmbedder
from src.storage_chroma import ChromaStorage


def run_processing_pipeline():
    """Run CSV loading, cleaning, merging, embedding, normalization and storage in Chroma."""
    print("\n Starting data processing and embedding pipeline...")

    # --- LOAD ---
    loader = CSVLoader()
    df_true = loader.load_csv("data/raw/True.csv")
    df_fake = loader.load_csv("data/raw/Fake.csv")

    # --- CLEAN ---
    cleaner_true = DataCleaner(df_true)
    cleaned_df_true = (
        cleaner_true.add_label(1)
        .drop_empty_rows_and_duplicated()
        .remove_spaces()
        .lower_case()
        .date_format()
        .clean_all_text_columns()
        .get_df()
    )
    cleaner_true.save_csv("data/processed/cleaned_df_true.csv")

    cleaner_fake = DataCleaner(df_fake)
    cleaned_df_fake = (
        cleaner_fake.add_label(0)
        .drop_empty_rows_and_duplicated()
        .remove_spaces()
        .lower_case()
        .date_format()
        .clean_all_text_columns()
        .get_df()
    )
    cleaner_fake.save_csv("data/processed/cleaned_df_fake.csv")

    # --- MERGE ---
    merger = DatasetMerger()
    combined_df = merger.merge([cleaned_df_true, cleaned_df_fake])
    combined_df.to_csv("data/processed/cleaned_df_all.csv", index=False)
    print(f"[INFO] Merge completed: {combined_df.shape[0]} articles combined.")

    # --- EMBEDDING ---
    output_path = "data/processed/embedded_chunks_normalized.csv"
    if not os.path.exists(output_path):
        print("\n[INFO] Starting embedding with Ollama...")
        try:
            embedder = OllamaEmbedder(model_name="all-minilm", chunk_size=100, overlap=30)
            embedded_df = embedder.embed_dataframe(
                combined_df[:1000], text_col="text", output_path=output_path
            )
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            exit(1)
    else:
        print(f"[INFO] Embeddings already exist: {output_path}")

    # --- INSERT INTO CHROMA ---
    storage = ChromaStorage(persist_dir="data/vector_db", collection_name="articles")
    df_loaded = storage.load_embedded_data(csv_path=output_path)
    storage.insert_into_chroma(df_loaded)

    print("\n [SUCCESS] Processing + Embedding + Storage completed successfully.")


def run_retrieval_and_rag():
    """Run retrieval and RAG loop on user inputs."""
    print("\n🔍 Running retrieval and RAG...")

    client = chromadb.PersistentClient(path="data/vector_db")
    collection_name = "articles"

    try:
        collection = client.get_collection(name=collection_name)
    except chromadb.errors.NotFoundError:
        print(f"[ERROR]  Collection '{collection_name}' does not exist. Run processing first.")
        return

    while True:
        user_input = input("\n Enter your article text (or type 'exit' to quit): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("\n Exiting RAG session. Goodbye!\n")
            break

        # --- CLEAN ---
        def clean(text: str):
            text = text.lower().strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-z0-9\s]', '', text)
            return text

        cleaned_text = clean(user_input)

        # --- EMBEDDING ---
        embedding_result = ollama.embed(model="all-minilm", input=cleaned_text)
        embedding_vector = embedding_result["embeddings"][0]

        # --- NORMALIZE ---
        olla = OllamaEmbedder()
        normalized_vector = olla.normalize_vector(embedding_vector)

        # --- QUERY ---
        query_results = collection.query(query_embeddings=[normalized_vector], n_results=2)

        if query_results['documents'] and query_results['documents'][0]:
            context_docs = "\n\n".join(query_results['documents'][0])
        else:
            context_docs = "No relevant data found."

        # --- PROMPT ---
        prompt = (
            "You are given a candidate article (User Article) and a context composed of similar "
            "document chunks retrieved from a trusted article database. Use the context to judge "
            "whether the User Article is TRUE (credible) or FAKE (misinformation). "
            "Return exactly one of the labels: TRUE or FAKE, then on the next lines provide a "
            "brief justification (2-4 bullet points) citing which context chunks support your decision.\n\n"
            "CONTEXT (retrieved chunks):\n"
            f"{context_docs}\n\n"
            "USER ARTICLE:\n"
            f"{user_input}\n\n"
            "OUTPUT FORMAT:\n"
            "Label: <TRUE or FAKE>\n"
            "Justification:\n"
            "- bullet 1 (cite source header or chunk snippet)\n"
            "- bullet 2\n\n"
            "Be concise, factual, and do not hallucinate. If the evidence is inconclusive, say 'INCONCLUSIVE' "
            "instead of TRUE/FAKE and explain why.\n\n"
            "Start now.\n"
        )

        print("\n Generating model response...\n")

        output = ollama.generate(model="phi3:mini", prompt=prompt)
        print("\n MODEL RESPONSE:\n")
        print(output.get("response", "No response generated."))

        # Ask if user wants to continue
        again = input("\n Do you want to analyze another article? (y/n): ").strip().lower()
        if again != "y":
            print("\n Exiting RAG session. Goodbye!\n")
            break
