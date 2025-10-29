from src.rag_pipeline import run_processing_pipeline, run_retrieval_and_rag


if __name__ == "__main__":
    print("=== Fake News Detection System ===")
    print(" Run Processing + Embedding + Insert into Chroma")
    print(" Run Retrieval + RAG")
    choice = input("\nChoose an option (1 or 2): ").strip()

    if choice == "1":
        run_processing_pipeline()
    elif choice == "2":
        run_retrieval_and_rag()
    else:
        print("Invalid choice. Please choose 1 or 2.")
