"""CLI principal pour piloter deux fonctionnalités:
- build-db: pipeline complet pour construire la base vectorielle (prétraitement → embeddings → insertion ChromaDB)
- rag: système RAG pour analyser un article via le contexte retrouvé dans Chroma

Le fichier expose des fonctions réutilisables (run_build_vector_db, run_rag_system)
et un point d'entrée (main) utilisable via `python -m src.cli ...`.
"""

import argparse
import sys
from pathlib import Path

from src.preprocessing import CSVLoader, DataCleaner, DatasetMerger
from src.embedding import OllamaEmbedder
from src.storage_chroma import ChromaStorage
from src.rag_pipeline import RAGPipeline


def run_build_vector_db(
    true_csv: str = "data/raw/True.csv",
    fake_csv: str = "data/raw/Fake.csv",
    processed_true_out: str = "data/processed/cleaned_df_true.csv",
    processed_fake_out: str = "data/processed/cleaned_df_fake.csv",
    processed_merged_out: str = "data/processed/cleaned_df_all.csv",
    embedded_out: str = "data/processed/embedded_chunks_normalized.csv",
    chroma_path: str = "data/vector_db",
    collection_name: str = "articles",
    embedding_model: str = "all-minilm",
    chunk_size: int = 300,
    overlap: int = 30,
):
    """Exécute le pipeline de création de la base vectorielle.

    Étapes:
    1) Chargement des CSV True/Fake
    2) Nettoyage/normalisation + sauvegarde
    3) Fusion des datasets
    4) Chunking + embeddings (Ollama) + éventuelle sauvegarde CSV
    5) Création/chargement de la collection Chroma et insertion par lots
    """
    print("[CLI] Lancement du pipeline de construction de la base vectorielle...")

    loader = CSVLoader()
    # 1) Chargement
    df_true = loader.load_csv(true_csv)
    df_fake = loader.load_csv(fake_csv)

    # 2) Nettoyage et normalisation
    cleaner_true = DataCleaner(df_true)
    cleaned_df_true = (
        cleaner_true
        .add_label(1)
        .drop_empty_rows_and_duplicated()
        .remove_spaces()
        .lower_case()
        .date_format()
        .clean_all_text_columns()
        .get_df()
    )
    cleaner_true.save_csv(processed_true_out)

    cleaner_fake = DataCleaner(df_fake)
    cleaned_df_fake = (
        cleaner_fake
        .add_label(0)
        .drop_empty_rows_and_duplicated()
        .remove_spaces()
        .lower_case()
        .date_format()
        .clean_all_text_columns()
        .get_df()
    )
    cleaner_fake.save_csv(processed_fake_out)

    # 3) Fusion des deux jeux de données nettoyés
    merger = DatasetMerger()
    combined_df = merger.merge([cleaned_df_true, cleaned_df_fake])
    Path(processed_merged_out).parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(processed_merged_out, index=False)
    print(f"[INFO] Fusion terminée : {combined_df.shape[0]} articles combinés.")

    # 4) Embeddings + sauvegarde si demandé
    if not Path(embedded_out).exists():
        print("\n[INFO] Démarrage de la vectorisation avec Ollama...")
        try:
            embedder = OllamaEmbedder(model_name=embedding_model, chunk_size=chunk_size, overlap=overlap)
            embedded_df = embedder.embed_dataframe(combined_df, text_col="text", output_path=embedded_out)
            if embedded_df is None or embedded_df.empty:
                print("[ERREUR] Aucune embedding générée.")
                sys.exit(1)
        except Exception as e:
            print(f"[ERREUR] Échec de la vectorisation : {e}")
            sys.exit(1)
    else:
        print(f"[INFO] Embeddings déjà existants : {embedded_out}")

    # 5) Création/chargement de la collection et insertion par lots
    storage = ChromaStorage(persist_dir=chroma_path, collection_name=collection_name)
    df_loaded = storage.load_embedded_data(csv_path=embedded_out)
    storage.insert_into_chroma(df_loaded)

    print("\n[SUCCESS] Base vectorielle prête !")


def run_rag_system(
    chroma_path: str = "data/vector_db",
    collection_name: str = "articles",
    embedding_model: str = "all-minilm",
    generation_model: str = "llama3.2",
    n_results: int = 3,
    text: str | None = None,
):
    """Lance le système RAG pour analyser un texte.

    Source du texte:
    - `--text` (argument direct)
    - entrée interactive (stdin) si aucune source n'est fournie
    """
    print("[CLI] Lancement du système RAG...")

    rag = RAGPipeline(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # Choix de la source du texte à analyser
    if text:
        user_text = text
    else:
        print("\n=== Analyse d'article ===")
        print("Collez votre article puis validez par une ligne vide :")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "":
                break
            lines.append(line)
        user_text = "\n".join(lines)

    if not user_text.strip():
        print("[ERREUR] Aucun texte fourni. Collez un article.")
        sys.exit(2)

    # Envoi au pipeline RAG pour vectorisation requête → recherche → génération
    print("\n[INFO] Lancement de l'analyse RAG...")
    result = rag.analyze_article(user_text, model_name=generation_model, n_results=n_results)
    print("\n====== RÉPONSE DU MODÈLE ======")
    print(result.strip())


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments avec deux sous-commandes: build-db et rag."""
    parser = argparse.ArgumentParser(
        prog="fake-news-rag",
        description="CLI pour construire la base vectorielle et lancer le système RAG",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sous-commande: build-db
    p_build = subparsers.add_parser(
        "build-db",
        help="Exécute le pipeline de construction de la base vectorielle",
    )
    # Paramètres de données et sortie
    p_build.add_argument("--true-csv", default="data/raw/True.csv")
    p_build.add_argument("--fake-csv", default="data/raw/Fake.csv")
    p_build.add_argument("--processed-true-out", default="data/processed/cleaned_df_true.csv")
    p_build.add_argument("--processed-fake-out", default="data/processed/cleaned_df_fake.csv")
    p_build.add_argument("--processed-merged-out", default="data/processed/cleaned_df_all.csv")
    p_build.add_argument("--embedded-out", default="data/processed/embedded_chunks_normalized.csv")
    # Paramètres Chroma/embedding
    p_build.add_argument("--chroma-path", default="data/vector_db")
    p_build.add_argument("--collection", default="articles")
    p_build.add_argument("--embedding-model", default="all-minilm")
    p_build.add_argument("--chunk-size", type=int, default=300)
    p_build.add_argument("--overlap", type=int, default=30)

    # Sous-commande: rag
    p_rag = subparsers.add_parser(
        "rag",
        help="Lance le système RAG pour analyser un article",
    )
    p_rag.add_argument("--chroma-path", default="data/vector_db")
    p_rag.add_argument("--collection", default="articles")
    p_rag.add_argument("--embedding-model", default="all-minilm")
    p_rag.add_argument("--generation-model", default="llama3.2")
    p_rag.add_argument("--n-results", type=int, default=3)
    # Une seule source de texte possible
    src = p_rag.add_mutually_exclusive_group()
    src.add_argument("--text", help="Texte de l'article à analyser")

    return parser


def main(argv: list[str] | None = None):
    """Point d'entrée du module CLI.

    Décode les arguments et délègue aux fonctions dédiées.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-db":
        run_build_vector_db(
            true_csv=args.true_csv,
            fake_csv=args.fake_csv,
            processed_true_out=args.processed_true_out,
            processed_fake_out=args.processed_fake_out,
            processed_merged_out=args.processed_merged_out,
            embedded_out=args.embedded_out,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    elif args.command == "rag":
        run_rag_system(
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model,
            n_results=args.n_results,
            text=args.text,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



