"""Point d'entrée principal de l'application.

Deux modes de lancement:
- Sans arguments: menu interactif pour choisir entre build-db et rag
- Avec sous-commandes: interface équivalente à `python -m src.cli ...`

Ce fichier délègue la logique métier aux fonctions exposées par `src.cli`.
"""

import argparse
import sys
from src.cli import run_build_vector_db, run_rag_system

# === CONFIGURATION ===
CHROMA_PATH = "data/vector_db"
COLLECTION_NAME = "articles"
EMBEDDING_MODEL = "all-minilm"
GENERATION_MODEL = "llama3.2"

def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments pour `main.py`.

    Les sous-commandes reflètent celles de `src.cli` afin de garder un
    comportement homogène quelque soit le point d'entrée utilisé.
    """
    parser = argparse.ArgumentParser(
        prog="fake-news-rag",
        description="Point d'entrée principal. Choisissez build-db ou rag.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # build-db
    p_build = subparsers.add_parser("build-db", help="Construire la base vectorielle")
    p_build.add_argument("--true-csv", default="data/raw/True.csv")
    p_build.add_argument("--fake-csv", default="data/raw/Fake.csv")
    p_build.add_argument("--processed-true-out", default="data/processed/cleaned_df_true.csv")
    p_build.add_argument("--processed-fake-out", default="data/processed/cleaned_df_fake.csv")
    p_build.add_argument("--processed-merged-out", default="data/processed/cleaned_df_all.csv")
    p_build.add_argument("--embedded-out", default="data/processed/embedded_chunks_normalized.csv")
    p_build.add_argument("--chroma-path", default=CHROMA_PATH)
    p_build.add_argument("--collection", default=COLLECTION_NAME)
    p_build.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    p_build.add_argument("--chunk-size", type=int, default=300)
    p_build.add_argument("--overlap", type=int, default=30)

    # rag
    p_rag = subparsers.add_parser("rag", help="Lancer l'analyse RAG")
    p_rag.add_argument("--chroma-path", default=CHROMA_PATH)
    p_rag.add_argument("--collection", default=COLLECTION_NAME)
    p_rag.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    p_rag.add_argument("--generation-model", default=GENERATION_MODEL)
    p_rag.add_argument("--n-results", type=int, default=3)
    src = p_rag.add_mutually_exclusive_group()
    src.add_argument("--text", help="Texte de l'article à analyser")

    return parser


def interactive_menu():
    """Affiche un menu simple quand aucun argument n'est fourni."""
    print("=== Fake News RAG - Menu ===")
    print("1) Construire la base vectorielle (build-db)")
    print("2) Lancer le système RAG (rag)")
    choice = input("Votre choix [1/2] : ").strip()
    return choice


def main(argv: list[str] | None = None):
    """Point d'entrée du programme.

    - Si aucune sous-commande n'est fournie, on bascule en mode interactif.
    - Sinon, on délègue au lanceur correspondant (build-db ou rag).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Mode interactif si aucune commande fournie
    if not args.command:
        choice = interactive_menu()
        if choice == "1":
            # utiliser les valeurs par défaut
            run_build_vector_db()
            return
        elif choice == "2":
            run_rag_system(
                chroma_path=CHROMA_PATH,
                collection_name=COLLECTION_NAME,
                embedding_model=EMBEDDING_MODEL,
                generation_model=GENERATION_MODEL,
                n_results=5,
            )
            return
        else:
            print("Choix invalide.")
            parser.print_help()
            return 1

    # Mode sous-commandes
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
