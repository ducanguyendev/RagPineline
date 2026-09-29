"""
RAG Retriever

Load vector database and test similarity search.
"""

import argparse
import gc
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8"
    )


from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings



# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "BAAI/bge-m3"

DEFAULT_DB = "data/vectorstore/chroma"



# ============================================================
# LOAD VECTOR DB
# ============================================================

_VECTOR_DB_CACHE = {}


def clear_vector_db_cache(db_path: Path | None = None) -> int:
    """Drop cached Chroma references, optionally for one persistent directory."""
    if db_path is None:
        keys = list(_VECTOR_DB_CACHE)
    else:
        target = str(Path(db_path).resolve())
        keys = [key for key in _VECTOR_DB_CACHE if key[0] == target]

    removed = [_VECTOR_DB_CACHE.pop(key) for key in keys]
    removed_count = len(removed)
    del removed
    gc.collect()
    return removed_count

def load_vector_db(
    db_path: Path,
    model_name: str
):
    cache_key = (str(Path(db_path).resolve()), model_name)
    if cache_key in _VECTOR_DB_CACHE:
        return _VECTOR_DB_CACHE[cache_key]

    print("=" * 60)
    print(f"Loading embedding model: {model_name}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("Loading Chroma database...")

    db = Chroma(
        persist_directory=str(db_path),
        collection_name="rag_documents",
        embedding_function=embeddings
    )

    print("Vector database loaded")
    print("=" * 60)

    _VECTOR_DB_CACHE[cache_key] = db
    return db



# ============================================================
# SEARCH
# ============================================================

def search(
    db,
    query: str,
    k: int = 4
):

    print("\nQUERY:")
    print(query)

    print("\nSEARCH RESULTS")
    print("=" * 60)


    results = db.similarity_search_with_score(

        query,

        k=k

    )


    for index, (doc, score) in enumerate(
        results,
        start=1
    ):


        print(
            f"\n--- RESULT {index} ---"
        )


        print(
            f"Score: {score:.4f}"
        )


        print(
            "Metadata:"
        )


        for key, value in doc.metadata.items():

            print(
                f"  {key}: {value}"
            )


        print(
            "\nContent:"
        )


        print(
            doc.page_content[:1000]
        )



    return results



# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Test RAG retrieval"
    )


    parser.add_argument(

        "--db",

        type=Path,

        default=Path(
            DEFAULT_DB
        )

    )


    parser.add_argument(

        "--model",

        type=str,

        default=DEFAULT_MODEL

    )


    parser.add_argument(

        "--query",

        type=str,

        required=True

    )


    parser.add_argument(

        "--k",

        type=int,

        default=4

    )


    args = parser.parse_args()



    db = load_vector_db(

        args.db,

        args.model

    )


    search(

        db,

        args.query,

        args.k

    )



if __name__ == "__main__":
    main()
