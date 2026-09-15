"""
RAG Indexer

Input:
    chunked.jsonl

Output:
    Chroma Vector Database

Embedding:
    BAAI/bge-m3
"""

import argparse
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8"
    )


from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings



# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "BAAI/bge-m3"

DEFAULT_DB = "data/vectorstore/chroma"



# ============================================================
# LOAD JSONL
# ============================================================

def load_chunks(
    path: Path
):

    documents = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            data = json.loads(
                line
            )

            raw_meta = {
                **data.get(
                    "metadata",
                    {}
                ),
                "chunk_id":
                    data.get(
                        "chunk_id",
                        ""
                    ),
                "doc_id":
                    data.get(
                        "doc_id",
                        ""
                    )
            }

            clean_meta = {}
            for k, v in raw_meta.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    if not v:
                        continue
                    clean_meta[k] = ",".join(str(x) for x in v)
                else:
                    clean_meta[k] = v

            documents.append(

                Document(

                    page_content=
                        data["chunk_content"],

                    metadata=clean_meta

                )

            )

    return documents



# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_index(

    input_file: Path,

    persist_dir: Path,

    model_name: str

):


    print("=" * 60)

    print("Loading chunks...")

    documents = load_chunks(
        input_file
    )


    print(
        f"Documents loaded: {len(documents)}"
    )


    print("=" * 60)

    print(
        f"Loading embedding model: {model_name}"
    )


    embeddings = HuggingFaceEmbeddings(

        model_name=model_name,

        model_kwargs={
            "device": "cuda"
        },

        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 1
        }

    )


    print("Building vector database...")


    persist_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    db = Chroma.from_documents(

        documents=documents,

        embedding=embeddings,

        persist_directory=str(
            persist_dir
        ),

        collection_name="rag_documents"

    )


    print("=" * 60)

    print("INDEX COMPLETE")

    print(
        f"Vectors stored: {len(documents)}"
    )

    print(
        f"Database: {persist_dir}"
    )

    print("=" * 60)


    return db



# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Build RAG vector index"
    )


    parser.add_argument(

        "--input",

        type=Path,

        default=Path(
            "data/processed/chunked.jsonl"
        )

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


    args = parser.parse_args()



    if not args.input.exists():

        raise FileNotFoundError(
            args.input
        )


    build_index(

        args.input,

        args.db,

        args.model

    )



if __name__ == "__main__":
    main()