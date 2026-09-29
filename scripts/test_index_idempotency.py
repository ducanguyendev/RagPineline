"""Verify stable-ID Chroma indexing without touching the production database."""

from __future__ import annotations

import gc
import hashlib
import math
import sys
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.indexer import (
    COLLECTION_NAME,
    count_vectors,
    load_chunks,
    upsert_documents,
)


SAMPLE_SIZE = 20


class DeterministicTestEmbeddings(Embeddings):
    """Small offline embeddings used only to test Chroma ID lifecycle."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [(value - 127.5) / 127.5 for value in digest[:16]]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _assert_separate_test_path(test_path: Path, production_path: Path) -> None:
    test_path = test_path.resolve()
    production_path = production_path.resolve()
    if (
        test_path == production_path
        or production_path in test_path.parents
        or test_path in production_path.parents
    ):
        raise RuntimeError(
            f"Refusing to run: test path {test_path} is not separate from "
            f"production path {production_path}."
        )


def main() -> None:
    input_file = ROOT / "data" / "processed" / "chunked.jsonl"
    production_path = ROOT / "data" / "vectorstore" / "chroma"
    documents = load_chunks(input_file, limit=SAMPLE_SIZE)
    if not documents:
        raise ValueError(f"No chunks were found in {input_file}.")

    with tempfile.TemporaryDirectory(
        prefix="ragpipeline-chroma-phase2-",
        ignore_cleanup_errors=True,
    ) as temp_dir:
        test_path = Path(temp_dir) / "chroma_phase2_test"
        _assert_separate_test_path(test_path, production_path)

        db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=DeterministicTestEmbeddings(),
            persist_directory=str(test_path),
        )
        upsert_documents(db, documents, batch_size=8, show_progress=False)
        first_count = count_vectors(db)

        upsert_documents(db, documents, batch_size=8, show_progress=False)
        second_count = count_vectors(db)
        duplicate_count = second_count - first_count

        print("=" * 60)
        print("CHROMA IDEMPOTENCY TEST")
        print("=" * 60)
        print(f"Input chunks : {len(documents)}")
        print(f"First count  : {first_count}")
        print(f"Second count : {second_count}")
        print(f"Duplicate    : {duplicate_count}")

        if first_count != len(documents):
            raise AssertionError(
                f"Expected {len(documents)} vectors after first index, got {first_count}."
            )
        if second_count != first_count:
            raise AssertionError(
                f"Second index changed vector count from {first_count} to {second_count}."
            )

        print("PASS")
        print("=" * 60)
        del db
        gc.collect()


if __name__ == "__main__":
    main()
