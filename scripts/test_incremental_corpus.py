"""Fast isolated integration test for the Phase 3 incremental corpus lifecycle."""

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

from src.corpus_manager import (
    atomic_write_jsonl,
    bootstrap_existing_corpus,
    CorpusConsistencyError,
    CorpusPaths,
    EmbeddingModelMismatchError,
    IncrementalCorpusManager,
    document_id_for_source,
    load_manifest,
    processing_fingerprint,
    read_jsonl,
    validate_manifest_corpus,
)
from src.data.chunker_optimized import ChunkConfig
from src.rag.bm25_retriever import clear_bm25_cache, load_bm25_retriever
from src.rag.indexer import COLLECTION_NAME, count_vectors


class DeterministicTestEmbeddings(Embeddings):
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


class FailOnceVectorDB:
    """Delegate to Chroma, but fail once after a real upsert for rollback testing."""

    def __init__(self, db: Chroma) -> None:
        self.db = db
        self.fail_next_add = True

    def get(self, *args, **kwargs):
        return self.db.get(*args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.db.delete(*args, **kwargs)

    def add_documents(self, *args, **kwargs):
        result = self.db.add_documents(*args, **kwargs)
        if self.fail_next_add:
            self.fail_next_add = False
            raise RuntimeError("intentional post-upsert failure")
        return result


def make_chunks(source: str, count: int, revision: str) -> list[dict]:
    document_id = document_id_for_source(source)
    records = []
    for index in range(1, count + 1):
        chunk_id = f"{document_id}:{revision}:{index:04d}"
        records.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "doc_id": document_id,
                "chunk_index": index,
                "chunk_content": (
                    f"Corpus marker {source} revision {revision} chunk {index}. "
                    f"Keyword {source[0].lower()}_document."
                ),
                "metrics": {"char_count": 1, "token_count": 1},
                "metadata": {
                    "source": source,
                    "source_relative_path": source,
                    "document_id": document_id,
                    "page": index,
                    "total_chunks": count,
                },
            }
        )
    return records


def assert_test_paths_are_isolated(paths: CorpusPaths) -> None:
    production = CorpusPaths.from_root(ROOT)
    for test_path, production_path in (
        (paths.vector_dir, production.vector_dir),
        (paths.canonical_chunks, production.canonical_chunks),
        (paths.manifest_path, production.manifest_path),
    ):
        if test_path.resolve() == production_path.resolve():
            raise RuntimeError(f"Test path points to production: {test_path}")


def main() -> None:
    config = ChunkConfig(
        strategy="structure_block_aware",
        target_tokens=600,
        token_overlap=100,
    )
    changed_config = ChunkConfig(
        strategy="structure_block_aware",
        target_tokens=700,
        token_overlap=100,
    )
    assert processing_fingerprint("a" * 64, config) != processing_fingerprint(
        "a" * 64, changed_config
    )
    with tempfile.TemporaryDirectory(
        prefix="ragpipeline-corpus-phase3-",
        ignore_cleanup_errors=True,
    ) as temp_dir:
        paths = CorpusPaths.from_root(Path(temp_dir))
        assert_test_paths_are_isolated(paths)
        paths.vector_dir.mkdir(parents=True, exist_ok=True)
        db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=DeterministicTestEmbeddings(),
            persist_directory=str(paths.vector_dir),
        )
        manager = IncrementalCorpusManager(
            paths,
            "BAAI/bge-m3",
            vector_db=db,
        )

        chunks_a = make_chunks("document-a.pdf", 20, "v1")
        baseline = manager.index_preprocessed_document(
            source_relative_path="document-a.pdf",
            pdf_sha256="a" * 64,
            file_size=100,
            config=config,
            records=chunks_a,
        )
        assert baseline["status"] == "new"
        assert count_vectors(db) == 20

        chunks_b_v1 = make_chunks("document-b.pdf", 7, "v1")
        added = manager.index_preprocessed_document(
            source_relative_path="document-b.pdf",
            pdf_sha256="b" * 64,
            file_size=200,
            config=config,
            records=chunks_b_v1,
        )
        assert added["status"] == "new"
        assert count_vectors(db) == 27
        primed_bm25 = load_bm25_retriever(paths.canonical_chunks)
        assert primed_bm25.N == 27

        unchanged = manager.index_preprocessed_document(
            source_relative_path="document-b.pdf",
            pdf_sha256="b" * 64,
            file_size=200,
            config=config,
            records=chunks_b_v1,
        )
        assert unchanged["status"] == "unchanged"
        assert unchanged["skipped"] is True
        assert count_vectors(db) == 27

        old_b_ids = {record["chunk_id"] for record in chunks_b_v1}
        chunks_b_v2 = make_chunks("document-b.pdf", 9, "v2")
        updated = manager.index_preprocessed_document(
            source_relative_path="document-b.pdf",
            pdf_sha256="c" * 64,
            file_size=220,
            config=config,
            records=chunks_b_v2,
        )
        assert updated["status"] == "updated"
        assert updated["stale_chunks_removed"] == 7
        assert count_vectors(db) == 29

        stale_ids = set(db.get(ids=sorted(old_b_ids), include=[]).get("ids", []))
        a_ids = {record["chunk_id"] for record in chunks_a}
        present_a_ids = set(db.get(ids=sorted(a_ids), include=[]).get("ids", []))
        assert not stale_ids
        assert present_a_ids == a_ids

        refreshed_bm25 = load_bm25_retriever(paths.canonical_chunks)
        assert refreshed_bm25 is not primed_bm25
        assert refreshed_bm25.N == 29
        bm25_sources = {
            (record.get("metadata") or {}).get("source")
            for record in refreshed_bm25.documents
        }
        assert bm25_sources == {"document-a.pdf", "document-b.pdf"}

        manifest = load_manifest(paths.manifest_path)
        assert manifest is not None
        canonical = read_jsonl(paths.canonical_chunks)
        validated = validate_manifest_corpus(manifest, paths)
        canonical_ids = [record["chunk_id"] for record in canonical]
        assert len(canonical) == 29
        assert len(set(canonical_ids)) == 29
        assert [record["chunk_id"] for record in validated] == canonical_ids
        assert sum(doc["chunk_count"] for doc in manifest["documents"].values()) == 29

        manifest_before_failure = paths.manifest_path.read_bytes()
        canonical_before_failure = paths.canonical_chunks.read_bytes()
        failing_manager = IncrementalCorpusManager(
            paths,
            "BAAI/bge-m3",
            vector_db=FailOnceVectorDB(db),
        )
        try:
            failing_manager.index_preprocessed_document(
                source_relative_path="document-c.pdf",
                pdf_sha256="d" * 64,
                file_size=300,
                config=config,
                records=make_chunks("document-c.pdf", 3, "v1"),
            )
        except RuntimeError as exc:
            assert "intentional post-upsert failure" in str(exc)
        else:
            raise AssertionError("Intentional vector failure did not propagate.")
        assert count_vectors(db) == 29
        assert paths.manifest_path.read_bytes() == manifest_before_failure
        assert paths.canonical_chunks.read_bytes() == canonical_before_failure

        duplicate_chunks = make_chunks("document-c.pdf", 1, "duplicate")
        duplicate_chunks[0]["chunk_id"] = chunks_a[0]["chunk_id"]
        try:
            manager.index_preprocessed_document(
                source_relative_path="document-c.pdf",
                pdf_sha256="e" * 64,
                file_size=301,
                config=config,
                records=duplicate_chunks,
            )
        except CorpusConsistencyError:
            pass
        else:
            raise AssertionError("Global duplicate chunk ID was not rejected.")
        assert count_vectors(db) == 29

        bootstrap_paths = CorpusPaths.from_root(Path(temp_dir) / "bootstrap-fixture")
        assert_test_paths_are_isolated(bootstrap_paths)
        bootstrap_paths.raw_dir.mkdir(parents=True, exist_ok=True)
        (bootstrap_paths.raw_dir / "bootstrap-a.pdf").write_bytes(b"fixture-pdf-a")
        (bootstrap_paths.raw_dir / "bootstrap-b.pdf").write_bytes(b"fixture-pdf-b")
        bootstrap_records = (
            make_chunks("bootstrap-a.pdf", 2, "stable")
            + make_chunks("bootstrap-b.pdf", 1, "stable")
        )
        atomic_write_jsonl(bootstrap_paths.canonical_chunks, bootstrap_records)
        bootstrap_paths.vector_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=DeterministicTestEmbeddings(),
            persist_directory=str(bootstrap_paths.vector_dir),
        )
        bootstrap_db.add_texts(
            [record["chunk_content"] for record in bootstrap_records],
            metadatas=[record["metadata"] for record in bootstrap_records],
            ids=[record["chunk_id"] for record in bootstrap_records],
        )
        bootstrap_canonical_before = bootstrap_paths.canonical_chunks.read_bytes()
        bootstrap_dry_run = bootstrap_existing_corpus(
            bootstrap_paths,
            "BAAI/bge-m3",
            config,
            dry_run=True,
        )
        assert bootstrap_dry_run["detected_documents"] == 2
        assert not bootstrap_paths.manifest_path.exists()
        bootstrap_apply = bootstrap_existing_corpus(
            bootstrap_paths,
            "BAAI/bge-m3",
            config,
            dry_run=False,
        )
        assert bootstrap_apply["current_chroma_vectors"] == 3
        assert count_vectors(bootstrap_db) == 3
        assert bootstrap_paths.canonical_chunks.read_bytes() == bootstrap_canonical_before
        bootstrap_manifest = load_manifest(bootstrap_paths.manifest_path)
        assert bootstrap_manifest is not None
        assert len(bootstrap_manifest["documents"]) == 2
        assert len(validate_manifest_corpus(bootstrap_manifest, bootstrap_paths)) == 3

        wrong_model_manager = IncrementalCorpusManager(
            paths,
            "another/embedding-model",
            vector_db=db,
        )
        try:
            wrong_model_manager.index_preprocessed_document(
                source_relative_path="document-c.pdf",
                pdf_sha256="d" * 64,
                file_size=300,
                config=config,
                records=make_chunks("document-c.pdf", 1, "v1"),
            )
        except EmbeddingModelMismatchError:
            pass
        else:
            raise AssertionError("Embedding model mismatch was not blocked.")

        print("=" * 60)
        print("INCREMENTAL CORPUS TEST")
        print("=" * 60)
        print(f"Baseline A             : {baseline['vectors_after']} vectors")
        print(f"Add B                  : {added['vectors_after']} vectors")
        print(f"Re-index unchanged B   : {unchanged['vectors_after']} vectors")
        print(f"Update B               : {updated['vectors_after']} vectors")
        print(f"Canonical chunks       : {len(canonical)}")
        print(f"Unique chunk IDs       : {len(set(canonical_ids))}")
        print("BM25 corpus            : synchronized")
        print(f"Stale vector IDs       : {len(stale_ids)}")
        print("Embedding model guard  : PASS")
        print("Failure rollback       : PASS")
        print("Global duplicate guard : PASS")
        print("Bootstrap fixture      : PASS")
        print("PASS")
        print("=" * 60)

        clear_bm25_cache(paths.canonical_chunks)
        del manager
        del db
        del bootstrap_db
        gc.collect()


if __name__ == "__main__":
    main()
