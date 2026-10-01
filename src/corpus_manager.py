"""Persistent multi-document corpus lifecycle for incremental RAG indexing."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from langchain_chroma import Chroma

from src.data.chunker_optimized import ChunkConfig
from src.rag.bm25_retriever import clear_bm25_cache
from src.rag.indexer import (
    COLLECTION_NAME,
    build_index,
    count_vectors,
    initialize_embeddings,
    load_chunks,
    upsert_documents,
)
from src.rag.retriever import clear_vector_db_cache


MANIFEST_VERSION = 1
PIPELINE_SCHEMA_VERSION = "incremental-corpus-v1"
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"


class CorpusError(RuntimeError):
    """Base error for corpus lifecycle failures."""


class CorpusBootstrapRequiredError(CorpusError):
    """Raised when a legacy canonical corpus has no manifest yet."""


class CorpusConsistencyError(CorpusError):
    """Raised when manifest, canonical chunks, or vectors disagree."""


class EmbeddingModelMismatchError(CorpusError):
    """Raised when an incremental operation would mix embedding models."""


@dataclass(frozen=True)
class CorpusPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    documents_dir: Path
    canonical_chunks: Path
    manifest_path: Path
    vector_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "CorpusPaths":
        root = Path(root).resolve()
        processed_dir = root / "data" / "processed"
        return cls(
            root=root,
            raw_dir=root / "data" / "raw",
            processed_dir=processed_dir,
            documents_dir=processed_dir / "documents",
            canonical_chunks=processed_dir / "chunked.jsonl",
            manifest_path=processed_dir / "corpus_manifest.json",
            vector_dir=root / "data" / "vectorstore" / "chroma",
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_source_path(source_relative_path: str | Path) -> str:
    value = unicodedata.normalize("NFC", str(source_relative_path).replace("\\", "/"))
    normalized = PurePosixPath(value).as_posix().lstrip("./")
    if not normalized or normalized == "." or ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"Unsafe or empty source path: {source_relative_path!r}")
    return normalized


def _source_identity_key(source_relative_path: str | Path) -> str:
    return normalize_source_path(source_relative_path).casefold()


def document_id_for_source(source_relative_path: str | Path) -> str:
    normalized = normalize_source_path(source_relative_path)
    identity_hash = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()[:16]
    ascii_stem = (
        unicodedata.normalize("NFKD", PurePosixPath(normalized).stem)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", ascii_stem).strip("-_").lower()
    slug = (slug or "document")[:48]
    return f"{slug}-{identity_hash}"


def chunk_config_payload(config: ChunkConfig) -> dict[str, Any]:
    return {
        "strategy": str(config.strategy),
        "target_tokens": int(config.target_tokens),
        "token_overlap": int(config.token_overlap),
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
    }


def processing_fingerprint(pdf_sha256: str, config: ChunkConfig) -> str:
    payload = {
        "pdf_sha256": pdf_sha256,
        **chunk_config_payload(config),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorpusConsistencyError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
    return records


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_bytes(path, serialized.encode("utf-8"))


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    serialized = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    _atomic_write_bytes(path, serialized.encode("utf-8"))


def load_manifest(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusConsistencyError(f"Cannot read corpus manifest {path}: {exc}") from exc
    if manifest.get("version") != MANIFEST_VERSION:
        raise CorpusConsistencyError(
            f"Unsupported corpus manifest version: {manifest.get('version')!r}."
        )
    if not isinstance(manifest.get("documents"), dict):
        raise CorpusConsistencyError("Corpus manifest field 'documents' must be an object.")
    return manifest


def new_manifest(embedding_model: str) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "embedding_model": embedding_model,
        "updated_at": utc_now(),
        "documents": {},
    }


def _record_chunk_id(record: dict[str, Any]) -> str:
    raw_id = record.get("chunk_id")
    return str(raw_id).strip() if raw_id is not None else ""


def validate_document_chunks(
    records: list[dict[str, Any]],
    *,
    document_label: str,
) -> list[str]:
    if not records:
        raise CorpusConsistencyError(f"Document {document_label!r} produced no chunks.")

    chunk_ids: list[str] = []
    seen: set[str] = set()
    for position, record in enumerate(records, start=1):
        chunk_id = _record_chunk_id(record)
        if not chunk_id:
            raise CorpusConsistencyError(
                f"Document {document_label!r} has a missing chunk_id at record {position}."
            )
        if chunk_id in seen:
            raise CorpusConsistencyError(
                f"Document {document_label!r} contains duplicate chunk_id {chunk_id!r}."
            )
        seen.add(chunk_id)
        chunk_ids.append(chunk_id)
    return chunk_ids


def _resolve_processed_path(paths: CorpusPaths, relative_path: str) -> Path:
    candidate = (paths.processed_dir / PurePosixPath(relative_path)).resolve()
    boundary = paths.processed_dir.resolve()
    if candidate == boundary or boundary not in candidate.parents:
        raise CorpusConsistencyError(
            f"Manifest chunks_path escapes processed directory: {relative_path!r}."
        )
    return candidate


def validate_manifest_corpus(
    manifest: dict[str, Any],
    paths: CorpusPaths,
    *,
    overrides: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    overrides = overrides or {}
    global_owners: dict[str, str] = {}
    canonical_records: list[dict[str, Any]] = []

    document_items = sorted(
        manifest["documents"].items(),
        key=lambda item: _source_identity_key(item[1]["source_relative_path"]),
    )
    for document_id, document in document_items:
        if document.get("status") != "indexed":
            continue
        if document_id in overrides:
            records = overrides[document_id]
        else:
            chunks_path = _resolve_processed_path(paths, document["chunks_path"])
            if not chunks_path.exists():
                raise CorpusConsistencyError(
                    f"Missing per-document chunks for {document_id}: {chunks_path}"
                )
            records = read_jsonl(chunks_path)

        chunk_ids = validate_document_chunks(records, document_label=document_id)
        manifest_ids = [str(value) for value in document.get("chunk_ids", [])]
        if chunk_ids != manifest_ids:
            raise CorpusConsistencyError(
                f"Manifest chunk IDs do not match per-document chunks for {document_id}."
            )
        if len(chunk_ids) != int(document.get("chunk_count", -1)):
            raise CorpusConsistencyError(
                f"Manifest chunk_count does not match chunks for {document_id}."
            )

        for chunk_id in chunk_ids:
            previous_owner = global_owners.get(chunk_id)
            if previous_owner is not None and previous_owner != document_id:
                raise CorpusConsistencyError(
                    f"Duplicate global chunk ID {chunk_id!r} belongs to both "
                    f"{previous_owner!r} and {document_id!r}."
                )
            global_owners[chunk_id] = document_id
        canonical_records.extend(records)

    return canonical_records


def read_chroma_vector_count(
    vector_dir: Path,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Read a persistent Chroma count through immutable, read-only SQLite."""
    sqlite_path = Path(vector_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return 0

    uri = f"{sqlite_path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        try:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM embeddings AS e
                JOIN segments AS s ON e.segment_id = s.id
                JOIN collections AS c ON s.collection = c.id
                WHERE c.name = ?
                """,
                (collection_name,),
            ).fetchone()
        except sqlite3.DatabaseError:
            row = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0]) if row else 0
    finally:
        connection.close()


def read_chroma_vector_ids(
    vector_dir: Path,
    collection_name: str = COLLECTION_NAME,
) -> set[str]:
    """Read persistent Chroma IDs without opening the database in write mode."""
    sqlite_path = Path(vector_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return set()

    uri = f"{sqlite_path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        try:
            rows = connection.execute(
                """
                SELECT e.embedding_id
                FROM embeddings AS e
                JOIN segments AS s ON e.segment_id = s.id
                JOIN collections AS c ON s.collection = c.id
                WHERE c.name = ?
                """,
                (collection_name,),
            ).fetchall()
        except sqlite3.DatabaseError:
            rows = connection.execute("SELECT embedding_id FROM embeddings").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        connection.close()


def _safe_remove_tree(path: Path, boundary: Path) -> None:
    resolved_path = Path(path).resolve()
    resolved_boundary = Path(boundary).resolve()
    if resolved_path == resolved_boundary or resolved_boundary not in resolved_path.parents:
        raise CorpusError(f"Refusing to remove unsafe path: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _find_raw_pdf(raw_dir: Path, source: str) -> Path | None:
    if not raw_dir.exists():
        return None
    target_key = _source_identity_key(source)
    for candidate in raw_dir.rglob("*.pdf"):
        relative = candidate.relative_to(raw_dir).as_posix()
        if _source_identity_key(relative) == target_key:
            return candidate
    return None


def _serialized_update(function):
    """Serialize manifest/canonical/vector mutations inside one app process."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self._update_lock:
            return function(self, *args, **kwargs)

    return wrapped


class IncrementalCorpusManager:
    """Coordinate per-document processing, Chroma updates, and corpus metadata."""

    def __init__(
        self,
        paths: CorpusPaths,
        embedding_model: str,
        *,
        vector_db: Chroma | None = None,
        vector_db_factory: Callable[[], Chroma] | None = None,
    ) -> None:
        self.paths = paths
        self.embedding_model = embedding_model
        self._injected_vector_db = vector_db
        self._vector_db_factory = vector_db_factory
        self._last_runtime: dict[str, Any] = {}
        self._update_lock = threading.RLock()

    def _load_manifest_for_update(self) -> dict[str, Any]:
        manifest = load_manifest(self.paths.manifest_path)
        if manifest is None:
            if self.paths.canonical_chunks.exists() and read_jsonl(
                self.paths.canonical_chunks
            ):
                raise CorpusBootstrapRequiredError(
                    "Canonical chunks already exist but corpus_manifest.json is missing. "
                    "Run scripts/bootstrap_corpus_manifest.py after review before "
                    "incremental indexing."
                )
            return new_manifest(self.embedding_model)
        self._assert_embedding_model(manifest)
        self._ensure_canonical_matches_manifest(manifest)
        return manifest

    def _assert_embedding_model(self, manifest: dict[str, Any]) -> None:
        existing_model = manifest.get("embedding_model")
        if existing_model != self.embedding_model:
            raise EmbeddingModelMismatchError(
                "Embedding model mismatch. "
                f"Existing corpus: {existing_model}. Requested: {self.embedding_model}. "
                "Full rebuild required."
            )

    def _ensure_canonical_matches_manifest(self, manifest: dict[str, Any]) -> None:
        expected_records = validate_manifest_corpus(manifest, self.paths)
        if not self.paths.canonical_chunks.exists():
            raise CorpusConsistencyError(
                f"Canonical corpus is missing: {self.paths.canonical_chunks}"
            )
        actual_records = read_jsonl(self.paths.canonical_chunks)
        expected_ids = [_record_chunk_id(record) for record in expected_records]
        actual_ids = validate_document_chunks(
            actual_records, document_label="canonical corpus"
        )
        if len(expected_ids) != len(actual_ids) or set(expected_ids) != set(actual_ids):
            raise CorpusConsistencyError(
                "Canonical chunked.jsonl does not match active manifest chunk IDs."
            )

    def _source_relative_path(self, pdf_path: Path) -> str:
        pdf_path = Path(pdf_path).resolve()
        raw_dir = self.paths.raw_dir.resolve()
        try:
            relative = pdf_path.relative_to(raw_dir)
        except ValueError as exc:
            raise CorpusError(f"PDF must be inside {raw_dir}: {pdf_path}") from exc
        return normalize_source_path(relative.as_posix())

    def _read_vector_count(self) -> int:
        if self._injected_vector_db is not None:
            return count_vectors(self._injected_vector_db)
        return read_chroma_vector_count(self.paths.vector_dir)

    def _open_vector_db(self) -> Chroma:
        if self._injected_vector_db is not None:
            return self._injected_vector_db
        if self._vector_db_factory is not None:
            return self._vector_db_factory()

        runtime = initialize_embeddings(model_name=self.embedding_model)
        self._last_runtime = {
            "device": runtime.device,
            "device_name": runtime.device_name,
            "precision": runtime.precision,
            "embedding_batch_size": runtime.batch_size,
        }
        self.paths.vector_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=runtime.embeddings,
            persist_directory=str(self.paths.vector_dir),
        )

    def _corpus_chunk_count(self, manifest: dict[str, Any]) -> int:
        return sum(
            int(document.get("chunk_count", 0))
            for document in manifest["documents"].values()
            if document.get("status") == "indexed"
        )

    def _ensure_vector_count_matches(
        self,
        manifest: dict[str, Any],
        vector_count: int,
    ) -> None:
        expected = self._corpus_chunk_count(manifest)
        if vector_count != expected:
            raise CorpusConsistencyError(
                "Dense corpus is not synchronized with the manifest. "
                f"Manifest chunks: {expected}; Chroma vectors: {vector_count}."
            )

    def _unchanged_result(
        self,
        manifest: dict[str, Any],
        document_id: str,
        source: str,
        document: dict[str, Any],
        started_at: float,
    ) -> dict[str, Any]:
        vectors = self._read_vector_count()
        self._ensure_vector_count_matches(manifest, vectors)
        result = {
            "document_id": document_id,
            "source": source,
            "status": "unchanged",
            "skipped": True,
            "chunks_before": int(document["chunk_count"]),
            "chunks_after": int(document["chunk_count"]),
            "vectors_before": vectors,
            "vectors_after": vectors,
            "stale_chunks_removed": 0,
            "timings": {
                "loading": 0.0,
                "captioning": 0.0,
                "chunking": 0.0,
                "embedding_indexing": 0.0,
                "total": round((time.perf_counter() - started_at) * 1000, 2),
            },
        }
        print(f"Document       : {source}")
        print("Status         : UNCHANGED")
        print("Processing     : SKIPPED")
        print(f"Vectors before : {vectors}")
        print(f"Vectors after  : {vectors}")
        return result

    def _prepare_document_context(
        self,
        source_relative_path: str,
        pdf_sha256: str,
        file_size: int,
        config: ChunkConfig,
        started_at: float,
    ) -> tuple[dict[str, Any], str, str, dict[str, Any] | None, dict[str, Any] | None]:
        source_relative_path = normalize_source_path(source_relative_path)
        document_id = document_id_for_source(source_relative_path)
        fingerprint = processing_fingerprint(pdf_sha256, config)
        manifest = self._load_manifest_for_update()
        previous = manifest["documents"].get(document_id)

        if previous and previous.get("processing_fingerprint") == fingerprint:
            return (
                manifest,
                document_id,
                fingerprint,
                previous,
                self._unchanged_result(
                    manifest,
                    document_id,
                    PurePosixPath(source_relative_path).name,
                    previous,
                    started_at,
                ),
            )
        return manifest, document_id, fingerprint, previous, None

    @_serialized_update
    def index_document(self, pdf_path: Path, config: ChunkConfig) -> dict[str, Any]:
        """Parse and incrementally index one new or changed PDF."""
        started_at = time.perf_counter()
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)

        source_relative_path = self._source_relative_path(pdf_path)
        pdf_sha256 = sha256_file(pdf_path)
        file_size = pdf_path.stat().st_size
        manifest, document_id, fingerprint, previous, unchanged = (
            self._prepare_document_context(
                source_relative_path,
                pdf_sha256,
                file_size,
                config,
                started_at,
            )
        )
        if unchanged is not None:
            return unchanged

        staging_dir, final_dir = self._new_version_paths(document_id, fingerprint)
        timings = {
            "loading": 0.0,
            "captioning": 0.0,
            "chunking": 0.0,
            "embedding_indexing": 0.0,
        }
        try:
            records = self._process_pdf_to_staging(
                pdf_path,
                source_relative_path,
                document_id,
                config,
                staging_dir,
                final_dir,
                timings,
            )
            return self._commit_document_update(
                manifest=manifest,
                previous=previous,
                document_id=document_id,
                source_relative_path=source_relative_path,
                pdf_sha256=pdf_sha256,
                file_size=file_size,
                fingerprint=fingerprint,
                config=config,
                records=records,
                staging_dir=staging_dir,
                final_dir=final_dir,
                timings=timings,
                started_at=started_at,
            )
        except Exception:
            if staging_dir.exists():
                _safe_remove_tree(staging_dir, self.paths.documents_dir)
            raise

    @_serialized_update
    def index_preprocessed_document(
        self,
        *,
        source_relative_path: str,
        pdf_sha256: str,
        file_size: int,
        config: ChunkConfig,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Index deterministic preprocessed chunks; used by fast lifecycle tests."""
        started_at = time.perf_counter()
        manifest, document_id, fingerprint, previous, unchanged = (
            self._prepare_document_context(
                source_relative_path,
                pdf_sha256,
                file_size,
                config,
                started_at,
            )
        )
        if unchanged is not None:
            return unchanged

        staging_dir, final_dir = self._new_version_paths(document_id, fingerprint)
        staging_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_jsonl(staging_dir / "chunks.jsonl", records)
        timings = {
            "loading": 0.0,
            "captioning": 0.0,
            "chunking": 0.0,
            "embedding_indexing": 0.0,
        }
        try:
            return self._commit_document_update(
                manifest=manifest,
                previous=previous,
                document_id=document_id,
                source_relative_path=normalize_source_path(source_relative_path),
                pdf_sha256=pdf_sha256,
                file_size=file_size,
                fingerprint=fingerprint,
                config=config,
                records=records,
                staging_dir=staging_dir,
                final_dir=final_dir,
                timings=timings,
                started_at=started_at,
            )
        except Exception:
            if staging_dir.exists():
                _safe_remove_tree(staging_dir, self.paths.documents_dir)
            raise

    def _new_version_paths(self, document_id: str, fingerprint: str) -> tuple[Path, Path]:
        staging_root = self.paths.documents_dir / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        version_token = f"{fingerprint[:16]}-{uuid.uuid4().hex[:8]}"
        staging_dir = staging_root / f"{document_id}-{version_token}"
        final_dir = (
            self.paths.documents_dir
            / document_id
            / "versions"
            / version_token
        )
        return staging_dir, final_dir

    def _process_pdf_to_staging(
        self,
        pdf_path: Path,
        source_relative_path: str,
        document_id: str,
        config: ChunkConfig,
        staging_dir: Path,
        final_dir: Path,
        timings: dict[str, float],
    ) -> list[dict[str, Any]]:
        from src.data.chunker_optimized import process_chunking
        from src.data.image_captioner import caption_elements
        from src.data.pdf_loader import convert_pdfs

        staging_dir.mkdir(parents=True, exist_ok=False)
        pdf_extract = staging_dir / "pdf_extract.jsonl"
        elements = staging_dir / "elements.jsonl"
        chunks = staging_dir / "chunks.jsonl"
        images_dir = staging_dir / "images"

        try:
            final_relative = final_dir.resolve().relative_to(self.paths.root.resolve())
        except ValueError as exc:
            raise CorpusError("Document artifact directory must be inside project root.") from exc
        image_path_prefix = f"{final_relative.as_posix()}/images"

        loading_start = time.perf_counter()
        convert_pdfs(
            pdf_path,
            pdf_extract,
            elements,
            images_dir=images_dir,
            image_path_prefix=image_path_prefix,
        )
        timings["loading"] = round((time.perf_counter() - loading_start) * 1000, 2)
        if not elements.exists() or not read_jsonl(elements):
            raise CorpusError(f"PDF parsing produced no elements for {pdf_path.name}.")

        caption_start = time.perf_counter()
        try:
            caption_elements(
                elements,
                model_name=DEFAULT_CAPTION_MODEL,
                image_root_override=images_dir,
            )
        except Exception as exc:
            # Preserve the existing pipeline behavior: captions are best-effort.
            print(f"! Captioner warning: {exc}")
        timings["captioning"] = round(
            (time.perf_counter() - caption_start) * 1000, 2
        )

        chunking_start = time.perf_counter()
        process_chunking(elements, chunks, config)
        timings["chunking"] = round(
            (time.perf_counter() - chunking_start) * 1000, 2
        )
        raw_records = read_jsonl(chunks)
        records = self._normalize_processed_chunks(
            raw_records,
            document_id=document_id,
            source_relative_path=source_relative_path,
        )
        atomic_write_jsonl(chunks, records)
        return records

    def _normalize_processed_chunks(
        self,
        records: list[dict[str, Any]],
        *,
        document_id: str,
        source_relative_path: str,
    ) -> list[dict[str, Any]]:
        source_name = PurePosixPath(source_relative_path).name
        normalized: list[dict[str, Any]] = []
        total_chunks = len(records)
        for chunk_index, original in enumerate(records, start=1):
            record = copy.deepcopy(original)
            chunk_id = f"{document_id}:chunk:{chunk_index:06d}"
            record["chunk_id"] = chunk_id
            record["document_id"] = document_id
            record["doc_id"] = document_id
            record["chunk_index"] = chunk_index
            metadata = dict(record.get("metadata") or {})
            metadata.update(
                {
                    "source": source_name,
                    "source_relative_path": source_relative_path,
                    "document_id": document_id,
                    "total_chunks": total_chunks,
                }
            )
            record["metadata"] = metadata
            normalized.append(record)
        validate_document_chunks(normalized, document_label=document_id)
        return normalized

    def _ids_present(self, db: Chroma, ids: set[str]) -> set[str]:
        if not ids:
            return set()
        present: set[str] = set()
        id_list = sorted(ids)
        for start in range(0, len(id_list), 1000):
            result = db.get(ids=id_list[start : start + 1000], include=[])
            present.update(str(value) for value in result.get("ids", []))
        return present

    def _rollback_vectors(
        self,
        db: Chroma,
        new_ids: set[str],
        old_chunks_path: Path | None,
    ) -> None:
        if new_ids:
            db.delete(ids=sorted(new_ids))
        if old_chunks_path is not None:
            old_documents = load_chunks(old_chunks_path)
            batch_size = max(1, int(os.getenv("CHROMA_BATCH_SIZE", "128")))
            upsert_documents(
                db,
                old_documents,
                batch_size=batch_size,
                show_progress=False,
            )

    def _commit_document_update(
        self,
        *,
        manifest: dict[str, Any],
        previous: dict[str, Any] | None,
        document_id: str,
        source_relative_path: str,
        pdf_sha256: str,
        file_size: int,
        fingerprint: str,
        config: ChunkConfig,
        records: list[dict[str, Any]],
        staging_dir: Path,
        final_dir: Path,
        timings: dict[str, float],
        started_at: float,
    ) -> dict[str, Any]:
        new_ids_list = validate_document_chunks(records, document_label=document_id)
        new_ids = set(new_ids_list)
        old_ids = set(str(value) for value in (previous or {}).get("chunk_ids", []))
        old_chunks_path = None
        if previous is not None:
            old_chunks_path = _resolve_processed_path(
                self.paths, previous["chunks_path"]
            )
            if not old_chunks_path.exists():
                raise CorpusConsistencyError(
                    f"Previous chunks are missing for {document_id}: {old_chunks_path}"
                )

        final_chunks_path = final_dir / "chunks.jsonl"
        chunks_relative_path = final_chunks_path.relative_to(
            self.paths.processed_dir
        ).as_posix()
        now = utc_now()
        document_record = {
            "document_id": document_id,
            "source": PurePosixPath(source_relative_path).name,
            "source_relative_path": source_relative_path,
            "sha256": pdf_sha256,
            "file_size": int(file_size),
            "processing_fingerprint": fingerprint,
            "processing": chunk_config_payload(config),
            "chunk_count": len(new_ids_list),
            "chunk_ids": new_ids_list,
            "chunks_path": chunks_relative_path,
            "status": "indexed",
            "embedding_model": self.embedding_model,
            "indexed_at": (previous or {}).get("indexed_at", now),
            "updated_at": now,
        }
        candidate_manifest = copy.deepcopy(manifest)
        candidate_manifest["updated_at"] = now
        candidate_manifest["documents"][document_id] = document_record
        canonical_records = validate_manifest_corpus(
            candidate_manifest,
            self.paths,
            overrides={document_id: records},
        )

        old_canonical_bytes = (
            self.paths.canonical_chunks.read_bytes()
            if self.paths.canonical_chunks.exists()
            else None
        )
        db = self._open_vector_db()
        vectors_before = count_vectors(db)
        self._ensure_vector_count_matches(manifest, vectors_before)

        existing_new_ids = self._ids_present(db, new_ids)
        unexpected_collisions = existing_new_ids - old_ids
        if unexpected_collisions:
            sample = sorted(unexpected_collisions)[:3]
            raise CorpusConsistencyError(
                "New chunk IDs already exist outside this document: " + ", ".join(sample)
            )
        if previous is not None:
            present_old_ids = self._ids_present(db, old_ids)
            missing_old_ids = old_ids - present_old_ids
            if missing_old_ids:
                raise CorpusConsistencyError(
                    f"Chroma is missing {len(missing_old_ids)} old IDs for {document_id}."
                )

        mutation_started = False
        canonical_replaced = False
        embedding_start = time.perf_counter()
        try:
            mutation_started = True
            documents = load_chunks(staging_dir / "chunks.jsonl")
            batch_size = max(1, int(os.getenv("CHROMA_BATCH_SIZE", "128")))
            upsert_documents(db, documents, batch_size=batch_size)

            stale_ids = old_ids - new_ids
            if stale_ids:
                db.delete(ids=sorted(stale_ids))

            vectors_after = count_vectors(db)
            expected_after = len(canonical_records)
            if vectors_after != expected_after:
                raise CorpusConsistencyError(
                    "Incremental vector count mismatch. "
                    f"Expected {expected_after}; got {vectors_after}."
                )
            timings["embedding_indexing"] = round(
                (time.perf_counter() - embedding_start) * 1000, 2
            )

            final_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_dir, final_dir)
            atomic_write_jsonl(self.paths.canonical_chunks, canonical_records)
            canonical_replaced = True
            atomic_write_json(self.paths.manifest_path, candidate_manifest)
        except Exception as original_error:
            rollback_error: Exception | None = None
            if mutation_started:
                try:
                    self._rollback_vectors(db, new_ids, old_chunks_path)
                except Exception as exc:
                    rollback_error = exc
            if canonical_replaced:
                try:
                    if old_canonical_bytes is None:
                        self.paths.canonical_chunks.unlink(missing_ok=True)
                    else:
                        _atomic_write_bytes(
                            self.paths.canonical_chunks, old_canonical_bytes
                        )
                except Exception as exc:
                    rollback_error = rollback_error or exc
            clear_vector_db_cache(self.paths.vector_dir)
            if rollback_error is not None:
                raise CorpusError(
                    f"Corpus update failed ({original_error}); rollback also failed "
                    f"({rollback_error}). Manual recovery is required."
                ) from original_error
            raise

        clear_bm25_cache(self.paths.canonical_chunks)
        clear_vector_db_cache(self.paths.vector_dir)
        status = "updated" if previous is not None else "new"
        stale_count = len(old_ids - new_ids)
        timings["total"] = round((time.perf_counter() - started_at) * 1000, 2)
        result = {
            "document_id": document_id,
            "source": document_record["source"],
            "status": status,
            "skipped": False,
            "chunks_before": len(old_ids),
            "chunks_after": len(new_ids),
            "vectors_before": vectors_before,
            "vectors_after": vectors_after,
            "stale_chunks_removed": stale_count,
            "timings": timings,
            **self._last_runtime,
        }
        print(f"Document       : {document_record['source']}")
        print(f"Status         : {status.upper()}")
        print(f"Old chunks     : {len(old_ids)}")
        print(f"New chunks     : {len(new_ids)}")
        print(f"Stale IDs      : {stale_count}")
        print(f"Vectors before : {vectors_before}")
        print(f"Vectors after  : {vectors_after}")
        return result

    def list_documents(self) -> list[dict[str, Any]]:
        manifest = load_manifest(self.paths.manifest_path)
        if manifest is None:
            return []
        self._assert_embedding_model(manifest)
        return sorted(
            (copy.deepcopy(value) for value in manifest["documents"].values()),
            key=lambda item: _source_identity_key(item["source_relative_path"]),
        )

    def file_status(self, pdf_path: Path) -> str:
        manifest = load_manifest(self.paths.manifest_path)
        if manifest is None:
            if (
                self.paths.canonical_chunks.exists()
                and self.paths.canonical_chunks.stat().st_size > 0
            ):
                return "bootstrap_required"
            return "new"
        self._assert_embedding_model(manifest)
        source_relative_path = self._source_relative_path(pdf_path)
        document_id = document_id_for_source(source_relative_path)
        document = manifest["documents"].get(document_id)
        if document is None:
            return "new"
        stat = Path(pdf_path).stat()
        if int(document.get("file_size", -1)) != stat.st_size:
            return "changed"
        return "indexed" if document.get("sha256") == sha256_file(pdf_path) else "changed"

    def summary(self) -> dict[str, Any]:
        manifest = load_manifest(self.paths.manifest_path)
        if manifest is not None:
            self._assert_embedding_model(manifest)
            documents = len(manifest["documents"])
            chunks = self._corpus_chunk_count(manifest)
            manifest_ready = True
        else:
            records = (
                read_jsonl(self.paths.canonical_chunks)
                if self.paths.canonical_chunks.exists()
                else []
            )
            sources = {
                str((record.get("metadata") or {}).get("source") or "").strip()
                for record in records
            }
            documents = len({source for source in sources if source})
            chunks = len(records)
            manifest_ready = False
        return {
            "documents": documents,
            "chunks": chunks,
            "vectors": self._read_vector_count(),
            "manifest_ready": manifest_ready,
            "embedding_model": self.embedding_model,
        }

    @_serialized_update
    def full_rebuild(self, *, confirm: bool) -> dict[str, Any]:
        """Explicitly rebuild the entire canonical corpus, never one selected PDF."""
        if not confirm:
            raise CorpusError("Full corpus rebuild requires confirm=true.")
        manifest = self._load_manifest_for_update()
        canonical_records = validate_manifest_corpus(manifest, self.paths)
        canonical_file_records = read_jsonl(self.paths.canonical_chunks)
        canonical_ids = validate_document_chunks(
            canonical_file_records, document_label="canonical corpus"
        )
        rebuilt_ids = [_record_chunk_id(record) for record in canonical_records]
        if len(rebuilt_ids) != len(canonical_ids) or set(rebuilt_ids) != set(canonical_ids):
            raise CorpusConsistencyError("Canonical corpus does not match the manifest.")

        vector_parent = (self.paths.root / "data" / "vectorstore").resolve()
        resolved_vector_dir = self.paths.vector_dir.resolve()
        if vector_parent not in resolved_vector_dir.parents:
            raise CorpusError(f"Unsafe vector directory: {resolved_vector_dir}")

        vectors_before = self._read_vector_count()
        clear_vector_db_cache(self.paths.vector_dir)
        if self.paths.vector_dir.exists():
            _safe_remove_tree(self.paths.vector_dir, vector_parent)
        db = build_index(
            input_file=self.paths.canonical_chunks,
            persist_dir=self.paths.vector_dir,
            model_name=self.embedding_model,
        )
        vectors_after = count_vectors(db)
        if vectors_after != len(canonical_records):
            raise CorpusConsistencyError(
                f"Full rebuild expected {len(canonical_records)} vectors, got {vectors_after}."
            )
        clear_vector_db_cache(self.paths.vector_dir)
        return {
            "status": "rebuilt",
            "documents": len(manifest["documents"]),
            "chunks": len(canonical_records),
            "vectors_before": vectors_before,
            "vectors_after": vectors_after,
        }


def bootstrap_existing_corpus(
    paths: CorpusPaths,
    embedding_model: str,
    config: ChunkConfig,
    *,
    dry_run: bool,
    force: bool = False,
) -> dict[str, Any]:
    """Create a manifest and per-document chunks without embedding anything."""
    records = read_jsonl(paths.canonical_chunks)
    all_ids = validate_document_chunks(records, document_label="canonical corpus")
    if len(all_ids) != len(set(all_ids)):
        raise CorpusConsistencyError("Canonical corpus contains duplicate chunk IDs.")

    groups: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        source = str(metadata.get("source") or "").strip()
        if not source:
            raise CorpusConsistencyError(
                f"Chunk {_record_chunk_id(record)!r} has no metadata.source."
            )
        source_key = _source_identity_key(source)
        groups.setdefault(source_key, []).append(record)
        sources.setdefault(source_key, source)

    manifest = new_manifest(embedding_model)
    overrides: dict[str, list[dict[str, Any]]] = {}
    for source_key, document_records in groups.items():
        source = sources[source_key]
        raw_pdf = _find_raw_pdf(paths.raw_dir, source)
        source_relative_path = (
            normalize_source_path(raw_pdf.relative_to(paths.raw_dir).as_posix())
            if raw_pdf is not None
            else normalize_source_path(source)
        )
        document_id = document_id_for_source(source_relative_path)
        if document_id in manifest["documents"]:
            raise CorpusConsistencyError(
                f"Multiple sources resolve to document ID {document_id!r}."
            )
        pdf_sha256 = sha256_file(raw_pdf) if raw_pdf is not None else "unavailable"
        file_size = raw_pdf.stat().st_size if raw_pdf is not None else 0
        fingerprint = processing_fingerprint(pdf_sha256, config)
        chunks_relative_path = (
            Path("documents")
            / document_id
            / "versions"
            / f"bootstrap-{fingerprint[:16]}"
            / "chunks.jsonl"
        ).as_posix()
        chunk_ids = validate_document_chunks(
            document_records, document_label=document_id
        )
        now = utc_now()
        manifest["documents"][document_id] = {
            "document_id": document_id,
            "source": PurePosixPath(source_relative_path).name,
            "source_relative_path": source_relative_path,
            "sha256": pdf_sha256,
            "file_size": file_size,
            "processing_fingerprint": fingerprint,
            "processing": chunk_config_payload(config),
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "chunks_path": chunks_relative_path,
            "status": "indexed",
            "embedding_model": embedding_model,
            "indexed_at": now,
            "updated_at": now,
        }
        overrides[document_id] = document_records

    rebuilt_records = validate_manifest_corpus(manifest, paths, overrides=overrides)
    if len(rebuilt_records) != len(records):
        raise CorpusConsistencyError("Bootstrap aggregation changed total chunk count.")
    rebuilt_ids = [_record_chunk_id(record) for record in rebuilt_records]
    if set(rebuilt_ids) != set(all_ids):
        raise CorpusConsistencyError("Bootstrap aggregation changed canonical chunk IDs.")

    vector_ids_before = read_chroma_vector_ids(paths.vector_dir)
    if vector_ids_before != set(all_ids):
        missing = len(set(all_ids) - vector_ids_before)
        unexpected = len(vector_ids_before - set(all_ids))
        raise CorpusConsistencyError(
            "Canonical and Chroma stable IDs differ during bootstrap. "
            f"Missing in Chroma: {missing}; unexpected in Chroma: {unexpected}."
        )
    vectors_before = len(vector_ids_before)
    report = {
        "existing_canonical_chunks": len(records),
        "unique_chunk_ids": len(set(all_ids)),
        "detected_documents": len(groups),
        "current_chroma_vectors": vectors_before,
        "embedding_required": False,
        "vector_db_modification": False,
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    existing_manifest = load_manifest(paths.manifest_path)
    if existing_manifest is not None and not force:
        raise CorpusError(
            f"Manifest already exists at {paths.manifest_path}; use --force only after review."
        )
    for document_id, document_records in overrides.items():
        chunks_path = _resolve_processed_path(
            paths, manifest["documents"][document_id]["chunks_path"]
        )
        if chunks_path.exists() and not force:
            existing_ids = [
                _record_chunk_id(record) for record in read_jsonl(chunks_path)
            ]
            if existing_ids != manifest["documents"][document_id]["chunk_ids"]:
                raise CorpusError(f"Bootstrap chunks already differ at {chunks_path}.")
        else:
            atomic_write_jsonl(chunks_path, document_records)

    # The existing canonical file is already valid. Leaving it byte-for-byte
    # unchanged is safer than rewriting 5006 production records during bootstrap.
    atomic_write_json(paths.manifest_path, manifest)
    vector_ids_after = read_chroma_vector_ids(paths.vector_dir)
    if vector_ids_after != vector_ids_before:
        raise CorpusConsistencyError(
            "Chroma IDs changed during bootstrap, which must be metadata-only."
        )
    report["manifest_path"] = str(paths.manifest_path)
    return report
