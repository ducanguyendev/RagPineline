"""Build a persistent Chroma index from project chunk data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.device_utils import clear_device_cache, get_device_name, resolve_device


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DB = "data/vectorstore/chroma"
DEFAULT_INDEX_DEVICE = "auto"
DEFAULT_INDEX_DTYPE = "auto"
DEFAULT_INDEX_BATCH_SIZE = 8
DEFAULT_CHROMA_BATCH_SIZE = 128
COLLECTION_NAME = "rag_documents"


@dataclass(frozen=True)
class EmbeddingRuntime:
    embeddings: HuggingFaceEmbeddings
    device: str
    device_name: str
    precision: str
    batch_size: int
    model_load_seconds: float
    fallback_reason: str | None = None


def _first_nonempty(*values: object) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _fallback_chroma_id(data: dict[str, object], content: str) -> str:
    """Build a deterministic ID when legacy chunk data has no ``chunk_id``."""
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    document_id = _first_nonempty(
        data.get("document_id"),
        data.get("doc_id"),
        metadata.get("document_id"),
        metadata.get("doc_id"),
        metadata.get("source"),
        "unknown-document",
    )
    identity = {
        "document_id": document_id,
        "chunk_index": data.get("chunk_index"),
        "page_start": metadata.get("page_start", metadata.get("page")),
        "section": metadata.get("section"),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    serialized = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"fallback-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _stable_chroma_id(data: dict[str, object], content: str, line_number: int) -> str:
    raw_chunk_id = data.get("chunk_id")
    if raw_chunk_id is not None and str(raw_chunk_id).strip():
        return str(raw_chunk_id).strip()

    fallback_id = _fallback_chroma_id(data, content)
    print(
        "WARNING: "
        f"Missing chunk_id on line {line_number}; using deterministic Chroma ID "
        f"{fallback_id}."
    )
    return fallback_id


def load_chunks(path: Path, *, limit: int | None = None) -> list[Document]:
    """Load project JSONL chunks as LangChain documents."""
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be greater than zero, got {limit}.")

    documents: list[Document] = []
    id_lines: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            data = json.loads(line)
            content = data["chunk_content"]
            stable_id = _stable_chroma_id(data, content, line_number)
            previous_line = id_lines.get(stable_id)
            if previous_line is not None:
                raise ValueError(
                    f"Duplicate stable Chroma ID {stable_id!r} on lines "
                    f"{previous_line} and {line_number}. chunk_id values must be "
                    "globally unique within a collection."
                )
            id_lines[stable_id] = line_number

            raw_meta = {
                **data.get("metadata", {}),
                "chunk_id": stable_id,
                "document_id": data.get("document_id", data.get("doc_id", "")),
                "doc_id": data.get("doc_id", ""),
            }

            clean_meta: dict[str, object] = {}
            for key, value in raw_meta.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    clean_meta[key] = ",".join(str(item) for item in value)
                else:
                    clean_meta[key] = value

            documents.append(
                Document(
                    id=stable_id,
                    page_content=content,
                    metadata=clean_meta,
                )
            )

            if limit is not None and len(documents) >= limit:
                break

    return documents


def document_chroma_ids(documents: list[Document]) -> list[str]:
    """Return validated, globally unique Chroma IDs for a document batch."""
    ids: list[str] = []
    seen: set[str] = set()

    for position, document in enumerate(documents, start=1):
        stable_id = _first_nonempty(
            getattr(document, "id", None),
            document.metadata.get("chunk_id"),
        )
        if not stable_id:
            raise ValueError(
                f"Document {position} has no stable Chroma ID. "
                "Load documents through load_chunks() or provide Document.id."
            )
        if stable_id in seen:
            raise ValueError(
                f"Duplicate stable Chroma ID {stable_id!r} in indexing input. "
                "IDs must be globally unique within a collection."
            )
        seen.add(stable_id)
        ids.append(stable_id)

    return ids


def count_vectors(db: Chroma) -> int:
    """Count collection IDs through the public LangChain Chroma API."""
    result = db.get(include=[])
    return len(result.get("ids", []))


def upsert_documents(
    db: Chroma,
    documents: list[Document],
    batch_size: int,
    *,
    show_progress: bool = True,
) -> list[str]:
    """Upsert documents with stable IDs through Chroma's public API."""
    resolved_batch_size = _positive_int(batch_size, "CHROMA_BATCH_SIZE")
    ids = document_chroma_ids(documents)

    with tqdm(
        total=len(documents),
        desc="Indexing",
        unit="chunk",
        disable=not show_progress,
    ) as progress:
        for start in range(0, len(documents), resolved_batch_size):
            batch = documents[start : start + resolved_batch_size]
            batch_ids = ids[start : start + resolved_batch_size]
            db.add_documents(batch, ids=batch_ids)
            progress.update(len(batch))

    return ids


def _positive_int(value: str | int, setting_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting_name} must be a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{setting_name} must be greater than zero, got {parsed}.")
    return parsed


def resolve_precision(device: str, preferred: str = "auto") -> str:
    """Resolve an embedding precision for the selected device."""
    requested = (preferred or "auto").strip().lower()
    if requested == "auto":
        return "float16" if device.startswith(("xpu", "cuda")) else "float32"

    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    try:
        return aliases[requested]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported INDEX_DTYPE '{preferred}'. "
            "Use auto, float16, float32, or bfloat16."
        ) from exc


def _torch_dtype(precision: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[precision]


def _create_embeddings(
    model_name: str,
    device: str,
    precision: str,
    batch_size: int,
) -> HuggingFaceEmbeddings:
    """Create embeddings using the API installed in the project environment."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
            # SentenceTransformers 6.x forwards this nested dictionary to
            # transformers.AutoModel.from_pretrained().
            "model_kwargs": {"torch_dtype": _torch_dtype(precision)},
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size,
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size,
        },
    )


def _log_cpu_fallback(device: str, reason: Exception) -> None:
    accelerator = device.split(":", maxsplit=1)[0].upper()
    print("WARNING:")
    print(f"{accelerator} embedding initialization failed.")
    print("Falling back to CPU float32.")
    print(f"Reason: {type(reason).__name__}: {reason}")


def initialize_embeddings(
    model_name: str = DEFAULT_MODEL,
    preferred_device: str | None = None,
    preferred_dtype: str | None = None,
    batch_size: int | None = None,
    *,
    allow_cpu_fallback: bool = True,
    log_configuration: bool = True,
) -> EmbeddingRuntime:
    """Load the embedding model on the requested accelerator."""
    requested_device = preferred_device or os.getenv(
        "INDEX_DEVICE", DEFAULT_INDEX_DEVICE
    )
    requested_dtype = preferred_dtype or os.getenv(
        "INDEX_DTYPE", DEFAULT_INDEX_DTYPE
    )
    resolved_batch_size = _positive_int(
        batch_size
        if batch_size is not None
        else os.getenv("INDEX_BATCH_SIZE", DEFAULT_INDEX_BATCH_SIZE),
        "INDEX_BATCH_SIZE",
    )

    load_start = time.perf_counter()
    fallback_reason: str | None = None

    try:
        device = resolve_device(requested_device)
        precision = resolve_precision(device, requested_dtype)
        embeddings = _create_embeddings(
            model_name=model_name,
            device=device,
            precision=precision,
            batch_size=resolved_batch_size,
        )
    except Exception as exc:
        normalized_request = (requested_device or "auto").strip().lower()
        if not allow_cpu_fallback or normalized_request == "cpu":
            raise

        failed_device = str(locals().get("device", normalized_request))
        _log_cpu_fallback(failed_device, exc)
        clear_device_cache(failed_device)

        device = "cpu"
        precision = "float32"
        fallback_reason = f"{type(exc).__name__}: {exc}"
        embeddings = _create_embeddings(
            model_name=model_name,
            device=device,
            precision=precision,
            batch_size=resolved_batch_size,
        )

    runtime = EmbeddingRuntime(
        embeddings=embeddings,
        device=device,
        device_name=get_device_name(device),
        precision=precision,
        batch_size=resolved_batch_size,
        model_load_seconds=time.perf_counter() - load_start,
        fallback_reason=fallback_reason,
    )

    if log_configuration:
        print("-" * 60)
        print(f"Embedding model : {model_name}")
        print(f"Device          : {runtime.device}")
        print(f"Device name     : {runtime.device_name}")
        print(f"Precision       : {runtime.precision}")
        print(f"Batch size      : {runtime.batch_size}")
        print("-" * 60)

    return runtime


def build_index(input_file: Path, persist_dir: Path, model_name: str):
    """Embed all chunks and add them to a persistent Chroma collection."""
    print("=" * 60)
    print("Loading chunks...")
    documents = load_chunks(input_file)
    print(f"Documents loaded: {len(documents)}")

    if not documents:
        raise ValueError(f"No chunks were found in {input_file}.")

    embedding_batch_size = _positive_int(
        os.getenv("INDEX_BATCH_SIZE", DEFAULT_INDEX_BATCH_SIZE),
        "INDEX_BATCH_SIZE",
    )
    chroma_batch_size = _positive_int(
        os.getenv("CHROMA_BATCH_SIZE", DEFAULT_CHROMA_BATCH_SIZE),
        "CHROMA_BATCH_SIZE",
    )

    runtime = initialize_embeddings(
        model_name=model_name,
        batch_size=embedding_batch_size,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=runtime.embeddings,
        persist_directory=str(persist_dir),
    )
    vectors_before = count_vectors(db)

    print("Building vector database...")
    indexing_start = time.perf_counter()
    upsert_documents(db, documents, chroma_batch_size)

    indexing_seconds = time.perf_counter() - indexing_start
    chunk_count = len(documents)
    chunks_per_second = chunk_count / indexing_seconds if indexing_seconds else 0.0
    milliseconds_per_chunk = (
        indexing_seconds * 1000 / chunk_count if chunk_count else 0.0
    )
    vectors_after = count_vectors(db)

    print("=" * 60)
    print("INDEX COMPLETE")
    print("=" * 60)
    print(f"Model            : {model_name}")
    print(f"Device           : {runtime.device}")
    print(f"Precision        : {runtime.precision}")
    print(f"Embedding batch  : {runtime.batch_size}")
    print(f"Chroma batch     : {chroma_batch_size}")
    print(f"Input chunks     : {chunk_count}")
    print(f"Vectors before   : {vectors_before}")
    print(f"Vectors after    : {vectors_after}")
    print(f"Model load       : {runtime.model_load_seconds:.2f} s")
    print(f"Indexing         : {indexing_seconds:.2f} s")
    print(f"Throughput       : {chunks_per_second:.2f} chunks/s")
    print(f"Per chunk        : {milliseconds_per_chunk:.1f} ms")
    print(f"Database         : {persist_dir}")
    print("=" * 60)

    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG vector index")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/chunked.jsonl"),
    )
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    build_index(args.input, args.db, args.model)


if __name__ == "__main__":
    main()
