"""Build a persistent Chroma index from project chunk data."""

from __future__ import annotations

import argparse
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


def load_chunks(path: Path) -> list[Document]:
    """Load project JSONL chunks as LangChain documents."""
    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            data = json.loads(line)
            raw_meta = {
                **data.get("metadata", {}),
                "chunk_id": data.get("chunk_id", ""),
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
                    page_content=data["chunk_content"],
                    metadata=clean_meta,
                )
            )

    return documents


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

    print("Building vector database...")
    indexing_start = time.perf_counter()
    with tqdm(total=len(documents), desc="Indexing", unit="chunk") as progress:
        for start in range(0, len(documents), chroma_batch_size):
            batch = documents[start : start + chroma_batch_size]
            db.add_documents(batch)
            progress.update(len(batch))

    indexing_seconds = time.perf_counter() - indexing_start
    chunk_count = len(documents)
    chunks_per_second = chunk_count / indexing_seconds if indexing_seconds else 0.0
    milliseconds_per_chunk = (
        indexing_seconds * 1000 / chunk_count if chunk_count else 0.0
    )
    vectors_stored = int(db._collection.count())

    print("=" * 60)
    print("INDEX COMPLETE")
    print("=" * 60)
    print(f"Model            : {model_name}")
    print(f"Device           : {runtime.device}")
    print(f"Precision        : {runtime.precision}")
    print(f"Embedding batch  : {runtime.batch_size}")
    print(f"Chroma batch     : {chroma_batch_size}")
    print(f"Chunks           : {chunk_count}")
    print(f"Model load       : {runtime.model_load_seconds:.2f} s")
    print(f"Indexing         : {indexing_seconds:.2f} s")
    print(f"Throughput       : {chunks_per_second:.2f} chunks/s")
    print(f"Per chunk        : {milliseconds_per_chunk:.1f} ms")
    print(f"Vectors stored   : {vectors_stored}")
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
