"""Benchmark BGE-M3 embedding without modifying the project vector database."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.device_utils import clear_device_cache, synchronize_device  # noqa: E402
from src.rag.indexer import DEFAULT_MODEL, initialize_embeddings  # noqa: E402


def _contains_chunks(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as file:
        return any(line.strip() for line in file)


def find_chunk_file(requested: Path | None) -> Path:
    """Find a non-empty chunk JSONL file without moving or changing data."""
    if requested is not None:
        candidate = requested if requested.is_absolute() else ROOT / requested
        if not _contains_chunks(candidate):
            raise FileNotFoundError(f"Chunk file is missing or empty: {candidate}")
        return candidate

    primary = ROOT / "data" / "processed" / "chunked.jsonl"
    candidates = [primary]
    data_dir = ROOT / "data"
    if data_dir.exists():
        candidates.extend(sorted(data_dir.rglob("chunked*.jsonl")))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _contains_chunks(candidate):
            return candidate

    raise FileNotFoundError("No non-empty chunked JSONL file was found under data/.")


def load_chunk_texts(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("chunk_content") or record.get("content")
            if text:
                texts.append(str(text))
            if len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"No usable chunk text was found in {path}.")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BGE-M3 embedding")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.limit <= 0:
        parser.error("--limit must be greater than zero")

    load_dotenv(ROOT / ".env")
    chunk_file = find_chunk_file(args.input)
    texts = load_chunk_texts(chunk_file, args.limit)
    print(f"Chunk data      : {chunk_file}")

    runtime = initialize_embeddings(
        model_name=DEFAULT_MODEL,
        preferred_device=args.device,
        preferred_dtype=args.dtype,
        batch_size=args.batch_size,
        allow_cpu_fallback=False,
        log_configuration=False,
    )

    warmup_size = min(args.batch_size, len(texts))
    runtime.embeddings.embed_documents(texts[:warmup_size])
    synchronize_device(runtime.device)

    encode_start = time.perf_counter()
    embeddings = runtime.embeddings.embed_documents(texts)
    synchronize_device(runtime.device)
    encode_seconds = time.perf_counter() - encode_start

    chunk_count = len(embeddings)
    chunks_per_second = chunk_count / encode_seconds if encode_seconds else 0.0
    milliseconds_per_chunk = (
        encode_seconds * 1000 / chunk_count if chunk_count else 0.0
    )

    print("=" * 60)
    print("EMBEDDING BENCHMARK")
    print("=" * 60)
    print(f"Device          : {runtime.device}")
    print(f"Device name     : {runtime.device_name}")
    print(f"Precision       : {runtime.precision}")
    print(f"Batch size      : {runtime.batch_size}")
    print(f"Chunks          : {chunk_count}")
    print(f"Model load      : {runtime.model_load_seconds:.2f} s")
    print(f"Encode          : {encode_seconds:.2f} s")
    print(f"Throughput      : {chunks_per_second:.2f} chunks/s")
    print(f"Per chunk       : {milliseconds_per_chunk:.1f} ms")
    print("=" * 60)

    clear_device_cache(runtime.device)


if __name__ == "__main__":
    main()
