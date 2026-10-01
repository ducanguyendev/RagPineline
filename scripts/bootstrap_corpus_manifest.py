"""Bootstrap Phase 3 corpus metadata without embedding existing chunks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus_manager import CorpusPaths, bootstrap_existing_corpus
from src.data.chunker_optimized import ChunkConfig


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description=(
            "Create corpus_manifest.json and per-document chunk files without "
            "modifying Chroma or re-embedding chunks."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and validate only; write nothing.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Write per-document chunks and the manifest after validation.",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--strategy", default="structure_block_aware")
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--token-overlap", type=int, default=100)
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing manifest only after explicit review.",
    )
    args = parser.parse_args()

    paths = CorpusPaths.from_root(args.root)
    config = ChunkConfig(
        strategy=args.strategy,
        target_tokens=args.target_tokens,
        token_overlap=args.token_overlap,
    )
    report = bootstrap_existing_corpus(
        paths,
        args.model,
        config,
        dry_run=args.dry_run,
        force=args.force,
    )

    print("=" * 60)
    print("CORPUS MANIFEST BOOTSTRAP")
    print("=" * 60)
    print(f"Mode                      : {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Existing canonical chunks : {report['existing_canonical_chunks']}")
    print(f"Unique chunk IDs          : {report['unique_chunk_ids']}")
    print(f"Detected documents        : {report['detected_documents']}")
    print(f"Current Chroma vectors    : {report['current_chroma_vectors']}")
    print("Embedding required        : NO")
    print("Vector DB modification    : NO")
    if args.dry_run:
        print("Filesystem modification   : NO")
    else:
        print(f"Manifest                  : {report['manifest_path']}")
    print("PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
