"""
Optimized RAG Chunker (Wrapper for Structure & Block-Aware Pipeline)

Input:
    elements.jsonl or pdf_extract.jsonl

Output:
    chunked.jsonl
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.chunker import process_structure_chunking, StructureChunkConfig


@dataclass
class ChunkConfig:
    strategy: str = "token"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    target_tokens: int = 600
    token_overlap: int = 100
    min_size: int = 150
    max_size: int = 2500
    add_header: bool = True


def process_chunking(
    input_path: Path,
    output_path: Path,
    config: ChunkConfig
):
    struct_config = StructureChunkConfig(
        target_tokens=getattr(config, "target_tokens", 600),
        token_overlap=getattr(config, "token_overlap", 100),
        hard_max_tokens=800
    )
    process_structure_chunking(input_path, output_path, struct_config)


def main():
    parser = argparse.ArgumentParser(description="Optimized RAG Chunker")
    parser.add_argument("--input", type=Path, default=Path("data/processed/elements.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunked.jsonl"))
    parser.add_argument("--strategy", default="token")
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--token-overlap", type=int, default=100)

    args = parser.parse_args()

    # Kiểm tra xem elements.jsonl có tồn tại không, nếu không fallback pdf_extract.jsonl
    input_file = args.input
    if not input_file.exists():
        fallback = input_file.parent / "pdf_extract.jsonl"
        if fallback.exists():
            input_file = fallback

    config = ChunkConfig(
        strategy=args.strategy,
        target_tokens=args.target_tokens,
        token_overlap=args.token_overlap
    )

    process_chunking(input_file, args.output, config)


if __name__ == "__main__":
    main()