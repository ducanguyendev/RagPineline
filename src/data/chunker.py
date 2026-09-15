"""
Structure & Block-Aware RAG Chunker (Main Orchestrator)

Thực hiện pipeline chunking 3 tầng:
1. Heading / Section Hard Boundary
2. Block-Aware Packing (Paragraph, Table, Image)
3. Selective Block Splitting (Paragraph đệ quy, Table row-aware)
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.section_builder import build_sections
from src.data.block_normalizer import normalize_blocks
from src.data.block_packer import pack_blocks
from src.data.metadata_builder import build_chunk_record


@dataclass
class StructureChunkConfig:
    target_tokens: int = 600
    token_overlap: int = 100
    hard_max_tokens: int = 800


def process_structure_chunking(
    input_path: Path,
    output_path: Path,
    config: StructureChunkConfig
):
    """
    Đọc elements.jsonl hoặc pdf_extract.jsonl và tiến hành chunking theo 3 tầng ưu tiên.
    """
    print("=" * 60)
    print(f"Loading input data: {input_path}")

    raw_records: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            raw_records.append(json.loads(line))

    if not raw_records:
        print("! Input file is empty.")
        return

    # Chuyển đổi định dạng nếu input là pdf_extract.jsonl cũ (mỗi record = 1 trang)
    elements: List[Dict[str, Any]] = []
    first_rec = raw_records[0] if raw_records else {}
    document_id = first_rec.get("document_id") or first_rec.get("metadata", {}).get("title") or input_path.stem
    file_name = first_rec.get("metadata", {}).get("source") or f"{document_id}.pdf"

    for idx, rec in enumerate(raw_records):
        if "type" in rec:
            elements.append(rec)
        else:
            # Tự chuyển đổi trang thành các paragraph element
            text_content = rec.get("content") or rec.get("text") or ""
            page = rec.get("metadata", {}).get("page", idx + 1)
            doc_id = rec.get("document_id") or rec.get("metadata", {}).get("title") or document_id
            lines = [l.strip() for l in text_content.split("\n") if l.strip()]
            for l_idx, line_str in enumerate(lines):
                is_head = line_str.startswith("#") or line_str.startswith("##") or line_str.startswith("###")
                elements.append({
                    "document_id": doc_id,
                    "element_id": f"p{page}_e{l_idx+1}",
                    "type": "heading" if is_head else "paragraph",
                    "text": line_str,
                    "page": page
                })

    print(f"✓ Total Elements: {len(elements)}")

    # TẦNG 1: Heading / Section Boundary
    print("Building section hierarchy (Tầng 1)...")
    section_elements = build_sections(elements)

    # Chuẩn hóa các elements thành các Block
    print("Normalizing blocks (Paragraph, Table, Image)...")
    blocks = normalize_blocks(section_elements)
    print(f"✓ Total Normalized Blocks: {len(blocks)}")

    # TẦNG 2 & TẦNG 3: Block Packing & Selective Splitting
    print("Packing blocks & performing selective splitting (Tầng 2 & 3)...")
    packed_chunks = pack_blocks(
        blocks,
        target_tokens=config.target_tokens,
        overlap_tokens=config.token_overlap,
        hard_max_tokens=config.hard_max_tokens
    )
    print(f"✓ Generated {len(packed_chunks)} Chunks")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = len(packed_chunks)

    with output_path.open("w", encoding="utf-8") as outfile:
        for p_chunk in packed_chunks:
            chunk_doc_id = p_chunk.doc_id if (p_chunk.doc_id and p_chunk.doc_id != "default_doc") else document_id
            chunk_file_name = f"{chunk_doc_id}.pdf"
            record = build_chunk_record(p_chunk, chunk_doc_id, chunk_file_name, total_chunks)
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("✅ STRUCTURE-AWARE CHUNKING COMPLETE")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total Chunks: {total_chunks}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Structure & Block-Aware RAG Chunker")
    parser.add_argument("--input", type=Path, default=Path("data/processed/elements.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunked.jsonl"))
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--token-overlap", type=int, default=100)
    parser.add_argument("--hard-max-tokens", type=int, default=800)

    args = parser.parse_args()

    config = StructureChunkConfig(
        target_tokens=args.target_tokens,
        token_overlap=args.token_overlap,
        hard_max_tokens=args.hard_max_tokens
    )

    process_structure_chunking(args.input, args.output, config)


if __name__ == "__main__":
    main()
