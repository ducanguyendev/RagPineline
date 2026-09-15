"""
Metadata Builder

Đóng gói thông tin rich metadata chuẩn hóa cho từng Chunk đầu ra.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any, Dict
from src.data.block_packer import PackedChunk
from src.data.token_counter import count_tokens


def build_chunk_record(
    packed_chunk: PackedChunk,
    document_id: str,
    file_name: str,
    total_chunks: int
) -> Dict[str, Any]:
    """
    Tạo bản ghi JSON hoàn chỉnh cho một chunk bao gồm metrics và metadata.
    """
    token_cnt = count_tokens(packed_chunk.content)
    char_cnt = len(packed_chunk.content)

    effective_doc_id = packed_chunk.doc_id if (packed_chunk.doc_id and packed_chunk.doc_id != "default_doc") else document_id
    clean_doc_id = effective_doc_id.replace(".pdf", "")
    effective_file_name = f"{clean_doc_id}.pdf" if clean_doc_id != document_id.replace(".pdf", "") else file_name
    chunk_id = f"{clean_doc_id}_sec_{packed_chunk.chunk_index:02d}_chunk_{packed_chunk.chunk_index:02d}"

    metadata = {
        "source": effective_file_name,
        "title": clean_doc_id,
        "section": packed_chunk.section,
        "section_path": packed_chunk.section_path,
        "page_start": packed_chunk.page_start,
        "page_end": packed_chunk.page_end,
        "page": packed_chunk.page_start,
        "token_count": token_cnt,
        "total_chunks": total_chunks,
        "chunk_strategy": "structure_block_aware"
    }

    if packed_chunk.element_ids:
        metadata["element_ids"] = ",".join(packed_chunk.element_ids)
    if packed_chunk.element_types:
        metadata["element_types"] = ",".join(packed_chunk.element_types)
    if packed_chunk.table_ids:
        metadata["table_ids"] = ",".join(packed_chunk.table_ids)
    if packed_chunk.image_ids:
        metadata["image_ids"] = ",".join(packed_chunk.image_ids)
    if packed_chunk.image_paths:
        metadata["image_paths"] = ",".join(packed_chunk.image_paths)

    record = {
        "chunk_id": chunk_id,
        "document_id": clean_doc_id,
        "doc_id": clean_doc_id,
        "chunk_index": packed_chunk.chunk_index,
        "chunk_content": packed_chunk.content,
        "metrics": {
            "char_count": char_cnt,
            "token_count": token_cnt
        },
        "metadata": metadata
    }

    return record
