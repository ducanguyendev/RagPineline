"""
Block Packer (Tầng 2: Block-Aware Packing trong cùng Section)

Thực hiện gom đóng gói các Block cùng Section:
- Ranh giới Section là ranh giới cứng: Không bao giờ pack các Block thuộc 2 Section khác nhau.
- Target tokens: 500-600 tokens (hard max: 800 tokens).
- Ưu tiên giữ nguyên vẹn Block mà không cắt vụn nếu chưa cần thiết.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from typing import Any, Dict, List
from src.data.block_normalizer import Block
from src.data.paragraph_splitter import split_large_paragraph
from src.data.table_splitter import split_large_table


@dataclass
class PackedChunk:
    chunk_index: int
    section: str
    section_path: str
    content: str
    token_count: int
    page_start: int
    page_end: int
    doc_id: str = "default_doc"
    element_ids: List[str] = field(default_factory=list)
    element_types: List[str] = field(default_factory=list)
    table_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)


def pack_blocks(
    blocks: List[Block],
    target_tokens: int = 600,
    overlap_tokens: int = 100,
    hard_max_tokens: int = 800
) -> List[PackedChunk]:
    """
    Gom các block thành danh sách các PackedChunk theo nguyên tắc Tầng 1 và Tầng 2.
    """
    if not blocks:
        return []

    # 1. Phân nhóm các Block theo Document và Section (Tầng 1: Hard Boundary)
    section_groups: List[List[Block]] = []
    current_group: List[Block] = []

    for block in blocks:
        if not current_group:
            current_group.append(block)
        elif current_group[-1].doc_id == block.doc_id and current_group[-1].section_path == block.section_path:
            current_group.append(block)
        else:
            section_groups.append(current_group)
            current_group = [block]

    if current_group:
        section_groups.append(current_group)

    packed_chunks: List[PackedChunk] = []
    chunk_counter = 1

    # 2. Xử lý đóng gói (Packing) trong từng Section (Tầng 2 & Tầng 3)
    for group in section_groups:
        processed_blocks: List[Block] = []
        for block in group:
            if block.token_count > hard_max_tokens:
                if block.block_type == "paragraph":
                    processed_blocks.extend(
                        split_large_paragraph(block, target_tokens, overlap_tokens, hard_max_tokens)
                    )
                elif block.block_type == "table":
                    processed_blocks.extend(
                        split_large_table(block, target_tokens, hard_max_tokens)
                    )
                else:
                    processed_blocks.append(block)
            else:
                processed_blocks.append(block)

        current_chunk_blocks: List[Block] = []
        current_tokens = 0

        def emit_chunk(c_blocks: List[Block]) -> PackedChunk:
            nonlocal chunk_counter
            section = c_blocks[0].section
            section_path = c_blocks[0].section_path
            doc_id = c_blocks[0].doc_id
            pages = [b.page for b in c_blocks if b.page]
            page_start = min(pages) if pages else 1
            page_end = max(pages) if pages else page_start

            elem_ids = []
            elem_types = set()
            table_ids = []
            image_ids = []
            image_paths = []

            content_parts = []
            for b in c_blocks:
                content_parts.append(b.content)
                elem_ids.extend(b.element_ids)
                elem_types.add(b.block_type)
                if b.block_type == "table":
                    table_ids.append(b.block_id)
                elif b.block_type == "image":
                    image_ids.append(b.block_id)
                    if b.image_data and b.image_data.get("image_path"):
                        image_paths.append(b.image_data["image_path"])

            header_prefix = f"Section: {section_path}\n---\n"
            full_content = header_prefix + "\n\n".join(content_parts)

            chunk = PackedChunk(
                chunk_index=chunk_counter,
                section=section,
                section_path=section_path,
                content=full_content,
                token_count=len(full_content) // 4,
                page_start=page_start,
                page_end=page_end,
                doc_id=doc_id,
                element_ids=list(dict.fromkeys(elem_ids)),
                element_types=sorted(list(elem_types)),
                table_ids=table_ids,
                image_ids=image_ids,
                image_paths=list(dict.fromkeys(image_paths))
            )
            chunk_counter += 1
            return chunk

        for b in processed_blocks:
            if not current_chunk_blocks:
                current_chunk_blocks.append(b)
                current_tokens = b.token_count
            elif current_tokens + b.token_count <= target_tokens or current_tokens + b.token_count <= hard_max_tokens:
                current_chunk_blocks.append(b)
                current_tokens += b.token_count
            else:
                packed_chunks.append(emit_chunk(current_chunk_blocks))
                current_chunk_blocks = [b]
                current_tokens = b.token_count

        if current_chunk_blocks:
            packed_chunks.append(emit_chunk(current_chunk_blocks))

    return packed_chunks
