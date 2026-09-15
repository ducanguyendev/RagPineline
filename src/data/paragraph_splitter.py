"""
Paragraph Splitter (Tầng 3: Selective Splitting cho Paragraph quá khổ)

Chỉ khi một Paragraph Block tự nó vượt quá hard_max_tokens (ví dụ > 800 tokens),
mới tiến hành Recursive Split theo ranh giới đoạn/câu và áp dụng overlap (80-100 tokens).
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data.block_normalizer import Block
from src.data.token_counter import count_tokens, TokenCounter


def split_large_paragraph(
    block: Block,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
    hard_max_tokens: int = 800
) -> List[Block]:
    """
    Cắt nhỏ một đoạn văn bản quá khổ thành danh sách các sub-Block nhỏ hơn.
    """
    if block.token_count <= hard_max_tokens:
        return [block]

    # Quy đổi token sang số ký tự ước tính để làm chunk_size / chunk_overlap
    avg_char_per_token = len(block.content) / max(1, block.token_count)
    chunk_size_chars = int(target_tokens * avg_char_per_token)
    chunk_overlap_chars = int(overlap_tokens * avg_char_per_token)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(200, chunk_size_chars),
        chunk_overlap=max(50, chunk_overlap_chars),
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )

    sub_texts = splitter.split_text(block.content)
    sub_blocks: List[Block] = []

    for index, sub_text in enumerate(sub_texts, start=1):
        if not sub_text.strip():
            continue
            
        sub_tokens = count_tokens(sub_text)
        sub_blocks.append(
            Block(
                block_id=f"{block.block_id}_sub_{index}",
                block_type="paragraph",
                content=sub_text.strip(),
                section=block.section,
                section_path=block.section_path,
                page=block.page,
                element_ids=block.element_ids,
                token_count=sub_tokens
            )
        )

    return sub_blocks if sub_blocks else [block]
