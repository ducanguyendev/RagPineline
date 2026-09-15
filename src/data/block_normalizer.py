"""
Block Normalizer

Gom nhóm và chuẩn hóa các elements thành các Block cấu trúc:
- ParagraphBlock: Đoạn văn bản thuần
- TableBlock: Bảng biểu (Atomic Block)
- ImageBlock: Hình ảnh + Caption gốc + AI Caption (Atomic Block)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from src.data.token_counter import count_tokens


@dataclass
class Block:
    block_id: str
    block_type: Literal["paragraph", "table", "image"]
    content: str
    section: str
    section_path: str
    page: int
    doc_id: str = "default_doc"
    element_ids: List[str] = field(default_factory=list)
    token_count: int = 0
    table_data: Optional[Dict[str, Any]] = None
    image_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.token_count == 0 and self.content:
            self.token_count = count_tokens(self.content)


def normalize_blocks(elements: List[Dict[str, Any]]) -> List[Block]:
    """
    Biến đổi danh sách elements thành danh sách các Block.
    Tự động ghép nối Image với Caption gốc và AI Caption từ BLIP.
    """
    blocks: List[Block] = []
    idx = 0
    total = len(elements)

    while idx < total:
        elem = elements[idx]
        elem_type = elem.get("type", "paragraph").lower()
        elem_id = elem.get("element_id", f"e_{idx:04d}")
        text = elem.get("text", "").strip()
        section = elem.get("section", "Overview")
        section_path = elem.get("section_path", "Overview")
        page = elem.get("page", 1)
        doc_id = elem.get("document_id") or elem.get("doc_id") or "default_doc"

        if elem_type == "heading" and not text:
            idx += 1
            continue

        # Xử lý TABLE BLOCK
        if elem_type == "table":
            table_info = elem.get("table_data", {})
            markdown_content = table_info.get("markdown", text)
            caption = elem.get("caption", "").strip()
            
            full_content = ""
            if caption:
                full_content += f"Table Caption: {caption}\n"
            full_content += markdown_content

            blocks.append(
                Block(
                    block_id=f"block_table_{idx:04d}",
                    block_type="table",
                    content=full_content,
                    section=section,
                    section_path=section_path,
                    page=page,
                    doc_id=doc_id,
                    element_ids=[elem_id],
                    table_data=table_info
                )
            )
            idx += 1
            continue

        # Xử lý IMAGE BLOCK (Original Caption + Description + AI Caption, KHÔNG embed image_path)
        if elem_type in ["picture", "image", "figure"]:
            image_id = elem.get("image_id") or f"fig_{idx:03d}"
            image_path = elem.get("image_path", "")
            original_caption = elem.get("caption") or elem.get("text") or ""
            description = elem.get("description", "").strip()
            ai_caption = elem.get("ai_caption", "").strip()

            element_ids = [elem_id]
            # Kiểm tra element tiếp theo có phải là caption độc lập không
            if idx + 1 < total and elements[idx + 1].get("type", "").lower() == "caption":
                next_caption = elements[idx + 1].get("text", "").strip()
                if next_caption:
                    original_caption = f"{original_caption} {next_caption}".strip() if original_caption else next_caption
                element_ids.append(elements[idx + 1].get("element_id", f"e_{idx+1:04d}"))
                idx += 1

            # Ghép nội dung semantic cho ImageBlock
            parts = []
            if original_caption.strip():
                parts.append(original_caption.strip())
            if description:
                parts.append(f"Description:\n{description}")
            if ai_caption:
                parts.append(f"AI Description:\n{ai_caption}")

            if not parts:
                parts.append("[Image Content]")

            image_content = "\n\n".join(parts)

            image_data = {
                "image_id": image_id,
                "image_path": image_path,
                "original_caption": original_caption,
                "description": description,
                "ai_caption": ai_caption
            }

            blocks.append(
                Block(
                    block_id=image_id,
                    block_type="image",
                    content=image_content,
                    section=section,
                    section_path=section_path,
                    page=page,
                    doc_id=doc_id,
                    element_ids=element_ids,
                    image_data=image_data
                )
            )
            idx += 1
            continue

        # Xử lý PARAGRAPH BLOCK
        if text:
            content = f"### {text}" if elem_type == "heading" else text

            blocks.append(
                Block(
                    block_id=f"block_para_{idx:04d}",
                    block_type="paragraph",
                    content=content,
                    section=section,
                    section_path=section_path,
                    page=page,
                    doc_id=doc_id,
                    element_ids=[elem_id]
                )
            )

        idx += 1

    return blocks
