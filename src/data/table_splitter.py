"""
Table Splitter (Tầng 3: Row-aware Split cho Bảng biểu quá khổ)

Nếu một Bảng (Table Block) quá lớn (> hard_max_tokens ~800 tokens):
- KHÔNG dùng RecursiveCharacterTextSplitter để tránh làm rách cấu trúc bảng.
- Thực hiện Row-aware Split (cắt theo từng dòng dữ liệu).
- BẮT BUỘC lặp lại Header của Bảng (và Caption) ở tất cả các sub-chunk.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List
from src.data.block_normalizer import Block
from src.data.token_counter import count_tokens


def split_large_table(
    block: Block,
    target_tokens: int = 600,
    hard_max_tokens: int = 800
) -> List[Block]:
    """
    Cắt một bảng quá lớn theo từng hàng dữ liệu, tự động lặp lại Header và Caption trên mỗi sub-chunk.
    """
    if block.token_count <= hard_max_tokens:
        return [block]

    content_lines = block.content.strip().split("\n")
    if len(content_lines) <= 3:
        return [block]

    # Phân tách phần Header / Caption với các dòng dữ liệu (Rows)
    header_lines: List[str] = []
    row_lines: List[str] = []
    in_table = False

    for line in content_lines:
        line_str = line.strip()
        if line_str.startswith("|") and "|" in line_str[1:]:
            in_table = True
            if len(header_lines) < 2:  # Thường line 1 là tên cột, line 2 là |---|---|
                header_lines.append(line)
            else:
                row_lines.append(line)
        else:
            if not in_table:
                header_lines.append(line)  # Lưu Caption / Tiêu đề bảng ở trên
            else:
                row_lines.append(line)

    if not row_lines:
        return [block]

    header_text = "\n".join(header_lines) + "\n"
    header_tokens = count_tokens(header_text)

    sub_blocks: List[Block] = []
    current_rows: List[str] = []
    current_tokens = header_tokens
    sub_index = 1

    for row in row_lines:
        row_tok = count_tokens(row)
        if current_rows and (current_tokens + row_tok > target_tokens or current_tokens + row_tok > hard_max_tokens):
            # Đóng sub-chunk hiện tại
            table_text = header_text + "\n".join(current_rows)
            sub_blocks.append(
                Block(
                    block_id=f"{block.block_id}_sub_{sub_index}",
                    block_type="table",
                    content=table_text,
                    section=block.section,
                    section_path=block.section_path,
                    page=block.page,
                    element_ids=block.element_ids,
                    token_count=count_tokens(table_text),
                    table_data=block.table_data
                )
            )
            sub_index += 1
            current_rows = [row]
            current_tokens = header_tokens + row_tok
        else:
            current_rows.append(row)
            current_tokens += row_tok

    if current_rows:
        table_text = header_text + "\n".join(current_rows)
        sub_blocks.append(
            Block(
                block_id=f"{block.block_id}_sub_{sub_index}",
                block_type="table",
                content=table_text,
                section=block.section,
                section_path=block.section_path,
                page=block.page,
                element_ids=block.element_ids,
                token_count=count_tokens(table_text),
                table_data=block.table_data
            )
        )

    return sub_blocks if sub_blocks else [block]
