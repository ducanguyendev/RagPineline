"""
Section Builder (Tầng 1: Chunking theo Heading/Section)

Xây dựng cây phân cấp Section/Heading và gán ngữ cảnh section cho từng phần tử (element).
Ranh giới Section là ranh giới cứng (Hard Boundary).
"""

import re
from typing import Any, Dict, List


def parse_heading_level(text: str, default_level: int = 1) -> int:
    """
    Xác định level của heading dựa vào định dạng Markdown hoặc số hiệu.
    Ví dụ: # (level 1), ## (level 2), III. (level 1), III.2. (level 2), 1. (level 1), 1.1 (level 2)
    """
    clean_t = text.strip()
    
    # Check Markdown hashes: e.g. "## Heading"
    if clean_t.startswith("#"):
        hashes = len(clean_t) - len(clean_t.lstrip("#"))
        return max(1, min(hashes, 6))

    # Check Roman numerals or numbers: e.g. "II.2.1."
    match = re.match(r"^([I|V|X]+|\d+)(\.([I|V|X]+|\d+))*", clean_t)
    if match:
        dot_count = clean_t[:match.end()].count(".")
        if dot_count > 0:
            return min(dot_count, 6)

    return default_level


def build_sections(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gán section và section_path cho từng element theo ranh giới Heading.
    """
    processed_elements = []
    
    # Stack lưu trữ các cặp (level, title) đại diện cho hierarchy hiện tại
    heading_stack: List[tuple[int, str]] = []
    current_doc_id = None

    for index, elem in enumerate(elements):
        doc_id = elem.get("document_id") or elem.get("doc_id")
        if current_doc_id is not None and doc_id != current_doc_id:
            heading_stack = []
        current_doc_id = doc_id

        elem_type = elem.get("type", "paragraph").lower()
        text = elem.get("text", "").strip()

        is_heading = elem_type == "heading" or elem.get("is_heading", False)

        if is_heading and text:
            level = elem.get("level") or parse_heading_level(text)
            
            # Pop các heading có level cao hơn hoặc bằng level hiện tại
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, text))

        # Xây dựng section title và section_path
        if heading_stack:
            current_section = heading_stack[-1][1]
            section_path = " > ".join(title for _, title in heading_stack)
        else:
            current_section = "Overview"
            section_path = "Overview"

        elem_copy = dict(elem)
        elem_copy["section"] = current_section
        elem_copy["section_path"] = section_path
        processed_elements.append(elem_copy)

    return processed_elements
