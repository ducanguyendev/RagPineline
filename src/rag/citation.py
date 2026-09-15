"""
Citation Subsystem

1. Citation Validation: Validates [SOURCE_X] tags in LLM answer against source_mapping.
2. Source Resolution: Resolves real metadata (file, page, section) and physical image URLs.
"""

import re
from typing import Any, Dict, List, Tuple


def validate_and_resolve_citations(
    answer_text: str,
    source_mapping: Dict[str, Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Xác minh các nhãn [SOURCE_X] trong câu trả lời từ LLM và trích xuất danh sách nguồn & đường dẫn ảnh thực tế.
    
    Args:
        answer_text: Câu trả lời văn bản sinh bởi LLM
        source_mapping: Bảng ánh xạ SOURCE_ID -> Metadata thực tế từ Context Builder
        
    Returns:
        (cleaned_answer, citations_list, image_urls_list)
    """
    if not answer_text or not source_mapping:
        return answer_text, [], []

    # 1. Tìm tất cả các thẻ [SOURCE_X] trong câu trả lời
    found_tags = set(re.findall(r"\[SOURCE_\d+\]", answer_text))
    
    valid_citations = []
    referenced_source_ids = set()

    for tag in sorted(found_tags):
        clean_id = tag.replace("[", "").replace("]", "")
        if clean_id in source_mapping:
            referenced_source_ids.add(clean_id)
            info = source_mapping[clean_id]
            valid_citations.append({
                "source_id": clean_id,
                "file": info.get("source_file", ""),
                "page": info.get("page", 1),
                "section": info.get("section", ""),
                "chunk_id": info.get("chunk_id", ""),
                "rerank_score": info.get("rerank_score", 0.0)
            })
        else:
            # Nếu LLM bịa ra [SOURCE_99] không có trong mapping, xóa khỏi câu trả lời
            print(f"⚠️ [Citation Hallucination Filter] Removed non-existent source tag: {tag}")
            answer_text = answer_text.replace(tag, "")

    # 2. Thu thập danh sách ảnh thực tế từ các Nguồn được trích dẫn (hoặc từ tất cả Nguồn được chọn)
    image_urls = []
    seen_images = set()

    # Ưu tiên các nguồn được LLM trích dẫn cụ thể, nếu không có thì lấy toàn bộ nguồn trong context
    target_sources = referenced_source_ids if referenced_source_ids else source_mapping.keys()

    for s_id in target_sources:
        info = source_mapping.get(s_id, {})
        img_paths = info.get("image_paths", [])
        for p in img_paths:
            clean_p = p.replace("\\", "/")
            if clean_p in seen_images:
                continue
            seen_images.add(clean_p)

            # Chuyển đổi "data/images/doc/fig_001.png" -> "/images/doc/fig_001.png"
            if "data/images/" in clean_p:
                rel_url = "/images/" + clean_p.split("data/images/")[1]
                image_urls.append(rel_url)
            elif clean_p.startswith("/images/"):
                image_urls.append(clean_p)

    # Làm sạch khoảng trắng thừa trong câu trả lời nếu có
    cleaned_answer = re.sub(r"[ \t]+", " ", answer_text).strip()

    return cleaned_answer, valid_citations, image_urls
