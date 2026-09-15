"""
Context Builder Subsystem

1. Deduplication: Removes overlapping context text.
2. Token Budget: Enforces MAX_CONTEXT_TOKENS hard limit.
3. SOURCE ID Mapping: Assigns explicit [SOURCE_X] IDs for citation tracking.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.token_counter import count_tokens


def is_duplicate(text: str, existing_texts: List[str], threshold: float = 0.82) -> bool:
    """Kiểm tra xem nội dung đoạn văn có trùng lặp > threshold với các đoạn đã thêm không."""
    for s in existing_texts:
        # Nếu đoạn text này nằm gọn trong đoạn khác hoặc similarity quá cao
        if text in s or s in text:
            return True
        ratio = SequenceMatcher(None, text[:300], s[:300]).ratio()
        if ratio >= threshold:
            return True
    return False


def build_context(
    documents: List[Dict[str, Any]],
    max_tokens: int = 4000
) -> Tuple[str, Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Xây dựng khối Context chuẩn hóa kèm theo SOURCE IDs.
    
    Args:
        documents: Danh sách các reranked chunks
        max_tokens: Hạn mức token tối đa cho khối Context
        
    Returns:
        (formatted_context_str, source_mapping_dict, selected_chunks)
    """
    if not documents:
        return "", {}, []

    context_blocks = []
    source_mapping = {}
    selected_chunks = []
    existing_contents = []
    current_tokens = 0

    source_counter = 1

    for doc in documents:
        content = (doc.get("content") or doc.get("page_content") or "").strip()
        if not content:
            continue

        # 1. Khử trùng lặp (Deduplication)
        if is_duplicate(content, existing_contents):
            print(f"⏩ [Context Deduplication] Skipped duplicate content in chunk {doc.get('chunk_id')}")
            continue

        # Đếm token của block này
        meta = doc.get("metadata") or {}
        source_file = meta.get("source") or doc.get("source") or "Document"
        page = meta.get("page") or doc.get("page") or 1
        section = meta.get("section_path") or meta.get("section") or doc.get("section") or "Overview"

        source_id = f"SOURCE_{source_counter}"

        block_header = f"[{source_id}]\nFile: {source_file} | Page: {page} | Section: {section}\nContent:\n"
        full_block = f"{block_header}{content}\n"
        block_tokens = count_tokens(full_block)

        # 2. Kiểm tra Token Budget (MAX_CONTEXT_TOKENS)
        if current_tokens + block_tokens > max_tokens and context_blocks:
            print(f"⚠️ [Token Budget Reached] Stopping context assembly at {current_tokens} tokens (Limit: {max_tokens})")
            break

        # Thêm vào danh sách context
        context_blocks.append(full_block)
        existing_contents.append(content)
        current_tokens += block_tokens
        selected_chunks.append(doc)

        # Trích xuất image_paths nếu có
        raw_img_paths = meta.get("image_paths") or doc.get("image_paths") or ""
        image_paths_list = []
        if isinstance(raw_img_paths, str) and raw_img_paths.strip():
            image_paths_list = [p.strip() for p in raw_img_paths.split(",") if p.strip()]
        elif isinstance(raw_img_paths, list):
            image_paths_list = raw_img_paths

        source_mapping[source_id] = {
            "source_id": source_id,
            "source_file": source_file,
            "page": page,
            "section": section,
            "chunk_id": doc.get("chunk_id") or meta.get("chunk_id"),
            "image_paths": image_paths_list,
            "rerank_score": doc.get("rerank_score", 0.0),
            "raw_doc": doc
        }

        source_counter += 1

    formatted_context = "\n----------------------------------------\n".join(context_blocks)
    return formatted_context, source_mapping, selected_chunks


if __name__ == "__main__":
    sample_docs = [
        {
            "content": "RAG gồm 3 bước: Indexing, Retrieval và Generation.",
            "metadata": {"source": "rag.pdf", "page": 5, "section": "II. Kiến trúc RAG"}
        }
    ]
    ctx, mapping, sel = build_context(sample_docs, max_tokens=1000)
    print("FORMATTED CONTEXT:\n", ctx)
    print("MAPPING:\n", mapping)
