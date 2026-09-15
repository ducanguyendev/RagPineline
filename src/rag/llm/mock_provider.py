"""
Mock Provider for local testing when no API key is set.
"""

from typing import Any, Dict, List
from src.rag.llm.base import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")

        # Tìm các mã [SOURCE_X] có trong ngữ cảnh
        import re
        sources = re.findall(r"\[SOURCE_\d+\]", user_content)
        src_tag = sources[0] if sources else "[SOURCE_1]"

        return (
            f"Dựa trên tài liệu được cung cấp, hệ thống RAG vận hành theo quy trình gồm 3 giai đoạn chính: "
            f"Indexing (chuẩn bị dữ liệu), Retrieval (truy xuất ngữ cảnh) và Generation (sinh câu trả lời tự nhiên) {src_tag}."
        )
