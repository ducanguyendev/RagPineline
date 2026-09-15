"""
Token Counter Utility for RAG Pipeline
Uses tiktoken cl100k_base with fallback calculation.
"""

from typing import Optional
import tiktoken


class TokenCounter:
    _instance: Optional["TokenCounter"] = None

    def __init__(self, model_name: str = "cl100k_base"):
        try:
            self.encoder = tiktoken.get_encoding(model_name)
        except Exception:
            self.encoder = None

    @classmethod
    def get_instance(cls) -> "TokenCounter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        # Fallback estimation for Vietnamese/English
        return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    return TokenCounter.get_instance().count(text)
