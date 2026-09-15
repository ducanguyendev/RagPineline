"""
Base Interface for LLM Providers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseLLMProvider(ABC):
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Sinh câu trả lời từ danh sách tin nhắn [system, user, assistant].
        
        Args:
            messages: List of message dicts {"role": "system"/"user", "content": "..."}
            kwargs: Extra parameters like temperature, max_tokens
            
        Returns:
            Generated text string
        """
        pass
