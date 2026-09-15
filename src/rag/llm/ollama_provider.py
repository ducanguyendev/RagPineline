"""
Ollama Local Provider for RAG
"""

import json
import os
import urllib.request
from typing import Any, Dict, List
from src.rag.llm.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = None, model_name: str = "qwen2.5:7b"):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("LLM_MODEL", "qwen2.5:7b")

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 1200)
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            raise RuntimeError(f"Không thể kết nối dịch vụ Ollama Local ({self.base_url}): {e}")
