"""
OpenAI Provider for RAG
"""

import json
import os
import urllib.request
from typing import Any, Dict, List
from src.rag.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY không được để trống!")

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1200)
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()

        return "Không nhận được phản hồi từ OpenAI API."
