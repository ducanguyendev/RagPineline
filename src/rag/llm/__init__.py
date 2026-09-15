"""
LLM Provider Factory
"""

import os
from pathlib import Path
from src.rag.llm.base import BaseLLMProvider
from src.rag.llm.gemini_provider import GeminiProvider
from src.rag.llm.openai_provider import OpenAIProvider
from src.rag.llm.ollama_provider import OllamaProvider
from src.rag.llm.mock_provider import MockProvider


def _load_env_file():
    curr = Path(__file__).resolve()
    for parent in [curr.parent, curr.parents[1], curr.parents[2], curr.parents[3]]:
        env_p = parent / ".env"
        if env_p.exists():
            with env_p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        parts = line.split("=", 1)
                        k = parts[0].strip()
                        v = parts[1].strip().strip("'").strip('"')
                        if k and v:
                            os.environ[k] = v
            break


def get_llm_provider(provider_name: str = None) -> BaseLLMProvider:
    _load_env_file()
    name = (provider_name or os.getenv("LLM_PROVIDER", "gemini")).lower()

    if name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("! Warning: GEMINI_API_KEY chưa được cấu hình. Chuyển sang MockProvider để test.")
            return MockProvider()
        return GeminiProvider(api_key=api_key)
    elif name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("! Warning: OPENAI_API_KEY chưa được cấu hình. Chuyển sang MockProvider để test.")
            return MockProvider()
        return OpenAIProvider(api_key=api_key)
    elif name == "ollama":
        return OllamaProvider()
    elif name == "mock":
        return MockProvider()
    else:
        print(f"! Provider '{name}' không hợp lệ. Sử dụng MockProvider mặc định.")
        return MockProvider()
