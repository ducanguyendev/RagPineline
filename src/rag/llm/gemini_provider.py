"""
Gemini Provider using official google-genai SDK
"""

import json
import os
import sys
from typing import Any, Dict, List
from src.rag.llm.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY không được để trống. Vui lòng cấu hình API Key trong file .env!")

        system_instruction = ""
        user_prompt = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                user_prompt = content

        # 1. Thử sử dụng SDK chính thức mới nhất google.genai
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_instruction if system_instruction else None,
                temperature=kwargs.get("temperature", 0.1),
                max_output_tokens=kwargs.get("max_tokens", 1200)
            )

            # Chuẩn hóa tên mô hình nếu người dùng truyền chuỗi dạng "Gemini 3.5 Flash Lite"
            raw_model = self.model_name.lower().strip()
            lite_candidates = []
            if "lite" in raw_model or "flash" in raw_model:
                lite_candidates = ["gemini-2.0-flash-lite", "gemini-1.5-flash-8b", "gemini-2.0-flash", "gemini-1.5-flash"]

            # Tự động lấy danh sách active models hỗ trợ generateContent từ API
            active_models = []
            try:
                for m in client.models.list():
                    name_clean = m.name.replace("models/", "")
                    if any(k in name_clean for k in ["flash", "lite", "pro", "gemma"]):
                        active_models.append(name_clean)
            except Exception:
                pass

            models_to_try = lite_candidates + [self.model_name] + active_models + ["gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
            last_err = None

            for m in list(dict.fromkeys([x for x in models_to_try if x])):
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=user_prompt,
                        config=config
                    )
                    if response and response.text:
                        print(f"✓ Gemini API response received using model: {m}")
                        return response.text.strip()
                except Exception as e:
                    last_err = e
                    continue

            if last_err:
                raise last_err

        except Exception as e_genai:
            # 2. Fallback sang legacy SDK google.generativeai nếu có
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                for m in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
                    try:
                        mdl = legacy_genai.GenerativeModel(
                            model_name=m,
                            system_instruction=system_instruction if system_instruction else None
                        )
                        res = mdl.generate_content(
                            user_prompt,
                            generation_config=legacy_genai.types.GenerationConfig(
                                temperature=kwargs.get("temperature", 0.1),
                                max_output_tokens=kwargs.get("max_tokens", 1200)
                            )
                        )
                        if res and res.text:
                            return res.text.strip()
                    except Exception:
                        continue
            except Exception:
                pass

            err_msg = str(e_genai)
            if "401" in err_msg or "UNAUTHENTICATED" in err_msg:
                print("❌ [Gemini API Error] GEMINI_API_KEY không hợp lệ! API Key chuẩn từ Google AI Studio phải bắt đầu bằng 'AIzaSy...'. Vui lòng tạo key mới tại https://aistudio.google.com/app/apikey")
                raise RuntimeError("GEMINI_API_KEY không hợp lệ (Lỗi 401 UNAUTHENTICATED). API Key chuẩn từ Google AI Studio phải bắt đầu bằng 'AIzaSy...'. Vui lòng tạo key tại https://aistudio.google.com/app/apikey")

            raise RuntimeError(f"Lỗi gọi Gemini SDK ({e_genai})")

        return "Không nhận được phản hồi từ Gemini API."
