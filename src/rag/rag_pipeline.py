"""
Master RAG Pipeline Orchestrator (Online Serving Stage)

Pipeline Flow:
User Query -> Dense Retrieval -> Re-ranking -> Relevance Threshold -> Context Builder -> Prompt Builder -> LLM Provider -> Citation Validation -> Answer + Sources + Physical Images
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag.retriever import load_vector_db
from src.rag.reranker import rerank_documents
from src.rag.context_builder import build_context
from src.rag.prompt_builder import build_messages
from src.rag.llm import get_llm_provider
from src.rag.citation import validate_and_resolve_citations

NO_INFO_FALLBACK = "Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này."


def _load_env_file():
    env_path = ROOT / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    parts = line.split("=", 1)
                    k = parts[0].strip()
                    v = parts[1].strip().strip("'").strip('"')
                    if k:
                        os.environ[k] = v


def answer_query(
    query: str,
    top_k: int = None,
    top_n: int = None,
    threshold: float = None,
    max_context_tokens: int = None,
    provider_name: str = None,
    vector_dir: Path = None,
    embedding_model: str = None
) -> Dict[str, Any]:
    """
    Thực thi luồng RAG hoàn chỉnh End-to-End.
    """
    _load_env_file()
    start_time = time.perf_counter()
    query = query.strip()

    if not query:
        return {
            "query": query,
            "answer": "Vui lòng nhập câu hỏi hợp lệ.",
            "citations": [],
            "image_urls": [],
            "logs": []
        }

    # Load cấu hình mặc định từ file .env / tham số
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "15"))
    top_n = top_n or int(os.getenv("RERANK_TOP_N", "5"))
    threshold = threshold if threshold is not None else float(os.getenv("RERANK_THRESHOLD", "0.05"))
    max_context_tokens = max_context_tokens or int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
    embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    vector_dir = vector_dir or (ROOT / "data" / "vectorstore" / "chroma")

    # Giới hạn an toàn tham số (Principle 18)
    top_k = max(1, min(top_k, 30))
    top_n = max(1, min(top_n, 10))
    if top_n > top_k:
        top_n = top_k

    logs = []

    # BƯỚC 1: HYBRID RETRIEVAL (Dense Vector + Sparse BM25 + RRF Fusion)
    retrieval_start = time.perf_counter()
    from src.rag.hybrid_retriever import hybrid_search

    candidates = hybrid_search(
        query=query,
        top_k_dense=top_k,
        top_k_sparse=top_k,
        rrf_k=60,
        vector_dir=vector_dir,
        embedding_model=embedding_model
    )

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    logs.append({"stage": "hybrid_retrieval", "count": len(candidates), "duration_ms": round(retrieval_ms, 2)})

    # BƯỚC 2 & 3: RE-RANKING & RELEVANCE THRESHOLD
    rerank_start = time.perf_counter()
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranked_docs, passed_threshold = rerank_documents(
        query=query,
        documents=candidates,
        top_n=top_n,
        threshold=threshold,
        model_name=reranker_model
    )
    rerank_ms = (time.perf_counter() - rerank_start) * 1000
    logs.append({"stage": "re_ranking", "passed_threshold": passed_threshold, "count": len(reranked_docs), "duration_ms": round(rerank_ms, 2)})

    # NẾU KHÔNG CÓ CHUNK ĐẠT THRESHOLD -> NGẮT SỚM (EARLY EXIT - NO LLM CALL)
    if not passed_threshold or not reranked_docs:
        total_ms = (time.perf_counter() - start_time) * 1000
        print("⛔ [Early Exit] No candidate passed relevance threshold. Skipping LLM call.")
        return {
            "query": query,
            "answer": NO_INFO_FALLBACK,
            "citations": [],
            "image_urls": [],
            "total_duration_ms": round(total_ms, 2),
            "logs": logs
        }

    # BƯỚC 4: CONTEXT BUILDER (Deduplication, MAX_CONTEXT_TOKENS & SOURCE IDs)
    context_text, source_mapping, selected_chunks = build_context(reranked_docs, max_tokens=max_context_tokens)

    # BƯỚC 5: PROMPT BUILDER
    messages = build_messages(query, context_text)

    # BƯỚC 6: LLM GENERATION
    llm_start = time.perf_counter()
    llm = get_llm_provider(provider_name)
    raw_answer = llm.generate(messages)
    llm_ms = (time.perf_counter() - llm_start) * 1000
    logs.append({"stage": "llm_generation", "provider": llm.__class__.__name__, "duration_ms": round(llm_ms, 2)})

    # BƯỚC 7: CITATION VALIDATION & SOURCE RESOLUTION
    cleaned_answer, citations, image_urls = validate_and_resolve_citations(raw_answer, source_mapping)

    total_ms = (time.perf_counter() - start_time) * 1000

    return {
        "query": query,
        "answer": cleaned_answer,
        "citations": citations,
        "image_urls": image_urls,
        "total_duration_ms": round(total_ms, 2),
        "logs": logs,
        "debug": {
            "selected_chunks_count": len(selected_chunks),
            "source_ids": list(source_mapping.keys())
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Test Full RAG Pipeline End-to-End")
    parser.add_argument("--query", required=True, type=str, help="Question query")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM provider: gemini | openai | ollama | mock")
    args = parser.parse_args()

    print(f"\nQUERY: {args.query}\n" + "=" * 60)
    result = answer_query(args.query, provider_name=args.provider)
    print("\nANSWER:\n", result["answer"])
    print("\nCITATIONS:\n", json.dumps(result["citations"], ensure_ascii=False, indent=2))
    print("\nIMAGES:\n", result["image_urls"])
    print("\nTOTAL DURATION:", result["total_duration_ms"], "ms")


if __name__ == "__main__":
    main()
