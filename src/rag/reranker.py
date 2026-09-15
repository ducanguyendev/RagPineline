"""
Re-ranker Subsystem for RAG Pipeline

Uses BAAI/bge-reranker-v2-m3 Cross-Encoder model to score query-chunk pairs,
sort by relevance, and apply RERANK_THRESHOLD.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_RERANKER_MODEL_CACHE = None


def get_reranker_model(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global _RERANKER_MODEL_CACHE
    if _RERANKER_MODEL_CACHE is not None:
        return _RERANKER_MODEL_CACHE

    device = "cpu"
    print(f"Loading Re-ranker model: {model_name} on {device.upper()}...")

    try:
        from sentence_transformers import CrossEncoder
        _RERANKER_MODEL_CACHE = CrossEncoder(model_name, device=device)
        print(f"✓ Re-ranker model loaded successfully on device: {device.upper()}")
        return _RERANKER_MODEL_CACHE
    except Exception as e:
        print(f"! Warning: Failed to load CrossEncoder on {device} ({e}). Falling back to Transformers pipeline.")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        class FallbackReranker:
            def __init__(self, tok, mdl):
                self.tokenizer = tok
                self.model = mdl

            def predict(self, pairs: List[List[str]]) -> List[float]:
                scores = []
                import torch
                with torch.no_grad():
                    for pair in pairs:
                        inputs = self.tokenizer(
                            pair[0], pair[1],
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt"
                        )
                        out = self.model(**inputs)
                        score = out.logits[0][0].item()
                        scores.append(score)
                return scores

        _RERANKER_MODEL_CACHE = FallbackReranker(tokenizer, model)
        print("✓ Fallback Re-ranker loaded on CPU")
        return _RERANKER_MODEL_CACHE


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: int = 5,
    threshold: float = 0.35,
    model_name: str = "BAAI/bge-reranker-v2-m3"
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Tái xếp hạng danh sách chunks bằng Cross-Encoder.
    
    Args:
        query: Câu hỏi của người dùng
        documents: Danh sách candidate chunks từ Dense Retrieval
        top_n: Số lượng chunk giữ lại tối đa
        threshold: Ngưỡng điểm tối thiểu để được coi là relevant
        
    Returns:
        (reranked_chunks, passed_threshold)
    """
    if not documents or not query.strip():
        return [], False

    reranker = get_reranker_model(model_name)

    # Chuẩn bị cặp (query, content)
    pairs = []
    for doc in documents:
        content = doc.get("content") or doc.get("page_content") or ""
        pairs.append([query, content])

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_scores = reranker.predict(pairs)

    # Ghi nhận rerank_score vào từng document
    scored_docs = []
    for idx, doc in enumerate(documents):
        doc_copy = dict(doc)
        raw_score = float(raw_scores[idx])
        # Chuẩn hóa score bằng sigmoid nếu raw score thuộc dạng logit
        import math
        sig_score = 1.0 / (1.0 + math.exp(-raw_score)) if raw_score < 0 or raw_score > 1 else raw_score

        doc_copy["retrieval_score"] = doc.get("score") or doc.get("distance") or 0.0
        doc_copy["rerank_score"] = round(sig_score, 4)
        doc_copy["raw_rerank_score"] = round(raw_score, 4)
        scored_docs.append(doc_copy)

    # Sắp xếp giảm dần theo rerank_score
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Lọc theo threshold
    valid_docs = [d for d in scored_docs if d["rerank_score"] >= threshold]

    if not valid_docs:
        print(f"⚠️ [Re-ranker Threshold] Top score ({scored_docs[0]['rerank_score'] if scored_docs else 0}) < Threshold ({threshold}). Early exit!")
        return [], False

    # Giới hạn Top-N
    final_docs = valid_docs[:top_n]
    return final_docs, True


if __name__ == "__main__":
    test_query = "Kiến trúc RAG hiện đại gồm những giai đoạn nào?"
    test_docs = [
        {"content": "Một hệ thống RAG tiêu chuẩn hiện nay gồm 3 giai đoạn chính: Indexing, Retrieval và Generation.", "score": 0.8},
        {"content": "Hà Nội là thủ đô của Việt Nam với lịch sử ngàn năm văn hiến.", "score": 0.5}
    ]
    res, passed = rerank_documents(test_query, test_docs, top_n=2, threshold=0.3)
    print(f"Passed: {passed}")
    for d in res:
        print(f"Score: {d['rerank_score']} | Content: {d['content']}")
