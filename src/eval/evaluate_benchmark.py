"""
Benchmark Evaluator for vi-coze RAG Pipeline

Evaluates the RAG system against a golden testset (data/eval_dataset.json).
Calculates:
- Answer Semantic Similarity (via Embedding Model)
- Early Exit Rate / No-Info Detection
- Citation Accuracy & Image Resolution Counts
- End-to-End Latency Metrics
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

from src.rag.rag_pipeline import answer_query
from sentence_transformers import SentenceTransformer, util

_SIM_MODEL = None


def get_similarity_model():
    global _SIM_MODEL
    if _SIM_MODEL is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        print(f"Loading similarity model for evaluation: {model_name} on CPU...")
        _SIM_MODEL = SentenceTransformer(model_name, device="cpu")
    return _SIM_MODEL


def compute_semantic_similarity(str1: str, str2: str) -> float:
    """Tính độ tương đồng ngữ nghĩa (Cosine Similarity) giữa 2 câu bằng bge-m3 trên CPU."""
    if not str1.strip() or not str2.strip():
        return 0.0
    model = get_similarity_model()
    emb1 = model.encode(str1, convert_to_tensor=True)
    emb2 = model.encode(str2, convert_to_tensor=True)
    cosine_score = float(util.cos_sim(emb1, emb2)[0][0])
    return max(0.0, min(1.0, round(cosine_score, 4)))


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    provider: str = "gemini",
    limit: int = None
) -> Dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    if limit and limit > 0:
        dataset = dataset[:limit]

    print("=" * 70)
    print(f"🚀 RUNNING BENCHMARK EVALUATION ({len(dataset)} items)")
    print(f"Provider: {provider} | Dataset: {dataset_path}")
    print("=" * 70)

    results = []
    total_latency = 0.0
    total_similarity = 0.0
    pass_count = 0  # Similarity >= 0.70
    early_exit_count = 0

    for idx, item in enumerate(dataset, start=1):
        q_id = item.get("id", idx)
        query = item.get("query", "")
        expected = item.get("expected_answer", "")

        print(f"\n[{idx}/{len(dataset)}] ID {q_id}: {query}")

        start_t = time.perf_counter()
        rag_res = answer_query(query=query, provider_name=provider)
        duration_ms = (time.perf_counter() - start_t) * 1000

        gen_answer = rag_res.get("answer", "")
        citations = rag_res.get("citations", [])
        image_urls = rag_res.get("image_urls", [])

        # Kiểm tra xem có rơi vào Early Exit không
        is_early_exit = "không tìm thấy thông tin" in gen_answer.lower()
        if is_early_exit:
            early_exit_count += 1

        # Tính similarity ngữ nghĩa với Ground Truth
        sim_score = compute_semantic_similarity(gen_answer, expected)
        is_pass = sim_score >= 0.70
        if is_pass:
            pass_count += 1

        total_latency += duration_ms
        total_similarity += sim_score

        status_str = "✅ PASS" if is_pass else "❌ LOW SIM"
        print(f"   ↳ {status_str} | Similarity: {sim_score:.4f} | Latency: {duration_ms:.2f}ms | Citations: {len(citations)}")
        print(f"   ↳ Ans: {gen_answer[:100]}...")

        results.append({
            "id": q_id,
            "query": query,
            "expected_answer": expected,
            "generated_answer": gen_answer,
            "similarity_score": sim_score,
            "is_pass": is_pass,
            "is_early_exit": is_early_exit,
            "citations_count": len(citations),
            "citations": citations,
            "image_urls_count": len(image_urls),
            "image_urls": image_urls,
            "latency_ms": round(duration_ms, 2)
        })

    avg_latency = round(total_latency / len(dataset), 2)
    avg_similarity = round(total_similarity / len(dataset), 4)
    pass_rate = round((pass_count / len(dataset)) * 100, 2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "total_test_cases": len(dataset),
        "avg_latency_ms": avg_latency,
        "avg_similarity_score": avg_similarity,
        "pass_count": pass_count,
        "pass_rate_percent": pass_rate,
        "early_exit_count": early_exit_count,
        "details": results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Testcases      : {len(dataset)}")
    print(f"Average Latency      : {avg_latency} ms")
    print(f"Avg Semantic Match   : {avg_similarity * 100:.2f}%")
    print(f"Pass Rate (>=0.70)   : {pass_rate}% ({pass_count}/{len(dataset)})")
    print(f"Early Exit Count     : {early_exit_count}")
    print(f"Saved Results        : {output_path}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline against benchmark dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval_results.json"))
    parser.add_argument("--provider", type=str, default="gemini")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of testcases to run")

    args = parser.parse_args()
    run_evaluation(args.dataset, args.output, args.provider, args.limit)


if __name__ == "__main__":
    main()
