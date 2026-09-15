"""
Retrieval & Re-ranking Subsystem Evaluator for vi-coze RAG Pipeline

Evaluates:
- Context Precision: Evaluates rank ordering accuracy of retrieved chunks.
- Context Recall: Evaluates fact/context coverage against ground truth reference.

Features:
- Bypasses LLM Generation completely (Fast, cost-effective, offline-capable).
- Categorized breakdown by content type: Text, Table, Image.
- Configurable RERANK_THRESHOLD to evaluate precision/recall trade-offs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag.hybrid_retriever import hybrid_search
from src.rag.reranker import rerank_documents
from sentence_transformers import SentenceTransformer, util

_SIM_MODEL = None


def get_similarity_model():
    global _SIM_MODEL
    if _SIM_MODEL is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        print(f"Loading embedding similarity model for retrieval eval: {model_name} on CPU...")
        _SIM_MODEL = SentenceTransformer(model_name, device="cpu")
    return _SIM_MODEL


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts using bge-m3."""
    if not text1.strip() or not text2.strip():
        return 0.0
    model = get_similarity_model()
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    cosine_score = float(util.cos_sim(emb1, emb2)[0][0])
    return max(0.0, min(1.0, round(cosine_score, 4)))


def compute_context_metrics(
    retrieved_contexts: List[str],
    reference_answer: str,
    reference_contexts: List[str],
    relevance_threshold: float = 0.55
) -> Tuple[float, float]:
    """
    Computes Context Recall & Context Precision.

    - Context Recall: Ratio of key facts/sentences in reference_contexts covered by at least one retrieved chunk.
    - Context Precision: Mean Average Precision (MAP) of relevant chunks in retrieved_contexts.
    """
    if not retrieved_contexts:
        return 0.0, 0.0

    # Build reference text set to check coverage against
    ref_targets = reference_contexts if reference_contexts else [reference_answer]
    
    # 1. COMPUTE CONTEXT RECALL
    # For each reference target sentence/paragraph, find max similarity score in retrieved_contexts
    target_coverages = []
    for ref_t in ref_targets:
        if not ref_t.strip():
            continue
        max_sim = 0.0
        for ctx in retrieved_contexts:
            sim = compute_semantic_similarity(ref_t, ctx)
            if sim > max_sim:
                max_sim = sim
        target_coverages.append(max_sim)

    context_recall = sum(target_coverages) / len(target_coverages) if target_coverages else 0.0

    # 2. COMPUTE CONTEXT PRECISION (Mean Average Precision - MAP)
    # Determine relevance of each retrieved chunk at rank k
    relevance_flags = []
    for ctx in retrieved_contexts:
        # Check if chunk matches any reference context or answer
        max_rel = 0.0
        for ref_t in ref_targets:
            sim = compute_semantic_similarity(ctx, ref_t)
            if sim > max_rel:
                max_rel = sim
        is_relevant = max_rel >= relevance_threshold
        relevance_flags.append(is_relevant)

    total_relevant = sum(1 for flag in relevance_flags if flag)
    if total_relevant == 0:
        context_precision = 0.0
    else:
        running_relevant = 0
        precision_at_k_sum = 0.0
        for k, is_rel in enumerate(relevance_flags, start=1):
            if is_rel:
                running_relevant += 1
                precision_at_k = running_relevant / k
                precision_at_k_sum += precision_at_k
        context_precision = precision_at_k_sum / total_relevant

    return round(context_recall, 4), round(context_precision, 4)


def run_retrieval_evaluation(
    dataset_path: Path,
    output_path: Path,
    top_k: int = 15,
    top_n: int = 5,
    threshold: float = 0.05
) -> Dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    print("=" * 75)
    print(f"🎯 RUNNING RETRIEVAL & RE-RANKING EVALUATION ({len(dataset)} items)")
    print(f"Top-K Retrieval: {top_k} | Top-N Rerank: {top_n} | Threshold: {threshold}")
    print(f"Dataset: {dataset_path}")
    print("=" * 75)

    results = []
    category_stats: Dict[str, Dict[str, Any]] = {}

    start_all = time.perf_counter()

    for idx, item in enumerate(dataset, start=1):
        q_id = item.get("id", idx)
        category = item.get("category", "text").lower()
        user_input = item.get("user_input") or item.get("question", "")
        reference = item.get("reference") or item.get("ground_truth", "")
        ref_contexts = item.get("reference_contexts", [])

        print(f"\n[{idx}/{len(dataset)}] Category: [{category.upper()}] | ID {q_id}")
        print(f"   Query: {user_input}")

        t0 = time.perf_counter()

        # BƯỚC 1: HYBRID RETRIEVAL (Dense + Sparse BM25 + RRF)
        candidates = hybrid_search(
            query=user_input,
            top_k_dense=top_k,
            top_k_sparse=top_k,
            rrf_k=60
        )

        # BƯỚC 2: RE-RANKING & RELEVANCE THRESHOLD FILTERING
        reranked_docs, passed_threshold = rerank_documents(
            query=user_input,
            documents=candidates,
            top_n=top_n,
            threshold=threshold
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract retrieved contexts
        retrieved_contexts = [doc.get("content", "") for doc in reranked_docs]

        # BƯỚC 3: COMPUTE EVALUATION METRICS
        recall_score, precision_score = compute_context_metrics(
            retrieved_contexts=retrieved_contexts,
            reference_answer=reference,
            reference_contexts=ref_contexts
        )

        print(f"   ↳ Retrived: {len(retrieved_contexts)} chunks | Recall: {recall_score:.4f} | Precision: {precision_score:.4f} | Latency: {latency_ms:.2f}ms")

        record = {
            "id": q_id,
            "category": category,
            "user_input": user_input,
            "reference": reference,
            "reference_contexts": ref_contexts,
            "retrieved_chunks_count": len(retrieved_contexts),
            "retrieved_contexts": retrieved_contexts,
            "context_recall": recall_score,
            "context_precision": precision_score,
            "latency_ms": round(latency_ms, 2)
        }
        results.append(record)

        # Update Category Stats
        if category not in category_stats:
            category_stats[category] = {
                "count": 0,
                "total_recall": 0.0,
                "total_precision": 0.0,
                "total_latency_ms": 0.0
            }
        category_stats[category]["count"] += 1
        category_stats[category]["total_recall"] += recall_score
        category_stats[category]["total_precision"] += precision_score
        category_stats[category]["total_latency_ms"] += latency_ms

    total_duration_s = time.perf_counter() - start_all

    # Overall Averages
    avg_overall_recall = round(sum(r["context_recall"] for r in results) / len(results), 4)
    avg_overall_precision = round(sum(r["context_precision"] for r in results) / len(results), 4)
    avg_overall_latency = round(sum(r["latency_ms"] for r in results) / len(results), 2)

    # Category Summaries
    category_summary = {}
    for cat, stat in category_stats.items():
        cnt = stat["count"]
        category_summary[cat] = {
            "count": cnt,
            "avg_context_recall": round(stat["total_recall"] / cnt, 4),
            "avg_context_precision": round(stat["total_precision"] / cnt, 4),
            "avg_latency_ms": round(stat["total_latency_ms"] / cnt, 2)
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": len(results),
        "rerank_threshold": threshold,
        "overall_metrics": {
            "avg_context_recall": avg_overall_recall,
            "avg_context_precision": avg_overall_precision,
            "avg_latency_ms": avg_overall_latency,
            "total_duration_s": round(total_duration_s, 2)
        },
        "by_category": category_summary,
        "details": results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print("📊 RETRIEVAL EVALUATION SUMMARY (CLASSIFIED BY CATEGORY)")
    print("=" * 75)
    print(f"{'Category':<12} | {'Count':<6} | {'Context Recall':<16} | {'Context Precision':<18} | {'Avg Latency':<12}")
    print("-" * 75)
    for cat, metrics in category_summary.items():
        print(f"{cat.upper():<12} | {metrics['count']:<6} | {metrics['avg_context_recall']*100:>14.2f}% | {metrics['avg_context_precision']*100:>16.2f}% | {metrics['avg_latency_ms']:>9.2f} ms")
    print("-" * 75)
    print(f"{'OVERALL':<12} | {len(results):<6} | {avg_overall_recall*100:>14.2f}% | {avg_overall_precision*100:>16.2f}% | {avg_overall_latency:>9.2f} ms")
    print("=" * 75)
    print(f"📁 Output Saved To: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval & Re-ranking Subsystem")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_retrieval_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval_retrieval_results.json"))
    parser.add_argument("--top-k", type=int, default=15, help="Number of candidates from Dense+BM25 Search")
    parser.add_argument("--top-n", type=int, default=5, help="Number of candidates after Re-ranker")
    parser.add_argument("--threshold", type=float, default=0.05, help="Re-ranker relevance threshold")

    args = parser.parse_args()
    run_retrieval_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        top_k=args.top_k,
        top_n=args.top_n,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
