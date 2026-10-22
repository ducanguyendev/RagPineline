"""Run offline RAG evaluation with RAGAS and a Gemini judge."""

import argparse
import inspect
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_environment() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required to load ROOT/.env") from exc
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. RAGAS Gemini evaluation cannot run.")
    model = (os.getenv("RAGAS_LLM_MODEL") or os.getenv("LLM_MODEL") or "gemini-2.5-flash").strip()
    return model


def _load_ragas_api():
    try:
        import ragas
        from ragas.metrics import collections
    except Exception as exc:
        raise RuntimeError(f"RAGAS import failure: {exc}") from exc

    classes = {
        "faithfulness": getattr(collections, "Faithfulness", None),
        "context_precision": getattr(collections, "ContextPrecision", None),
        "context_recall": getattr(collections, "ContextRecall", None),
    }
    missing = [name for name, cls in classes.items() if cls is None]
    if missing:
        raise RuntimeError(
            "Installed RAGAS does not expose collection metrics: " + ", ".join(missing)
        )
    try:
        from ragas.llms import llm_factory
    except Exception as exc:
        raise RuntimeError(f"RAGAS llm_factory is unavailable: {exc}") from exc
    return ragas, classes, llm_factory


def create_ragas_llm(model: str, api_key: Optional[str] = None, llm_factory=None):
    """Construct RAGAS's Gemini judge adapter for the installed RAGAS API."""
    if llm_factory is None:
        try:
            from ragas.llms import llm_factory
        except Exception as exc:
            raise RuntimeError(f"RAGAS llm_factory is unavailable: {exc}") from exc
    api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. RAGAS Gemini evaluation cannot run.")
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError(f"google-genai import failed: {exc}") from exc

    try:
        client = genai.Client(api_key=api_key)
        signature = inspect.signature(llm_factory)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize Gemini judge client: {exc}") from exc
    kwargs = {}
    if "provider" in signature.parameters:
        kwargs["provider"] = "google"
    if "client" in signature.parameters:
        kwargs["client"] = client
    elif "api_key" in signature.parameters:
        kwargs["api_key"] = api_key
    try:
        return llm_factory(model, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not configure RAGAS Gemini judge with model {model!r}: {exc}"
        ) from exc


def _metric_instance(metric_class, llm):
    signature = inspect.signature(metric_class)
    kwargs = {}
    if "llm" in signature.parameters:
        kwargs["llm"] = llm
    if "embeddings" in signature.parameters:
        kwargs["embeddings"] = None
    try:
        return metric_class(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize RAGAS metric {metric_class.__name__}: {exc}") from exc


def _score_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    for attr in ("value", "score"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    raise TypeError(f"RAGAS returned an unsupported score value: {type(value).__name__}")


def _metric_score(metric, row: Dict[str, Any]) -> float:
    method = getattr(metric, "ascore", None) or getattr(metric, "score", None)
    if method is None:
        raise RuntimeError(f"Metric {type(metric).__name__} exposes neither ascore() nor score()")
    parameters = inspect.signature(method).parameters
    if "single_turn" in parameters:
        from ragas.dataset_schema import SingleTurnSample
        sample = SingleTurnSample(**row)
        result = method(single_turn=sample)
    else:
        kwargs = {key: row[key] for key in ("user_input", "response", "reference", "retrieved_contexts")}
        result = method(**kwargs)
    if inspect.isawaitable(result):
        import asyncio
        result = asyncio.run(result)
    return _score_value(result)


def _question_and_reference(item: Dict[str, Any], testcase_id: Any):
    question = item.get("user_input") or item.get("query") or item.get("question")
    reference = item.get("reference") or item.get("expected_answer") or item.get("ground_truth")
    if not question:
        raise ValueError(f"Test case {testcase_id} is missing question (user_input/query/question)")
    if not reference:
        raise ValueError(f"Test case {testcase_id} is missing reference (reference/expected_answer/ground_truth)")
    return str(question), str(reference)


def _avg(details, key):
    values = [item[key] for item in details if isinstance(item.get(key), (int, float))]
    return round(statistics.mean(values), 6) if values else None


def _corpus_warning():
    # Lightweight metadata check only; never opens or modifies the vector index.
    candidates = [ROOT / "data" / "processed" / "manifest.json", ROOT / "data" / "processed" / "chunked.jsonl"]
    corpus_text = " ".join(path.read_text(encoding="utf-8", errors="ignore")[:200000] for path in candidates if path.is_file()).lower()
    if corpus_text and not any(term in corpus_text for term in ("retrieval-augmented", "hallucination", "grounding", "rag system")):
        print("WARNING: Evaluation dataset appears unrelated to the indexed corpus.\nRAGAS scores would not be meaningful.", file=sys.stderr)


def run_evaluation(dataset_path: Path, output_path: Path, limit: Optional[int] = None, include_contexts: bool = False):
    ragas, metric_classes, llm_factory = _load_ragas_api()
    model = _load_environment()
    if not dataset_path.is_file():
        raise RuntimeError(f"Dataset not found: {dataset_path}")
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read evaluation dataset {dataset_path}: {exc}") from exc
    if not isinstance(dataset, list):
        raise RuntimeError("Evaluation dataset must be a JSON array of test cases")
    # Validate every row before any retrieval or API call so malformed dataset is a setup failure.
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Test case {index} must be a JSON object")
        _question_and_reference(item, item.get("id", index))
    if limit is not None:
        dataset = dataset[:limit]
    llm = create_ragas_llm(model, llm_factory=llm_factory)
    metrics = {name: _metric_instance(cls, llm) for name, cls in metric_classes.items()}
    # Validate the installed score API before beginning the RAG run.
    for metric in metrics.values():
        if not (callable(getattr(metric, "ascore", None)) or callable(getattr(metric, "score", None))):
            raise RuntimeError(f"Metric {type(metric).__name__} exposes neither ascore() nor score()")
    _corpus_warning()

    from src.rag.rag_pipeline import answer_query

    details = []
    for index, item in enumerate(dataset, start=1):
        testcase_id = item.get("id", index) if isinstance(item, dict) else index
        print(f"[{index}/{len(dataset)}] {testcase_id}", flush=True)
        record = {
            "id": testcase_id,
            "query": None,
            "reference": None,
            "response": None,
            "retrieved_contexts_count": 0,
            "retrieved_context_ids": [],
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
            "rag_latency_ms": None,
            "is_early_exit": False,
            "evaluation_status": "failed",
            "error": None,
        }
        try:
            if not isinstance(item, dict):
                raise ValueError(f"Test case {testcase_id} must be a JSON object")
            question, reference = _question_and_reference(item, testcase_id)
            record["query"], record["reference"] = question, reference
            started = time.perf_counter()
            result = answer_query(question, provider_name="gemini", include_eval_trace=True)
            record["rag_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            record["response"] = result.get("answer", "")
            trace = result.get("eval_trace") or {}
            contexts = trace.get("retrieved_contexts") or []
            record["retrieved_contexts_count"] = len(contexts)
            record["retrieved_context_ids"] = trace.get("retrieved_context_ids") or []
            if include_contexts:
                record["retrieved_contexts"] = contexts
            if not contexts:
                record["is_early_exit"] = True
                record["error"] = "No retrieved context passed reranking threshold."
            else:
                row = {"user_input": question, "response": record["response"], "reference": reference, "retrieved_contexts": contexts}
                for name, metric in metrics.items():
                    record[name] = _metric_score(metric, row)
            record["evaluation_status"] = "success"
            print(f"[{index}/{len(dataset)}] SUCCESS")
        except Exception as exc:
            record["error"] = str(exc)
            print(f"[{index}/{len(dataset)}] FAILED\nReason: {exc}")
        details.append(record)

    successful = sum(item["evaluation_status"] == "success" for item in details)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "provider": "gemini",
        "ragas_version": getattr(ragas, "__version__", "unknown"),
        "ragas_model": model,
        "total_test_cases": len(details),
        "successful": successful,
        "failed": len(details) - successful,
        "early_exit_count": sum(item["is_early_exit"] for item in details),
        "metrics": {
            "avg_faithfulness": _avg(details, "faithfulness"),
            "avg_context_precision": _avg(details, "context_precision"),
            "avg_context_recall": _avg(details, "context_recall"),
        },
        "details": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("\n========================================\nRAGAS EVALUATION SUMMARY\n========================================")
    print(f"Total               : {summary['total_test_cases']}")
    print(f"Successful          : {summary['successful']}")
    print(f"Failed              : {summary['failed']}")
    print(f"Early Exit          : {summary['early_exit_count']}\n")
    for label, key in (("Faithfulness", "avg_faithfulness"), ("Context Precision", "avg_context_precision"), ("Context Recall", "avg_context_recall")):
        value = summary["metrics"][key]
        print(f"{label:<20}: {value:.4f}" if value is not None else f"{label:<20}: n/a")
    print(f"\nOutput:\n{output_path}\n========================================")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS and Gemini")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "eval_dataset.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "ragas_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-contexts", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        run_evaluation(args.dataset, args.output, args.limit, args.include_contexts)
    except RuntimeError as exc:
        parser.exit(1, f"RAGAS setup error: {exc}\n")


if __name__ == "__main__":
    main()
