"""
Sparse Keyword Retriever (BM25Okapi) for vi-coze RAG Pipeline

Implements BM25 scoring over chunked.jsonl documents.
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


VIETNAMESE_STOPWORDS = {
    "trong", "tài", "liệu", "được", "hiểu", "là", "gì", "của", "và", "các", "những",
    "một", "với", "cho", "này", "đó", "theo", "như", "đã", "đang", "sẽ", "về", "ra",
    "sao", "nào", "bằng", "tại", "từ", "đến", "khi", "để", "thì", "hoặc", "có", "không",
    "như", "thế", "nào", "ở", "gồm", "những", "bởi", "vốn", "tạo"
}


def tokenize(text: str) -> List[str]:
    """Tách từ tiếng Việt / Tiếng Anh bằng regex và lọc bỏ từ dừng (stop words)."""
    if not text:
        return []
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    # Lọc bỏ stop words để tập trung vào từ khóa nội dung thực sự
    filtered = [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]
    return filtered if filtered else tokens


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_tokens = []
        self.doc_lens = []
        df: Dict[str, int] = {}

        for doc in documents:
            content = doc.get("chunk_content") or doc.get("content") or ""
            tokens = tokenize(content)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))

            # Đếm số lượng tài liệu chứa từ t
            seen = set(tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        self.N = len(documents)
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0

        # Tính IDF theo công thức Robertson-Spärck Jones
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 15) -> List[Tuple[Dict[str, Any], float]]:
        query_tokens = tokenize(query)
        if not query_tokens or self.N == 0:
            return []

        scores = [0.0] * self.N

        for idx, doc_toks in enumerate(self.doc_tokens):
            doc_len = self.doc_lens[idx]
            if doc_len == 0:
                continue

            # Đếm tần suất các từ trong document
            freq_map: Dict[str, int] = {}
            for t in doc_toks:
                freq_map[t] = freq_map.get(t, 0) + 1

            doc_score = 0.0
            for q_term in query_tokens:
                if q_term not in freq_map:
                    continue
                f = freq_map[q_term]
                idf_val = self.idf.get(q_term, 0.0)
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                doc_score += idf_val * (numerator / denominator)

            scores[idx] = doc_score

        # Sắp xếp danh sách giảm dần theo điểm số BM25
        ranked_indices = sorted(range(self.N), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked_indices[:top_k]:
            if scores[i] <= 0:
                continue
            results.append((self.documents[i], round(scores[i], 4)))

        return results


_BM25_CACHE = {}


def clear_bm25_cache(input_file: Path = None) -> int:
    """Clear all cached BM25 indexes or only entries for one chunk file."""
    if input_file is None:
        keys = list(_BM25_CACHE)
    else:
        target = str(Path(input_file).resolve())
        keys = [key for key in _BM25_CACHE if key[0] == target]

    for key in keys:
        _BM25_CACHE.pop(key, None)
    return len(keys)


def load_bm25_retriever(input_file: Path = None) -> BM25Retriever:
    if input_file is None:
        input_file = ROOT / "data" / "processed" / "chunked.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(f"Input chunked file not found: {input_file}")

    resolved_path = str(input_file.resolve())
    stat = input_file.stat()
    cache_key = (resolved_path, stat.st_mtime_ns, stat.st_size)
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    # A rewritten chunk file must not leave its previous in-memory index active.
    clear_bm25_cache(input_file)

    documents = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    bm25 = BM25Retriever()
    bm25.fit(documents)
    _BM25_CACHE[cache_key] = bm25
    print(f"✓ Loaded BM25 Index with {len(documents)} chunks from {input_file.name}")
    return bm25


if __name__ == "__main__":
    bm25 = load_bm25_retriever()
    res = bm25.search("Grounding trong tài liệu được hiểu là gì?", top_k=5)
    print("BM25 RESULTS:")
    for doc, score in res:
        print(f"Score: {score:.4f} | Chunk ID: {doc.get('chunk_id')} | Content: {doc.get('chunk_content')[:100]}...")
