"""
Hybrid Retriever (Dense Vector + Sparse BM25 + Reciprocal Rank Fusion)

Combines ChromaDB Dense Vector Search with BM25 Keyword Search using RRF.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag.retriever import load_vector_db
from src.rag.bm25_retriever import load_bm25_retriever


def hybrid_search(
    query: str,
    top_k_dense: int = 15,
    top_k_sparse: int = 15,
    rrf_k: int = 60,
    vector_dir: Path = None,
    embedding_model: str = "BAAI/bge-m3",
    chunked_file: Path = None
) -> List[Dict[str, Any]]:
    """
    Thực hiện Hybrid Search: Dense Vector + Sparse BM25 -> RRF Fusion.
    
    Args:
        query: Câu hỏi của người dùng
        top_k_dense: Số candidate từ Vector DB
        top_k_sparse: Số candidate từ BM25 Search
        rrf_k: Hằng số làm mượt RRF (mặc định 60)
        
    Returns:
        Danh sách candidate chunks đã hợp nhất và xếp hạng theo RRF Score
    """
    query = query.strip()
    if not query:
        return []

    if vector_dir is None:
        vector_dir = ROOT / "data" / "vectorstore" / "chroma"

    # 1. DENSE VECTOR RETRIEVAL (ChromaDB + BGE-M3)
    db = load_vector_db(vector_dir, embedding_model)
    dense_results = db.similarity_search_with_score(query, k=top_k_dense)

    dense_candidates = []
    for rank, (doc, distance) in enumerate(dense_results, start=1):
        meta = doc.metadata or {}
        cid = meta.get("chunk_id") or f"dense_doc_{rank}"
        dense_candidates.append({
            "chunk_id": cid,
            "content": doc.page_content,
            "rank_dense": rank,
            "dense_score": 1.0 / (1.0 + max(float(distance), 0.0)),
            "metadata": meta
        })

    # 2. SPARSE KEYWORD RETRIEVAL (BM25Okapi)
    bm25 = load_bm25_retriever(chunked_file)
    sparse_results = bm25.search(query, top_k=top_k_sparse)

    sparse_candidates = []
    for rank, (doc_dict, score) in enumerate(sparse_results, start=1):
        cid = doc_dict.get("chunk_id") or f"sparse_doc_{rank}"
        meta = doc_dict.get("metadata") or {}
        sparse_candidates.append({
            "chunk_id": cid,
            "content": doc_dict.get("chunk_content") or doc_dict.get("content") or "",
            "rank_sparse": rank,
            "bm25_score": score,
            "metadata": meta
        })

    # 3. RECIPROCAL RANK FUSION (RRF)
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Đưa kết quả Dense vào map
    for item in dense_candidates:
        cid = item["chunk_id"]
        rank_d = item["rank_dense"]
        rrf_score = 1.0 / (rrf_k + rank_d)

        doc_map[cid] = {
            "chunk_id": cid,
            "content": item["content"],
            "metadata": item["metadata"],
            "rank_dense": rank_d,
            "rank_sparse": None,
            "dense_score": item["dense_score"],
            "bm25_score": 0.0,
            "rrf_score": rrf_score
        }

    # Đưa kết quả Sparse vào map và cộng gộp RRF score
    for item in sparse_candidates:
        cid = item["chunk_id"]
        rank_s = item["rank_sparse"]
        rrf_contrib = 1.0 / (rrf_k + rank_s)

        if cid in doc_map:
            doc_map[cid]["rank_sparse"] = rank_s
            doc_map[cid]["bm25_score"] = item["bm25_score"]
            doc_map[cid]["rrf_score"] += rrf_contrib
        else:
            doc_map[cid] = {
                "chunk_id": cid,
                "content": item["content"],
                "metadata": item["metadata"],
                "rank_dense": None,
                "rank_sparse": rank_s,
                "dense_score": 0.0,
                "bm25_score": item["bm25_score"],
                "rrf_score": rrf_contrib
            }

    # 4. SẮP XẾP VÀ TRẢ VỀ THEO RRF SCORE
    merged_docs = list(doc_map.values())
    merged_docs.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Chuẩn hóa định dạng trả về tương thích với reranker
    final_candidates = []
    for d in merged_docs:
        meta = d["metadata"]
        final_candidates.append({
            "content": d["content"],
            "score": round(d["rrf_score"], 6),
            "source": meta.get("source"),
            "page": meta.get("page") or meta.get("page_start"),
            "section": meta.get("section") or meta.get("section_path"),
            "chunk_id": d["chunk_id"],
            "metadata": meta,
            "hybrid_info": {
                "rrf_score": round(d["rrf_score"], 6),
                "rank_dense": d["rank_dense"],
                "rank_sparse": d["rank_sparse"],
                "bm25_score": round(d["bm25_score"], 4)
            }
        })

    return final_candidates


if __name__ == "__main__":
    test_q = "Grounding trong tài liệu được hiểu là gì?"
    print(f"Testing Hybrid Search for: '{test_q}'")
    results = hybrid_search(test_q, top_k_dense=15, top_k_sparse=15)
    print(f"\nTop {len(results)} Hybrid Candidates:")
    for idx, r in enumerate(results[:5], start=1):
        print(f"[{idx}] RRF Score: {r['score']:.6f} | Chunk ID: {r['chunk_id']} | Dense Rank: {r['hybrid_info']['rank_dense']} | Sparse Rank: {r['hybrid_info']['rank_sparse']}")
        print(f"    Content: {r['content'][:120]}...\n")
