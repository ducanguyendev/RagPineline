from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.data.pdf_loader import convert_pdfs
from src.data.image_captioner import caption_elements
from src.data.chunker_optimized import ChunkConfig, process_chunking
from src.rag.device_utils import (
    get_device_name,
    is_cuda_available,
    is_xpu_available,
    resolve_device,
)
from src.rag.indexer import build_index
from src.rag.retriever import load_vector_db
from src.rag.rag_pipeline import answer_query
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = ROOT / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PDF_EXTRACT = PROCESSED_DIR / "pdf_extract.jsonl"
ELEMENTS_FILE = PROCESSED_DIR / "elements.jsonl"
CHUNKED = PROCESSED_DIR / "chunked.jsonl"
VECTOR_DIR = ROOT / "data" / "vectorstore" / "chroma"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "rag_documents"


def accelerator_status() -> dict[str, Any]:
    requested_device = os.getenv("INDEX_DEVICE", "auto")
    try:
        index_device = resolve_device(requested_device)
        device_name = get_device_name(index_device)
        device_error = None
    except Exception as exc:
        index_device = resolve_device("auto")
        device_name = get_device_name(index_device)
        device_error = f"{type(exc).__name__}: {exc}"

    status = {
        "xpu_available": is_xpu_available(),
        "cuda_available": is_cuda_available(),
        "index_device": index_device,
        "device_name": device_name,
        # Keep this key for the current frontend status badge.
        "device": index_device,
    }
    if device_error:
        status["index_device_error"] = device_error
    return status


app = FastAPI(
    title="Vietnamese PDF RAG Pipeline Demo",
    version="1.0.0",
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadRequest(BaseModel):
    file_name: str


class ChunkRequest(BaseModel):
    strategy: str = "structure_block_aware"
    target_tokens: int = Field(default=600, ge=100, le=4000)
    token_overlap: int = Field(default=100, ge=0, le=1000)


class EmbedRequest(BaseModel):
    reset_db: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_chunk_preview(limit: int = 3) -> list[dict[str, Any]]:
    if not CHUNKED.exists():
        return []
    result = []
    with CHUNKED.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            result.append(
                {
                    "chunk_id": data.get("chunk_id"),
                    "content": data.get("chunk_content", "")[:500],
                    "metadata": data.get("metadata", {}),
                    "metrics": data.get("metrics", {}),
                }
            )
            if len(result) >= limit:
                break
    return result


def vector_count() -> int:
    if not VECTOR_DIR.exists():
        return 0
    try:
        db = load_vector_db(VECTOR_DIR, EMBEDDING_MODEL)
        return int(db._collection.count())
    except Exception:
        return 0


@app.get("/health")
def health():
    return {
        "status": "ok",
        **accelerator_status(),
    }


@app.get("/api/files")
def list_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [
            {
                "name": p.name,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            }
            for p in RAW_DIR.glob("*.pdf")
            if p.is_file()
        ],
        key=lambda x: x["name"].lower(),
    )

    return {
        "directory": str(RAW_DIR),
        "files": files,
    }


@app.get("/api/pipeline/status")
def pipeline_status():
    accelerator = accelerator_status()
    return {
        "pdf_documents": count_jsonl(PDF_EXTRACT),
        "elements": count_jsonl(ELEMENTS_FILE),
        "chunks": count_jsonl(CHUNKED),
        "vectors": vector_count(),
        "embedding_model": EMBEDDING_MODEL,
        **accelerator,
        "paths": {
            "raw": str(RAW_DIR),
            "elements": str(ELEMENTS_FILE),
            "pdf_extract": str(PDF_EXTRACT),
            "chunked": str(CHUNKED),
            "vector_db": str(VECTOR_DIR),
        },
    }


@app.post("/api/pipeline/load")
def run_load(request: LoadRequest):
    pdf_path = RAW_DIR / Path(request.file_name).name

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy file: {pdf_path}",
        )

    start = time.perf_counter()

    try:
        convert_pdfs(pdf_path, PDF_EXTRACT, ELEMENTS_FILE)
        try:
            caption_elements(ELEMENTS_FILE)
        except Exception as cap_err:
            print(f"! Captioner warning: {cap_err}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Loader lỗi: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000
    elements_cnt = count_jsonl(ELEMENTS_FILE)

    return {
        "stage": "loading",
        "status": "success",
        "message": f"Đã load {request.file_name} ({elements_cnt} elements & BLIP captions)",
        "duration_ms": round(duration_ms, 2),
        "details": {
            "elements": elements_cnt,
            "documents": count_jsonl(PDF_EXTRACT),
            "output": str(ELEMENTS_FILE),
        },
    }


@app.post("/api/pipeline/chunk")
def run_chunk(request: ChunkRequest):
    input_file = ELEMENTS_FILE if ELEMENTS_FILE.exists() else PDF_EXTRACT

    if not input_file.exists():
        raise HTTPException(
            status_code=400,
            detail="Chưa có elements.jsonl hoặc pdf_extract.jsonl. Hãy chạy Loading trước.",
        )

    config = ChunkConfig(
        strategy=request.strategy,
        target_tokens=request.target_tokens,
        token_overlap=request.token_overlap,
    )

    start = time.perf_counter()

    try:
        process_chunking(input_file, CHUNKED, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunking lỗi: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000
    chunks = count_jsonl(CHUNKED)

    return {
        "stage": "chunking",
        "status": "success",
        "message": f"Đã tạo {chunks} chunks (Structure & Block-Aware)",
        "duration_ms": round(duration_ms, 2),
        "details": {
            "chunks": chunks,
            "strategy": request.strategy,
            "target_tokens": request.target_tokens,
            "token_overlap": request.token_overlap,
            "output": str(CHUNKED),
            "preview": read_chunk_preview(),
        },
    }


@app.post("/api/pipeline/embed")
def run_embed(request: EmbedRequest):
    if not CHUNKED.exists():
        raise HTTPException(
            status_code=400,
            detail="Chưa có chunked.jsonl. Hãy chạy Chunking trước.",
        )

    if request.reset_db and VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)

    start = time.perf_counter()

    try:
        db = build_index(
            input_file=CHUNKED,
            persist_dir=VECTOR_DIR,
            model_name=EMBEDDING_MODEL,
        )
        vectors = int(db._collection.count())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding/Indexing lỗi: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000

    return {
        "stage": "embedding",
        "status": "success",
        "message": f"Đã embedding và lưu {vectors} vectors",
        "duration_ms": round(duration_ms, 2),
        "details": {
            "vectors": vectors,
            "model": EMBEDDING_MODEL,
            **accelerator_status(),
            "database": str(VECTOR_DIR),
        },
    }


@app.post("/api/retrieval/search")
def search_top_k(request: SearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query không được để trống.")

    if not VECTOR_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Chưa có Vector Database. Hãy chạy Embedding trước.",
        )

    logs = []
    total_start = time.perf_counter()

    logs.append(
        {
            "stage": "query",
            "status": "success",
            "message": f'Nhận query: "{query}"',
            "duration_ms": 0,
        }
    )

    load_start = time.perf_counter()
    try:
        db = load_vector_db(VECTOR_DIR, EMBEDDING_MODEL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không load được ChromaDB: {exc}") from exc

    logs.append(
        {
            "stage": "vector_db",
            "status": "success",
            "message": "Đã load ChromaDB và embedding model",
            "duration_ms": round((time.perf_counter() - load_start) * 1000, 2),
        }
    )

    search_start = time.perf_counter()

    try:
        from src.rag.hybrid_retriever import hybrid_search
        hybrid_results = hybrid_search(
            query=query,
            top_k_dense=request.top_k,
            top_k_sparse=request.top_k,
            rrf_k=60,
            vector_dir=VECTOR_DIR,
            embedding_model=EMBEDDING_MODEL
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hybrid Retrieval lỗi: {exc}") from exc

    retrieval_ms = (time.perf_counter() - search_start) * 1000

    results = []

    for rank, item in enumerate(hybrid_results[:request.top_k], start=1):
        metadata = item.get("metadata", {})
        display_score = item.get("score", 0.0)

        image_urls = []
        raw_paths = metadata.get("image_paths", "")
        if raw_paths:
            path_list = [p.strip() for p in raw_paths.split(",") if p.strip()]
            for p in path_list:
                clean_p = p.replace("\\", "/")
                if "data/images/" in clean_p:
                    rel = clean_p.split("data/images/")[1]
                    image_urls.append(f"/images/{rel}")
                elif clean_p.startswith("/images/"):
                    image_urls.append(clean_p)

        results.append(
            {
                "rank": rank,
                "score": round(display_score, 6),
                "distance": 0.0,
                "content": item.get("content", ""),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_id": item.get("chunk_id") or metadata.get("chunk_id"),
                "image_urls": image_urls,
                "metadata": metadata,
            }
        )

    logs.append(
        {
            "stage": "retrieval",
            "status": "success",
            "message": f"Đã lấy Top-{len(results)} chunks",
            "duration_ms": round(retrieval_ms, 2),
        }
    )

    total_ms = (time.perf_counter() - total_start) * 1000

    logs.append(
        {
            "stage": "done",
            "status": "success",
            "message": "Hoàn thành Retrieval",
            "duration_ms": round(total_ms, 2),
        }
    )

    return {
        "query": query,
        "top_k": request.top_k,
        "total_vectors": int(db._collection.count()),
        "total_duration_ms": round(total_ms, 2),
        "logs": logs,
        "results": results,
    }


class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi người dùng")
    top_k: int = Field(default=15, description="Số lượng candidate từ Vector Search")
    top_n: int = Field(default=5, description="Số lượng chunk giữ lại sau Re-ranking")
    threshold: float = Field(default=0.35, description="Ngưỡng điểm RERANK_THRESHOLD")
    provider: str = Field(default="gemini", description="LLM Provider: gemini | openai | ollama | mock")


@app.post("/api/rag/chat")
def run_chat(request: ChatRequest):
    try:
        res = answer_query(
            query=request.query,
            top_k=request.top_k,
            top_n=request.top_n,
            threshold=request.threshold,
            provider_name=request.provider
        )
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG Chat lỗi: {exc}") from exc

