Được. Dưới đây là bản `README.md` tiếng Việt, viết lại theo đúng pipeline bạn đang có.

```markdown
# Vietnamese PDF RAG Pipeline

Hệ thống xử lý dữ liệu RAG (Retrieval-Augmented Generation) cho tài liệu PDF tiếng Việt.

Pipeline hiện tại tập trung vào:

- Đọc và trích xuất nội dung từ PDF
- Chuẩn hóa dữ liệu
- Chia nhỏ tài liệu thành các đoạn (chunk)
- Tạo vector embedding
- Lưu trữ trong Vector Database
- Tìm kiếm ngữ nghĩa (Semantic Retrieval)

Phiên bản hiện tại hoàn thành phần **Data Pipeline + Retrieval Pipeline**.

Chưa bao gồm bước sinh câu trả lời bằng LLM.

---

# 1. Kiến trúc tổng quan

```

PDF gốc

```
|
v
```

PDF Loader
(pdf_loader.py)

```
|
v
```

pdf_extract.jsonl

```
|
v
```

Chunking
(chunker_optimized.py)

```
|
v
```

chunked.jsonl

```
|
v
```

Embedding Model
(BAAI/bge-m3)

```
|
v
```

Vector Database
(ChromaDB)

```
|
v
```

Retriever
(retriever.py)

```
|
v
```

Các đoạn tài liệu liên quan

```

---

# 2. Cấu trúc thư mục

```

C:\RAG

│
├── data/
│   │
│   ├── raw/
│   │   └── [Reading]-RAG-System.pdf
│   │
│   ├── processed/
│   │   ├── pdf_extract.jsonl
│   │   └── chunked.jsonl
│   │
│   └── vectorstore/
│       └── chroma/
│
│
├── src/
│   │
│   ├── data/
│   │   │
│   │   ├── pdf_loader.py
│   │   └── chunker_optimized.py
│   │
│   │
│   └── rag/
│       │
│       ├── indexer.py
│       └── retriever.py
│
│
├── requirements.txt
└── README.md

````

---

# 3. Chuẩn bị môi trường

## Yêu cầu

- Python >= 3.10
- CUDA (tùy chọn, dùng GPU để tăng tốc embedding)

Tạo môi trường ảo:

```bash
python -m venv .venv
````

Kích hoạt:

Windows:

```bash
.venv\Scripts\activate
```

---

# 4. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Các thư viện chính:

```
docling
langchain
langchain-text-splitters
sentence-transformers
langchain-huggingface
chromadb
langchain-chroma
tiktoken
```

---

# 5. Pipeline xử lý dữ liệu

## Bước 1: Trích xuất PDF

### File:

```
src/data/pdf_loader.py
```

### Chức năng:

Chuyển đổi tài liệu PDF thành JSONL có cấu trúc.

Input:

```
data/raw/[Reading]-RAG-System.pdf
```

Output:

```
data/processed/pdf_extract.jsonl
```

Ví dụ:

```json
{
"id":"Reading_page_5",
"content":"Một hệ thống RAG hoàn chỉnh...",
"metadata":{
    "source":"[Reading]-RAG-System.pdf",
    "page":5,
    "title":"Reading-RAG-System"
}
}
```

Chạy:

```bash
python src/data/pdf_loader.py \
--input data/raw/[Reading]-RAG-System.pdf
```

---

# 6. Chunking tài liệu

## File:

```
src/data/chunker_optimized.py
```

## Chức năng:

Chia nhỏ tài liệu thành các đoạn phù hợp cho việc tìm kiếm.

Tính năng:

* Token-aware chunking
* Hỗ trợ tiếng Việt
* Giữ metadata
* Theo dõi thống kê chunk
* Kiểm soát kích thước chunk

Input:

```
data/processed/pdf_extract.jsonl
```

Output:

```
data/processed/chunked.jsonl
```

Chạy:

```bash
python src/data/chunker_optimized.py \
--input data/processed/pdf_extract.jsonl \
--output data/processed/chunked.jsonl \
--strategy token \
--target-tokens 600 \
--token-overlap 100
```

Ví dụ output:

```json
{
"chunk_id":"page_5_chunk_1",

"chunk_content":
"[DOCUMENT CONTEXT]

Title: Reading RAG System

Page:5

Một hệ thống RAG gồm ba giai đoạn..."
,
"metrics":{
    "token_count":520
},

"metadata":{
    "page":5
}
}
```

---

# 7. Tạo Vector Database

## File:

```
src/rag/indexer.py
```

## Chức năng:

Chuyển đổi chunk thành vector embedding và lưu vào ChromaDB.

Embedding model:

```
BAAI/bge-m3
```

Input:

```
data/processed/chunked.jsonl
```

Output:

```
data/vectorstore/chroma/
```

Chạy:

```bash
python src/rag/indexer.py
```

Ví dụ kết quả:

```
Documents loaded: 177

INDEX COMPLETE

Vectors stored: 177
```

---

# 8. Retrieval - Tìm kiếm tài liệu

## File:

```
src/rag/retriever.py
```

## Chức năng:

Tìm các đoạn tài liệu có ý nghĩa gần nhất với câu hỏi người dùng.

Luồng hoạt động:

```
Câu hỏi người dùng

        |
        v

Embedding câu hỏi

        |
        v

Vector Similarity Search

        |
        v

Top-K đoạn tài liệu phù hợp
```

Ví dụ:

```bash
python src/rag/retriever.py \
--query "RAG hiện đại gồm những giai đoạn nào?"
```

Kết quả:

```
Page: 42

Một pipeline RAG hoàn chỉnh vận hành dựa trên
3 giai đoạn chính:

1. Indexing
2. Retrieval
3. Generation
```

---

# 9. Kết quả hiện tại

Tài liệu thử nghiệm:

```
[Reading]-RAG-System.pdf
```

Kết quả pipeline:

| Thành phần      | Kết quả     |
| --------------- | ----------- |
| PDF xử lý       | Thành công  |
| Số document     | 116         |
| Số chunk        | 177         |
| Embedding model | BAAI/bge-m3 |
| Vector Database | ChromaDB    |
| Retrieval       | Thành công  |

---

# 10. Quyết định thiết kế

## Vì sao sử dụng Docling?

Docling được sử dụng để:

* Đọc PDF
* Trích xuất nội dung
* Giữ cấu trúc tài liệu
* Hỗ trợ bảng và layout

## Vì sao sử dụng BGE-M3?

Lý do:

* Hỗ trợ đa ngôn ngữ
* Phù hợp tiếng Việt
* Có thể chạy local

## Vì sao sử dụng ChromaDB?

Lý do:

* Dễ triển khai
* Phù hợp thử nghiệm RAG
* Tích hợp tốt với LangChain

---

# 11. Giới hạn hiện tại

Đã triển khai:

✅ PDF ingestion
✅ Text extraction
✅ Chunking
✅ Embedding
✅ Vector Database
✅ Semantic Retrieval

Chưa triển khai:

❌ Xử lý hình ảnh trong PDF
❌ Vision model
❌ Reranker
❌ LLM sinh câu trả lời

---

# 12. Hướng phát triển tiếp theo

## Multimodal RAG

Bổ sung:

```
Hình ảnh trong PDF

        |
        v

Vision Model

        |
        v

Caption

        |
        v

Embedding
```

## Cải thiện Retrieval

Có thể thêm:

* Cross Encoder Reranker
* BM25 Hybrid Search
* Metadata filtering

## Thêm lớp sinh câu trả lời

Kiến trúc hoàn chỉnh:

```
Retriever

    +

LLM Generator

    =

RAG Chatbot
```

---

# 13. Trạng thái hệ thống

Hiện tại:

```
PDF

↓

Loader

↓

JSONL

↓

Chunking

↓

Embedding

↓

Vector Database

↓

Retriever
```

Trạng thái:

✅ Hoàn thành Data Pipeline
✅ Hoàn thành Indexing Pipeline
✅ Hoàn thành Retrieval Pipeline

Sẵn sàng tích hợp lớp LLM Generation.

```

Bạn có thể lưu trực tiếp nội dung này thành:

```

C:\RAG\README.md

```

Bản này phản ánh đúng trạng thái hiện tại của project, không ghi dư phần multimodal hay chatbot vì bạn đã quyết định dừng ở Retrieval.
```
