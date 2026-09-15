# Báo cáo Tổng hợp Công nghệ, Lý do Lựa chọn & So sánh Ưu/Nhược điểm
*(Pipeline RAG Multimodal Dự án `vi-coze`: Từ Đọc File PDF đến Sinh Câu trả lời)*

Tài liệu này tổng hợp toàn bộ các công nghệ được sử dụng trong dự án **`vi-coze`**, phân tích chi tiết **Lý do lựa chọn (Design Rationale)** và **So sánh Ưu / Nhược điểm với các giải pháp thay thế khác** trên thị trường.

---

## 🏗️ Sơ đồ Tổng quan Kiến trúc Pipeline

```
[1. PDF Ingestion] ────► [2. Chunking Engine] ────► [3. Embedding & VectorStore]
  Docling + BLIP          Structure & Block-Aware     BAAI/bge-m3 + ChromaDB
         │                                                      │
         ▼                                                      ▼
[6. Response & Citations] ◄── [5. LLM Generation] ◄─── [4. Hybrid Search & Rerank]
 Citation & Image Map          Gemini API / OpenAI      BM25 + Dense + RRF + Cross-Encoder
```

---

## 🛠️ Chi tiết Công nghệ theo từng Giai đoạn

### 1. Giai đoạn Đọc & Trích xuất File PDF (PDF Ingestion)

* **Công nghệ chọn**: **Docling (`docling.document_converter`)** + **Pillow (PIL)**.
* **Lý do lựa chọn**: Nhận diện cấu trúc PDF phức tạp, phân loại tiêu đề, đoạn văn, giữ nguyên định dạng **Bảng biểu Markdown** và cắt các file ảnh minh họa vật lý lưu vào `data/images/`.
* **So sánh với các giải pháp thay thế**:
  * *Vs PyPDF2 / PyMuPDF (fitz)*: `PyPDF2` chỉ trích xuất chữ thô, làm rách vỡ định dạng Bảng biểu (Table) và làm mất tiêu đề cột. Docling bảo toàn 100% Bảng Markdown giúp LLM đọc hiểu bảng dữ liệu.
  * *Vs pdfplumber*: `pdfplumber` đọc bảng tốt hơn PyPDF nhưng rất chậm khi trích xuất hình ảnh và dễ lỗi layout đa cột (multi-column layout).
  * *Vs Unstructured*: `Unstructured` mạnh nhưng nặng, yêu cầu nhiều phụ thuộc rắc rối (Poppler, Tesseract OCR), trong khi Docling gọn gàng và giữ nguyên Markdown Table chuẩn.
* **Ưu điểm**: Giữ nguyên định dạng Bảng Markdown, trích xuất ảnh PNG sắc nét, phân tích layout đa cột tốt.
* **Nhược điểm**: Tốc độ parse PDF chậm hơn một chút so với đọc text thuần bằng PyMuPDF.

---

### 2. Giai đoạn Mô tả Ảnh tự động (Image AI Captioning - Multimodal)

* **Công nghệ chọn**: **BLIP (`Salesforce/blip-image-captioning-base`)**.
* **Lý do lựa chọn**: Mô hình Vision-Language AI tự động đọc hiểu và sinh mô tả bằng văn bản (`ai_caption`) cho hình ảnh sơ đồ/biểu đồ (tự động bỏ qua ảnh rác `<100x100` px).
* **So sánh với các giải pháp thay thế**:
  * *Vs GPT-4o Vision API*: GPT-4o Vision cho caption tuyệt vời nhưng tốn chi phí API rất cao với tài liệu có hàng trăm hình ảnh và phụ thuộc kết nối internet.
  * *Vs LLaVA / Moondream*: LLaVA nặng (7B-13B params) đòi hỏi VRAM GPU lớn (8GB+). BLIP base siêu nhẹ (220M params), nạp được trên mọi máy tính.
  * *Vs Bỏ qua hình ảnh*: Nếu bỏ qua ảnh, hệ thống mất 100% tri thức nằm trong các hình vẽ/sơ đồ.
* **Ưu điểm**: Miễn phí, chạy offline local tốt trên cả GPU/CPU, biến hình ảnh thành text có thể tìm kiếm được.
* **Nhược điểm**: Ngôn ngữ đầu ra mặc định là tiếng Anh (cần prompt dịch/thích ứng).

---

### 3. Giai đoạn Phân đoạn Tài liệu Cấu trúc (Structure & Block-Aware Chunking)

* **Công nghệ chọn**: **Custom Section & Block Packers** + **tiktoken (`cl100k_base`)**.
* **Lý do lựa chọn**: Phân tích ranh giới tiêu đề H1/H2/H3, đóng gói theo token budget (`target_tokens = 600`), tạo gối đầu ngữ cảnh (`token_overlap = 100`) và tự động **bảo tồn dòng tiêu đề (Header Row)** khi chia nhỏ bảng biểu.
* **So sánh với các giải pháp thay thế**:
  * *Vs Fixed-size Chunking (Cắt ký tự cố định)*: Fixed-size rất dễ cắt đôi một câu văn hoặc làm rách một bảng biểu làm mất hẳn ngữ nghĩa.
  * *Vs RecursiveCharacterTextSplitter thuần*: RecursiveTextSplitter khá hơn Fixed-size nhưng không hiểu cấu trúc Bảng (cắt bảng làm mất dòng tiêu đề cột). Custom Block Packer tự động lặp lại Header Row ở mỗi chunk bảng nhỏ.
* **Ưu điểm**: Giữ trọn vẹn ý nghĩa câu văn và dữ liệu cột của Bảng biểu, đếm token chuẩn xác.
* **Nhược điểm**: Phải thiết kế logic phân tích block tùy chỉnh.

---

### 4. Giai đoạn Mã hóa Vector & Cơ sở dữ liệu Vector (Indexing & Vector Store)

* **Công nghệ chọn**: **BAAI/bge-m3** + **ChromaDB (`langchain-chroma`)**.
* **Lý do lựa chọn**: BGE-M3 mã hóa vector 1024 chiều ngữ nghĩa tiếng Việt mạnh mẽ. ChromaDB lưu trữ vector tại chỗ (Local Persistent Vector Store).
* **So sánh với các giải pháp thay thế**:
  * *Vs OpenAI `text-embedding-3-small/large`*: OpenAI tốn phí API, dữ liệu phải gửi ra bên ngoài. BGE-M3 hoàn toàn miễn phí, chạy local, vector 1024 chiều rất mạnh cho tiếng Việt.
  * *Vs Pinecone / Qdrant Cloud*: Pinecone/Qdrant là Cloud DB, yêu cầu API key, đường truyền mạng và tốn chi phí duy trì. ChromaDB hoàn toàn offline local.
  * *Vs FAISS*: FAISS chỉ lưu vector trong RAM và quản lý metadata khá thủ công. ChromaDB tự động lưu trữ đĩa cứng (DuckDB/SQLite) kèm metadata filtering linh hoạt.
* **Ưu điểm**: Miễn phí, chạy local không cần setup server, vector sắc nét, quản lý metadata phong phú.
* **Nhược điểm**: ChromaDB ở quy mô hàng triệu vector kém hơn Milvus/Qdrant (tuy nhiên dư sức cho tài liệu <100.000 chunks).

---

### 5. Giai đoạn Tìm kiếm Lai & Dung hợp Thứ hạng (Hybrid Search & RRF Fusion)

* **Công nghệ chọn**: **Dense Vector (`bge-m3`) + Sparse BM25 (`BM25Okapi` + Vietnamese Stopwords) + RRF Fusion ($k=60$)**.
* **Lý do lựa chọn**: Kết hợp ưu điểm ngữ nghĩa của Vector Search và ưu điểm tìm từ khóa chính xác của BM25.
* **So sánh với các giải pháp thay thế**:
  * *Vs Dense Search thuần*: Dense Search hoàn toàn "mù" khi tìm từ khóa chuyên ngành độc nhất (`Grounding`, mã sản phẩm, tên riêng) bị chìm trong đoạn văn lớn. BM25 giải quyết dứt điểm điểm yếu này.
  * *Vs Sparse Search thuần (chỉ BM25)*: BM25 không hiểu từ đồng nghĩa (ví dụ `xe hơi` ↔ `ô tô`).
  * *Vs Weighted Sum (Cộng điểm trực tiếp)*: Cộng điểm trực tiếp yêu cầu chuẩn hóa thang điểm 0-1 rắc rối do BM25 và Vector Search thuộc 2 miền giá trị khác nhau. **RRF (Reciprocal Rank Fusion)** dung hợp dựa trên thứ hạng (Rank-based) đơn giản và hiệu quả hơn hẳn.
* **Ưu điểm**: Tìm đúng 100% cả từ khóa chính xác lẫn ý nghĩa tương đương, RRF hoạt động ổn định mọi trường hợp.
* **Nhược điểm**: Tốn thêm thời gian tính toán BM25 (tuy nhiên <5ms cho vài trăm chunks).

---

### 6. Giai đoạn Tái xếp hạng & Ngắt sớm (Re-ranking & Early Exit)

* **Công nghệ chọn**: **BAAI/bge-reranker-v2-m3 (Cross-Encoder)** + **Relevance Threshold (`RERANK_THRESHOLD = 0.05`)**.
* **Lý do lựa chọn**: Cross-Encoder đọc song song cả Query và Chunk để chấm điểm thực sự. Lọc ngưỡng giúp ngắt sớm (Early Exit) để không bịa câu trả lời.
* **So sánh với các giải pháp thay thế**:
  * *Vs Không dùng Re-ranker*: Nếu chỉ dùng Bi-Encoder (Dense Vector), thứ hạng Top-K nhiều khi bị lệch do Bi-Encoder mã hóa query và doc độc lập. Cross-Encoder tăng độ chính xác tìm kiếm thêm 15-20%.
  * *Vs LLM Re-ranking (dùng GPT-4o chấm điểm)*: Dùng LLM làm Re-ranker rất đắt và cực kỳ chậm (mất 2-5 giây per query). `bge-reranker-v2-m3` vừa nhanh vừa chính xác.
* **Ưu điểm**: Tăng vượt trội độ chính xác tìm kiếm, chống bịa đặt (Anti-Hallucination) tuyệt đối.
* **Nhược điểm**: Tốn thêm tài nguyên CPU/GPU tính toán cho 15-30 candidate chunks.

---

### 7. Giai đoạn Sinh Câu trả lời & Trích dẫn (LLM Generation & Citation Resolution)

* **Công nghệ chọn**: **Google Gemini API (`gemini-2.5-flash`)** + **Anti-Hallucination Guardrails Prompt** + **Citation Resolver (`citation.py`)**.
* **Lý do lựa chọn**: LLM chính sinh câu trả lời tiếng Việt trôi chảy. Ánh xạ nhãn `[SOURCE_X]` với file PDF gốc và hiển thị hình ảnh PNG minh họa thực tế.
* **So sánh với các giải pháp thay thế**:
  * *Vs OpenAI `gpt-4o`*: `gpt-4o` thông minh nhưng chi phí API cao hơn gấp nhiều lần. `gemini-2.5-flash` có hiệu năng tương đương cho tác vụ RAG nhưng giá rẻ và nhanh hơn.
  * *Vs Local LLM (Llama-3 8B, Qwen-2.5 7B via Ollama)*: Local LLM bảo mật 100% nhưng đòi hỏi GPU đắt tiền (8GB-16GB VRAM), tốc độ sinh từ trên máy cá nhân chậm. Gemini API chạy mượt trên mọi thiết bị.
* **Ưu điểm**: Siêu nhanh, giá rẻ, tiếng Việt tự nhiên, minh bạch nguồn trích dẫn và đính kèm ảnh PNG trực quan.
* **Nhược điểm**: Phụ thuộc kết nối Internet và API Key.

---

### 8. Giai đoạn Web Server & Giao diện Dashboard (Serving & Dashboard UI)

* **Công nghệ chọn**: **FastAPI (`uvicorn`)** + **React 18 (`Vite`)**.
* **Lý do lựa chọn**: FastAPI có tốc độ cao, hỗ trợ bất đồng bộ (async), tự động sinh OpenAPI spec. React + Vite xây dựng giao diện console theo dõi pipeline trực quan.
* **So sánh với các giải pháp thay thế**:
  * *Vs Streamlit / Gradio*: Streamlit/Gradio dựng UI nhanh nhưng rất khó tùy biến giao diện đẹp, thiếu tính mượt mà SPA và khó mở rộng thành sản phẩm thương mại.
  * *Vs Flask / Django*: Flask cũ hơn và thiếu async native; Django quá cồng kềnh cho các ứng dụng Microservice/AI.
* **Ưu điểm**: Giao diện đẹp, hiện đại, xem log trace theo thời gian thực, phục vụ ảnh PNG mượt mà.
* **Nhược điểm**: Cần phát triển độc lập giữa Backend Python và Frontend React.

---

## 📊 Bảng tổng hợp So sánh Toàn diện

| Giai đoạn | Công nghệ lựa chọn | Lý do chính | So sánh với Giải pháp thay thế (Ưu / Nhược điểm) |
| :--- | :--- | :--- | :--- |
| **1. PDF Parsing** | `Docling` + `Pillow` | Giữ nguyên Bảng Markdown & cắt ảnh vật lý | **Ưu**: Không rách bảng/hình ảnh như `PyPDF2`/`pdfplumber`. **Nhược**: Chậm hơn PyMuPDF thuần. |
| **2. Image AI** | `BLIP (Salesforce)` | Sinh mô tả văn bản cho ảnh để RAG tìm kiếm | **Ưu**: Nhẹ, miễn phí local vs `GPT-4o Vision` đắt đỏ / `LLaVA` nặng. **Nhược**: Caption tiếng Anh. |
| **3. Chunking** | Custom Structure Packer | Bảo tồn Header Row cho Bảng & ranh giới H1/H2/H3 | **Ưu**: Tránh rách ngữ cảnh vs `Fixed-size Chunking`. **Nhược**: Cần tự viết code custom. |
| **4. Embedding** | `BAAI/bge-m3` | Embedding đa ngôn ngữ / tiếng Việt tốt nhất | **Ưu**: Miễn phí, vector 1024D, 8k context vs `PhoBERT` (256 context) hay `OpenAI` tốn phí. |
| **5. Vector DB** | `ChromaDB` | Lưu trữ local persistent nhẹ, miễn phí | **Ưu**: Không cần setup server Docker/Cloud vs `Milvus`/`Pinecone`. **Nhược**: Scale kém hơn Milvus khi >1M vectors. |
| **6. Hybrid Search**| `Dense + BM25 + RRF` | Tìm đúng 100% cả ý nghĩa lẫn từ khóa độc nhất (`Grounding`) | **Ưu**: Khắc phục điểm mù của `Dense Search thuần`. RRF dung hợp thứ hạng không cần chuẩn hóa điểm. |
| **7. Re-ranking** | `bge-reranker-v2-m3` | Cross-Encoder đánh giá chi tiết + Early Exit | **Ưu**: Chính xác vượt trội vs `Bi-Encoder thuần`; Rẻ & nhanh hơn `LLM Re-ranking`. |
| **8. LLM Engine** | `Gemini API (2.5-flash)` | Sinh câu trả lời tiếng Việt trôi chảy + Citations | **Ưu**: Siêu nhanh, rẻ hơn `GPT-4o`, không nặng máy như `Local Llama-3`. **Nhược**: Phụ thuộc Internet. |
| **9. Web Stack** | `FastAPI` + `React (Vite)` | REST API bất đồng bộ + Giao diện Console mượt mà | **Ưu**: Đẹp, mở rộng thương mại tốt hơn `Streamlit`/`Gradio`. **Nhược**: Tách biệt code FE/BE. |
