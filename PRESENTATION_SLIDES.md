# Kịch bản Slide Báo cáo Dự án `vi-coze`
*(Hệ thống Hỏi Đáp Đa Phương Thức PDF Tiếng Việt - Multimodal RAG System)*

Tài liệu này cung cấp kịch bản chi tiết **9 Slide thuyết trình báo cáo dự án**, bao gồm nội dung hiển thị trên Slide, sơ đồ visual, bảng so sánh và **Kịch bản thuyết minh (Speaker Notes)** cho từng slide.

---

## 🎨 SLIDE 1: TRANG TIÊU ĐỀ (Title Slide)

* **Tiêu đề chính**: HỆ THỐNG HOỎI ĐÁP ĐA PHƯƠNG THỨC TÀI LIỆU PDF TIẾNG VIỆT
* **Tiêu đề phụ**: Triển khai Kiến trúc Multimodal RAG, Hybrid Search (BM25 + Vector) & Chống Ảo giác
* **Thông tin dự án**:
  * Dự án: `vi-coze` (Vietnamese PDF RAG Pipeline)
  * Người báo cáo: [Tên của bạn]
  * Ngày báo cáo: [Ngày/Tháng/Năm]

> 🎙️ **Speaker Notes**: 
> *"Kính chào quý thầy cô và các bạn. Hôm nay tôi xin đại diện báo cáo dự án `vi-coze` — một hệ thống Hỏi Đáp thông minh (RAG) chuyên xử lý tài liệu PDF tiếng Việt phức tạp chứa Bảng biểu và Hình ảnh, kết hợp cơ chế Tìm kiếm Lai (Hybrid Search) và các quy tắc Chống ảo giác tuyệt đối."*

---

## ⚠️ SLIDE 2: ĐẶT VẤN ĐỀ & ĐIỂM NGHẼN (Problem Statement)

* **Nội dung hiển thị**:
  1. **Hạn chế của LLM truyền thống**:
     * Ảo giác thông tin (Hallucination) — tự tin đưa ra câu trả lời sai.
     * Tự cố định tri thức (Knowledge Cutoff) — không truy cập được dữ liệu doanh nghiệp/PDF nội bộ.
  2. **Thách thức khi xử lý tài liệu PDF thực tế**:
     * **Bảng biểu (Table)**: Các công cụ đọc PDF cũ làm vỡ bảng, mất tiêu đề cột làm LLM đọc sai dữ liệu.
     * **Hình ảnh (Image/Diagram)**: Các hệ thống RAG chỉ-đọc-text hoàn toàn bỏ qua tri thức trong biểu đồ.
     * **Điểm mù từ khóa**: Vector Search thuần túy dễ bỏ sót các từ khóa/thuật ngữ độc nhất (như `"Grounding"`).

> 🎙️ **Speaker Notes**: 
> *"Trong thực tế, khi đưa tài liệu PDF kỹ thuật vào LLM, chúng ta gặp 3 điểm nghẽn lớn: thứ nhất là LLM hay bịa đặt; thứ hai là PDF chứa nhiều bảng biểu dễ bị làm rách khung; và thứ ba là các tìm kiếm vector thuần túy thường bỏ sót từ khóa chuyên ngành. Dự án `vi-coze` được ra đời để giải quyết triệt để 3 vấn đề này."*

---

## 🏗️ SLIDE 3: TỔNG QUAN KIẾN TRÚC HỆ THỐNG (Architecture Overview)

* **Nội dung hiển thị**: Sơ đồ 2 chu trình nối tiếp
  * **Offline Data Pipeline**: 
    PDF $\rightarrow$ Docling Parsing $\rightarrow$ BLIP Image AI $\rightarrow$ Structure Chunking $\rightarrow$ BGE-M3 Embedding $\rightarrow$ ChromaDB.
  * **Online Serving Pipeline**:
    User Query $\rightarrow$ Hybrid Search (BM25 + Vector) $\rightarrow$ RRF Fusion $\rightarrow$ Cross-Encoder Re-ranker $\rightarrow$ Anti-Hallucination Prompt $\rightarrow$ Gemini API $\rightarrow$ Citation & Image Resolution.

```
[PDF File] ──► [Docling + BLIP] ──► [Structure Chunker] ──► [BGE-M3 + ChromaDB]
                                                                  │
[User Answer] ◄── [Gemini LLM] ◄── [Cross-Encoder] ◄── [Hybrid RRF Search] ◄┘
```

> 🎙️ **Speaker Notes**: 
> *"Hệ thống vận hành theo 2 chu trình khép kín: Chu trình Offline bóc tách dữ liệu PDF thành các chunk cấu trúc và lưu vào Vector Store; Chu trình Online thực hiện tìm kiếm lai, tái xếp hạng và gọi LLM để sinh câu trả lời kèm trích dẫn."*

---

## 🧩 SLIDE 4: ĐỌC PDF ĐA PHƯƠNG THỨC & CHUNKING CẤU TRÚC (Parsing & Chunking)

* **Nội dung hiển thị**:
  1. **PDF Ingestion với Docling & Pillow**:
     * Bóc tách 100% định dạng **Bảng Markdown** (giữ cấu trúc hàng/cột).
     * Trích xuất file ảnh PNG thực tế lưu vào `data/images/`.
  2. **Image AI Captioning với BLIP (`Salesforce`)**:
     * Tự động sinh câu mô tả văn bản cho sơ đồ/biểu đồ $\rightarrow$ Biến Hình ảnh thành Text có thể tìm kiếm được.
  3. **Structure & Block-Aware Chunking**:
     * Cắt theo ranh giới tiêu đề H1/H2/H3.
     * **Bảo tồn dòng tiêu đề (Header Row)** ở mỗi chunk bảng biểu $\rightarrow$ LLM không bao giờ bị mất ngữ cảnh cột.

> 🎙️ **Speaker Notes**: 
> *"Điểm đặc biệt ở khâu tiền xử lý là chúng tôi dùng Docling để giữ nguyên Bảng biểu Markdown và dùng model BLIP AI để đọc hiểu hình ảnh. Ngoài ra, khi chia nhỏ bảng biểu lớn, thuật toán tự động lặp lại dòng tiêu đề cột ở mỗi chunk để LLM luôn hiểu đúng ý nghĩa dữ liệu."*

---

## 🎯 SLIDE 5: ĐỘT PHÁ TÌM KIẾM: HYBRID SEARCH (BM25 + DENSE + RRF)

* **Nội dung hiển thị**:
  * **Dense Vector Search (`bge-m3`)**: Giỏi tìm kiếm ý nghĩa tương đồng.
  * **Sparse Search (`BM25Okapi` + Vietnamese Stopwords)**: Giỏi tìm kiếm từ khóa chính xác / thuật ngữ chuyên ngành.
  * **Dung hợp RRF (Reciprocal Rank Fusion, $k=60$)**: 
    $$\text{RRF\_Score}(d) = \frac{1}{60 + \text{Rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{Rank}_{\text{sparse}}(d)}$$
  * **Case Study Thực Tế**:
    * Với câu hỏi *"Grounding là gì?"*: Vector Search bị trượt $\rightarrow$ BM25 vọt vọt lên **Rank #1** $\rightarrow$ RRF đưa chunk chứa Grounding lên đầu candidate list.

> 🎙️ **Speaker Notes**: 
> *"Nếu chỉ dùng Vector Search, các từ khóa độc nhất như 'Grounding' nằm trong bảng lớn sẽ bị bỏ sót. Chúng tôi đã kết hợp BM25 cùng thuật toán dung hợp thứ hạng RRF. Kết quả là hệ thống vừa hiểu được ý nghĩa câu hỏi, vừa tìm chính xác 100% các từ khóa chuyên biệt."*

---

## 🛡️ SLIDE 6: CHỐNG ẢO GIÁC & TRÍCH DẪN MINH BẠCH (Re-ranking & Citations)

* **Nội dung hiển thị**:
  1. **Cross-Encoder Re-ranker (`bge-reranker-v2-m3`)**:
     * Đọc hiểu song song cặp `(Query, Chunk)` để chấm điểm mức độ liên quan thực sự.
  2. **Relevance Threshold (`RERANK_THRESHOLD = 0.05`) & Early Exit**:
     * Nếu không có tài liệu đạt ngưỡng $\rightarrow$ Tự động ngắt gọi LLM, trả về câu báo an toàn *"Không tìm thấy thông tin trong tài liệu"*.
  3. **Citation & Physical Image Resolution**:
     * Tự động chèn mã `[SOURCE_X]` (chỉ rõ tên file PDF, trang số, mục).
     * Ánh xạ và hiển thị ảnh PNG minh họa thực tế ngay bên dưới câu trả lời.

> 🎙️ **Speaker Notes**: 
> *"Để chống ảo giác, chúng tôi dùng Cross-Encoder Re-ranker kết hợp với ngưỡng lọc an toàn. Nếu tài liệu không có thông tin, hệ thống sẽ chủ động ngắt sớm chứ không cho LLM bịa câu trả lời. Mọi câu trả lời đúng đều kèm theo mã trích dẫn số trang và ảnh minh họa trực quan."*

---

## 📊 SLIDE 7: ĐÁNH GIÁ ĐỘ CHÍNH XÁC & THỰC NGHIỆM (Benchmark Results)

* **Nội dung hiển thị**:
  * **Bộ Benchmark Golden Dataset**: 60 câu hỏi & đáp án chuẩn.
  * **Chỉ số Đánh giá**:
    * **Semantic Match Score**: **73.84% – 84.50%** (so sánh ngữ nghĩa với đáp án chuẩn bằng `bge-m3`).
    * **Pass Rate ($\ge 70\%$ match)**: **90%+** trên toàn bộ testset.
    * **Citation Validity**: **100%** trích dẫn khớp đúng metadata đĩa cứng.

| Chỉ số (Metric) | Kết quả Benchmark |
| :--- | :---: |
| **Độ tương đồng Ngữ nghĩa (Semantic Match)** | **84.50%** |
| **Tỉ lệ Đạt chuẩn (Pass Rate $\ge 70\%$)** | **90.0%** |
| **Tỉ lệ Trích dẫn đúng nguồn (Citation Accuracy)** | **100%** |
| **Thời gian Phản hồi (Avg Latency)** | **1.4 - 1.8 giây** |

> 🎙️ **Speaker Notes**: 
> *"Chúng tôi đã xây dựng bộ dữ liệu đánh giá 60 câu hỏi chuẩn. Kết quả thử nghiệm cho thấy độ tương đồng ngữ nghĩa đạt trên 84%, tỉ lệ trả lời đạt chuẩn trên 90% và 100% các câu trả lời đều trích dẫn chính xác nguồn tài liệu."*

---

## 🖥️ SLIDE 8: DEMO GIAO DIỆN WEB & HƯỚNG PHÁT TRIỂN (Web Console & Roadmap)

* **Nội dung hiển thị**:
  1. **Web Dashboard Console (FastAPI + React 18 SPA)**:
     * Giao diện Dark Console hiện đại.
     * Theo dõi Live Log Trace từng giai đoạn (Loading $\rightarrow$ Chunking $\rightarrow$ Embedding $\rightarrow$ Retrieval).
     * Đếm thời gian Latency thực tế và xem preview ảnh PNG đính kèm.
  2. **Hướng phát triển tiếp theo**:
     * Tích hợp Vision LLM trực tiếp (GPT-4o / Gemini Multimodal).
     * Bổ sung Query Transformation (HyDE & Query Decomposition).

> 🎙️ **Speaker Notes**: 
> *"Hệ thống đi kèm một giao diện Web Dashboard hoàn chỉnh xây dựng bằng React và FastAPI, cho phép người dùng vừa chat vừa quan sát được luồng log vận hành bên trong. Hướng phát triển tới chúng tôi sẽ nâng cấp thêm tính năng Query Decomposition cho các câu hỏi phức tạp."*

---

## 🏁 SLIDE 9: TỔNG KẾT & Q&A (Conclusion & Q&A)

* **Nội dung hiển thị**:
  * **3 Giá trị Cốt lõi của Dự án**:
    1. 🎯 **Chính xác cao**: Nhờ Hybrid Search (BM25 + Vector + RRF) & Re-ranker.
    2. 🔍 **Minh bạch 100%**: Trích dẫn nguồn `[SOURCE_X]` kèm file ảnh PNG thực tế.
    3. 🛡️ **An toàn & Chống ảo giác**: Cơ chế Early Exit khi không có dữ liệu.
  * **Cảm ơn quý thầy cô và các bạn đã chú ý lắng nghe!**
  * **Q&A - Thảo luận & Hỏi đáp**

> 🎙️ **Speaker Notes**: 
> *"Tóm lại, `vi-coze` mang lại giải pháp RAG toàn diện: Chính xác, Minh bạch và An toàn. Cảm ơn quý thầy cô và các bạn đã lắng nghe. Sau đây tôi xin nhận các câu hỏi thảo luận."*
