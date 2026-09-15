"""
Prompt Builder Subsystem

Constructs Grounded System Prompt & User Message with Anti-Hallucination Guardrails.
"""

from typing import Any, Dict, List


SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp phân tích tài liệu kỹ thuật cho hệ thống RAG (vi-coze).
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác, rõ ràng và trung thực bằng tiếng Việt.

QUY TẮC BẮT BUỘC (ANTI-HALLUCINATION GUARDRAILS):
1. CHỈ sử dụng thông tin được cung cấp trong các khối ngữ cảnh [SOURCE_X] bên dưới để trả lời.
2. KHÔNG tự suy đoán, bịa đặt hoặc sử dụng kiến thức bên ngoài không có trong ngữ cảnh.
3. Mỗi khi đưa ra một ý hoặc thông tin, BẮT BUỘC chèn ký hiệu trích dẫn mã [SOURCE_X] ngay phía sau câu hoặc mệnh đề tương ứng (Ví dụ: "RAG hiện đại vận hành theo mô hình Retrieve and Prompt [SOURCE_1].").
4. Nếu ngữ cảnh được cung cấp KHÔNG chứa thông tin để trả lời câu hỏi, bạn phải thành thật trả lời đúng mẫu sau:
   "Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này."
5. Tuyệt đối KHÔNG tự sáng tác thêm mã [SOURCE_X] giả không tồn tại trong danh sách ngữ cảnh.
6. Giữ nguyên tính chính xác của các thuật ngữ chuyên môn, bảng biểu và số liệu kỹ thuật.
"""


def build_messages(query: str, context_text: str) -> List[Dict[str, str]]:
    """
    Tạo danh sách các tin nhắn (System Prompt & User Prompt) chuẩn hóa gửi cho LLM.
    """
    user_content = f"""Dưới đây là các khối tài liệu ngữ cảnh được trích xuất:

========================================
NGỮ CẢNH TÀI LIỆU:
========================================
{context_text}
========================================

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

Hãy trả lời câu hỏi bằng tiếng Việt kèm theo trích dẫn mã [SOURCE_X] chuẩn xác theo đúng quy tắc trên.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content.strip()}
    ]


if __name__ == "__main__":
    msgs = build_messages("RAG gồm những giai đoạn nào?", "[SOURCE_1]\nContent: RAG gồm 3 giai đoạn.")
    print("SYSTEM:", msgs[0]["content"])
    print("USER:", msgs[1]["content"])
