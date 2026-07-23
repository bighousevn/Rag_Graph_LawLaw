"""
Bước 4: Dựa vào các điều khoản đã tìm được (output/query_result.json) + câu hỏi GỐC của người
dùng (input/1_question.txt), gọi GPT-4o trả lời — BẮT BUỘC có căn cứ trích dẫn, TUYỆT ĐỐI không
tự đoán/bổ sung nội dung luật ngoài các điều khoản đã cung cấp.

Input:  input/1_question.txt      câu hỏi GỐC (chưa chuẩn hoá) — trả lời đúng ý người dùng hỏi,
                                   không phải bản đã viết lại kiểu luật ở Bước 1
        output/query_result.json  kết quả truy vấn, lấy final.sections (path/document_name/
                                   text_content — văn bản GỐC của điều khoản, không phải rewritten)
Output: output/4_answer.txt       câu trả lời cuối, có trích dẫn Điều/Khoản/Điểm
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
QUESTION_FILE = os.path.join(BASE_DIR, "input", "1_question.txt")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
RESULT_FILE   = os.path.join(OUTPUT_DIR, "query_result.json")
OUT_ANSWER    = os.path.join(OUTPUT_DIR, "4_answer.txt")

MODEL = "gpt-4o"

SYSTEM_PROMPT = """\
Bạn là trợ lý pháp lý giao thông đường bộ Việt Nam. Nhiệm vụ: trả lời câu hỏi của người dùng CHỈ
dựa vào các điều khoản pháp luật được cung cấp trong tin nhắn — đây là kết quả tìm kiếm từ 1 hệ
thống truy vấn đồ thị tri thức, KHÔNG chắc chắn đã đầy đủ hoặc hoàn toàn đúng.

QUY TẮC BẮT BUỘC — TUYỆT ĐỐI TUÂN THỦ:
1. CHỈ dùng thông tin có trong các điều khoản được cung cấp. TUYỆT ĐỐI KHÔNG dùng kiến thức pháp
   luật khác ngoài các điều khoản đó, KHÔNG tự suy đoán/bổ sung nội dung luật không có trong đoạn
   trích — kể cả khi bạn nghĩ mình biết đáp án đúng.
2. Mọi khẳng định trong câu trả lời PHẢI trích dẫn nguồn ngay sau nó, dạng "(Điều X, Khoản Y,
   Điểm Z - <tên văn bản>)", lấy đúng từ trường "path" của điều khoản tương ứng.
3. Câu hỏi thường có NHIỀU khía cạnh khác nhau (vd: hành vi gì, tính số lần vi phạm ra sao, mức
   phạt bao nhiêu, ai xử phạt...). Hãy trả lời ĐẦY ĐỦ MỌI khía cạnh MÀ điều khoản cung cấp CÓ căn
   cứ — LỖI CẦN TRÁNH NHẤT là từ chối trả lời TOÀN BỘ câu hỏi chỉ vì 1 khía cạnh nào đó không có
   căn cứ. Ví dụ: câu hỏi vừa hỏi hành vi có vi phạm không, vừa hỏi cách tính số lần vi phạm khi
   bị ghi nhận ở nhiều nơi trong 1 ngày — nếu điều khoản chỉ trả lời được phần hành vi/mức phạt mà
   không nói gì về cách tính số lần, vẫn PHẢI trả lời trọn vẹn phần hành vi/mức phạt đó.
4. Với khía cạnh KHÔNG có căn cứ trong điều khoản cung cấp, nói ngắn gọn "các điều khoản không đề
   cập tới việc [khía cạnh cụ thể]" — rồi tiếp tục trả lời trọn vẹn các khía cạnh khác có căn cứ.
   TUYỆT ĐỐI KHÔNG bịa nội dung cho riêng khía cạnh thiếu căn cứ đó.
5. Chỉ khi TOÀN BỘ câu hỏi không có bất kỳ khía cạnh nào được điều khoản đề cập tới, mới trả lời
   kiểu "các điều khoản tìm được không đủ căn cứ để trả lời câu hỏi này".
6. Không tự suy đoán mức phạt/chế tài cụ thể nếu điều khoản cung cấp không nêu rõ con số.
7. Nếu nhiều điều khoản cùng liên quan nhưng khác nhau ở mức độ/ngưỡng, liệt kê rõ từng trường
   hợp, không gộp chung thành 1 câu trả lời mơ hồ.
8. Văn phong ngắn gọn, rõ ràng, dễ hiểu cho người không chuyên luật — không dùng thuật ngữ hành
   chính rườm rà không cần thiết."""


def log(msg: str) -> None:
    print(msg, flush=True)


def build_user_message(question: str, sections: list[dict]) -> str:
    if not sections:
        return f"Câu hỏi: {question}\n\nKHÔNG tìm được điều khoản nào liên quan trong hệ thống."

    parts = [f"Câu hỏi: {question}", "", "Các điều khoản tìm được:"]
    for i, sec in enumerate(sections, 1):
        parts.append(f"\n[{i}] {sec['path']} - {sec['document_name']}")
        parts.append(sec["text_content"])
    return "\n".join(parts)


def main():
    with open(QUESTION_FILE, encoding="utf-8") as f:
        question = f.read().strip()
    if not question:
        log(f"File câu hỏi rỗng: {QUESTION_FILE}")
        return

    with open(RESULT_FILE, encoding="utf-8") as f:
        result = json.load(f)
    sections = result.get("final", {}).get("sections", [])
    log(f"Câu hỏi: {question}")
    log(f"Số điều khoản làm căn cứ: {len(sections)}")

    user_message = build_user_message(question, sections)

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUT_ANSWER, "w", encoding="utf-8") as f:
        f.write(answer)

    log(f"\n=== CÂU TRẢ LỜI ===\n{answer}")
    log(f"\nĐã lưu → {OUT_ANSWER}")


if __name__ == "__main__":
    main()
