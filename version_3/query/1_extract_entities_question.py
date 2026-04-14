import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Đóng vai trò: Bạn là một hệ thống Trích xuất Thực thể (Entity Extractor) SIÊU TRỪU TƯỢNG chuyên dụng cho cơ sở dữ liệu đồ thị (GraphRAG).

Nhiệm vụ: Đọc câu hỏi của người dùng, phân tích ngầm theo cấu trúc Triplet (S-V-O), sau đó trích xuất các thành phần đã được TRỪU TƯỢNG HÓA TỘT ĐỘ vào một danh sách phẳng.

Quy tắc bắt buộc (Strict Rules):
1. Trừu tượng hóa Thực thể (Entity Abstraction): Gom nhóm tất cả chủ thể và đối tượng về các siêu lớp (Superclasses) cơ bản nhất.
   - Ví dụ: dùng "Người", "Tổ chức", "Cơ quan", "Tài sản", "Thủ tục", "Hợp đồng", "Quyền lợi", "Giấy phép".
   - TUYỆT ĐỐI KHÔNG dùng các từ chỉ định cụ thể hay diễn giải dài dòng (Cấm dùng: "người lao động", "hợp đồng mua bán ô tô", "cổng dịch vụ hành chính công", "xe máy").
2. Động từ nguyên thể (Root Verbs Only): Bắt buộc đưa mọi hành động về gốc. Tuyệt đối loại bỏ mọi trợ động từ, từ chỉ trạng thái, ý định, hoặc thời gian (bỏ các từ: muốn, sẽ, đang, bị, được, đã).
   - Ví dụ: "muốn đăng ký" -> "đăng ký"; "làm thủ tục" -> "thực hiện"; "đến trụ sở" -> "đến"; "nộp lệ phí" -> "nộp".
3. Xác định Ẩn số (Query Target): Phần câu hỏi hoặc mục đích tìm kiếm của người dùng phải được biến thành một thực thể đích chứa biến ?.
   - Ví dụ: "?Cách thức", "?Mức phạt", "?Trách nhiệm", "?Thủ tục".
4. Loại bỏ nhiễu: Bỏ qua hoàn toàn các yếu tố không tạo nên cấu trúc logic của sự việc (thời gian, địa điểm, cảm xúc, lời chào hỏi, lý do cá nhân).

Định dạng đầu ra:
Chỉ xuất ra một file JSON chứa mảng "entities" phẳng. Các từ trong mảng phải là chữ thường, không lặp lại. Không giải thích thêm bất cứ điều gì.

Ví dụ Output JSON:
{
    "entities": [
        "người",
        "cơ quan",
        "hợp đồng",
        "tài sản",
        "thủ tục",
        "thực hiện",
        "nộp",
        "đăng ký",
        "?mức phạt"
    ]
}
"""

def load_question(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return ""

    # Tuong thich nguoc voi dinh dang JSON cu, nhung uu tien file text thuần.
    if content.startswith("{"):
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return str(data.get("question", "")).strip()
        except json.JSONDecodeError:
            pass

    return content

def extract_entities(question_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={ "type": "json_object" },
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]
    )
    return json.loads(response.choices[0].message.content)

def normalize_to_output_format(raw_result):
    """
    Chuyển đổi mảng entities thành định dạng [{ "id": "T01", "name": "..." }]
    """
    raw_entities = raw_result.get("entities", [])
    seen = set()
    final_output = []
    counter = 1

    for name in raw_entities:
        name = str(name).strip().lower()
        if not name or name in seen:
            continue

        seen.add(name)
        final_output.append({
            "id": f"T{counter:02d}",
            "name": name
        })
        counter += 1

    return final_output

def main():
    input_file = "./version_3/query/1_question.txt"
    output_file = "./version_3/query/2_entities_question.json"

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    question = load_question(input_file)
    if not question:
        print("Lỗi: File câu hỏi rỗng hoặc không hợp lệ.")
        return

    print("🔍 Đang phân tích câu hỏi và trích xuất siêu trừu tượng...")

    try:
        raw_result = extract_entities(question)
        final_entities = normalize_to_output_format(raw_result)
    except Exception as e:
        print(f"❌ Lỗi khi gọi OpenAI hoặc chuẩn hóa dữ liệu: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_entities, f, ensure_ascii=False, indent=4)

    print("🎉 Trích xuất thành công! Dữ liệu đã đạt chuẩn siêu trừu tượng.")
    print(f"Lưu kết quả tại: {output_file}")

    # In ra terminal để xem luôn kết quả
    print("\n--- KẾT QUẢ ĐÃ TRỪU TƯỢNG HÓA ---")
    for entity in final_entities:
        print(f"[{entity['id']}] {entity['name']}")

if __name__ == "__main__":
    main()
