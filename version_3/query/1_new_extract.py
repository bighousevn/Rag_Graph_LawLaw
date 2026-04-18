import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Đóng vai trò: Bạn là chuyên gia Hệ thống hóa Ngôn ngữ và Trích xuất Tri thức (Knowledge Extractor) cho hệ thống GraphRAG.

Nhiệm vụ: Thực hiện tuần tự 2 bước trên câu hỏi/tình huống của người dùng và trả về định dạng JSON nghiêm ngặt.

--- BƯỚC 1: TRỪU TƯỢNG HÓA & LÀM SẠCH NGỮ NGHĨA ---
1. Trừu tượng hóa (Abstraction): Thay thế toàn bộ tên riêng, địa danh, thời gian bằng các Siêu lớp (Superclasses):
   - Người/M/A/B... -> Người
   - UBND/Công an/Chủ tịch/Tòa án... -> Cơ quan
   - Văn hóa phẩm/Xe máy/Đất đai... -> Tài sản
   - Biên bản/Thông báo/Quyết định... -> Văn bản / Quyết định
   - Phạt tiền/Tịch thu/Tiêu hủy... -> Biện pháp
2. Lọc nhiễu (Denoising): Xóa bỏ hoàn toàn ngày tháng, các trạng từ chỉ thời gian, cảm xúc.
3. Diễn đạt lại (Reformulation): Kết nối các thực thể thành một chuỗi logic chặt chẽ.
4. Xác định Ẩn số (Query Target): Chuyển câu hỏi về dạng tìm kiếm mục tiêu (ví dụ: tính hợp pháp, mức phạt, trách nhiệm). TUYỆT ĐỐI KHÔNG DÙNG dấu chấm hỏi (?) trước các từ này.

--- BƯỚC 2: TRÍCH XUẤT TRIPLET (S-V-O) - YÊU CẦU ĐỘ SÂU TỐI ĐA ---
Từ đoạn văn bản đã chuẩn hóa, trích xuất CHI TIẾT TẤT CẢ các bộ ba tri thức (Subject - Verb - Object).
- KHAI THÁC TRIỆT ĐỂ (Exhaustive): Chẻ nhỏ văn bản thành các đơn vị ý nghĩa nhỏ nhất. Một câu dài phải được tách thành nhiều mối quan hệ bắc cầu. Không được bỏ sót bất kỳ hành động nhỏ nào (như lập, ghi nhận, gồm, tác động).
- Cấu trúc: "s" (Chủ thể) - "v" (Hành động gốc) - "o" (Đối tượng / Ẩn số).
- Toàn bộ giá trị S, V, O phải được viết thường (lowercase) và KHÔNG chứa dấu chấm hỏi (?).

--- VÍ DỤ CHUẨN ĐẦU RA (JSON BẮT BUỘC) ---
{
  "normalized_text": "cơ quan phát hiện người thực hiện hành vi liên quan tài sản. cơ quan lập văn bản và ban hành quyết định áp dụng biện pháp tiền và tiêu hủy tài sản. xác định tính hợp pháp của quyết định.",
  "triplets": [
    { "s": "cơ quan", "v": "phát hiện", "o": "hành vi" },
    { "s": "cơ quan", "v": "lập", "o": "văn bản" },
    { "s": "văn bản", "v": "ghi nhận", "o": "hành vi" },
    { "s": "người", "v": "thực hiện", "o": "hành vi" },
    { "s": "cơ quan", "v": "ban hành", "o": "quyết định" },
    { "s": "quyết định", "v": "xử phạt", "o": "người" },
    { "s": "quyết định", "v": "áp dụng", "o": "biện pháp" },
    { "s": "biện pháp", "v": "gồm", "o": "tiền" },
    { "s": "biện pháp", "v": "gồm", "o": "tiêu hủy" },
    { "s": "tiêu hủy", "v": "tác động", "o": "tài sản" },
    { "s": "quyết định", "v": "có", "o": "tính hợp pháp" }
  ]
}
"""

def load_question(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def process_graph_extraction(question_text):
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

def main():
    input_file = "./version_3/query/1_question.txt"
    output_text_file = "./version_3/query/1.2_normalized_question.txt"
    output_triplet_file = "./version_3/query/1.2_triplets.json"
    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    question = load_question(input_file)
    if not question:
        print("❌ Lỗi: File câu hỏi rỗng.")
        return

    print("🔍 Đang tiền xử lý ngữ nghĩa và trích xuất Triplet...")

    try:
        result = process_graph_extraction(question)
        normalized_text = result.get("normalized_text", "")
        triplets = result.get("triplets", [])
    except Exception as e:
        print(f"❌ Lỗi khi gọi API hoặc parse JSON: {e}")
        return

    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(normalized_text)
    print(f"✅ Đã lưu văn bản chuẩn hóa tại: {output_text_file}")

    with open(output_triplet_file, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu triplets tại: {output_triplet_file}")

    print("\n" + "="*40)
    print("📜 VĂN BẢN CHUẨN HÓA:")
    print(normalized_text)
    print("-" * 40)
    print("🔗 TRIPLETS ĐƯỢC TÁCH:")
    for t in triplets:
        print(f"({t.get('s', '')}) - [{t.get('v', '')}] -> ({t.get('o', '')})")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()