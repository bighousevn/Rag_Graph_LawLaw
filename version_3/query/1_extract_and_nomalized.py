import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Đóng vai trò: Bạn là một hệ thống Trích xuất Bộ ba Tri thức (Knowledge Triplet Extractor) SIÊU TRỪU TƯỢNG, chuyên xử lý CÂU HỎI của người dùng cho hệ thống GraphRAG.

Mục tiêu: Đưa các tình huống cụ thể của người dùng về CÙNG MỘT MẶT BẰNG TRỪU TƯỢNG với cơ sở dữ liệu đồ thị luật pháp để thuật toán so khớp (Graph Matching) hoạt động chính xác 100%.

!!! CẢNH BÁO TỐI THƯỢNG VỀ TRỪU TƯỢNG HÓA !!!
Nhiệm vụ quan trọng nhất của bạn là TRỪU TƯỢNG HÓA TỘT ĐỘ các thành phần Subject (s), Verb (v), Object (o).
- TUYỆT ĐỐI KHÔNG dùng từ ngữ cụ thể của người dùng (VD: "ông Nguyễn Văn A", "công an phường", "sổ đỏ", "xe máy", "Luật đất đai") để làm "s" hoặc "o".
- Chủ thể (s) và Đối tượng (o) CHỈ ĐƯỢC PHÉP LÀ MỘT KHÁI NIỆM SIÊU LỚP (Superclass) đại diện chung nhất (VD: cơ quan, tổ chức, người, tài sản, thủ tục, hành vi, hình phạt, giấy phép, quyền lợi...).
- BỎ QUA MỌI TỪ BỔ SUNG NGỮ NGHĨA VÀ TỪ PHỦ ĐỊNH: Chỉ giữ lại danh từ/động từ lõi. Tuyệt đối không kèm theo tính từ, trạng từ, từ phủ định, hoặc phương hướng. (VD: "địa điểm khác" -> "địa điểm"; "không vi phạm" -> "vi phạm"; "mức phạt nặng" -> "mức phạt"; "cấp lại" -> "cấp").
- Toàn bộ các giá trị s, v, o phải được viết thường (lowercase) và KHÔNG chứa dấu chấm hỏi (?).

QUY TRÌNH TƯ DUY BẮT BUỘC (Thực hiện tuần tự 2 bước):

--- BƯỚC 1: TRỪU TƯỢNG HÓA & LÀM SẠCH NGỮ NGHĨA (Tạo normalized_text) ---
1. Áp dụng Siêu lớp: Đọc từng thực thể, tự hỏi nó thuộc nhóm chung nào trong luật? Thay thế tên riêng bằng siêu lớp (VD: "UBND Quận" -> "cơ quan", "văn hóa phẩm" -> "tài sản", "M" -> "người").
2. Lọc nhiễu: Bỏ qua mọi yếu tố thời gian, địa điểm, cảm xúc.
3. Xác định Ẩn số (Target): Tự hỏi người dùng muốn tìm kiếm điều gì? (VD: tìm "mức phạt", "tính hợp pháp"). Đưa ẩn số này thành một đối tượng siêu lớp.
4. Diễn đạt lại: Viết lại thành một đoạn văn ngắn gọn, logic chỉ chứa các siêu lớp, kết thúc bằng câu hỏi tìm ẩn số.

--- BƯỚC 2: TRÍCH XUẤT TRIPLET TỪ CÂU ĐÃ CHUẨN HÓA (S-V-O) ---
Từ chính đoạn "normalized_text" vừa tạo ở Bước 1, tiến hành tách Triplet:
1. Động từ gốc và Danh từ gốc: Lấy các cụm từ hành động/thực thể, gọt sạch các đuôi bổ ngữ ("thuộc về" -> "thuộc", "cấp lại" -> "cấp", "không đồng ý" -> "đồng ý"). Điền vào "s", "v", "o".
2. Khai thác triệt để: Chẻ nhỏ văn bản thành các đơn vị ý nghĩa nhỏ nhất. Một câu dài phải tách thành nhiều mối quan hệ.
3. Lắp ghép: (s) - [v] -> (o).

--- ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC (Không giải thích thêm) ---
{
  "normalized_text": "cơ quan phát hiện người thực hiện hành vi liên quan tài sản. cơ quan lập văn bản và ban hành quyết định áp dụng biện pháp tiền và tiêu hủy tài sản. xác định tính hợp pháp của quyết định.",
  "target": "tính hợp pháp",
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

    print("🔍 Đang tiến hành [BƯỚC 1] Chuẩn hóa và [BƯỚC 2] Trích xuất Triplet...")

    try:
        result = process_graph_extraction(question)
        normalized_text = result.get("normalized_text", "")
        triplets = result.get("triplets", [])
        target = result.get("target", "")
    except Exception as e:
        print(f"❌ Lỗi khi gọi API hoặc parse JSON: {e}")
        return

    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(normalized_text)
    print(f"✅ Đã lưu văn bản chuẩn hóa tại: {output_text_file}")

    with open(output_triplet_file, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu triplets tại: {output_triplet_file}")

    print("\n" + "="*50)
    print(f"🎯 ẨN SỐ CẦN TÌM (TARGET): {target.upper()}")
    print("-" * 50)
    print("📜 [BƯỚC 1] VĂN BẢN ĐÃ CHUẨN HÓA VỀ SIÊU LỚP:")
    print(normalized_text)
    print("-" * 50)
    print("🔗 [BƯỚC 2] TRIPLETS S-V-O TÁCH TỪ VĂN BẢN TRÊN:")
    for t in triplets:
        print(f"({t.get('s', '')}) - [{t.get('v', '')}] -> ({t.get('o', '')})")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()