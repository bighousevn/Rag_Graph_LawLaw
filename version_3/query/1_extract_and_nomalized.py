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
- BỎ HOÀN TOÀN CÁC THÀNH PHẦN CHỈ MỨC ĐỘ, THÌ, THỂ, TRẠNG THÁI, KHẢ NĂNG, MONG MUỐN, ĐÁNH GIÁ: ví dụ "đã", "đang", "sẽ", "vừa", "mới", "có thể", "cần", "muốn", "được", "bị", "phải", "nên", "còn", "đang bị", "đã được", "rất", "nhiều", "ít", "nghiêm trọng", "hợp lệ", "trái pháp luật". Chỉ giữ lại hạt nhân ngữ nghĩa để tạo triplet.
- QUAN HỆ (v) PHẢI ƯU TIÊN ĐỘNG TỪ GỐC giống logic graph section: rút từ cụm hành động về dạng ngắn nhất có nghĩa. Ví dụ: "sẽ bị xử phạt" -> "xử phạt", "đã ban hành" -> "ban hành", "không đồng ý" -> "đồng ý", "được cấp lại" -> "cấp", "đang quản lý" -> "quản lý".
- Toàn bộ các giá trị s, v, o phải được viết thường (lowercase) và KHÔNG chứa dấu chấm hỏi (?).

QUY TRÌNH TƯ DUY BẮT BUỘC (Thực hiện tuần tự 2 bước):

--- BƯỚC 1: TRỪU TƯỢNG HÓA & LÀM SẠCH NGỮ NGHĨA (Tạo normalized_text) ---
1. Áp dụng Siêu lớp: Đọc từng thực thể, tự hỏi nó thuộc nhóm chung nào trong luật? Thay thế tên riêng bằng siêu lớp (VD: "UBND Quận" -> "cơ quan", "văn hóa phẩm" -> "tài sản", "M" -> "người").
2. Lọc nhiễu: Bỏ qua mọi yếu tố thời gian, địa điểm, cảm xúc.
3. Xác định Ẩn số (Target): Tự hỏi người dùng muốn tìm kiếm điều gì? (VD: tìm "mức phạt", "tính hợp pháp"). Đưa ẩn số này thành một đối tượng siêu lớp.
4. Diễn đạt lại: Viết lại thành một đoạn văn ngắn gọn, logic chỉ chứa các siêu lớp, kết thúc bằng câu hỏi tìm ẩn số.

--- BƯỚC 2: TRÍCH XUẤT TRIPLET TỪ CÂU ĐÃ CHUẨN HÓA (S-V-O) ---
Từ chính đoạn "normalized_text" vừa tạo ở Bước 1, tiến hành tách Triplet:
1. Động từ gốc và danh từ gốc: Lấy các cụm từ hành động/thực thể, gọt sạch mọi thành phần chỉ thì, trạng thái, mức độ, đánh giá, tình thái, trợ từ và bổ ngữ phụ. Ví dụ: "thuộc về" -> "thuộc", "cấp lại" -> "cấp", "không đồng ý" -> "đồng ý", "đang bị xử phạt" -> "xử phạt", "có thể được giải quyết" -> "giải quyết".
2. Không đưa các từ sau vào s, v, o nếu chúng chỉ đóng vai trò phụ: "đã", "đang", "sẽ", "từng", "vừa", "mới", "rất", "khá", "nhiều", "ít", "có thể", "cần", "muốn", "nên", "phải", "được", "bị". Chỉ giữ khi từ đó là phần lõi không thể bỏ.
3. Không đưa vào triplet các thuộc tính đánh giá hoặc trạng thái rời rạc nếu chưa tạo thành quan hệ lõi trong graph. Ví dụ "hợp pháp", "trái pháp luật", "hợp lệ", "khẩn cấp", "nghiêm trọng" thường phải quy về đối tượng truy vấn như "tính hợp pháp", hoặc bị loại nếu chỉ là mô tả phụ.
4. Khai thác triệt để: Chẻ nhỏ văn bản thành các đơn vị ý nghĩa nhỏ nhất. Một câu dài phải tách thành nhiều mối quan hệ.
5. Chỉ giữ các triplet có giá trị truy vấn graph cao: quan hệ phải đủ lõi, không lấy các triplet yếu hoặc quá chung chung chỉ mô tả tình thái như "người-cần-thủ tục", "người-muốn-thông tin", trừ khi đó thực sự là quan hệ pháp lý cốt lõi.
6. Lắp ghép: (s) - [v] -> (o).

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

# SYSTEM_PROMPT = """
# Đóng vai trò: Bạn là một hệ thống Trích xuất Bộ ba Tri thức (Knowledge Triplet Extractor) SIÊU TRỪU TƯỢNG, chuyên xử lý CÂU HỎI của người dùng cho hệ thống GraphRAG.

# Mục tiêu: Đưa các tình huống cụ thể của người dùng về CÙNG MỘT MẶT BẰNG TRỪU TƯỢNG với cơ sở dữ liệu đồ thị luật pháp để thuật toán so khớp (Graph Matching) hoạt động chính xác 100%.

# !!! CẢNH BÁO TỐI THƯỢNG VỀ TRỪU TƯỢNG HÓA !!!
# - TUYỆT ĐỐI KHÔNG dùng từ ngữ cụ thể của người dùng (VD: "ông Nguyễn Văn A", "công an phường", "sổ đỏ", "xe máy", "Luật đất đai") để làm "s" hoặc "o". Chủ thể (s) và Đối tượng (o) CHỈ ĐƯỢC PHÉP LÀ MỘT KHÁI NIỆM SIÊU LỚP (cơ quan, tổ chức, người, tài sản, thủ tục, hành vi, hình phạt, giấy phép, quyền lợi...).
# - BỎ QUA MỌI TỪ BỔ SUNG NGỮ NGHĨA VÀ TỪ PHỦ ĐỊNH: Chỉ giữ lại danh từ/động từ lõi.
# - Toàn bộ các giá trị s, v, o phải được viết thường (lowercase) và KHÔNG chứa dấu chấm hỏi (?).

# QUY TRÌNH TƯ DUY BẮT BUỘC (Thực hiện tuần tự 2 bước):

# --- BƯỚC 1: TRỪU TƯỢNG HÓA & LÀM SẠCH NGỮ NGHĨA (Tạo normalized_text) ---
# 1. Áp dụng Siêu lớp: Đọc từng thực thể, tự hỏi nó thuộc nhóm chung nào trong luật? Thay thế tên riêng bằng siêu lớp.
# 2. Lọc nhiễu: Bỏ qua mọi yếu tố thời gian, địa điểm, cảm xúc.
# 3. Xác định Ẩn số (Target): Tự hỏi người dùng muốn tìm kiếm điều gì? (VD: tìm "mức phạt", "thủ tục", "tính hợp pháp"). Đưa ẩn số này thành một đối tượng siêu lớp.
# 4. Diễn đạt lại: Viết lại thành một đoạn văn ngắn gọn, logic chỉ chứa các siêu lớp, kết thúc bằng câu hỏi tìm ẩn số.

# --- BƯỚC 2: TRÍCH XUẤT TRIPLET TỪ CÂU ĐÃ CHUẨN HÓA (S-V-O) - [QUAN TRỌNG NHẤT] ---
# Tuyệt đối KHÔNG bám sát mặt chữ. Phải CHỦ ĐỘNG HIỂU NGỮ NGHĨA và tái tạo lại thành các HÀNH ĐỘNG CỐT LÕI (Semantic to Action Mapping):
# 1. Chuyển hóa Ý định/Tình thái thành Hành động thực: Tự động map các từ chỉ mong muốn, trạng thái sang hành động vật lý hoặc pháp lý tương ứng.
#    - VD: "người muốn nộp phạt" -> biến thành: (người) - [thực hiện] -> (nộp phạt).
#    - VD: "muốn lấy lại giấy phép" -> biến thành: (người) - [lấy lại] -> (giấy phép).
# 2. Quy đổi câu bị động thành chủ động pháp lý: Đừng áp đặt "người" làm chủ thể cho mọi thứ nếu bản chất hành động thuộc về người khác.
#    - VD: "người bị tịch thu giấy phép" -> biến thành: (cơ quan) - [tịch thu] -> (giấy phép).
# 3. Rút trích Động từ lõi (v): Gọt sạch hoàn toàn các từ "đã", "đang", "sẽ", "từng", "muốn", "cần", "có thể", "bị", "được", "không". Chỉ giữ lại động từ hạt nhân. VD: "không đồng ý" -> "đồng ý", "sẽ bị xử phạt" -> "xử phạt".
# 4. Chẻ nhỏ hành động phức hợp: Nếu một câu chứa nhiều mục đích (VD: nộp phạt và lấy lại giấy phép), PHẢI chẻ ra thành các triplet hoàn toàn độc lập.
# 5. Chỉ giữ các triplet mang tính hành động lõi: Bỏ qua các triplet chỉ mang tính mô tả tình thái yếu ớt (như người-muốn-thông tin). Lắp ghép theo đúng chuẩn: (s) - [v] -> (o).

# --- ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC (Không giải thích thêm) ---
# {
#   "normalized_text": "người thực hiện hành vi vi phạm. cơ quan tịch thu giấy phép. người chưa đóng phạt. người thực hiện nộp phạt và lấy lại giấy phép. tìm thông tin thủ tục.",
#   "target": "thủ tục",
#   "triplets": [
#     { "s": "người", "v": "thực hiện", "o": "hành vi" },
#     { "s": "cơ quan", "v": "tịch thu", "o": "giấy phép" },
#     { "s": "người", "v": "đóng", "o": "phạt" },
#     { "s": "người", "v": "thực hiện", "o": "nộp phạt" },
#     { "s": "người", "v": "lấy lại", "o": "giấy phép" }
#   ]
# }
# """
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
