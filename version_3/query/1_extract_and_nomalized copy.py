import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Đóng vai trò: Bạn là hệ thống Tiền xử lý Câu hỏi (Query Pre-processor) cho GraphRAG pháp luật.
Nhiệm vụ: Tối ưu hóa câu hỏi bằng cách nhân bản các Triplet ngữ nghĩa để tăng tỷ lệ khớp (hit-rate) với đồ thị tri thức (KG).

!!! DANH MỤC SIÊU LỚP BẮT BUỘC CHO NODE (ONTOLOGY) !!!
Chủ thể (s) và Đối tượng (o) BẮT BUỘC phải chọn từ danh sách sau:
[Cơ quan, Tổ chức, Người, Phương tiện, Tài sản, Thủ tục, Hành vi, Hình phạt, Biện pháp, Giấy phép, Quyền lợi, Quy định, Luật, Thời gian, Địa điểm, Mức tiền, Trống]

!!! CHIẾN LƯỢC NHÂN BẢN TRIPLET NGỮ NGHĨA (MỚI - QUAN TRỌNG) !!!

1. ĐA DẠNG HÓA CHỦ THỂ (SUBJECT EXPANSION):
   - Nếu câu hỏi chứa chức danh cụ thể (VD: Chủ tịch xã, Chiến sĩ CAND), bạn phải sinh ra đồng thời:
     + Triplet dùng Siêu lớp: (Cơ quan)
     + Triplet dùng Thuật ngữ pháp lý: (cá nhân có thẩm quyền), (người có thẩm quyền).

2. ĐA DẠNG HÓA ĐỐI TƯỢNG (OBJECT EXPANSION):
   - Nếu câu hỏi chứa vật phẩm cụ thể (VD: hàng hóa, vũ khí, xe máy), bạn phải sinh ra đồng thời:
     + Triplet dùng Siêu lớp: (Tài sản), (Phương tiện) hoặc (Hành vi).
     + Triplet dùng Thuật ngữ luật định: (tang vật), (phương tiện vi phạm).
   - VD: "Chủ tịch xã tịch thu vũ khí"
     -> (Cơ quan) - [tịch thu] -> (Tài sản)
     -> (Cơ quan) - [tịch thu] -> (tang vật)

3. CHUẨN HÓA ĐỘNG TỪ (VERB HARMONIZATION):
   - Sử dụng động từ hạt nhân, không chia thì, không modal verbs (có quyền, được phép).
   - Ưu tiên: xử phạt, tịch thu, tạm giữ, cưỡng chế, khiếu nại.

4. KHÔI PHỤC CHIỀU CHỦ ĐỘNG:
   - Luôn đặt thực thể thực hiện hành động làm (s).

QUY TRÌNH TƯ DUY:
- Bước 1: Xác định hành động cốt lõi và các thực thể (S, O).
- Bước 2: Tìm các thực thể tương đương trong danh mục Ontology và trong ngôn ngữ pháp lý của Luật Xử lý vi phạm hành chính.
- Bước 3: Tạo tập hợp Triplet đa tầng để bao phủ mọi khả năng tìm kiếm trên đồ thị.

ĐỊNH DẠNG JSON ĐẦU RA:
{
  "normalized_text": "văn bản chuẩn hóa tóm tắt tình huống",
  "target": "mục tiêu truy vấn",
  "triplets": [
    { "s": "Cơ quan", "v": "tịch thu", "o": "Tài sản" },
    { "s": "Cơ quan", "v": "tịch thu", "o": "tang vật" },
    { "s": "người có thẩm quyền", "v": "tịch thu", "o": "tang vật" }
  ]
}
"""

def load_question(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def process_graph_extraction(question_text):
    try:
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
    except Exception as e:
        print(f"❌ Lỗi API: {e}")
        return {"normalized_text": "", "target": "không rõ", "triplets": []}

def main():
    # Giữ nguyên cấu trúc đường dẫn file của bạn
    input_file = "./version_3/query/1_question.txt"
    output_text_file = "./version_3/query/1.2_normalized_question.txt"
    output_triplet_file = "./version_3/query/1.2_triplets.json"

    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    question = load_question(input_file)
    if not question: return

    print(f"🔍 Đang phân tích và nhân bản Triplet cho: '{question[:100]}...'")

    result = process_graph_extraction(question)

    # Lưu dữ liệu
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(result.get("normalized_text", ""))

    with open(output_triplet_file, "w", encoding="utf-8") as f:
        json.dump(result.get("triplets", []), f, ensure_ascii=False, indent=4)

    print("\n" + "="*60)
    print("🔗 TRIPLET ĐA TẦNG (MULTILAYER TRIPLETS):")
    for t in result.get("triplets", []):
        print(f"   ({t['s']}) --[{t['v']}]--> ({t['o']})")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()