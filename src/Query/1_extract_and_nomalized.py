import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """Đóng vai trò: Bạn là một hệ thống Trích xuất Bộ ba Tri thức (Knowledge Triplet Extractor) chuyên xử lý CÂU HỎI của người dùng cho hệ thống GraphRAG pháp luật.

Nhiệm vụ của bạn:
1. Đọc và hiểu sâu sắc câu hỏi pháp lý của người dùng (thường có văn phong dân dã, tự do, kể chuyện).
2. Quy đổi các danh từ, động từ hội thoại dân dã đó thành các thuật ngữ pháp lý khái quát chung (Generalization) khớp với cơ sở dữ liệu luật đất đai.
3. Trích xuất thành các bộ ba nguyên tử (Atomic Triplets: s - v - o) cực kỳ đơn giản và độc lập.
4. Nếu trong mối quan hệ/bộ ba đó thiếu Chủ thể (s) hoặc Đối tượng (o) (ví dụ: do bị ẩn đi trong câu hỏi hoặc chính là ẩn số/mục tiêu giải quyết đang cần tìm), BẮT BUỘC thay thế thành phần thiếu đó bằng dấu "*".

QUY TẮC KHÁI QUÁT HÓA THÀNH THUẬT NGỮ PHÁP LÝ (Không dùng từ quá cụ thể):
- "nhà em", "nhà hàng xóm", "ông bà", "hàng xóm", "chủ đất", "người dân" -> khái quát thành "người sử dụng đất" hoặc "người" hoặc "cá nhân".
- "sổ", "sổ đỏ", "sổ hồng", "giấy tờ", "giấy tờ đất" -> khái quát thành "giấy chứng nhận" hoặc "giấy chứng nhận quyền sử dụng đất".
- "đất nhà em", "28m đất", "thửa đất", "khoảnh đất" -> khái quát thành "đất" hoặc "diện tích" hoặc "thửa đất".
- "đo sai", "đo thiếu", "đo lệch" -> khái quát thành "đo đạc" hoặc "sai lệch" hoặc "ranh giới".
- "lấn đất", "không chịu trả đất", "cãi nhau về đất" -> khái quát thành "tranh chấp" hoặc "tranh chấp đất đai" hoặc "lấn chiếm".
- "lấy lại đất", "đòi đất", "kiện ra tòa", "cách giải quyết" -> khái quát thành "giải quyết tranh chấp" hoặc "giải quyết tranh chấp đất đai".

QUY TẮC DÙNG DẤU "*" CHO THÀNH PHẦN KHUYẾT HOẶC TRUY VẤN:
- Thay vì dùng các từ nghi vấn hoặc đại từ bất định (ai, cái gì, thế nào, làm sao, cách nào), hãy dùng dấu "*" ở vị trí đó.
- Ví dụ: "Ai giải quyết tranh chấp?" -> (*, giải quyết, tranh chấp)

VÍ DỤ MẪU BẮT BUỘC HỌC THEO:
Câu hỏi: "Đất nhà em bị đo sai 28m đất qua hàng xóm, giờ làm giấy tờ mới phát hiện đất trong sổ bị thiếu so với ranh giới đất thực tế. Nhà hàng xóm không chịu cắt trả thì nhà em mất đất luôn ạ. Có cách nào lấy lại đất không?"
Các bộ ba trích xuất chuẩn xác:
[
  { "s": "người sử dụng đất", "v": "đo đạc", "o": "đất" },
  { "s": "giấy chứng nhận", "v": "ghi nhận", "o": "diện tích" },
  { "s": "diện tích", "v": "thiếu so với", "o": "ranh giới" },
  { "s": "người sử dụng đất", "v": "tranh chấp", "o": "ranh giới" },
  { "s": "*", "v": "giải quyết", "o": "tranh chấp đất đai" }
]

--- ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC (Không giải thích thêm) ---
{
  "normalized_text": "đoạn văn tóm tắt câu hỏi sử dụng các danh từ đã được khái quát hóa.",
  "target": "ẩn số chính cần tìm (ví dụ: giải quyết tranh chấp đất đai)",
  "triplets": [
    { "s": "chủ thể hoặc *", "v": "quan hệ", "o": "đối tượng hoặc *" }
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "input", "1_question.txt")
    output_text_file = os.path.join(base_dir, "output", "1.2_normalized_question.txt")
    output_triplet_file = os.path.join(base_dir, "output", "1.2_triplets.json")

    # Tạo các thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(input_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_text_file), exist_ok=True)

    # Nếu file câu hỏi chưa có, ghi một câu hỏi ví dụ mặc định
    if not os.path.exists(input_file):
        default_question = "Mọi người cho em hỏi : Đất nhà em bị đo sai 28m đất qua hàng xóm, giờ nhà em lên làm giấy tờ mới phát hiện đất trong sổ bị thiếu so với ranh giới đất thực tế ( đất này là đất từ thời ông bà em nên ranh giới đó là đúng ạ). Nhà hàng xóm thì k chịu cắt đất trả thì nhà em mất 28m đất đó luôn ạ. Có cách nào lấy lại đất ko mọi người tư vấn giúp em, em cảm ơn."
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(default_question)
        print(f"📝 Đã tạo file câu hỏi mẫu mặc định tại: {input_file}")

    question = load_question(input_file)
    if not question:
        print("❌ Lỗi: File câu hỏi rỗng.")
        return

    print(f"📖 Câu hỏi đầu vào: '{question}'")
    print("🔍 Đang tiến hành trích xuất Triplets và chuẩn hóa bằng LLM...")

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
    print("📜 VĂN BẢN ĐÃ CHUẨN HÓA VỀ SIÊU LỚP:")
    print(normalized_text)
    print("-" * 50)
    print("🔗 TRIPLETS S-V-O TRÍCH XUẤT ĐƯỢC (Bao gồm các phần tử khuyết/truy vấn '*'):")
    for t in triplets:
        print(f"({t.get('s', '')}) - [{t.get('v', '')}] -> ({t.get('o', '')})")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
