import py_vncorenlp
import os
import json
import time
import re
from collections import Counter

# 1. Đường dẫn cấu hình
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(base_dir, '../../vncorenlp_model'))
input_path = os.path.join(base_dir, 'material_for_triplets/sections_nghi_dinh_168_2024_1.json')
output_path = os.path.join(base_dir, 'material_for_triplets/keyphrases_nghi_dinh_168_2024_1.json')

print(f"Loading VnCoreNLP from: {model_dir}")
# Dùng annotator tách từ 'wseg' và gán nhãn 'pos' để trích xuất danh từ/thuật ngữ
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"], save_dir=model_dir)

# Danh sách từ loại danh từ hoặc thuật ngữ viết tắt
ALLOWED_POS = {"N", "Np", "Ny"}

# Danh sách stopwords tiếng Việt chung và các danh từ chỉ đơn vị thời gian/đơn vị đếm chung chung
STOPWORDS = {
    "của", "các", "những", "tại", "đối_với", "theo", "và", "hoặc", "này", "về",
    "thì", "là", "mà", "có", "từ", "cho", "với", "để", "trong", "ngoài", "do",
    "sự", "việc", "nơi", "bên", "ngày", "tháng", "năm", "khoản", "điều", "điểm",
    "bộ", "luật", "nghị_định", "pháp_luật", "quy_định", "trường_hợp", "sau_đây",
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "đáp_ứng",
    "trực_tiếp", "thực_hiện", "tiến_hành", "tổ_chức", "cá_nhân", "cơ_quan", "đơn_vị"
}

def clean_structural_text(text):
    """Xóa các ký tự cấu trúc ở đầu câu"""
    text = re.sub(r'^(Điều \d+\s*\.|[a-z]\s*\)|\d+\s*\.)\s*', '', text)
    return text

if not os.path.exists(input_path):
    print(f"❌ Không tìm thấy file input: {input_path}")
    exit(1)

print(f"Reading input file: {input_path}")
with open(input_path, 'r', encoding='utf-8') as f:
    sections = json.load(f)

total_sections = len(sections)
print(f"Loaded {total_sections} sections. Bắt đầu trích xuất các keyphrases...")

keyphrase_counter = Counter()

start_time = time.time()
for i, section in enumerate(sections, 1):
    original_text = section.get("text_content", "")
    if original_text.strip():
        try:
            # Làm sạch ký tự tiêu đề cấu trúc trước khi parse
            cleaned_text = clean_structural_text(original_text)

            # Phân tích cú pháp với VnCoreNLP
            annotated_doc = model.annotate_text(cleaned_text)

            # annotated_doc trả về dictionary, key là index câu, value là list các token
            for sentence_idx, sentence in annotated_doc.items():
                for token in sentence:
                    word = token['wordForm']
                    pos = token['posTag']
                    word_lower = word.lower()

                    # Trích xuất danh từ, danh từ riêng hoặc viết tắt làm keyphrase
                    if pos in ALLOWED_POS:
                        # Bỏ qua từ quá ngắn hoặc nằm trong Stopwords
                        if len(word_lower.replace("_", "")) < 2 or word_lower in STOPWORDS:
                            continue

                        # Chuẩn hóa về chữ thường để gom nhóm tần suất chính xác
                        keyphrase_counter[word] += 1

        except Exception as e:
            print(f"❌ Lỗi xử lý tại section ID {section.get('id', 'unknown')}: {e}")

    if i % 100 == 0 or i == total_sections:
        elapsed = time.time() - start_time
        print(f"-> Đã quét {i}/{total_sections} sections ({elapsed:.1f}s)...")

# Sắp xếp danh sách keyphrases theo tần suất từ cao xuống thấp
sorted_keyphrases = [
    {
        "keyphrase": item[0].replace("_", " "),
        "snake_case": item[0],
        "count": item[1]
    }
    for item in keyphrase_counter.most_common()
]

# Lưu kết quả
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sorted_keyphrases, f, ensure_ascii=False, indent=4)

print(f"\n✅ Hoàn thành! Đã trích xuất được {len(sorted_keyphrases)} keyphrases khác nhau.")
print(f"Đã lưu kết quả tại: {output_path}")

# In ra 30 keyphrases phổ biến nhất làm ví dụ
print("\n--- TOP 30 KEYPHRASES PHỔ BIẾN NHẤT ---")
for idx, item in enumerate(sorted_keyphrases[:30], 1):
    print(f"{idx:<2}. {item['keyphrase']:<35} (Tần suất: {item['count']})")
