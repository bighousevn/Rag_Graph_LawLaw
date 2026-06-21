import py_vncorenlp
import os
import json
import time
import re

# 1. Đường dẫn cấu hình
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(base_dir, '../../vncorenlp_model'))
input_path = os.path.join(base_dir, 'material_for_triplets/sections_nghi_dinh_168_2024_1.json')
output_path = os.path.join(base_dir, 'material_for_triplets/kg_cleaned_sections_nghi_dinh_168_2024_1.json')

print(f"Loading VnCoreNLP from: {model_dir}")
# CẬP NHẬT: Thêm tác vụ 'pos' (gán nhãn từ loại) bên cạnh 'wseg' (tách từ)
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"], save_dir=model_dir)

# 2. Định nghĩa các bộ lọc cho Đồ thị tri thức
# Chỉ giữ lại các từ loại mang ý nghĩa cốt lõi:
# N (Danh từ), Np (Danh từ riêng), V (Động từ), A (Tính từ)
ALLOWED_POS = {"N", "Np", "V", "A"}

# Danh sách từ dừng (Stopwords) pháp lý cơ bản không mang lại giá trị cho Node
STOPWORDS = {"của", "các", "những", "tại", "đối_với", "theo", "và", "hoặc", "này", "về", "thì", "là", "mà", "có", "từ", "cho", "với", "để", "trong", "ngoài", "do"}

# Danh sách các tiền tố ý định/tình thái cần bị loại bỏ để đảm bảo các
# quan hệ (relation/edge) trong đồ thị được chuẩn hóa về động từ gốc.
INTENTIONAL_PREFIXES = {"muốn", "cần", "phải", "được", "bị", "nên", "sẽ", "đang", "đã"}

def clean_structural_text(text):
    """Xóa các thành phần đánh dấu thứ tự cấu trúc ở đầu câu"""
    # Xóa "Điều 1.", "1.", "a)" v.v.
    text = re.sub(r'^(Điều \d+\s*\.|[a-z]\s*\)|\d+\s*\.)\s*', '', text)
    return text

if not os.path.exists(input_path):
    print(f"❌ Không tìm thấy file input: {input_path}")
    exit(1)

print(f"Reading input file: {input_path}")
with open(input_path, 'r', encoding='utf-8') as f:
    sections = json.load(f)

total_sections = len(sections)
print(f"Loaded {total_sections} sections. Bắt đầu phân tách và trích xuất thực thể...")

start_time = time.time()
for i, section in enumerate(sections, 1):
    original_text = section.get("text_content", "")
    if original_text.strip():
        try:
            # Bước 1: Dọn dẹp các ký tự cấu trúc văn bản
            cleaned_text = clean_structural_text(original_text)

            # Bước 2: Phân tích cú pháp với VnCoreNLP
            annotated_doc = model.annotate_text(cleaned_text)

            valid_tokens = []

            # annotated_doc trả về dictionary, key là index câu, value là list các token
            for sentence_idx, sentence in annotated_doc.items():
                for token in sentence:
                    word = token['wordForm']
                    pos = token['posTag']
                    word_lower = word.lower().replace("_", " ")

                    # Bước 3: Áp dụng các màng lọc ngữ nghĩa
                    if pos in ALLOWED_POS:
                        # 3.1 Bỏ qua Stopwords
                        if word_lower in STOPWORDS:
                            continue

                        # 3.2 Loại bỏ tiền tố ý định (để giữ lại root actions)
                        if word_lower in INTENTIONAL_PREFIXES:
                            continue

                        # Giữ lại định dạng ghép từ (có dấu gạch dưới _)
                        valid_tokens.append(word)

            # Bước 4: Cập nhật lại text_content bằng chuỗi các thực thể đã làm sạch
            section["text_content"] = " ".join(valid_tokens)

        except Exception as e:
            print(f"❌ Lỗi xử lý tại section ID {section.get('id', 'unknown')}: {e}")

    if i % 100 == 0 or i == total_sections:
        elapsed = time.time() - start_time
        print(f"-> Đã xử lý {i}/{total_sections} sections ({elapsed:.1f}s)...")

# 3. Lưu kết quả ra file mới
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sections, f, ensure_ascii=False, indent=4)

print(f"✅ Hoàn thành! File kết quả (đã làm sạch) được lưu tại: {output_path}")