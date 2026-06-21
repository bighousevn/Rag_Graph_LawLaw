import py_vncorenlp
import os
import json
import time

# Get model directory
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(base_dir, '../../vncorenlp_model'))

print(f"Loading VnCoreNLP from: {model_dir}...")
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "parse"], save_dir=model_dir)

# File paths
input_path = os.path.join(base_dir, "material_for_triplets/sections_rewritten_nghi_dinh_168_2024_1.json")
output_path = os.path.join(base_dir, "material_for_triplets/raw_vncore_output.json")

if not os.path.exists(input_path):
    print(f"❌ Không tìm thấy file input: {input_path}")
    os._exit(1)

print(f"Đọc dữ liệu từ: {input_path}")
with open(input_path, "r", encoding="utf-8") as f:
    sections = json.load(f)

results = []
total_sections = len(sections)

print(f"Bắt đầu xử lý và lưu định dạng đầu ra thô của mô hình cho {total_sections} sections...")
start_time = time.time()

for idx, section in enumerate(sections, 1):
    sec_id = section.get("id")
    propositions = section.get("rewritten_propositions", [])

    annotated_props = []
    for prop in propositions:
        if not prop or not isinstance(prop, str) or not prop.strip():
            continue
        try:
            # Dọn dẹp dấu gạch dưới nếu có và gọi mô hình
            cleaned_text = prop.replace("_", " ")
            raw_annotation = model.annotate_text(cleaned_text)

            annotated_props.append({
                "proposition": prop,
                "raw_annotation": raw_annotation
            })
        except Exception as e:
            print(f"❌ Lỗi khi annotate mệnh đề trong section {sec_id}: {e}")

    results.append({
        "id": sec_id,
        "document_name": section.get("document_name"),
        "level": section.get("level"),
        "path": section.get("path"),
        "original_text": section.get("text_content"),
        "annotated_propositions": annotated_props
    })

    # In ra ví dụ mẫu của section đầu tiên để kiểm tra định dạng thô
    if idx == 1:
        print("\n" + "="*60)
        print("VÍ DỤ ĐỊNH DẠNG ĐẦU RA THÔ CỦA MÔ HÌNH (Section s1, Mệnh đề 1):")
        if annotated_props:
            print(f"Mệnh đề: {annotated_props[0]['proposition']}")
            print(json.dumps(annotated_props[0]['raw_annotation'], ensure_ascii=False, indent=4))
        print("="*60 + "\n")

    if idx % 50 == 0 or idx == total_sections:
        elapsed = time.time() - start_time
        print(f"-> Đã xử lý {idx}/{total_sections} sections ({elapsed:.1f}s)...")

# Lưu định dạng thô ra file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n✅ Hoàn thành! Đã lưu định dạng thô của mô hình tại: {output_path}")
