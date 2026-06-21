import py_vncorenlp
import os

# Khởi tạo model với đầy đủ 3 annotator: tách từ, từ loại, và phân tích cú pháp
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(base_dir, '../../vncorenlp_model'))
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "parse"], save_dir=model_dir)

# Tập hợp các tiền tố ý định cần loại bỏ khỏi hành động cốt lõi
# Việc loại bỏ các từ này giúp chuẩn hóa quan hệ (Relation), ví dụ:
# "muốn thành lập" -> Relation chuẩn là "thành lập"
INTENTIONAL_PREFIXES = {"muốn", "cần", "phải", "được", "bị", "nên", "sẽ", "đang", "đã"}

def find_subject(verb_idx, sentence, tokens_by_index):
    # 1. Tìm chủ ngữ trực tiếp của động từ
    for token in sentence:
        if token['head'] == verb_idx and token['depLabel'] in ['sub', 'nsubj']:
            return token['wordForm'].replace("_", " ")

    # 2. Nếu không có, duyệt ngược lên cây quan hệ (traversing up) để tìm chủ ngữ thừa kế
    curr_idx = verb_idx
    visited = set()
    while curr_idx in tokens_by_index and curr_idx not in visited:
        visited.add(curr_idx)
        parent_idx = tokens_by_index[curr_idx]['head']
        if parent_idx == 0:
            break
        # Tìm chủ ngữ của nút cha
        for token in sentence:
            if token['head'] == parent_idx and token['depLabel'] in ['sub', 'nsubj']:
                return token['wordForm'].replace("_", " ")
        curr_idx = parent_idx
    return None

def extract_triplets(text):
    # Dọn dẹp dấu gạch dưới có sẵn để VnCoreNLP tự phân tách lại chuẩn xác
    cleaned_text = text.replace("_", " ")
    annotated_doc = model.annotate_text(cleaned_text)
    triplets = []

    for sentence_idx, sentence in annotated_doc.items():
        # Tạo dictionary tra cứu nhanh token theo index
        tokens_by_index = {token['index']: token for token in sentence}

        # 1. Tìm các Động từ làm hạt nhân (Relation)
        for token in sentence:
            word = token['wordForm'].lower().replace("_", " ")
            pos = token['posTag']
            idx = token['index']

            # Nếu là động từ chính (loại trừ các từ tình thái/thì)
            if pos == 'V' and word not in INTENTIONAL_PREFIXES:
                relation = token['wordForm'].replace("_", " ")

                # Kiểm tra xem động từ này có bị ảnh hưởng bởi động từ bị động ("bị", "được") không
                parent_idx = token['head']
                if parent_idx in tokens_by_index:
                    parent_token = tokens_by_index[parent_idx]
                    parent_word = parent_token['wordForm'].lower()
                    if parent_word in ["bị", "được"]:
                        relation = f"{parent_token['wordForm']} {relation}"

                # 2. Tìm Chủ thể (Subject) - sử dụng lan truyền chủ ngữ
                subject = find_subject(idx, sentence, tokens_by_index)

                # 3. Tìm Đối tượng (Object)
                object_ = None
                # 3.1 Tìm tân ngữ trực tiếp (dob)
                for other_token in sentence:
                    if other_token['head'] == idx and other_token['depLabel'] in ['dob', 'dobj']:
                        object_ = other_token['wordForm'].replace("_", " ")
                        break

                # 3.2 Nếu không có dob, tìm tân ngữ qua giới từ (pob)
                if not object_:
                    for prep_token in sentence:
                        if prep_token['head'] == idx and prep_token['posTag'] in ['E', 'I']:
                            # Tìm pob của giới từ này
                            for pob_token in sentence:
                                if pob_token['head'] == prep_token['index'] and pob_token['depLabel'] == 'pob':
                                    object_ = f"{prep_token['wordForm']} {pob_token['wordForm']}".replace("_", " ")
                                    break
                            if object_:
                                break

                # 4. Nếu đủ Subject - Relation - Object thì lưu lại
                if subject and relation and object_:
                    triplets.append({
                        "subject": subject,
                        "relation": relation,
                        "object": object_
                    })

    return triplets

# --- Chạy thử nghiệm ---
import json
import time

input_path = os.path.join(base_dir, "material_for_triplets/sections_rewritten_nghi_dinh_168_2024_1.json")
output_path = os.path.join(base_dir, "material_for_triplets/triplets_extracted_test.json")

if not os.path.exists(input_path):
    print(f"❌ Không tìm thấy file input: {input_path}")
    os._exit(1)

print(f"Đọc dữ liệu từ: {input_path}")
with open(input_path, "r", encoding="utf-8") as f:
    sections = json.load(f)

results = []
total_sections = len(sections)

print(f"Bắt đầu trích xuất triplets từ {total_sections} sections...")
start_time = time.time()

for idx, section in enumerate(sections, 1):
    sec_id = section.get("id")
    propositions = section.get("rewritten_propositions", [])

    section_triplets = []
    prop_to_triplets = {} # Cache to avoid duplicate extraction for print

    for prop in propositions:
        if not prop or not isinstance(prop, str) or not prop.strip():
            continue
        try:
            extracted = extract_triplets(prop)
            section_triplets.extend(extracted)
            prop_to_triplets[prop] = extracted
        except Exception as e:
            print(f"❌ Lỗi khi xử lý mệnh đề trong section {sec_id}: {e}")

    results.append({
        "id": sec_id,
        "document_name": section.get("document_name"),
        "level": section.get("level"),
        "path": section.get("path"),
        "original_text": section.get("text_content"),
        "rewritten_propositions": propositions,
        "triplets": section_triplets
    })

    # Chỉ in ra thông tin của 5 section đầu tiên làm mẫu để tránh tràn màn hình
    if idx <= 5:
        print("\n" + "="*60)
        print(f"Section {sec_id} | Path: {section.get('path')}")
        print(f"Văn bản gốc: {section.get('text_content')}")
        print("Mệnh đề & Triplets trích xuất được:")
        for prop in propositions:
            if prop in prop_to_triplets:
                print(f"  - Mệnh đề: {prop}")
                for triplet in prop_to_triplets[prop]:
                    print(f"    👉 ({triplet['subject']}) -[{triplet['relation']}]-> ({triplet['object']})")

    if idx % 50 == 0 or idx == total_sections:
        elapsed = time.time() - start_time
        print(f"-> Đã xử lý {idx}/{total_sections} sections ({elapsed:.1f}s)...")

# Lưu kết quả ra file mới
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n✅ Hoàn thành! Đã trích xuất và lưu triplets tại: {output_path}")
