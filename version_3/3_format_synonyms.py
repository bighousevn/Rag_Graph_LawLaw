import json
import os

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_synonyms(graph_data):
    synonyms_output = []
    synonym_counter = 1

    # Hàm xử lý chung cho cả Nodes và Relationships
    def process_entities(entities):
        nonlocal synonym_counter
        for entity in entities:
            entity_id = entity.get("id")
            entity_name = entity.get("name", "").strip()
            synonyms_list = entity.get("synonym", [])

            # Gộp mảng synonym và từ gốc (entity_name), sử dụng set() để loại bỏ trùng lặp
            all_syns = set([str(s).strip() for s in synonyms_list if str(s).strip()])
            if entity_name:
                all_syns.add(entity_name)

            # Sắp xếp theo thứ tự alphabet để dữ liệu xuất ra gọn gàng, dễ kiểm tra
            for syn in sorted(list(all_syns)):
                synonyms_output.append({
                    # Sinh ID tự động (Ví dụ: S0001, S0002...)
                    "synonymId": f"S{synonym_counter:04d}",
                    "entityId": entity_id,
                    "entityName": entity_name,
                    "synonym": syn
                })
                synonym_counter += 1

    # 1. Trích xuất từ vựng từ danh sách Nodes
    nodes = graph_data.get("nodes", [])
    process_entities(nodes)

    # 2. Trích xuất từ vựng từ danh sách Relationships (Edges)
    relationships = graph_data.get("relationships", [])
    process_entities(relationships)

    return synonyms_output

def main():
    input_file = './version_3/2_entitites_per_chunk.json'
    output_file = './version_3/4_synonyms_formatted.json'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    print("Đang nạp dữ liệu Đồ thị...")
    raw_data = load_data(input_file)

    # Tự động nhận diện cấu trúc mảng bọc ngoài
    if isinstance(raw_data, list) and len(raw_data) > 0:
        graph_data = raw_data[0]
    else:
        graph_data = raw_data

    print(f"Dữ liệu có {len(graph_data.get('nodes', []))} Nodes và {len(graph_data.get('relationships', []))} Relationships.")
    print("Đang tiến hành trích xuất và trải phẳng (flatten) danh sách từ đồng nghĩa...")

    # Chạy thuật toán chuyển đổi
    formatted_synonyms = format_synonyms(graph_data)

    # Lưu ra file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_synonyms, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*50}")
    print(f"🎉 HOÀN TẤT TẠO TỪ ĐIỂN ĐỒNG NGHĨA!")
    print(f"✅ Đã tạo thành công {len(formatted_synonyms)} bản ghi Synonym.")
    print(f"💾 Lưu tại: {output_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()