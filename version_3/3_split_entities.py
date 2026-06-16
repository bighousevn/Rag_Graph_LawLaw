import json
import os

def split_entities(input_file_path, output_file_path):
    print(f"Đang xử lý: {input_file_path}")

    if not os.path.exists(input_file_path):
        print(f"Lỗi: Không tìm thấy file {input_file_path}")
        return

    with open(input_file_path, "r", encoding="utf-8") as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Lỗi khi đọc file JSON {input_file_path}: {e}")
            return

    # Trích xuất tất cả các nodes gốc từ các batches
    original_nodes = []
    if isinstance(json_data, list):
        for batch in json_data:
            if isinstance(batch, dict):
                original_nodes.extend(batch.get("nodes", []))
            elif isinstance(batch, dict) and "nodes" not in batch:
                # Trường hợp danh sách phẳng các node trực tiếp (nếu có)
                original_nodes.append(batch)
    elif isinstance(json_data, dict):
        original_nodes = json_data.get("nodes", [])

    split_items = []
    name_count = 0
    alias_count = 0

    for node in original_nodes:
        entity_id = node.get("id")
        label = node.get("label")
        name = node.get("name")
        aliases = node.get("aliases", [])
        list_section_id = node.get("listSectionId", [])

        if not entity_id:
            continue

        # 1. Tạo item cho name (chỉ khi name tồn tại)
        if name:
            name_item = {
                "id": f"{entity_id}_name",
                "entity_id": entity_id,
                "name": name,
                "label": label,
                "type": "name",
                "listSectionId": list_section_id
            }
            split_items.append(name_item)
            name_count += 1

        # 2. Tạo item cho mỗi alias
        if isinstance(aliases, list):
            for i, alias in enumerate(aliases):
                if not alias:
                    continue
                alias_item = {
                    "id": f"{entity_id}_alias_{i}",
                    "entity_id": entity_id,
                    "name": alias,
                    "label": label,
                    "type": "alias",
                    "listSectionId": list_section_id
                }
                split_items.append(alias_item)
                alias_count += 1

    # Lưu kết quả
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(split_items, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu kết quả tại: {output_file_path}")
    print(f"  - Số lượng nodes gốc: {len(original_nodes)}")
    print(f"  - Số lượng name items: {name_count}")
    print(f"  - Số lượng alias items: {alias_count}")
    print(f"  - Tổng số items đồng cấp mới: {len(split_items)}")
    print("-" * 50)


if __name__ == "__main__":
    # Đường dẫn đến thư mục version_3
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Danh sách các file cần xử lý
    files_to_process = [
        # ("2_entities_per_chunk_vphc.json", "3_split_entities_vphc.json"),
        ("2_entities_per_chunk_dat_dai_1.json", "3_split_entities_dat_dai_1.json")
    ]

    for input_name, output_name in files_to_process:
        input_path = os.path.join(base_dir, input_name)
        output_path = os.path.join(base_dir, output_name)
        split_entities(input_path, output_path)
