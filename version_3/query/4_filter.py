import json
import os

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN FILE
# ==========================================
QUERY_RESULTS_FILE = "./version_3/query/4_seed_entities.json"
GRAPH_DATA_FILE = "./version_3/2_entities_per_chunk.json"
OUTPUT_FILE = "./version_3/query/5_filtered_triplets.json"

def main():
    print("="*70)
    print("🚀 BẮT ĐẦU TRÍCH XUẤT STRICT TRIPLETS & SECTION IDs")
    print("="*70)

    # ---------------------------------------------------------
    # BƯỚC 1: Lấy danh sách toàn bộ ID đã qua màng lọc Vector
    # ---------------------------------------------------------
    if not os.path.exists(QUERY_RESULTS_FILE):
        print(f"❌ Lỗi: Không tìm thấy file {QUERY_RESULTS_FILE}")
        return

    print("1. Đang nạp danh sách ID từ kết quả Vector Search...")
    with open(QUERY_RESULTS_FILE, "r", encoding="utf-8") as f:
        query_results = json.load(f)

    matched_ids = set()
    for query in query_results:
        for item in query.get("matched_items", []):
            matched_ids.add(item["entityId"])

    print(f"   => Có tổng cộng [ {len(matched_ids)} ] ID (Nodes/Edges) đạt điểm > 0.85.")

    # ---------------------------------------------------------
    # BƯỚC 2: Nạp Đồ thị gốc
    # ---------------------------------------------------------
    if not os.path.exists(GRAPH_DATA_FILE):
        print(f"❌ Lỗi: Không tìm thấy file {GRAPH_DATA_FILE}")
        return

    print("\n2. Đang nạp dữ liệu Đồ thị gốc...")
    with open(GRAPH_DATA_FILE, "r", encoding="utf-8") as f:
        graph_data_raw = json.load(f)

    if isinstance(graph_data_raw, list) and len(graph_data_raw) > 0:
        graph_data = graph_data_raw[0]
    else:
        graph_data = graph_data_raw

    all_edges = graph_data.get("relationships", [])
    # Chuyển Nodes thành dạng Dictionary để tra cứu nhanh ở Bước 4
    all_nodes_dict = {n["id"]: n for n in graph_data.get("nodes", [])}

    print(f"   => Đồ thị gốc có [ {len(all_edges)} ] Edges.")

    # ---------------------------------------------------------
    # BƯỚC 3: KIỂM TRA ĐIỀU KIỆN STRICT TRIPLET CỦA TỪNG EDGE
    # ---------------------------------------------------------
    print("\n3. Bắt đầu duyệt Edges và kiểm tra tính trọn vẹn (Triplet)...")

    valid_edges = []
    valid_node_ids = set()
    section_ids = set()

    for edge in all_edges:
        edge_id = edge.get("id")
        src_id = edge.get("source")
        tgt_id = edge.get("target")
        edge_name = edge.get("name", "")

        # ĐIỀU KIỆN 1: Bản thân Edge đó phải nằm trong file 2_output.json
        if edge_id in matched_ids:

            # ĐIỀU KIỆN 2: Source và Target CŨNG PHẢI nằm trong file 2_output.json
            if (src_id in matched_ids) and (tgt_id in matched_ids):
                valid_edges.append(edge)
                valid_node_ids.add(src_id)
                valid_node_ids.add(tgt_id)

                # Thu thập Section ID
                for sid in edge.get("listSectionId", []):
                    section_ids.add(sid)

                # Log thành công
                print(f"  [+] HỢP LỆ: Edge '{edge_name}' ({edge_id}) | Source ({src_id}) & Target ({tgt_id}) đều có mặt.")
            else:
                # Log thất bại và chỉ ra nguyên nhân
                missing = []
                if src_id not in matched_ids: missing.append(f"Source ({src_id})")
                if tgt_id not in matched_ids: missing.append(f"Target ({tgt_id})")
                print(f"  [-] BỊ LOẠI: Edge '{edge_name}' ({edge_id}) nằm trong output, nhưng thiếu {', '.join(missing)}.")

    print(f"\n   => Kết quả: Giữ lại được [ {len(valid_edges)} ] Edges hợp lệ.")

    # ---------------------------------------------------------
    # BƯỚC 4: Lọc Nodes và Đóng gói
    # ---------------------------------------------------------
    print("\n4. Đang dọn dẹp các Nodes mồ côi...")
    # Chỉ bốc ra những Nodes nằm trong tập hợp valid_node_ids của các Edge hợp lệ
    valid_nodes = [all_nodes_dict[nid] for nid in valid_node_ids if nid in all_nodes_dict]

    print(f"   => Giữ lại được [ {len(valid_nodes)} ] Nodes tạo thành các Triplet hoàn chỉnh.")

    # ---------------------------------------------------------
    # BƯỚC 5: Tổng hợp Section IDs
    # ---------------------------------------------------------
    sorted_sids = sorted(list(section_ids))
    print(f"\n5. Tổng hợp Section IDs từ các Triplet trên...")
    print(f"   => Tìm thấy [ {len(sorted_sids)} ] Section IDs.")
    print(f"   => Danh sách: {sorted_sids}")

    # Xuất file
    output_data = {
        "nodes": valid_nodes,
        "relationships": valid_edges,
        "relevant_section_ids": sorted_sids
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump([output_data], f, ensure_ascii=False, indent=4)

    print("\n" + "="*70)
    print(f"🎉 HOÀN TẤT! Dữ liệu cuối cùng đã được lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()