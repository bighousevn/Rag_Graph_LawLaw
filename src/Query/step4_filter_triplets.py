"""
Bước 4: Lọc Triplets từ kết quả Vector Search.
Input:  Kết quả từ Bước 3 (danh sách matched_nodes + matched_relations cho mỗi query triplet)
Output: Danh sách triplet đã lọc, sao cho S và O phải là source/target của V,
        và cả 3 thành phần phải cùng chia sẻ ít nhất 1 sectionId.
"""

import os
import json
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def filter_triplets(step3_results: List[Dict]) -> List[Dict]:
    """
    Lọc triplet từ kết quả vector search.
    Điều kiện:
      - Relation phải có source_id nằm trong matched_nodes (role=source)
      - Relation phải có target_id nằm trong matched_nodes (role=target)
      - Edge (relation) đã mang sẵn listSectionId, chính là các sectionId chung của bộ (S, V, O)
    """
    filtered_results = []

    for item in step3_results:
        query_triplet = item["query_triplet"]
        matched_nodes = item.get("matched_nodes", [])
        matched_relations = item.get("matched_relations", [])

        # Tạo set ID cho source và target
        source_ids = {n["id"] for n in matched_nodes if n.get("role") == "source"}
        target_ids = {n["id"] for n in matched_nodes if n.get("role") == "target"}

        valid_triplets = []

        for rel in matched_relations:
            src_id = rel.get("source_id")
            tgt_id = rel.get("target_id")
            section_ids = rel.get("listSectionId", [])

            # Điều kiện 1: source_id phải nằm trong danh sách node source đã match
            if src_id not in source_ids:
                continue

            # Điều kiện 2: target_id phải nằm trong danh sách node target đã match
            if tgt_id not in target_ids:
                continue

            # Điều kiện 3: Phải có ít nhất 1 sectionId
            if not section_ids:
                continue

            # Tìm thông tin node tương ứng
            s_node = next((n for n in matched_nodes if n["id"] == src_id and n.get("role") == "source"), None)
            o_node = next((n for n in matched_nodes if n["id"] == tgt_id and n.get("role") == "target"), None)

            valid_triplets.append({
                "s": {
                    "id": src_id,
                    "name": s_node.get("name") if s_node else "Unknown",
                    "score": s_node.get("score") if s_node else None
                },
                "v": {
                    "id": rel.get("id"),
                    "relation_id": rel.get("relation_id"),
                    "name": rel.get("name"),
                    "score": rel.get("score")
                },
                "o": {
                    "id": tgt_id,
                    "name": o_node.get("name") if o_node else "Unknown",
                    "score": o_node.get("score") if o_node else None
                },
                "listSectionId": section_ids
            })

        filtered_results.append({
            "query_triplet": query_triplet,
            "valid_triplets": valid_triplets
        })

    return filtered_results


if __name__ == "__main__":
    # Test bằng cách đọc output của step 3
    step3_out = os.path.join(BASE_DIR, "output", "3_vector_search.json")
    if not os.path.exists(step3_out):
        print(f"❌ Không tìm thấy file {step3_out}. Hãy chạy step 3 trước.")
    else:
        with open(step3_out, "r", encoding="utf-8") as f:
            step3_data = json.load(f)

        results = filter_triplets(step3_data)

        print("\n[KẾT QUẢ BƯỚC 4 - LỌC TRIPLETS]")
        for item in results:
            qt = item["query_triplet"]
            print(f"\n  Query: ({qt.get('s')}) - [{qt.get('v')}] -> ({qt.get('o')})")
            print(f"  Số triplet hợp lệ: {len(item['valid_triplets'])}")
            for vt in item["valid_triplets"]:
                print(f"    ({vt['s']['name']}) - [{vt['v']['name']}] -> ({vt['o']['name']})  |  sections: {vt['listSectionId']}")

        # Lưu output
        step4_out = os.path.join(BASE_DIR, "output", "4_filtered_triplets.json")
        with open(step4_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n💾 Đã lưu tại: {step4_out}")
