"""
Bước 5: Tổng hợp kết quả cuối cùng.
Input:  Kết quả từ Bước 4 (danh sách triplet đã lọc + sectionId)
Output: Danh sách triplet cuối cùng + danh sách sectionId chung cho TẤT CẢ triplet.
        Nếu không có sectionId nào chung hoàn toàn, trả về sectionId trùng nhiều nhất
        kèm thông báo trường hợp (exact / best_effort / none).
"""

import os
import json
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "5_final_results.json")


def aggregate_results(step4_results: List[Dict]) -> Dict:
    """
    Tổng hợp kết quả:
    - Gom tất cả triplet đã lọc.
    - Tìm sectionId chung cho TẤT CẢ query triplet.
    - Nếu không có sectionId nào chung hoàn toàn → lấy trùng nhiều nhất.
    """
    all_filtered_triplets = []
    section_id_sets = []  # Mỗi phần tử là set sectionId của 1 query triplet

    for item in step4_results:
        query_triplet = item["query_triplet"]
        valid_triplets = item.get("valid_triplets", [])

        # Gom tất cả sectionId của query triplet này
        triplet_section_ids = set()
        for vt in valid_triplets:
            triplet_section_ids.update(vt.get("listSectionId", []))

        all_filtered_triplets.append({
            "query_triplet": query_triplet,
            "valid_triplets": valid_triplets,
            "section_ids": sorted(list(triplet_section_ids))
        })

        if triplet_section_ids:
            section_id_sets.append(triplet_section_ids)

    total_query_triplets = len(step4_results)

    # Xây dựng kết quả
    result = {
        "total_query_triplets": total_query_triplets,
        "filtered_triplets": all_filtered_triplets
    }

    if not section_id_sets:
        # Trường hợp NONE: Không có triplet nào match được
        print("  ❌ KHÔNG tìm thấy bất kỳ sectionId nào liên quan.")
        result["match_type"] = "none"
        result["match_description"] = "Không tìm thấy bất kỳ điều khoản nào phù hợp với câu hỏi."
        result["common_section_ids"] = []
        return result

    # Tìm giao (intersection) của TẤT CẢ các set sectionId
    common_sids = section_id_sets[0]
    for s in section_id_sets[1:]:
        common_sids = common_sids & s

    if common_sids:
        # Trường hợp EXACT: Có sectionId chung cho tất cả triplet
        sorted_common = sorted(list(common_sids))
        print(f"  ✅ EXACT MATCH! Có {len(sorted_common)} điều khoản thỏa mãn toàn bộ {total_query_triplets} triplet:")
        print(f"     {sorted_common}")
        result["match_type"] = "exact"
        result["match_description"] = (
            f"Tìm thấy {len(sorted_common)} điều khoản (sectionId) chứa đầy đủ "
            f"tất cả {total_query_triplets} triplet từ câu hỏi."
        )
        result["common_section_ids"] = sorted_common
    else:
        # Trường hợp BEST_EFFORT: Không có giao hoàn toàn → đếm tần suất
        section_id_counts = {}
        for sid_set in section_id_sets:
            for sid in sid_set:
                section_id_counts[sid] = section_id_counts.get(sid, 0) + 1

        max_count = max(section_id_counts.values())
        best_sids = sorted([sid for sid, cnt in section_id_counts.items() if cnt == max_count])

        print(f"  ⚠️ BEST-EFFORT! Không có sectionId nào thỏa mãn toàn bộ {total_query_triplets} triplet.")
        print(f"     Các sectionId khớp nhiều nhất ({max_count}/{total_query_triplets} triplet): {best_sids}")

        # Phân tích triplet nào bị thiếu ở sectionId tốt nhất
        missing_triplets = []
        test_sid = best_sids[0]
        for i, sid_set in enumerate(section_id_sets):
            if test_sid not in sid_set:
                missing_triplets.append(step4_results[i]["query_triplet"])

        if missing_triplets:
            print(f"     Ví dụ: sectionId '{test_sid}' THIẾU các triplet sau:")
            for m in missing_triplets:
                print(f"       ({m.get('s')}) - [{m.get('v')}] -> ({m.get('o')})")

        result["match_type"] = "best_effort"
        result["match_description"] = (
            f"Không có điều khoản nào chứa đầy đủ tất cả {total_query_triplets} triplet. "
            f"Các điều khoản khớp nhiều nhất chứa {max_count}/{total_query_triplets} triplet."
        )
        result["common_section_ids"] = best_sids
        result["max_matched_count"] = max_count
        result["missing_triplets_example"] = missing_triplets

    return result


if __name__ == "__main__":
    # Test bằng cách đọc output của step 4
    step4_out = os.path.join(BASE_DIR, "output", "4_filtered_triplets.json")
    if not os.path.exists(step4_out):
        print(f"❌ Không tìm thấy file {step4_out}. Hãy chạy step 4 trước.")
    else:
        with open(step4_out, "r", encoding="utf-8") as f:
            step4_data = json.load(f)

        print("--- Đang xử lý Bước 5: Tổng hợp kết quả ---")
        result = aggregate_results(step4_data)

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"\n💾 Đã lưu kết quả cuối tại: {OUTPUT_FILE}")
