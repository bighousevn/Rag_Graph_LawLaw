import os
import json
from step1_normalize import normalize_question
from step2_extract_triplets import extract_triplets
from step3_vector_search import search_graph_from_triplets
from step4_filter_triplets import filter_triplets
from step5_aggregate import aggregate_results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("="*70)
    print("🚀 QUERY PIPELINE — 5 BƯỚC")
    print("="*70)

    # 1. Đọc file input
    input_file = os.path.join(BASE_DIR, "input", "1_question.txt")
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file input tại: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        question = f.read().strip()
    
    print(f"📖 Câu hỏi đầu vào: '{question}'\n")
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

    # ═══════════════════════════════════════════
    # BƯỚC 1: Chuẩn hóa câu hỏi
    # ═══════════════════════════════════════════
    print("▶ Bước 1: Chuẩn hóa câu hỏi")
    props = normalize_question(question)
    print(f"  - Kết quả: {props}")

    step1_out = os.path.join(BASE_DIR, "output", "1_normalized.json")
    with open(step1_out, "w", encoding="utf-8") as f:
        json.dump({"propositions": props}, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu: {step1_out}\n")

    # ═══════════════════════════════════════════
    # BƯỚC 2: Trích xuất triplet
    # ═══════════════════════════════════════════
    if not props:
        print("❌ Không có mệnh đề nào để trích xuất.")
        return

    print("▶ Bước 2: Trích xuất triplet")
    triplets = extract_triplets(props)
    for t in triplets:
        print(f"  - ({t.get('s')}) - [{t.get('v')}] -> ({t.get('o')})")
    
    step2_out = os.path.join(BASE_DIR, "output", "2_triplets.json")
    with open(step2_out, "w", encoding="utf-8") as f:
        json.dump({"triplets": triplets}, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu: {step2_out}\n")

    if not triplets:
        print("❌ Không trích xuất được triplet nào.")
        return

    # ═══════════════════════════════════════════
    # BƯỚC 3: Vector Search lấy candidates
    # ═══════════════════════════════════════════
    print("▶ Bước 3: Vector search lấy danh sách node & relation")
    step3_results = search_graph_from_triplets(triplets)

    step3_out = os.path.join(BASE_DIR, "output", "3_vector_search.json")
    with open(step3_out, "w", encoding="utf-8") as f:
        json.dump(step3_results, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu: {step3_out}\n")

    # ═══════════════════════════════════════════
    # BƯỚC 4: Lọc triplet (S-V-O cùng sectionId)
    # ═══════════════════════════════════════════
    print("▶ Bước 4: Lọc triplet (S phải là source, O phải là target của V, cùng sectionId)")
    step4_results = filter_triplets(step3_results)

    for item in step4_results:
        qt = item["query_triplet"]
        print(f"\n  Query: ({qt.get('s')}) - [{qt.get('v')}] -> ({qt.get('o')})")
        print(f"  Số triplet hợp lệ: {len(item['valid_triplets'])}")
        for vt in item["valid_triplets"]:
            print(f"    ({vt['s']['name']}) - [{vt['v']['name']}] -> ({vt['o']['name']})  |  sections: {vt['listSectionId']}")

    step4_out = os.path.join(BASE_DIR, "output", "4_filtered_triplets.json")
    with open(step4_out, "w", encoding="utf-8") as f:
        json.dump(step4_results, f, ensure_ascii=False, indent=4)
    print(f"\n💾 Đã lưu: {step4_out}\n")

    # ═══════════════════════════════════════════
    # BƯỚC 5: Tổng hợp kết quả (sectionId chung)
    # ═══════════════════════════════════════════
    print("▶ Bước 5: Tổng hợp kết quả cuối cùng")
    final_result = aggregate_results(step4_results)

    step5_out = os.path.join(BASE_DIR, "output", "5_final_results.json")
    with open(step5_out, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)
    print(f"\n💾 Đã lưu kết quả cuối cùng: {step5_out}")

    # In tóm tắt
    print("\n" + "="*70)
    print(f"📊 TÓM TẮT KẾT QUẢ")
    print(f"   Loại kết quả: {final_result.get('match_type', 'N/A').upper()}")
    print(f"   Mô tả: {final_result.get('match_description', '')}")
    print(f"   SectionId: {final_result.get('common_section_ids', [])}")
    print("="*70)

if __name__ == "__main__":
    main()
