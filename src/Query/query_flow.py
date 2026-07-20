import os
import json
import pymongo
from typing import List, Dict, Any, Set
from dotenv import load_dotenv
from openai import OpenAI
import torch
from sentence_transformers import SentenceTransformer

from step1_normalize import normalize_question
from step2_extract_triplets import extract_triplets
from step3_vector_search import search_graph_from_triplets

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(os.path.dirname(BASE_DIR), "Building_KG", "ontology", "output", "kg_triplets.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "query_results.json")

MONGO_URI = os.getenv("MONGO_URI", "").strip()
DB_NAME = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
VECTOR_INDEX_NAME = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index").strip()
VECTOR_FIELD_PATH = os.getenv("MONGO_VECTOR_FIELD_PATH", "vector").strip()
SCORE_THRESHOLD = float(os.getenv("MONGO_VECTOR_SCORE_THRESHOLD", "0.80"))
SEARCH_LIMIT = int(os.getenv("MONGO_VECTOR_SEARCH_LIMIT", "20"))
NUM_CANDIDATES = int(os.getenv("MONGO_VECTOR_NUM_CANDIDATES", "100"))

def step1_normalize_question(question: str) -> List[str]:
    print("▶ Bước 1: Chuẩn hóa câu hỏi")
    props = normalize_question(question)
    print(f"  - Kết quả: {props}")
    return props

def step2_extract_triplets(propositions: List[str]) -> List[Dict]:
    print("▶ Bước 2: Trích xuất triplet")
    triplets = extract_triplets(propositions)
    for t in triplets:
        print(f"  - ({t.get('s')}) - [{t.get('v')}] -> ({t.get('o')})")
    return triplets

def step3_and_4_search_and_filter(triplets: List[Dict]) -> List[Dict]:
    print("▶ Bước 3: Vector search lấy node và relation")
    results = search_graph_from_triplets(triplets)
    return results, {}

def step5_output_results(all_matched_triplets: List[Dict], section_id_counts: Dict[str, int], total_triplets: int):
    print("\n▶ Bước 5: Đánh giá và trả kết quả")
    
    # Tìm các section_id thỏa mãn mọi triplet (Exact match)
    exact_match_sids = [sid for sid, count in section_id_counts.items() if count == total_triplets]
    
    result = {
        "matched_triplets_details": all_matched_triplets,
        "total_query_triplets": total_triplets,
    }

    if exact_match_sids:
        print(f"  ✅ TÌM THẤY EXACT MATCH! Các điều khoản thỏa mãn toàn bộ ({total_triplets}/{total_triplets}) triplets:")
        print(f"  {exact_match_sids}")
        result["match_type"] = "exact"
        result["section_ids"] = exact_match_sids
    else:
        print(f"  ⚠️ KHÔNG có exact match. Chuyển sang chế độ BEST-EFFORT.")
        if section_id_counts:
            max_count = max(section_id_counts.values())
            best_sids = [sid for sid, count in section_id_counts.items() if count == max_count]
            print(f"  - Các điều khoản khớp nhiều nhất ({max_count}/{total_triplets} triplets):")
            print(f"  {best_sids}")
            
            # Phân tích triplet nào thiếu ở best_sids đầu tiên
            missing_in = []
            test_sid = best_sids[0]
            for t_idx, match_data in enumerate(all_matched_triplets):
                if test_sid not in match_data["triplet_section_ids"]:
                    missing_in.append(match_data["query_triplet"])

            print(f"  - Điều khoản {test_sid} THIẾU các triplet sau:")
            for m in missing_in:
                print(f"    ({m.get('s')}) - [{m.get('v')}] -> ({m.get('o')})")
            
            result["match_type"] = "best_effort"
            result["section_ids"] = best_sids
            result["max_matched_triplets"] = max_count
            result["missing_triplets_example"] = missing_in
        else:
            print("  ❌ Không tìm thấy bất kỳ section_id nào liên quan.")
            result["match_type"] = "none"
            result["section_ids"] = []

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"\n💾 Đã lưu kết quả tại: {OUTPUT_FILE}")

def main():
    print("="*70)
    print("🚀 QUERY FLOW PIPELINE — END TO END")
    print("="*70)

    # 1. Input
    question = input("Nhập câu hỏi pháp lý (để trống để đọc từ file input/1_question.txt): ")
    if not question.strip():
        input_file = os.path.join(BASE_DIR, "input", "1_question.txt")
        if os.path.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as f:
                question = f.read().strip()
            print(f"Đã đọc câu hỏi từ file: {input_file}")
            print(f"Câu hỏi: '{question}'")
        else:
            question = "Tôi lái xe máy vượt đèn đỏ thì bị xử lý như nào?"
            print(f"Dùng câu hỏi mặc định: '{question}'")

    # 2. Setup DB & Graph
    if not MONGO_URI:
        print("❌ Thiếu cấu hình MONGO_URI trong .env")
        return
    client_db = pymongo.MongoClient(MONGO_URI)
    db = client_db[DB_NAME]
    collection = db[COLLECTION_NAME]
    try:
        client_db.admin.command('ping')
    except Exception as e:
        print(f"❌ Lỗi kết nối tới MongoDB: {e}")
        return

    if not os.path.exists(GRAPH_FILE):
        print(f"❌ Không tìm thấy file đồ thị: {GRAPH_FILE}")
        return
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # Pipeline
    props = step1_normalize_question(question)
    
    # Save step 1 output
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    step1_out = os.path.join(BASE_DIR, "output", "1_normalized.json")
    with open(step1_out, "w", encoding="utf-8") as f:
        json.dump({"propositions": props}, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu kết quả chuẩn hóa tại: {step1_out}")

    triplets = step2_extract_triplets(props)
    
    # Save step 2 output
    step2_out = os.path.join(BASE_DIR, "output", "2_triplets.json")
    with open(step2_out, "w", encoding="utf-8") as f:
        json.dump({"triplets": triplets}, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu kết quả trích xuất Triplets tại: {step2_out}")
    
    if not triplets:
        print("❌ Không trích xuất được triplet nào.")
        return

    all_matched, section_id_counts = step3_and_4_search_and_filter(triplets, collection, graph_data)
    step5_output_results(all_matched, section_id_counts, len(triplets))

if __name__ == "__main__":
    main()
