"""
Bước 3: Trích xuất raw_entities + raw_triplets từ các mệnh đề đã chuẩn hoá.
Dùng LLM theo mô hình mạng lưới. KHÔNG gom concept (làm ở 4_build_ontology.py).

Tối ưu chi phí: gộp --batch-size sections thành 1 LLM call thay vì 1 call/section.
"""

import os
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in env variables or .env file.")

client = OpenAI(api_key=api_key)


EXTRACTION_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp lý tiếng Việt.

Bạn sẽ nhận nhiều section pháp lý, mỗi section có danh sách mệnh đề đánh số.
Với MỖI section và MỖI mệnh đề, hãy trích xuất THỰC THỂ và BỘ BA.

─── QUY TẮC THỰC THỂ (entity) ───
Entity CHỈ là danh từ / cụm danh từ THUẦN — KHÔNG chứa động từ, giới từ hoặc phó từ bên trong.
  ✓ ĐÚNG: "người", "xe ô tô", "giấy phép lái xe", "buồng lái", "số lượng quy định"
  ✗ SAI:  "người điều khiển ô tô"  (có động từ "điều khiển")
           "người trên buồng lái"   (có giới từ "trên")
           "quá số lượng quy định"  (có phó từ mức độ "quá")

Khi gặp các cụm phức tạp, phải tách:
  "người điều khiển ô tô"   → subject="người", relation="điều_khiển", object="ô tô"
  "người trên buồng lái"    → entity="người" + entity="buồng lái", thêm triplet (người, ở_trong, buồng lái)
  "chở quá số lượng"        → relation="chở_vượt_quá", object="số lượng quy định"

─── QUY TẮC BỘ BA (triplet) — MÔ HÌNH MẠNG LƯỚI ───
PHẢI trích xuất TẤT CẢ quan hệ — kể cả quan hệ trong phần ngữ cảnh/tiêu đề của mệnh đề.
KHÔNG được bỏ sót triplet với lý do "đây chỉ là background".

Ví dụ: "người điều khiển xe ô tô không đảm bảo khoảng cách an toàn với xe phía sau"
  entities: người | xe ô tô | khoảng cách an toàn | xe phía sau
  triplets:
    (người, điều_khiển, xe ô tô)               ← PHẢI có, dù là context
    (người, không_đảm_bảo, khoảng cách an toàn)
    (khoảng cách an toàn, với, xe phía sau)

KIỂM TRA SAU KHI SINH TRIPLETS: Mỗi entity PHẢI xuất hiện ít nhất 1 lần trong triplets
(làm subject hoặc object). Nếu có entity không xuất hiện trong bất kỳ triplet nào
→ đã bỏ sót relation, phải bổ sung triplet còn thiếu.

─── QUY TẮC RELATION ───
- Viết thường, dùng dấu gạch dưới: "điều_khiển", "không_đảm_bảo", "bị_tước"
- Phủ định PHẢI đưa vào relation: "không_có", "chưa_chấp_hành", "không_gắn"
- Giới từ vị trí ("trên/trong/tại/ở") → tách thành relation: "ở_trong", "ở_tại"
- Phó từ mức độ ("quá/vượt") → gộp vào relation: "chở_vượt_quá", "vượt_quá"

Trả về JSON duy nhất (key ngoài là chỉ số section "0","1",... key trong là chỉ số mệnh đề):
{
  "sections": {
    "0": {
      "results": {
        "0": {
          "entities": [
            {"text": "<danh từ thuần>", "role": "subject"},
            {"text": "<danh từ thuần>", "role": "object"}
          ],
          "triplets": [
            {"subject": "<danh từ thuần>", "relation": "<relation>", "object": "<danh từ thuần>"}
          ]
        }
      }
    },
    "1": { "results": { ... } }
  }
}
Bỏ qua mệnh đề trống hoặc không có thực thể/quan hệ rõ ràng."""


def call_with_retry(messages, max_retries=4, initial_backoff=2):
    backoff = initial_backoff
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content")
            return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def extract_batch(sections_batch):
    """One LLM call for a batch of sections. Returns dict keyed by section index str."""
    input_parts = []
    for i, section in enumerate(sections_batch):
        props = [p for p in section.get("rewritten_propositions", []) if p and p.strip()]
        if not props:
            continue
        input_parts.append(f"[Section {i} — id={section['id']}]")
        for j, prop in enumerate(props):
            input_parts.append(f"  {j}. \"{prop}\"")

    if not input_parts:
        return {}

    result = call_with_retry([
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": "Các section cần phân tích:\n" + "\n".join(input_parts)},
    ])
    return result.get("sections", {})


def build_section_result(section, prop_results):
    """Reconstruct per-section output from batch LLM results."""
    base = {
        "id": section.get("id"),
        "document_name": section.get("document_name"),
        "level": section.get("level"),
        "path": section.get("path"),
        "original_text": section.get("text_content"),
        "rewritten_propositions": section.get("rewritten_propositions", []),
    }

    propositions = [p for p in section.get("rewritten_propositions", []) if p and p.strip()]
    if not propositions or not prop_results:
        return {**base, "raw_entities": [], "raw_triplets": []}

    entity_set = set()
    raw_entities = []
    all_raw_triplets = []

    for j in range(len(propositions)):
        prop_result = prop_results.get(str(j), {})
        for e in prop_result.get("entities", []):
            text = e.get("text", "").strip()
            if text and text not in entity_set:
                entity_set.add(text)
                raw_entities.append({"text": text, "role": e.get("role", "")})
        all_raw_triplets.extend(prop_result.get("triplets", []))

    return {
        **base,
        "raw_entities": raw_entities,
        "raw_triplets": all_raw_triplets,
    }


def process_batch(sections_batch):
    """Process a batch of sections with one LLM call. Returns list of results."""
    batch_llm_results = extract_batch(sections_batch)
    results = []
    for i, section in enumerate(sections_batch):
        prop_results = batch_llm_results.get(str(i), {}).get("results", {})
        results.append(build_section_result(section, prop_results))
    return results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Extract triplets via LLM (batched, network model).")
    parser.add_argument(
        "--input",
        default=os.path.join(base_dir, "material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(base_dir, "material_for_triplets/3_triplets_extracted_nghi_dinh_168_2024_1.json"),
    )
    parser.add_argument("--threads", type=int, default=5, help="Parallel batch workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Sections per LLM call")
    parser.add_argument("--limit", type=int, default=0, help="Limit sections (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        sections = json.load(f)

    if args.limit > 0:
        sections = sections[: args.limit]
        print(f"Limiting to first {args.limit} sections.")

    print(f"Total sections: {len(sections)}")

    # Resume: load already-processed sections
    processed = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if "raw_triplets" in item:
                        processed[item["id"]] = item
            print(f"Resuming: {len(processed)} sections already done.")
        except Exception as e:
            print(f"Could not read existing output ({e}). Starting fresh.")

    to_process = [s for s in sections if s["id"] not in processed]
    print(f"Remaining: {len(to_process)} sections to process.")

    if not to_process:
        print("All sections already processed.")
        return

    batches = [
        to_process[i: i + args.batch_size]
        for i in range(0, len(to_process), args.batch_size)
    ]
    print(f"Batches: {len(batches)} × up to {args.batch_size} sections/call  |  threads={args.threads}")

    save_lock = threading.Lock()
    counts = {"success": len(processed), "fail": 0, "done": 0}

    def save_progress():
        ordered = [processed[s["id"]] for s in sections if s["id"] in processed]
        tmp = args.output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=4)
        os.replace(tmp, args.output)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            counts["done"] += 1
            try:
                results = future.result()
                with save_lock:
                    for result in results:
                        processed[result["id"]] = result
                        counts["success"] += 1
                    save_progress()
                ids = [r["id"] for r in results]
                n_triplets = sum(len(r.get("raw_triplets", [])) for r in results)
                print(f"[{counts['done']}/{len(batches)}] ✓ {ids} — {n_triplets} raw_triplets")
            except Exception as e:
                counts["fail"] += len(batch)
                ids = [s["id"] for s in batch]
                print(f"[{counts['done']}/{len(batches)}] ✗ {ids}: {e}")

    print(f"\nDone. Success: {counts['success']}, Failed: {counts['fail']}")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
