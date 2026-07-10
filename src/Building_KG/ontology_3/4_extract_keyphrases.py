"""
Bước 4 (pass 3/3): Trích keyphrase gốc cho từng subject/verb/object canonical đã có.

Nhiệm vụ DUY NHẤT của LLM: với mỗi triplet {s, v, o} đã trích, đối chiếu lại câu gốc
(propositions[].text — nay là text_content gốc) của chính section đó và tìm TẤT CẢ cụm từ trong
câu gốc đã được gộp vào từng tên canonical. Không tự đặt tên canonical mới, không tự suy diễn —
chỉ tìm phrase XUẤT HIỆN THẬT trong câu gốc.

Input:  output/triplets_refined.json   (từ 3_refine_atomicity.py)
Output: output/triplets_with_keyphrases.json
"""

import os
import json
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "output", "triplets_refined.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "triplets_with_keyphrases.json")

BATCH_SIZE = 15   # cần đọc lại câu gốc đầy đủ (không chỉ triplet ngắn) nên batch gần bằng Pass 1


SYSTEM_PROMPT = """\
Bạn là chuyên gia đối chiếu văn bản pháp lý giao thông đường bộ Việt Nam.

NHIỆM VỤ DUY NHẤT: với mỗi section, bạn được cho câu gốc và danh sách triplet {s, v, o} đã được
trích sẵn từ câu đó (tên canonical, Title Case, đã bỏ yếu tố bị động/phủ định — có thể KHÔNG khớp
nguyên văn với câu gốc do đã chuẩn hoá). Với MỖI phần tử s/v/o trong mỗi triplet, hãy tìm lại
TRONG CÂU GỐC cụm từ/cách diễn đạt GẦN NHẤT tương ứng với tên canonical đó (kể cả khi câu gốc viết
ở dạng bị động/phủ định — vẫn lấy đúng cụm gốc, không cần khớp y hệt nghĩa), trả về dưới dạng
"keyphrases" (danh sách chuỗi, giữ nguyên chữ thường/cách viết như trong câu gốc).

QUY TẮC:
1. Chỉ lấy phrase THỰC SỰ xuất hiện trong câu gốc — không tự bịa, không tự suy diễn thêm cách gọi
   không có trong câu.
2. Nếu tên canonical là kết quả GỘP NHIỀU cách gọi đồng nghĩa trong câu (ví dụ object "Ô tô" được
   gộp từ "xe ô tô", "xe chở người bốn bánh có gắn động cơ", "xe chở hàng bốn bánh có gắn động cơ",
   "xe tương tự xe ô tô" đều xuất hiện trong cùng 1 câu) → liệt kê ĐẦY ĐỦ tất cả các cụm đó trong
   "keyphrases", không chỉ lấy 1 cụm.
3. Nếu tên canonical chỉ ứng với 1 cụm duy nhất trong câu (ví dụ "Cồn" ứng với "nồng độ cồn") →
   "keyphrases" chỉ có 1 phần tử.
4. Relation đã bị chuẩn hoá bỏ "bị"/"được"/"không" — khi tìm keyphrase cho verb, lấy CỤM GỐC đầy
   đủ kể cả phần bị động/phủ định đã bị bỏ (vd verb canonical "Có" ứng với câu gốc "không có" →
   keyphrase vẫn là "không có", không phải "có").
5. Nếu không tìm được cụm nào khớp rõ ràng trong câu gốc (hiếm khi xảy ra) → dùng chính tên
   canonical (viết thường) làm keyphrase duy nhất, không để trống.
6. KHÔNG đổi tên canonical s/v/o đã cho, KHÔNG thêm/bớt/sửa triplet — chỉ bổ sung field
   "s_keyphrases", "v_keyphrases", "o_keyphrases" cho mỗi triplet.

Trả về JSON:
{
  "results": [
    {
      "id": "s117",
      "triplets": [
        {
          "s": "Người", "s_keyphrases": ["người điều khiển"],
          "v": "Điều khiển", "v_keyphrases": ["điều khiển"],
          "o": "Ô tô", "o_keyphrases": ["xe ô tô", "xe chở người bốn bánh có gắn động cơ",
                                        "xe chở hàng bốn bánh có gắn động cơ", "xe tương tự xe ô tô"]
        },
        {
          "s": "Người", "s_keyphrases": ["người"],
          "v": "Sử dụng", "v_keyphrases": ["có nồng độ cồn"],
          "o": "Cồn", "o_keyphrases": ["nồng độ cồn"]
        }
      ]
    }
  ]
}"""


def llm_call(messages: list, max_retries: int = 4) -> dict:
    backoff = 2
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("empty LLM response content")
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt+1}: {e}")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("llm_call: unreachable")


def _fallback_triplets(sec: dict) -> list[dict]:
    """Nếu LLM bỏ sót section, dùng chính tên canonical làm keyphrase duy nhất."""
    out = []
    for t in sec.get("triplets", []):
        out.append({
            "s": t["s"], "s_keyphrases": [t["s"].lower()],
            "v": t["v"], "v_keyphrases": [t["v"].lower()],
            "o": t["o"], "o_keyphrases": [t["o"].lower()],
        })
    return out


def extract_batch(sections: list[dict]) -> list[dict]:
    """sections: list of {id, path, propositions:[{text,segmented}], triplets:[{s,v,o}]}."""
    parts = []
    for sec in sections:
        if not sec.get("triplets"):
            continue
        texts = [p.get("text", "") for p in sec.get("propositions", []) if p.get("text")]
        text_block = " ".join(texts)
        trip_lines = "\n".join(f'  - ({t["s"]}, {t["v"]}, {t["o"]})' for t in sec["triplets"])
        parts.append(f"[{sec['id']}]\ncâu gốc: {text_block}\ntriplet:\n{trip_lines}")

    if not parts:
        return [{"id": s["id"], "path": s.get("path"), "propositions": s.get("propositions", []),
                  "triplets": _fallback_triplets(s)} for s in sections]

    result = llm_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Các section cần đối chiếu:\n\n" + "\n\n".join(parts)},
    ])

    sec_by_id = {s["id"]: s for s in sections}
    kp_by_id: dict[str, list[dict]] = {}
    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        sid = item.get("id", "")
        if sid not in sec_by_id:
            continue
        triplets = []
        for t in item.get("triplets", []):
            if not isinstance(t, dict):
                continue
            s = (t.get("s") or "").strip()
            v = (t.get("v") or "").strip()
            o = (t.get("o") or "").strip()
            if not (s and v and o):
                continue
            s_kps = [k.strip() for k in t.get("s_keyphrases", []) if isinstance(k, str) and k.strip()] or [s.lower()]
            v_kps = [k.strip() for k in t.get("v_keyphrases", []) if isinstance(k, str) and k.strip()] or [v.lower()]
            o_kps = [k.strip() for k in t.get("o_keyphrases", []) if isinstance(k, str) and k.strip()] or [o.lower()]
            triplets.append({
                "s": s, "s_keyphrases": s_kps,
                "v": v, "v_keyphrases": v_kps,
                "o": o, "o_keyphrases": o_kps,
            })
        kp_by_id[sid] = triplets

    output = []
    for sec in sections:
        sid = sec["id"]
        triplets = kp_by_id.get(sid) or _fallback_triplets(sec)
        output.append({
            "id": sid,
            "path": sec.get("path"),
            "propositions": sec.get("propositions", []),
            "triplets": triplets,
        })
    return output


def save(path: str, data: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--threads",    type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit",      type=int, default=0, help="0 = xử lý tất cả")
    parser.add_argument("--ids",        default="", help="Chỉ xử lý các id này, cách nhau bởi dấu phẩy")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    if args.ids:
        wanted = set(args.ids.split(","))
        sections = [s for s in sections if s["id"] in wanted]
    elif args.limit:
        sections = sections[:args.limit]

    done: dict[str, dict] = {}
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for item in json.load(f):
                done[item["id"]] = item

    to_process = [s for s in sections if s["id"] not in done]
    print(f"Total: {len(sections)} | Done: {len(done)} | Remaining: {len(to_process)}")

    if to_process:
        batches = [to_process[i:i+args.batch_size] for i in range(0, len(to_process), args.batch_size)]
        lock   = threading.Lock()
        counts = {"done": 0, "triplets": 0, "fail": 0}

        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = {ex.submit(extract_batch, b): b for b in batches}
            for fut in as_completed(futures):
                counts["done"] += 1
                try:
                    results = fut.result()
                    with lock:
                        for r in results:
                            done[r["id"]] = r
                        counts["triplets"] += sum(len(r["triplets"]) for r in results)
                        if counts["done"] % 5 == 0 or counts["done"] == len(batches):
                            ordered = [done[s["id"]] for s in sections if s["id"] in done]
                            save(args.output, ordered)
                    print(
                        f"\r[{counts['done']}/{len(batches)}] "
                        f"triplets={counts['triplets']} fail={counts['fail']}",
                        end=""
                    )
                except Exception as e:
                    counts["fail"] += 1
                    print(f"\n[FAIL] {e}")

        ordered = [done[s["id"]] for s in sections if s["id"] in done]
        save(args.output, ordered)
        print(f"\n\nDone. {len(ordered)} sections, {counts['triplets']} triplets → {args.output}")
    else:
        print("Không có section mới. Thoát.")


if __name__ == "__main__":
    main()
