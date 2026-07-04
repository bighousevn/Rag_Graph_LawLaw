"""
Bước 3 (pass 2/2): Rà soát tính NGUYÊN TỬ (atomic) của các triplet đã trích ở pass 1.

Nhiệm vụ DUY NHẤT của LLM ở bước này: với mỗi triplet {s, v, o} đã có sẵn, kiểm tra subject/
object có còn chứa QUAN HỆ ẨN bên trong không (một động từ hoặc giới từ/quan hệ từ nối 2 khái
niệm, ví dụ "Quy định về nhường đường", "Người có thẩm quyền"). Nếu có → tách thành chuỗi 2+
triplet nối tiếp. Nếu đã atomic → giữ nguyên, không đổi.

Input là danh sách triplet ngắn (không phải câu luật dài) — nhiệm vụ hẹp hơn nhiều so với pass 1,
nên batch được nhiều section/lượt gọi hơn mà không bị dilution.

Input:  output/triplets_raw.json       (từ 2_extract_triplets.py)
Output: output/triplets_refined.json
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
DEFAULT_INPUT  = os.path.join(BASE_DIR, "output", "triplets_raw.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "triplets_refined.json")

BATCH_SIZE = 25   # input ngắn hơn nhiều so với pass 1 (chỉ triplet, không phải câu dài) nên
                  # batch được lớn hơn mà không bị dilution như pass 1


SYSTEM_PROMPT = """\
Bạn là chuyên gia rà soát tính NGUYÊN TỬ (atomic) của các triplet (subject, verb, object) đã
được trích sẵn từ văn bản pháp lý giao thông Việt Nam.

NHIỆM VỤ DUY NHẤT: với mỗi triplet {s, v, o} cho sẵn, kiểm tra subject và object — nếu một trong
hai còn chứa QUAN HỆ ẨN bên trong (một động từ hoặc giới từ/quan hệ từ nối 2 khái niệm khác
nhau), hãy TÁCH triplet đó thành chuỗi 2 (hoặc nhiều) triplet nối tiếp. Nếu subject/object ĐÃ
atomic (là một khái niệm/vật cụ thể, không thể phân rã thêm mà không mất nghĩa), GIỮ NGUYÊN
không đổi — không viết lại, không đổi tên, không tự suy diễn thêm nội dung mới.

PHÉP THỬ atomic: tự hỏi "cụm này có thể viết lại thành chính nó = (X, quan hệ, Y) mà không mất
nghĩa không?". Nếu CÓ → chưa atomic, phải tách. Nếu KHÔNG → giữ nguyên.

VÍ DỤ CẦN TÁCH (sai → đúng):
(Người có thẩm quyền, Áp dụng, Hình thức tước quyền sử dụng giấy phép)
  →  (Người, Có, Thẩm quyền), (Người, Áp dụng, Hình thức tước quyền), (Quyền, Sử dụng, Giấy phép)
(Người, Không tuân thủ, Quy định về nhường đường)
  →  (Người, Không tuân thủ, Quy định), (Quy định, Về, Nhường đường)
(Người, Chạy, Xe dưới tốc độ tối thiểu)
  →  (Người, Chạy, Xe), (Xe, Dưới, Tốc độ tối thiểu)
(Người điều khiển, Vi phạm, Quy tắc)
  →  (Người, Điều khiển, ?), (Người, Vi phạm, Quy tắc)  — lưu ý: nếu object của "Điều khiển"
     không rõ trong câu gốc, bỏ triplet "Điều khiển" không rõ object, chỉ giữ lại phần chắc chắn.

VÍ DỤ ĐÃ ATOMIC — GIỮ NGUYÊN, KHÔNG được tách thêm dù cụm có 2-3 từ:
(Người, Điều khiển, Xe ô tô)
(Người, Vi phạm, Quy tắc giao thông đường bộ)
(Người, Không báo hiệu, Đèn khẩn cấp)
(Người, Sử dụng, Cồn)
(Người, Tước, Chứng chỉ hành nghề)
(Người, Ra, Quyết định xử phạt)
(Xe, Dưới, Tốc độ tối thiểu)

QUY TẮC:
1. Chỉ tách khi THỰC SỰ có quan hệ ẩn (verb hoặc giới từ/quan hệ từ) bên trong subject/object.
   KHÔNG tách chỉ vì cụm có nhiều hơn 1-2 từ — cụm danh từ ghép thuần tuý (không chứa quan hệ,
   như "Xe ô tô", "Chứng chỉ hành nghề", "Quyết định xử phạt", "Tốc độ tối thiểu", "Đèn khẩn
   cấp") KHÔNG được tách.
2. Giữ nguyên các triplet ĐÃ atomic — copy y nguyên, không viết lại/đổi tên/đổi thứ tự.
3. Không tự thêm hành vi/triplet mới ngoài nội dung đã có sẵn trong input — chỉ được TÁCH LẠI
   cấu trúc của triplet đã cho, không suy diễn thêm nội dung không có trong input.
4. Nếu tách 1 triplet thành nhiều, tổng số triplet của section đó tăng lên tương ứng — vẫn giữ
   đúng field "id".

Trả về JSON:
{
  "results": [
    {
      "id": "s117",
      "triplets": [
        {"s": "Người", "v": "Điều khiển", "o": "Ô tô"},
        {"s": "Người", "v": "Sử dụng", "o": "Cồn"}
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


def refine_batch(sections: list[dict]) -> list[dict]:
    """sections: list of {id, path, propositions, triplets:[{s,v,o}]}."""
    parts = []
    for sec in sections:
        triplets = sec.get("triplets", [])
        if not triplets:
            continue
        lines = "\n".join(f'  - ({t["s"]}, {t["v"]}, {t["o"]})' for t in triplets)
        parts.append(f"[{sec['id']}]\n{lines}")

    if not parts:
        return [{"id": s["id"], "path": s.get("path"), "propositions": s.get("propositions", []),
                  "triplets": s.get("triplets", [])} for s in sections]

    result = llm_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Các triplet cần rà soát:\n\n" + "\n\n".join(parts)},
    ])

    sec_by_id = {s["id"]: s for s in sections}
    refined_by_id: dict[str, list[dict]] = {}
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
            if s and v and o:
                triplets.append({"s": s, "v": v, "o": o})
        refined_by_id[sid] = triplets

    output = []
    for sec in sections:
        sid = sec["id"]
        # Nếu LLM bỏ sót section này trong response, giữ nguyên triplet gốc thay vì mất dữ liệu.
        triplets = refined_by_id.get(sid, sec.get("triplets", []))
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
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    if args.limit:
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
            futures = {ex.submit(refine_batch, b): b for b in batches}
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
