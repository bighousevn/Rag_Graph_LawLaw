"""
Test so sánh: trích triplet 100% bằng LLM, KHÔNG dùng câu đã tách từ VnCoreNLP —
chỉ đưa câu gốc (rewritten_proposition) cho LLM, để xem VnCoreNLP segment có thực sự
cần thiết hay không.

Input:  ../material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json
        (chỉ lấy 100 section từ index 50 đến 149, tức s51 → s150)
Output: output/triplets_pure_llm_test.json — cùng schema triplets_raw.json
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

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "..", "material_for_triplets",
                              "2_sections_rewritten_nghi_dinh_168_2024_1.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "triplets_pure_llm_test.json")

BATCH_SIZE = 10

SYSTEM_PROMPT = """\
Bạn là chuyên gia trích xuất triplet ngữ nghĩa cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

Với mỗi mệnh đề pháp lý dưới đây, hãy trích xuất TẤT CẢ các triplet (subject, verb, object) thể
hiện HÀNH VI pháp lý cốt lõi trong câu — chủ thể thực hiện, hành vi, đối tượng của hành vi.

QUAN TRỌNG: bạn sẽ nhận NHIỀU section trong 1 lượt. Áp dụng đầy đủ TẤT CẢ quy tắc bên dưới cho
MỖI section một cách ĐỘC LẬP và NHẤT QUÁN như thể mỗi section được xử lý riêng lẻ — TUYỆT ĐỐI
không được rút gọn/bỏ sót quy tắc ở các section phía sau chỉ vì đã áp dụng đầy đủ ở section đầu
tiên. Hai section có cấu trúc câu giống hệt nhau (chỉ khác mức định lượng, vd "chưa vượt quá
50mg" so với "vượt quá 50mg") PHẢI cho ra cùng một dạng triplet chuẩn hoá.

QUY TẮC BẮT BUỘC:
1. subject/object PHẢI là danh từ NGUYÊN TỬ (atomic) — không được ghép chủ thể+hành vi thành
   một cụm. Ví dụ "người điều khiển phương tiện" TUYỆT ĐỐI không phải 1 object/subject — phải
   tách thành 3 phần: subject "Người", verb "Điều khiển", object "Phương tiện".
2. Viết subject/verb/object dạng chuẩn hoá, viết hoa chữ cái đầu (Title Case), ngắn gọn, không
   kèm định ngữ/số lượng cụ thể không cần thiết (vd bỏ "hai bên", "trong vòng 10 ngày"...).
   Object cũng phải atomic — không được nhét cả mệnh đề phụ vào object (vd "Xe không đúng quy
   định" SAI, phải tách "Xe" làm object + phần "không đúng quy định" bỏ qua nếu không cốt lõi,
   hoặc tách thành triplet phụ riêng nếu bản thân là hành vi khác).
3. Một câu có thể sinh NHIỀU triplet nối tiếp nhau nếu diễn đạt một chuỗi hành vi (dạng mạng
   lưới) — ví dụ câu "Người sử dụng còi bị cấm tại khu đông dân cư" sinh 2 triplet nối tiếp:
   (Người, Sử dụng, Còi) và (Còi, Bị cấm tại, Khu đông dân cư).
4. Tự bỏ qua chi tiết không cốt lõi: số tiền phạt, thời hạn cụ thể, liệt kê minh hoạ phụ, các
   yếu tố không ảnh hưởng tới bản chất hành vi. Không có bước lọc nào khác làm việc này thay bạn.
5. Nếu một mệnh đề không chứa hành vi pháp lý rõ ràng nào (thuần thủ tục, định nghĩa phạm vi...)
   thì trả về "triplets": [].
6. Câu diễn đạt kiểu ĐO LƯỜNG/ĐỊNH LƯỢNG một chất trong cơ thể — ví dụ "trong máu hoặc hơi thở có
   nồng độ cồn", "có nồng độ cồn vượt quá X" — PHẢI được quy về đúng bản chất hành vi cốt lõi là
   Người SỬ DỤNG chất đó, KHÔNG được giữ nguyên cả cụm đo lường làm object. Sai (object không
   atomic, lẫn cả câu vào 1 cụm): (Người, Điều khiển, Xe trên đường có nồng độ cồn). Đúng (tách
   thành 2 triplet độc lập, atomic): (Người, Điều khiển, Xe) và (Người, Sử dụng, Cồn). Áp dụng
   tương tự cho các chất khác được đo lường trong cơ thể (ma túy, chất kích thích...).
7. RẤT NHIỀU mệnh đề trong bộ dữ liệu này dùng chung một KHUNG CÂU boilerplate lặp lại gần như
   nguyên văn ở hàng trăm điều khoản khác nhau, dạng: "Người điều khiển [danh sách loại xe] vi
   phạm quy tắc giao thông đường bộ bị xử phạt, trừ điểm giấy phép lái xe KHI [hành vi cụ thể]."
   — Phần khung này (chủ thể + "vi phạm quy tắc giao thông đường bộ" + "bị xử phạt"/"trừ điểm
   giấy phép lái xe") giống hệt nhau ở mọi điều khoản, KHÔNG giúp phân biệt điều khoản này với
   điều khoản khác → BỎ QUA, không sinh triplet cho phần khung này (không cần
   "(Người, Vi phạm, Quy tắc giao thông đường bộ)" hay "(Người, Bị xử phạt, Giấy phép lái xe)").
   PHẦN DUY NHẤT cần trích luôn là HÀNH VI CỤ THỂ nằm sau "khi"/"do"/"vì"/"khi thực hiện hành vi"
   — đây mới là nội dung phân biệt được điều khoản này với điều khoản khác, TUYỆT ĐỐI không được
   bỏ sót phần này dù nó nằm ở cuối câu dài. Vẫn giữ 1 triplet (Người, Điều khiển, <loại xe>) nếu
   câu có nêu loại xe, vì đó là subject/object thật sự dùng xuyên suốt đoạn văn.
8. Nếu bạn thấy nhiều mệnh đề trong cùng lượt gọi có PHẦN KHUNG giống hệt nhau nhưng phần "khi
   ..." khác nhau, thì các mệnh đề đó BẮT BUỘC phải cho ra các triplet hành-vi-cụ-thể KHÁC NHAU
   tương ứng — nếu bạn thấy mình sắp trả về cùng một bộ triplet cho nhiều mệnh đề có phần "khi..."
   khác nhau, dừng lại và đọc kỹ lại phần khác biệt đó trước khi trả lời.

VÍ DỤ MINH HOẠ về văn phong/độ chi tiết mong muốn (không nhất thiết liên quan input thực tế):
(Người, Sử dụng, Điện thoại)
(Người, Sử dụng, Cồn)
(Người, Chiếm dụng, Lòng đường)
(Người, Sử dụng, Chất kích thích)
(Pháp luật, Cấm, Chất kích thích)
(Người, Không chấp hành, Hiệu lệnh)
(Đèn giao thông, Ra, Hiệu lệnh)
(Người, Đi, Ngược chiều)
(Người, Điều khiển, Ô tô)
(Người, Sở hữu, Giấy phép lái xe)
(Giấy phép lái xe, Hết, Hạn sử dụng)
(Ô tô, Vận chuyển, Hành khách)
(Ô tô, Giữ, Khoảng cách an toàn)
(Ô tô, Xin, Vượt)

Trả về JSON:
{
  "results": [
    {
      "id": "s117",
      "triplets": [
        {"s": "Người", "v": "Điều khiển", "o": "Ô tô"},
        {"s": "Người", "v": "Sử dụng", "o": "Cồn"}
      ]
    },
    {
      "id": "s118",
      "triplets": []
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


def extract_batch(sections: list[dict]) -> list[dict]:
    """sections: list of {id, path, rewritten_propositions}. Chỉ dùng câu gốc, không segment."""
    parts = []
    for sec in sections:
        props = [p for p in sec.get("rewritten_propositions", []) if p and p.strip()]
        if props:
            lines = "\n".join(f"  - {p}" for p in props)
            parts.append(f"[{sec['id']}]\n{lines}")

    if not parts:
        return []

    result = llm_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Các mệnh đề cần trích xuất:\n\n" + "\n\n".join(parts)},
    ])

    sec_by_id = {s["id"]: s for s in sections}
    output = []
    for item in result.get("results", []):
        sid = item.get("id", "")
        sec = sec_by_id.get(sid)
        if not sec:
            continue
        triplets = []
        for t in item.get("triplets", []):
            s = (t.get("s") or "").strip()
            v = (t.get("v") or "").strip()
            o = (t.get("o") or "").strip()
            if s and v and o:
                triplets.append({"s": s, "v": v, "o": o})
        output.append({
            "id": sid,
            "path": sec.get("path"),
            "rewritten_propositions": sec.get("rewritten_propositions", []),
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
    parser.add_argument("--input",       default=DEFAULT_INPUT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index",   type=int, default=1085)
    parser.add_argument("--threads",     type=int, default=4)
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        all_sections = json.load(f)
    sections = all_sections[args.start_index:args.end_index]
    sections = [s for s in sections if s.get("rewritten_propositions")]
    print(f"Testing {len(sections)} sections (index {args.start_index}-{args.end_index}), "
          f"100% LLM, no VnCoreNLP segment.")

    done: dict[str, dict] = {}
    batches = [sections[i:i+args.batch_size] for i in range(0, len(sections), args.batch_size)]
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
                print(f"\r[{counts['done']}/{len(batches)}] triplets={counts['triplets']} fail={counts['fail']}", end="")
            except Exception as e:
                counts["fail"] += 1
                print(f"\n[FAIL] {e}")

    ordered = [done[s["id"]] for s in sections if s["id"] in done]
    save(args.output, ordered)
    print(f"\n\nDone. {len(ordered)} sections, {counts['triplets']} triplets → {args.output}")


if __name__ == "__main__":
    main()
