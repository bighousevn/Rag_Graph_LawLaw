"""
Bước 2 (pass 1/2): LLM trích raw triplet (s, v, o) trực tiếp từ mỗi section.

KHÔNG ràng buộc ontology có sẵn (ontology chưa tồn tại ở bước này). LLM tự đặt tên
canonical (Title Case) cho subject/relation/object và tự quyết định phần nào của câu
là nhiễu — không có bước lọc cứng nào can thiệp trước.

Đây chỉ là PASS 1 — trích nội dung, chưa cần atomic hoàn hảo. PASS 2 (3_refine_atomicity.py)
sẽ rà soát lại và tách nốt các subject/object còn chứa giới từ/quan hệ từ ẩn bên trong.

Input:  output/segmented.json   (từ 1_segment.py)
Output: output/triplets_raw.json
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
DEFAULT_INPUT  = os.path.join(BASE_DIR, "output", "segmented.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "triplets_raw.json")

BATCH_SIZE = 10   # sections per LLM call (batch lớn hơn dễ khiến LLM "lười" áp dụng quy tắc
                  # nhất quán cho các section phía sau — đã kiểm chứng qua test thủ công)


SYSTEM_PROMPT = """\
Bạn là chuyên gia trích xuất triplet ngữ nghĩa cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

Với mỗi mệnh đề pháp lý dưới đây, hãy trích xuất TẤT CẢ các triplet (subject, verb, object) thể
hiện HÀNH VI pháp lý cốt lõi trong câu — chủ thể thực hiện, hành vi, đối tượng của hành vi.

QUAN TRỌNG: bạn sẽ nhận NHIỀU section trong 1 lượt. Áp dụng đầy đủ TẤT CẢ quy tắc bên dưới cho
MỖI section một cách ĐỘC LẬP và NHẤT QUÁN như thể mỗi section được xử lý riêng lẻ — TUYỆT ĐỐI
không được rút gọn/bỏ sót quy tắc ở các section phía sau chỉ vì đã áp dụng đầy đủ ở section đầu
tiên. Hai section có cấu trúc câu giống hệt nhau (chỉ khác mức định lượng, vd "chưa vượt quá
50mg" so với "vượt quá 50mg") PHẢI cho ra cùng một dạng triplet chuẩn hoá.

Mỗi mệnh đề được cho ở 2 dạng: câu gốc và câu đã tách từ (từ ghép nối bằng "_", ví dụ
"giấy_phép lái_xe"). Dùng câu tách từ để nhận đúng ranh giới cụm từ ghép khi cần, nhưng luôn
viết subject/verb/object cuối cùng bằng tiếng Việt tự nhiên (có dấu cách, KHÔNG gạch dưới).

QUY TẮC BẮT BUỘC:
1. subject/object BẮT BUỘC là danh từ NGUYÊN TỬ — không được ghép chủ thể+hành vi thành 1 cụm.
   Ví dụ "người điều khiển phương tiện" TUYỆT ĐỐI không phải 1 object/subject — phải tách thành
   3 phần: subject "Người", verb "Điều khiển", object "Phương tiện". Tương tự "người có thẩm
   quyền" TUYỆT ĐỐI không phải 1 subject — tách thành subject "Người" + triplet riêng (Người, Có,
   Thẩm quyền). Đây là yêu cầu CƠ BẢN, áp dụng chặt chẽ, không có ngoại lệ.
   Riêng với trường hợp PHỨC TẠP hơn — object là 1 cụm dài chứa giới từ/quan hệ từ ẩn bên trong
   (vd "Quy định về nhường đường", "Xe dưới tốc độ tối thiểu") — không bắt buộc phải tách hết ở
   bước này, cứ giữ nguyên cả cụm làm object tạm thời, sẽ có một bước riêng phía sau rà soát và
   tách nốt phần này. Chỉ nới lỏng cho trường hợp phức tạp này, KHÔNG áp dụng nới lỏng cho case
   cơ bản (chủ thể+hành vi gộp chung) ở trên.
2. Nếu VnCoreNLP tách từ quá vụn (ví dụ "xe" và "ô_tô" bị tách rời dù cùng chỉ một loại xe),
   hãy tự ghép lại thành đúng 1 thực thể theo nghĩa câu gốc (ví dụ "Ô tô"), đừng để bị vỡ chỉ vì
   ranh giới tách từ.
3. Viết subject/verb/object dạng chuẩn hoá, viết hoa chữ cái đầu (Title Case), ngắn gọn, không
   kèm định ngữ/số lượng cụ thể không cần thiết (vd bỏ "hai bên", "trong vòng 10 ngày"...).
4. Một câu có thể sinh NHIỀU triplet nối tiếp nhau nếu diễn đạt một chuỗi hành vi (dạng mạng
   lưới) — ví dụ câu "Người sử dụng còi bị cấm tại khu đông dân cư" sinh 2 triplet nối tiếp:
   (Người, Sử dụng, Còi) và (Còi, Bị cấm tại, Khu đông dân cư).
5. Tự bỏ qua chi tiết không cốt lõi: số tiền phạt, thời hạn cụ thể, liệt kê minh hoạ phụ, các
   yếu tố không ảnh hưởng tới bản chất hành vi. Không có bước lọc nào khác làm việc này thay bạn.
6. Nếu một mệnh đề không chứa hành vi pháp lý rõ ràng nào (thuần thủ tục, định nghĩa phạm vi...)
   thì trả về "triplets": [].
7. Câu diễn đạt kiểu ĐO LƯỜNG/ĐỊNH LƯỢNG một chất trong cơ thể — ví dụ "trong máu hoặc hơi thở có
   nồng độ cồn", "có nồng độ cồn vượt quá X" — PHẢI được quy về đúng bản chất hành vi cốt lõi là
   Người SỬ DỤNG chất đó, KHÔNG được giữ nguyên cả cụm đo lường làm object. Sai (object không
   atomic, lẫn cả câu vào 1 cụm): (Người, Điều khiển, Xe trên đường có nồng độ cồn). Đúng (tách
   thành 2 triplet độc lập, atomic): (Người, Điều khiển, Xe) và (Người, Sử dụng, Cồn). Áp dụng
   tương tự cho các chất khác được đo lường trong cơ thể (ma túy, chất kích thích...).
8. RẤT NHIỀU mệnh đề trong bộ dữ liệu này dùng chung một KHUNG CÂU boilerplate lặp lại gần như
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
9. Nếu bạn thấy nhiều mệnh đề trong cùng lượt gọi có PHẦN KHUNG giống hệt nhau nhưng phần "khi
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
    """sections: list of {id, path, propositions:[{text, segmented}]}."""
    parts = []
    for sec in sections:
        lines = []
        for p in sec.get("propositions", []):
            text = (p.get("text") or "").strip()
            seg  = (p.get("segmented") or "").strip()
            if text:
                lines.append(f"  - gốc: {text}\n    tách từ: {seg}")
        if lines:
            parts.append(f"[{sec['id']}]\n" + "\n".join(lines))

    if not parts:
        return []

    result = llm_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Các mệnh đề cần trích xuất:\n\n" + "\n\n".join(parts)},
    ])

    sec_by_id = {s["id"]: s for s in sections}
    output = []
    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        sid = item.get("id", "")
        sec = sec_by_id.get(sid)
        if not sec:
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
    sections = [s for s in sections if s.get("propositions")]

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
