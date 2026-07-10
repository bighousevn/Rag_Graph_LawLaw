"""
Bước 2 (pass 1/3): LLM trích raw triplet (s, v, o) trực tiếp từ mỗi section (text_content gốc).

Mục tiêu: trích ĐẦY ĐỦ mạng lưới "ai/cái gì tác động gì lên ai/cái gì" trong điều luật — kể cả
hậu quả pháp lý (phạt tiền, trừ điểm giấy phép lái xe...), không lọc bớt vì lặp lại ở nhiều điều
khoản khác. Mục đích cuối là ROUTING câu hỏi người dùng tới đúng địa chỉ điều khoản — câu trả lời
chi tiết vẫn dựa vào văn bản gốc qua địa chỉ đó, nên triplet ưu tiên khả năng khớp/tìm kiếm hơn là
tuyệt đối chính xác ngữ nghĩa (vd bỏ luôn yếu tố phủ định "không").

Đây chỉ là PASS 1 — trích nội dung, chưa cần atomic hoàn hảo. PASS 2 (3_refine_atomicity.py)
rà soát lại và tách nốt các subject/object còn chứa giới từ/quan hệ từ ẩn bên trong.

Input:  output/segmented.json   (từ 1_segment.py, đã đọc từ text_content gốc)
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

from _normalize import strip_negation_passive

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "output", "segmented.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "triplets_raw.json")

BATCH_SIZE = 10   # prompt dài hơn ontology_2 (nhiều quy tắc hơn) — giữ batch nhỏ để giảm dilution


SYSTEM_PROMPT = """\
Bạn là chuyên gia trích xuất triplet ngữ nghĩa cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

Với mỗi section (văn bản luật gốc, có thể dài và có cấu trúc liệt kê), hãy trích xuất ĐẦY ĐỦ các
triplet (subject, verb, object) thể hiện MỌI quan hệ "ai/cái gì tác động gì lên ai/cái gì" — cả
hành vi vi phạm LẪN hậu quả pháp lý đi kèm (phạt tiền, trừ điểm giấy phép lái xe, tước quyền sử
dụng giấy phép, tạm giữ phương tiện...). Mục đích cuối cùng là dựng một đồ thị để ĐỊNH TUYẾN câu
hỏi người dùng tới đúng địa chỉ điều khoản — vì vậy ưu tiên trích được NHIỀU triplet có khả năng
khớp từ khoá cao, hơn là tối giản/lọc bớt.

QUAN TRỌNG: bạn sẽ nhận NHIỀU section trong 1 lượt. Áp dụng đầy đủ TẤT CẢ quy tắc bên dưới cho MỖI
section một cách ĐỘC LẬP và NHẤT QUÁN như thể mỗi section được xử lý riêng lẻ — TUYỆT ĐỐI không
được rút gọn/bỏ sót quy tắc ở các section phía sau chỉ vì đã áp dụng đầy đủ ở section đầu tiên.

Mỗi section được cho ở 2 dạng: câu gốc và câu đã tách từ (từ ghép nối bằng "_", ví dụ "giấy_phép
lái_xe"). Dùng câu tách từ để nhận đúng ranh giới cụm từ ghép khi cần, nhưng luôn viết
subject/verb/object cuối cùng bằng tiếng Việt tự nhiên (có dấu cách, KHÔNG gạch dưới).

═══════════════ QUY TẮC ═══════════════

── A. Tính NGUYÊN TỬ (atomic) ──
1. subject/object BẮT BUỘC là danh từ nguyên tử — không gộp chủ thể+hành vi thành 1 cụm KHI hành
   vi đó có TÂN NGỮ THAY ĐỔI theo từng câu (hành vi cụ thể đang được điều chỉnh). "người điều khiển
   phương tiện" TUYỆT ĐỐI không phải 1 subject/object — tách 3 phần: subject "Người", verb "Điều
   khiển", object "Phương tiện" (vì "phương tiện" thay đổi theo từng điều khoản — ô tô, xe máy...).
   NGOẠI LỆ — VAI TRÒ/THUỘC TÍNH CỐ ĐỊNH: nếu cụm mô tả một VAI TRÒ pháp lý cố định, không mang
   tân ngữ thay đổi theo câu (vd "người có thẩm quyền" — "thẩm quyền" không đổi theo từng điều
   khoản, đây là một chức danh/vai trò xuyên suốt văn bản, giống "Người đi bộ" là 1 concept riêng
   theo CLAUDE.md) → GIỮ NGUYÊN cụm đó làm 1 subject/object atomic (vd "Người có thẩm quyền"),
   ĐỒNG THỜI vẫn bổ sung thêm 1 triplet riêng thể hiện quan hệ nền: (Người, Có, Thẩm quyền). Tức
   là sinh CẢ HAI: subject "Người có thẩm quyền" dùng cho các triplet hành vi, VÀ triplet phụ
   (Người, Có, Thẩm quyền) để giữ liên kết ngữ nghĩa về gốc.
2. Nếu object là 1 cụm chứa quan hệ ẩn (giới từ/động từ nối 2 khái niệm) → tách thành chuỗi 2
   triplet nối tiếp (object của triplet đầu = subject của triplet sau):
   "đường có biển báo cấm" → (Người, Đi vào, Đường) + (Đường, Có, Biển báo cấm)
3. Nếu object là DANH SÁCH nhiều thực thể nối bằng dấu phẩy/"và"/"hoặc" (không phải 1 thực thể
   ghép, mà là nhiều thực thể riêng biệt liệt kê chung) → tách thành nhiều triplet song song, mỗi
   thực thể 1 triplet riêng, dùng chung subject+verb:
   "tạm giữ giấy phép, chứng chỉ hành nghề đã hết giá trị sử dụng" →
     (Người có thẩm quyền, Tạm giữ, Giấy phép) + (Người có thẩm quyền, Tạm giữ, Chứng chỉ hành nghề)
     + (Giấy phép, Hết, Giá trị sử dụng) + (Chứng chỉ hành nghề, Hết, Giá trị sử dụng)
   Ưu tiên tách phần danh sách (2 triplet đầu) ngay cả khi không kịp tách hết mệnh đề định ngữ
   phía sau — tách được 1 phần vẫn tốt hơn không tách gì.
4. Nếu VnCoreNLP tách từ quá vụn (vd "xe" và "ô_tô" bị tách rời dù cùng chỉ một loại xe), tự ghép
   lại thành đúng 1 thực thể theo nghĩa câu gốc (vd "Ô tô").
5. Viết subject/verb/object dạng chuẩn hoá, viết hoa chữ cái đầu (Title Case), ngắn gọn.

── B. Bị động và phủ định ──
6. QUAN HỆ BỊ ĐỘNG: bỏ "bị"/"được" khỏi tên relation, viết relation ở dạng chủ động — subject
   VẪN GIỮ NGUYÊN là đối tượng chịu tác động (không đổi sang chủ thể thực hiện):
     "bị phạt tiền" → (Người, Phạt, Tiền)          (không phải "Bị phạt")
     "bị trừ điểm giấy phép lái xe" → (Người, Trừ, Điểm giấy phép lái xe)
7. PHỦ ĐỊNH: bỏ hẳn "không"/"chưa"/"chẳng" khỏi verb VÀ object — viết ở dạng khẳng định thuần tuý,
   kể cả khi việc này khiến 2 hành vi trái ngược nhau (có/không có) cho ra CÙNG MỘT triplet. Đây
   là chủ đích: mục tiêu là định tuyến câu hỏi tới điều khoản, không phải mô tả chính xác tuyệt
   đối — chấp nhận đánh đổi độ chính xác ngữ nghĩa để tăng khả năng khớp truy vấn.
     "không có giấy phép lái xe" → (Người, Có, Giấy phép lái xe)
     "không quan sát" → (Người, Thực hiện, Quan sát)   (xem thêm mục D)
     "không giảm tốc độ" → (Người, Giảm, Tốc độ)

── C. Đầy đủ, không lọc bớt ──
8. TRÍCH ĐẦY ĐỦ mọi quan hệ — không bỏ qua một quan hệ chỉ vì nó lặp lại ở nhiều điều khoản khác
   (vd "vi phạm quy tắc giao thông đường bộ", "bị phạt tiền", "bị trừ điểm giấy phép lái xe" xuất
   hiện ở hầu hết mọi Điểm của Điều 6 — vẫn phải trích đầy đủ ở MỌI section, không chỉ giữ lại
   phần "đặc trưng" của riêng section đó).
9. Bỏ GIÁ TRỊ ĐỊNH LƯỢNG cụ thể (số tiền, %, mg, số ngày...) nhưng GIỮ bản chất hành vi/hậu quả mà
   giá trị đó gắn vào. "phạt tiền từ 6.000.000 đến 8.000.000 đồng" → bỏ số tiền, giữ (Người, Phạt,
   Tiền).
10. Bỏ tham chiếu chéo tới điều khoản khác (vd "trừ các hành vi vi phạm quy định tại điểm đ khoản
    11 Điều này") — không tạo triplet cho phần này.

── D. Danh sách hành vi và verb không có tân ngữ ──
11. Khi câu có cấu trúc "thực hiện một trong các hành vi vi phạm sau đây: ..." theo sau là danh
    sách hành vi nối bằng dấu chấm phẩy ";" hoặc "hoặc" — đây là các hành vi THAY THẾ NHAU (chỉ
    cần 1 trong nhiều). PHẢI tách mỗi hành vi/nhóm hành vi trong danh sách thành TRIPLET RIÊNG,
    KHÔNG được gộp cả danh sách thành 1 object dài — để một câu hỏi khớp bất kỳ hành vi nào trong
    danh sách đều tìm ra đúng điều khoản.
12. Trong danh sách đó, nếu một hành vi là ĐỘNG TỪ KHÔNG CÓ TÂN NGỮ rõ ràng và không thể quy chiếu
    an toàn về danh từ đã nêu trong CÙNG câu (vd "quan sát" — không nói rõ quan sát gì, và không
    có danh từ nào trong câu để gán vào) → dùng verb "Thực hiện" + chính hành vi đó (danh-động-từ-
    hoá, viết nguyên cụm) làm object: (Người, Thực hiện, Quan sát). Nếu tân ngữ có thể quy chiếu an
    toàn (vd "dừng lại" → "xe" đã nêu ở đầu câu "điều khiển xe...") thì dùng trực tiếp verb+object
    đó, KHÔNG cần "Thực hiện": (Người, Dừng, Xe).

═══════════════ VÍ DỤ ĐẦY ĐỦ (input thật → output đúng) ═══════════════

--- Ví dụ 1 ---
Input: "Điều 6. Xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn
bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm
quy tắc giao thông đường bộ. 6. Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với người điều
khiển xe thực hiện một trong các hành vi vi phạm sau đây: c) Điều khiển xe trên đường mà trong máu
hoặc hơi thở có nồng độ cồn nhưng chưa vượt quá 50 miligam/100 mililít máu hoặc chưa vượt quá 0,25
miligam/1 lít khí thở;"

Output:
(Người, Điều khiển, Ô tô)
(Người, Vi phạm, Quy tắc giao thông đường bộ)
(Người, Trừ, Điểm giấy phép lái xe)
(Người, Phạt, Tiền)
(Người, Sử dụng, Cồn)

--- Ví dụ 2 ---
Input: "Điều 6. Xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn
bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm
quy tắc giao thông đường bộ. 10. Phạt tiền từ 20.000.000 đồng đến 22.000.000 đồng đối với người
điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: a) Điều khiển xe không quan sát,
giảm tốc độ hoặc dừng lại để bảo đảm an toàn theo quy định mà gây tai nạn giao thông; điều khiển
xe chạy quá tốc độ quy định gây tai nạn giao thông; dừng xe, đỗ xe, quay đầu xe, lùi xe, tránh xe,
vượt xe, chuyển hướng, chuyển làn đường không đúng quy định gây tai nạn giao thông; không đi đúng
phần đường, làn đường, không giữ khoảng cách an toàn giữa hai xe theo quy định gây tai nạn giao
thông hoặc đi vào đường có biển báo hiệu có nội dung cấm đi vào đối với loại phương tiện đang điều
khiển gây tai nạn giao thông, trừ các hành vi vi phạm quy định tại điểm đ khoản 11 Điều này;"

Output:
(Người, Điều khiển, Ô tô)
(Người, Vi phạm, Quy tắc giao thông đường bộ)
(Người, Trừ, Điểm giấy phép lái xe)
(Người, Phạt, Tiền)
(Người, Thực hiện, Quan sát)
(Người, Giảm, Tốc độ)
(Người, Dừng, Xe)
(Người, Chạy quá, Tốc độ quy định)
(Người, Dừng, Xe)
(Người, Đỗ, Xe)
(Người, Quay đầu, Xe)
(Người, Lùi, Xe)
(Người, Tránh, Xe)
(Người, Vượt, Xe)
(Người, Chuyển, Hướng)
(Người, Chuyển, Làn đường)
(Người, Đi đúng, Phần đường)
(Người, Đi đúng, Làn đường)
(Người, Giữ, Khoảng cách an toàn)
(Người, Đi vào, Đường)
(Đường, Có, Biển báo cấm)
(Người, Gây, Tai nạn giao thông)

(Lưu ý: "trừ các hành vi vi phạm quy định tại điểm đ khoản 11 Điều này" KHÔNG tạo triplet — đây
là tham chiếu chéo, theo quy tắc C9.)

Trả về JSON:
{
  "results": [
    {"id": "s117", "triplets": [{"s": "Người", "v": "Điều khiển", "o": "Ô tô"}, ...]},
    {"id": "s118", "triplets": []}
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
        {"role": "user",   "content": "Các section cần trích xuất:\n\n" + "\n\n".join(parts)},
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
            s = strip_negation_passive(t.get("s") or "")
            v = strip_negation_passive(t.get("v") or "")
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
