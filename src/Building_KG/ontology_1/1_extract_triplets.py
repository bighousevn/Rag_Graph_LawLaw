"""
Bước 1 (pass 1/4): LLM trích raw triplet (s, v, o) trực tiếp từ mỗi section.

Input:  ../material_for_triplets/2_sections_rewritten_*.json, field `rewritten_propositions`
        (KHÔNG dùng text_content gốc, KHÔNG VnCoreNLP — theo đúng CLAUDE.md: rewritten_propositions
        đã lược bỏ chế tài/tham chiếu chéo, đủ sạch để GPT-4o-mini xử lý trực tiếp).
Output: output/triplets_raw.json

Đây chỉ là PASS 1 — trích nội dung theo các pattern cố định của miền, chưa cần atomic hoàn hảo.
PASS 2 (2_refine_atomicity.py) rà soát lại và tách nốt các subject/object còn chứa quan hệ ẩn.

Khác với các lần thử trước (xem CLAUDE.md "Quy trình tách triplet atomic"):
  - KHÔNG strip phủ định/bị động khỏi relation — "Không nhường" phải khác "Nhường", nếu không
    hai điều khoản đối lập sẽ bị trộn triplet.
  - KHÔNG trích "hậu quả pháp lý" (phạt tiền, trừ điểm...) — rewritten_propositions đã bỏ chế tài.
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
MATERIALS_DIR  = os.path.join(BASE_DIR, "..", "material_for_triplets")
DEFAULT_INPUT  = os.path.join(MATERIALS_DIR, "2_sections_rewritten_nghi_dinh_168_2024_1.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "triplets_raw.json")

BATCH_SIZE = 8   # nhiều quy tắc trong prompt — giữ batch nhỏ để giảm dilution


SYSTEM_PROMPT = """\
Bạn là chuyên gia trích xuất triplet ngữ nghĩa cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

Với mỗi section (1 hoặc nhiều mệnh đề pháp lý đã được chuẩn hoá Chủ thể–Hành vi–Đối tượng), hãy
trích xuất ĐẦY ĐỦ các triplet (subject, verb, object) thể hiện quan hệ "ai/cái gì tác động gì lên
ai/cái gì" đúng ý hành vi cốt lõi của mệnh đề. Mục đích cuối cùng là dựng một đồ thị để ĐỊNH TUYẾN
câu hỏi người dùng tới đúng địa chỉ điều khoản.

QUAN TRỌNG: bạn sẽ nhận NHIỀU section trong 1 lượt, thường là các Điểm/Khoản LIÊN TIẾP nhau trong
cùng 1 Điều (có thể là các điểm "anh em" trong cùng Khoản). Áp dụng đầy đủ TẤT CẢ quy tắc bên dưới
cho MỖI section một cách ĐỘC LẬP và NHẤT QUÁN — TUYỆT ĐỐI không rút gọn quy tắc ở section phía sau
chỉ vì đã áp dụng đầy đủ ở section đầu. Nếu 2 section trong cùng lượt gần giống hệt nhau nhưng khác
đúng một chi tiết (vd "có vạch kẻ đường" vs "không có vạch kẻ đường"), chi tiết đó BẮT BUỘC phải
tạo ra khác biệt rõ trong bộ triplet của từng section — không được để 2 section ra triplet giống hệt.

═══════════════ QUY TẮC ═══════════════

── 0. Chẻ mệnh đề TRƯỚC KHI làm bất cứ điều gì khác ──
0. Trước khi xác định relation/object, chẻ câu thành các mệnh đề ĐỘC LẬP theo dấu chấm phẩy ";",
   "hoặc", VÀ theo ranh giới verb-nội-tại/verb-ngoại-tác ngay cả khi không có dấu câu phân tách rõ
   ràng. Câu dạng "VERB1 X không VERB2 Y" trong đó VERB1 là verb nội tại (tự thân, xem mục B) và
   VERB2 là verb ngoại tác (tác động lên Y) LÀ HAI MỆNH ĐỘC LẬP — PHẢI tách thành 2 triplet riêng
   dùng 2 verb riêng, KHÔNG được gộp thành 1 tên relation ghép dài (vd "Chuyển hướng không nhường
   quyền đi trước" là SAI — phải tách "chuyển hướng" [nội tại, object = chính xe] và "không nhường
   quyền đi trước cho Y" [ngoại tác, object = Y] thành 2 triplet riêng biệt, xem Ví dụ 2).

── A. Tính NGUYÊN TỬ (atomic) ──
1. subject/object BẮT BUỘC là danh từ nguyên tử — không gộp chủ thể+hành vi thành 1 cụm KHI hành
   vi đó có TÂN NGỮ THAY ĐỔI theo từng câu. "người điều khiển xe ô tô" TUYỆT ĐỐI không phải 1
   subject/object — tách 3 phần: subject "Người", verb "Điều khiển", object "Ô tô". Danh sách liệt
   kê các tên đồng nhóm (vd "xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có
   gắn động cơ và các loại xe tương tự xe ô tô") gộp thành 1 object duy nhất "Ô tô" (không tách
   Concept riêng cho từng tên trong nhóm).
   NGOẠI LỆ — VAI TRÒ/THUỘC TÍNH CỐ ĐỊNH: nếu cụm mô tả một VAI TRÒ pháp lý cố định không đổi theo
   từng điều khoản (vd "người có thẩm quyền", "người đi bộ") → GIỮ NGUYÊN cụm đó làm 1 subject/
   object atomic, ĐỒNG THỜI vẫn bổ sung triplet phụ thể hiện quan hệ nền (vd (Người, Có, Thẩm quyền)).
2. Nếu object là 1 cụm chứa quan hệ ẩn (giới từ/động từ nối 2 khái niệm) → tách thành chuỗi 2
   triplet nối tiếp (object của triplet đầu = subject của triplet sau):
   "đường có biển báo cấm" → (Người, Đi vào, Đường) + (Đường, Có, Biển báo cấm)
3. Nếu object là DANH SÁCH nhiều thực thể RIÊNG BIỆT nối bằng dấu phẩy/"và"/"hoặc" (không phải 1
   thực thể ghép) → tách thành nhiều triplet song song, dùng chung subject+verb.
4. Viết subject/verb/object dạng chuẩn hoá, viết hoa chữ cái đầu (Title Case), ngắn gọn.

── B. Verb nội tại vs ngoại tác, GIỮ PHỦ ĐỊNH ──
5. Verb NỘI TẠI (tự thân thay đổi trạng thái của chính chủ thể/xe — quay đầu, chuyển hướng, rẽ
   trái/phải, dừng, đỗ, lùi, tránh) → object = chính chủ thể/xe đang thực hiện (vd (Người, Quay đầu,
   Ô tô)).
   Verb NGOẠI TÁC (tác động lên đối tượng khác — chuyển làn, không nhường, vượt) → object = thực
   thể bên ngoài chịu tác động (vd (Người, Không nhường, Người đi bộ)).
6. LUÔN GIỮ PHỦ ĐỊNH trong tên relation — "Không nhường" KHÁC "Nhường", "Không có" KHÁC "Có". TUYỆT
   ĐỐI không chuẩn hoá về dạng khẳng định — hai điều khoản đối lập (có/không có) phải cho ra 2 bộ
   triplet khác nhau, không được trộn lẫn.
7. Quan hệ BỊ ĐỘNG (không phải phủ định): bỏ "bị"/"được" khỏi tên relation, viết relation ở dạng
   chủ động, subject vẫn giữ nguyên là đối tượng chịu tác động: "giấy phép bị tước quyền sử dụng"
   → (Giấy phép, Tước, Quyền sử dụng).

── C. Cụm phụ theo mẫu cố định ──
8. `"X dành cho/của/thuộc Y"` → luôn tách `(X, Dành cho, Y)`, TRỪ KHI cả cụm nằm dưới phủ định-
   tồn-tại (`"không có X..."`) → BỎ HẲN, không tạo triplet, không thay thế bằng gì khác.
9. `"tại nơi có/không có Z"` → CHỈ tạo triplet `(S, Tại, Z)` khi Z THẬT SỰ tồn tại (không bị phủ
   định). Nếu là "không có Z" → không tạo triplet nào cho Z. subject của relation "Tại" luôn là
   thực thể vật lý (người/xe/biển báo), không bao giờ là một khái niệm hành động trừu tượng.
10. `"biển báo hiệu có nội dung cấm X"` → tách `(S, Tại, Biển báo hiệu)` + `(Biển báo hiệu, Cấm, X)`,
    dùng chung 1 tên "Biển báo hiệu" cho mọi loại biển. X tái dùng đúng tên hành vi đã dùng ở nhánh
    chính của câu (vd nếu hành vi chính là "Quay đầu" thì X = "Quay đầu", không đặt tên khác).
11. Ngưỡng số liệu (nồng độ cồn, khung giờ, tốc độ...) → GIỮ NGUYÊN giá trị nếu đó là yếu tố phân
    biệt với điều/khoản khác trong cùng lượt xử lý (vd 3 mốc nồng độ cồn khác nhau ở 3 Khoản khác
    nhau PHẢI cho object khác nhau, không được rút gọn về chung 1 tên "Cồn" mà mất mốc). Bỏ nếu chỉ
    là chi tiết minh hoạ không ảnh hưởng phân biệt.

── D. Lọc bỏ / danh sách hành vi / gộp đồng nghĩa ──
12. Bỏ tham chiếu chéo tới điều khoản khác (vd "trừ quy định tại điểm... khoản... Điều này") —
    không tạo triplet cho phần này. Bỏ định ngữ lặp không mang thông tin phân biệt riêng (vd "trái
    quy định" đứng một mình, không đi kèm nội dung cụ thể).
12b. Nếu 1 thực thể trong câu là ĐỒNG NGHĨA MIỀN gần nghĩa với 1 concept đã dùng trong CHÍNH câu đó
    hoặc trong ví dụ mẫu bên dưới (không phải tên gọi khác của cùng 1 vật, mà là 1 nhóm/biến thể
    thuộc cùng phạm trù hành vi) → dùng LUÔN tên concept đã có, không tạo tên mới. Ví dụ: "xe lăn
    của người khuyết tật" khi đi cùng "người đi bộ" trong cùng mệnh đề (cả hai cùng là đối tượng
    được nhường đường khi qua đường) → gộp vào object "Người đi bộ", KHÔNG tạo object riêng "Xe lăn
    của người khuyết tật".
13. Khi câu có cấu trúc liệt kê nhiều hành vi THAY THẾ NHAU nối bằng dấu chấm phẩy ";" hoặc "hoặc"
    → tách mỗi hành vi/mệnh đề độc lập thành TRIPLET RIÊNG, KHÔNG gộp cả danh sách thành 1 object dài.
14. Động từ KHÔNG CÓ TÂN NGỮ rõ ràng và không thể quy chiếu an toàn về danh từ đã nêu trong CÙNG
    câu → dùng verb "Thực hiện" + hành vi đó (danh-động-từ-hoá) làm object: (Người, Thực hiện, Quan
    sát). Nếu tân ngữ quy chiếu an toàn được (vd "dừng lại" khi "xe" đã nêu ở đầu câu) → dùng trực
    tiếp verb+object đó, không cần "Thực hiện": (Người, Dừng, Xe).

═══════════════ VÍ DỤ ĐẦY ĐỦ (input thật → output đúng) ═══════════════

--- Ví dụ 1: pattern "Biển báo hiệu + Cấm" (Khoản 4, Điểm k) ---
Input: "Người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có
gắn động cơ và các loại xe tương tự xe ô tô quay đầu xe tại nơi có biển báo hiệu có nội dung cấm
quay đầu đối với loại phương tiện đang điều khiển; điều khiển xe rẽ trái tại nơi có biển báo hiệu
có nội dung cấm rẽ trái đối với loại phương tiện đang điều khiển; điều khiển xe rẽ phải tại nơi có
biển báo hiệu có nội dung cấm rẽ phải đối với loại phương tiện đang điều khiển."

Output:
(Người, Điều khiển, Ô tô)
(Người, Quay đầu, Ô tô)
(Người, Tại, Biển báo hiệu)
(Biển báo hiệu, Cấm, Quay đầu)
(Người, Rẽ trái, Ô tô)
(Biển báo hiệu, Cấm, Rẽ trái)
(Người, Rẽ phải, Ô tô)
(Biển báo hiệu, Cấm, Rẽ phải)

--- Ví dụ 2: cặp điểm l/m — phủ định-tồn-tại của "dành cho" (Khoản 5, Điểm l và Điểm m) ---
Input điểm l: "Người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn
bánh có gắn động cơ và các loại xe tương tự xe ô tô chuyển hướng không nhường quyền đi trước cho
người đi bộ, xe lăn của người khuyết tật qua đường tại nơi có vạch kẻ đường dành cho người đi bộ;
xe thô sơ đang đi trên phần đường dành cho xe thô sơ."

Output điểm l:
(Người, Điều khiển, Ô tô)
(Người, Chuyển hướng, Ô tô)
(Người, Không nhường, Người đi bộ)
(Người đi bộ, Tại, Vạch kẻ đường)
(Vạch kẻ đường, Dành cho, Người đi bộ)
(Người, Không nhường, Xe thô sơ)
(Xe thô sơ, Tại, Phần đường)
(Phần đường, Dành cho, Xe thô sơ)

Input điểm m: "Người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn
bánh có gắn động cơ và các loại xe tương tự xe ô tô chuyển hướng không nhường đường cho các xe đi
ngược chiều; người đi bộ, xe thô sơ đang qua đường tại nơi không có vạch kẻ đường cho người đi bộ."

Output điểm m (KHÔNG có triplet nào về vạch kẻ đường — cụm nằm dưới "không có" nên bỏ hẳn theo
quy tắc C9, khác hẳn điểm l dù câu gần giống nhau):
(Người, Điều khiển, Ô tô)
(Người, Chuyển hướng, Ô tô)
(Người, Không nhường, Xe đi ngược chiều)
(Người, Không nhường, Người đi bộ)
(Người, Không nhường, Xe thô sơ)

--- Ví dụ 3: giữ ngưỡng số liệu vì phân biệt điều khoản (Khoản 6/9/11, Điểm c/a/a) ---
Input (Khoản 6 Điểm c): "Người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở
hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô điều khiển xe trên đường mà trong máu
hoặc hơi thở có nồng độ cồn nhưng chưa vượt quá 50 miligam/100 mililít máu hoặc chưa vượt quá 0,25
miligam/1 lít khí thở."

Output:
(Người, Điều khiển, Ô tô)
(Người, Sử dụng, Cồn)
(Cồn, Chưa vượt quá, 50mg/100ml máu)
(Cồn, Chưa vượt quá, 0,25mg/1L khí thở)

(Khoản 9 Điểm a — cùng cấu trúc câu, mốc khác — object của relation "Vượt quá" phải đổi theo đúng
mốc, KHÔNG được rút gọn về chung 1 object "Cồn vượt quá" mất số liệu):
(Người, Điều khiển, Ô tô)
(Người, Sử dụng, Cồn)
(Cồn, Vượt quá, 50-80mg/100ml máu)
(Cồn, Vượt quá, 0,25-0,4mg/1L khí thở)

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
    """sections: list of {id, path, document_name, rewritten_propositions}."""
    parts = []
    for sec in sections:
        props = [p.strip() for p in sec.get("rewritten_propositions", []) if p and p.strip()]
        if props:
            lines = "\n".join(f"  - {p}" for p in props)
            parts.append(f"[{sec['id']}]\n{lines}")

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
            s = (t.get("s") or "").strip()
            v = (t.get("v") or "").strip()
            o = (t.get("o") or "").strip()
            if s and v and o:
                triplets.append({"s": s, "v": v, "o": o})
        output.append({
            "id": sid,
            "path": sec.get("path"),
            "document_name": sec.get("document_name"),
            "propositions": sec.get("rewritten_propositions", []),
            "triplets": triplets,
        })

    return output


def save(path: str, data: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Bước 1: trích triplet thô từ rewritten_propositions")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--threads",    type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit",      type=int, default=0, help="0 = xử lý tất cả")
    parser.add_argument("--path-contains", default="", help="Chỉ xử lý section có path chứa chuỗi này (vd 'Điều 6')")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    if args.path_contains:
        sections = [s for s in sections if args.path_contains in (s.get("path") or "")]
    if args.limit:
        sections = sections[:args.limit]
    sections = [s for s in sections if s.get("rewritten_propositions")]

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
