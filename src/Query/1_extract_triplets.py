"""
Bước 1: Chuẩn hoá câu hỏi tự do của người dùng thành văn bản kiểu luật (loại bỏ tình tiết dư
thừa, văn phong kể chuyện), rồi tách thành các sub-triplet (s, v, o) — 2 lượt gọi GPT-4o RIÊNG
BIỆT, tách bạch 2 việc (giống cách Building_KG/2_rewrite_sections.py tách riêng "viết lại" khỏi
"trích triplet" ở phía xây kho ngữ liệu).

Input:  input/1_question.txt              câu hỏi tự do của người dùng (1 file text thô)
Output: output/1_normalized_question.txt  văn bản đã chuẩn hoá kiểu luật
        output/question_triplets.json     [{"s":.., "v":.., "o":.., "group":1}, ...] — đúng
                                           format mà 2_embedding_triplets.py đọc vào (Bước 2)

"group" (số nguyên) đánh dấu sub-triplet nào thuộc cùng 1 tình huống/hành vi độc lập trong câu hỏi
— câu hỏi ghép nhiều tình huống không liên quan (vd "không mang giấy tờ VÀ không có gương") sẽ có
nhiều group khác nhau. Bước 3 (3_query_flow.py) truy vấn RIÊNG từng group rồi hợp kết quả lại, để
không bị rớt mất tình huống nào khi câu hỏi cần nhiều điều khoản khác nhau mới trả lời đủ.

Nguyên tắc trích triplet bám theo tinh thần Building_KG/ontology/EXTRACTION_RULES.md (atomic
subject/object, relation LUÔN chuẩn hoá khẳng định, "*" cho thành phần khuyết/ẩn số) nhưng rút
gọn — chỉ xử lý 1 câu hỏi, không cần đầy đủ 16 rule chi tiết dùng cho việc xây cả kho ngữ liệu.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "input", "1_question.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUT_NORMALIZED = os.path.join(OUTPUT_DIR, "1_normalized_question.txt")
OUT_TRIPLETS   = os.path.join(OUTPUT_DIR, "question_triplets.json")

MODEL = "gpt-4o"

NORMALIZE_SYSTEM_PROMPT = """\
Bạn là chuyên gia pháp lý giao thông đường bộ Việt Nam. Nhiệm vụ: ĐỌC HIỂU tình huống thực tế
người dùng đang mô tả (dù diễn đạt lộn xộn, dùng từ đời thường, mô tả hình ảnh/thị giác cụ thể),
SUY LUẬN ra bản chất hành vi giao thông đang được hỏi, rồi viết lại thành 1 câu NGẮN GỌN đúng
văn phong pháp luật (giống cách diễn đạt trong Nghị định xử phạt vi phạm hành chính giao thông
đường bộ).

QUAN TRỌNG NHẤT — đây KHÔNG phải bài tập diễn giải lại câu chữ (paraphrase bề mặt), mà là bài
tập ĐỌC HIỂU + SUY LUẬN NGỮ CẢNH:
- Xác định rõ: (1) chủ thể là ai (người đi bộ / người điều khiển loại xe gì), (2) hành vi cốt lõi
  đang thực hiện là gì, (3) bối cảnh/điều kiện giao thông tại thời điểm đó thuộc PHẠM TRÙ PHÁP LÝ
  nào (loại biển báo, loại đèn tín hiệu, loại đường...) — ở mức khái niệm, KHÔNG mô tả lại chi
  tiết thị giác cụ thể.
- Luật không quan tâm chi tiết hình ảnh cụ thể (màu sắc, hình dạng, vị trí mũi tên trên đèn tín
  hiệu, hoa văn biển báo...) — chỉ quan tâm điều kiện đó thuộc LOẠI phạm trù pháp lý nào. Ví dụ:
  "mũi tên xanh cho rẽ phải nhưng đèn chính đỏ" → về bản chất pháp lý chỉ là "tại nơi có đèn tín
  hiệu giao thông" — KHÔNG cần mô tả lại cấu hình mũi tên/màu sắc trong câu viết lại.
- Nếu câu hỏi mô tả 1 tình huống dài dòng nhưng bản chất chỉ hỏi về 1 hành vi + 1 điều kiện, hãy
  QUY VỀ đúng hành vi + điều kiện đó, bỏ hết chi tiết không phải thuật ngữ pháp lý.
- NGƯỢC LẠI, nếu câu hỏi thật sự hỏi về NHIỀU hành vi/tình huống KHÁC NHAU, KHÔNG liên quan tới
  nhau (vd vừa hỏi không mang giấy tờ, vừa hỏi không có gương — 2 lỗi độc lập) → PHẢI giữ ĐỦ cả
  các hành vi đó trong câu viết lại, nối bằng "và"/";", TUYỆT ĐỐI không rút gọn/bỏ sót hành vi nào
  chỉ vì muốn câu ngắn gọn.

QUY TẮC KHÁC:
1. Loại bỏ hoàn toàn: xưng hô/kể chuyện ("nhà em", "ông bà em", "em ơi", "các anh chị cho em
   hỏi"), cảm thán, tình tiết không ảnh hưởng tới việc xác định hành vi.
2. KHÔNG bịa thêm SỰ KIỆN không có trong câu hỏi gốc (vd không tự thêm "gây tai nạn" nếu câu hỏi
   không nhắc tới) — nhưng ĐƯỢC PHÉP, và BẮT BUỘC, suy luận ra khái niệm pháp lý tương ứng với
   chi tiết thị giác/đời thường đã có sẵn trong câu hỏi (vd "mũi tên đèn xanh/đỏ" → khái niệm
   "đèn tín hiệu giao thông" — đây là quy đổi đúng bản chất, không phải bịa thêm).
3. Quy đổi từ ngữ đời thường sang thuật ngữ pháp lý giao thông chuẩn (vd "say xỉn" → "sử dụng
   rượu, bia"/"có nồng độ cồn"; "xe máy" giữ nguyên vì đã là thuật ngữ chuẩn).
4. Giữ đúng CỰC TÍNH (phủ định/khẳng định) như câu hỏi gốc — không tự ý đảo ngược nghĩa.
5. Câu viết lại nên bám khuôn mẫu phổ biến trong luật: "Người điều khiển [loại xe] [hành vi] tại
   nơi có [điều kiện/loại biển báo/loại đèn tín hiệu]." — điều chỉnh linh hoạt theo tình huống.

VÍ DỤ:
Input: "Các anh chị cho e hỏi mũi tên xanh đc rẽ phải nhưng lại có mũi lên đi thẳng và rẽ phải đỏ
thì oto có đc rẽ phải khi đèn đỏ ko ạ"
Suy luận: chủ thể = người điều khiển ô tô; hành vi cốt lõi = rẽ phải; bối cảnh = tại nơi có đèn
tín hiệu giao thông (mũi tên xanh/đỏ chỉ là chi tiết thị giác của đèn tín hiệu, không phải khái
niệm pháp lý riêng biệt cần giữ lại).
Output: "Người điều khiển ô tô rẽ phải tại nơi có đèn tín hiệu giao thông."

Trả về đúng JSON:
{"suy_luan": "chủ thể ... | hành vi ... | bối cảnh ...", "normalized_text": "câu đã viết lại"}"""

EXTRACT_SYSTEM_PROMPT = """\
Bạn là chuyên gia trích xuất triplet ngữ nghĩa cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

Từ đoạn văn bản đã chuẩn hoá, tách thành các sub-triplet (subject, verb, object) ĐỘC LẬP, atomic —
áp dụng đúng tinh thần các quy tắc sau (rút gọn từ EXTRACTION_RULES.md, dùng cho xây kho ngữ liệu).

MỖI sub-triplet PHẢI kèm thêm field "group" (số nguyên, bắt đầu từ 1) — đánh số theo MỆNH ĐỀ ĐỘC
LẬP mà sub-triplet đó thuộc về (đúng ranh giới chẻ mệnh đề ở rule 2 dưới đây): nếu văn bản hỏi về
NHIỀU tình huống/hành vi vi phạm KHÔNG LIÊN QUAN tới nhau (nối bằng "và"/","/"với lại"...) → mỗi
tình huống 1 group riêng. Các sub-triplet cùng mô tả 1 hành vi/tình huống (dù bị tách nhỏ do atomic
hoá) → giữ CHUNG 1 group. Nếu có 1 sub-triplet bối cảnh chung áp dụng cho MỌI tình huống (vd đang
điều khiển loại xe gì) → LẶP LẠI sub-triplet đó ở TỪNG group liên quan (không chỉ ghi 1 lần), để
mỗi group tự nó là 1 bộ sub-triplet ĐẦY ĐỦ, độc lập truy vấn được. Nếu văn bản chỉ có 1 tình huống
duy nhất → toàn bộ sub-triplet dùng chung group 1.

1. Subject/object phải là danh từ NGUYÊN TỬ — không gộp chủ thể+hành vi thành 1 cụm. "Người điều
   khiển xe ô tô" KHÔNG phải 1 subject — tách "Người", verb "Điều khiển", object "Ô tô".
2. Nếu văn bản mô tả NHIỀU hành vi/quan hệ độc lập (nối bằng dấu phẩy, ";", "và", hoặc nhiều mệnh
   đề) → tách thành NHIỀU sub-triplet riêng, KHÔNG gộp thành 1 object dài.
3. LUÔN CHUẨN HOÁ relation VỀ DẠNG KHẲNG ĐỊNH — bỏ "không"/"chưa"/"chẳng" khỏi tên relation (vd
   "chưa đủ kinh nghiệm" → relation "Đủ", object "Kinh nghiệm"; "không có giấy phép" → relation
   "Có", object "Giấy phép"). Ý nghĩa phủ định coi như đã được người dùng biết rõ (họ đang hỏi về
   đúng tình huống vi phạm đó), không cần giữ lại trong tên relation.
4. Nếu 1 thành phần (s hoặc o) KHÔNG được nêu trong văn bản — hoặc chính là ẩn số câu hỏi đang cần
   tìm (vd "thì bị gì", "phạt bao nhiêu", "ai xử lý") — dùng "*" thay cho thành phần đó. Verb (v)
   không dùng "*" — luôn phải xác định được hành vi cốt lõi.
5. Không thêm ngưỡng số liệu cụ thể (nồng độ cồn, tốc độ, khung giờ...) làm object riêng — chỉ
   giữ hành vi cốt lõi.
6. Viết s/v/o dạng chuẩn hoá, viết hoa chữ cái đầu, ngắn gọn — giống tên gọi trong luật, không
   phải nguyên văn cả cụm dài từ câu hỏi.
7. Verb NỘI TẠI (tự thân thay đổi trạng thái của chính chủ thể/xe — quay đầu, chuyển hướng, rẽ
   trái/phải, dừng, đỗ, lùi, tránh) → object PHẢI là chính chủ thể/xe đang thực hiện, KHÔNG PHẢI
   địa điểm/điều kiện đi kèm. Vd "rẽ phải tại nơi có đèn tín hiệu" → (Người, Rẽ phải, Ô tô), TUYỆT
   ĐỐI không gộp thành (Người, Rẽ phải, Nơi có đèn tín hiệu giao thông).
8. Cụm "tại nơi có Z" (Z = biển báo hiệu/đèn tín hiệu giao thông/vạch kẻ đường/loại đường...) LUÔN
   tách thành 1 sub-triplet RIÊNG dùng relation "Tại": (Người, Tại, Z) — độc lập với triplet hành
   vi chính (điều khiển/rẽ/dừng...), không gộp Z vào làm object của verb khác.

VÍ DỤ (1 tình huống duy nhất → toàn bộ chung group 1):
Input: "Người điều khiển xe ô tô có nồng độ cồn trong máu bị xử phạt như thế nào?"
Output: {"triplets": [
  {"s": "Người", "v": "Điều khiển", "o": "Ô tô", "group": 1},
  {"s": "Người", "v": "Sử dụng", "o": "Cồn", "group": 1},
  {"s": "*", "v": "Xử phạt", "o": "*", "group": 1}
]}

Input: "Tổ chức kinh doanh vận tải sử dụng lái xe điều khiển xe khách chưa đủ số năm kinh nghiệm
theo quy định."
Output: {"triplets": [
  {"s": "Tổ chức kinh doanh vận tải", "v": "Sử dụng", "o": "Lái xe", "group": 1},
  {"s": "Lái xe", "v": "Điều khiển", "o": "Xe khách", "group": 1},
  {"s": "Lái xe", "v": "Đủ", "o": "Số năm kinh nghiệm", "group": 1}
]}

Input: "Người điều khiển ô tô rẽ phải tại nơi có đèn tín hiệu giao thông."
Output: {"triplets": [
  {"s": "Người", "v": "Điều khiển", "o": "Ô tô", "group": 1},
  {"s": "Người", "v": "Rẽ phải", "o": "Ô tô", "group": 1},
  {"s": "Người", "v": "Tại", "o": "Đèn tín hiệu giao thông", "group": 1}
]}

VÍ DỤ (2 tình huống KHÔNG liên quan tới nhau → 2 group riêng, mỗi group lặp lại sub-triplet bối
cảnh chung "Điều khiển Xe" để tự nó đủ nghĩa):
Input: "Người điều khiển xe máy không mang giấy tờ xe và xe không có gương bị phạt bao nhiêu?"
Output: {"triplets": [
  {"s": "Người", "v": "Điều khiển", "o": "Xe máy", "group": 1},
  {"s": "Người", "v": "Mang", "o": "Giấy tờ xe", "group": 1},
  {"s": "Người", "v": "Điều khiển", "o": "Xe máy", "group": 2},
  {"s": "Xe máy", "v": "Có", "o": "Gương", "group": 2}
]}

Trả về đúng JSON: {"triplets": [{"s": "...", "v": "...", "o": "...", "group": 1}]}"""


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_question(text: str) -> tuple[str, str]:
    """Trả về (suy_luận, normalized_text). "suy_luận" chỉ để log/debug — ép model xác định rõ
    chủ thể/hành vi/bối cảnh TRƯỚC khi viết câu cuối, không phải phần dùng tiếp ở bước sau."""
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": NORMALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = resp.choices[0].message.content
    result = json.loads(content) if content else {}
    return result.get("suy_luan", ""), (result.get("normalized_text") or "").strip()


def extract_triplets(text: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = resp.choices[0].message.content
    result = json.loads(content) if content else {}
    return result.get("triplets", [])


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        question = f.read().strip()
    if not question:
        log(f"File câu hỏi rỗng: {INPUT_FILE}")
        return
    log(f"Câu hỏi gốc:\n  {question}")

    log("\nĐang chuẩn hoá câu hỏi (GPT-4o)...")
    suy_luan, normalized = normalize_question(question)
    log(f"Suy luận: {suy_luan}")
    log(f"Đã chuẩn hoá:\n  {normalized}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUT_NORMALIZED, "w", encoding="utf-8") as f:
        f.write(normalized)

    log("\nĐang tách triplet (GPT-4o)...")
    triplets = extract_triplets(normalized)

    with open(OUT_TRIPLETS, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    n_groups = len({t.get("group", 1) for t in triplets})
    log(f"\nĐã tách {len(triplets)} sub-triplet, {n_groups} group:")
    for t in triplets:
        log(f"  [group {t.get('group', 1)}] ({t.get('s')}, {t.get('v')}, {t.get('o')})")

    log(f"\nĐã lưu:\n  {OUT_NORMALIZED}\n  {OUT_TRIPLETS}")


if __name__ == "__main__":
    main()
