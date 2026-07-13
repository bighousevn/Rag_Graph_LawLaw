"""
Bước 2: Rewrite mỗi section thành mệnh đề chuẩn hóa Chủ thể–Hành vi–Đối tượng.
Input:  material_for_triplets/1_sections_*.json
Output: material_for_triplets/2_sections_rewritten_*.json
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
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "material_for_triplets/1_sections_nghi_dinh_168_2024_1.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json")

SYSTEM_PROMPT = """Bạn là một trợ lý phân tích ngôn ngữ pháp lý Việt Nam chuyên nghiệp.
Nhiệm vụ của bạn là đọc đoạn văn bản pháp luật (text_content) được cung cấp và DIỄN ĐẠT LẠI thành câu văn gọn gàng hơn, phục vụ việc điều hướng câu hỏi người dùng tới đúng điều khoản (không phải để trả lời trực tiếp câu hỏi).

Mục tiêu là giữ lại đúng phần nội dung mô tả CHỦ THỂ - HÀNH VI/QUAN HỆ - ĐỐI TƯỢNG, loại bỏ phần thông tin không cần thiết cho việc điều hướng. Cụ thể:

1. LƯỢC BỎ HOÀN TOÀN CHẾ TÀI/HẬU QUẢ PHÁP LÝ — không phân biệt LOẠI hay SỐ LIỆU. Chế tài là hậu quả áp dụng SAU KHI đã xác định vi phạm, không phải nội dung hành vi/quan hệ cần điều hướng — hầu hết các điểm/khoản trong cùng một Điều đều lặp lại cùng loại chế tài, giữ lại không giúp phân biệt điều khoản nào với điều khoản nào:
   - Xoá sạch mọi cụm chỉ hình thức xử phạt/biện pháp khắc phục, dù có kèm số liệu hay không, ví dụ: "bị phạt tiền (từ 200.000 đến 400.000 đồng)", "bị trừ (2) điểm giấy phép lái xe", "bị tước quyền sử dụng giấy phép lái xe (từ 10 đến 12 tháng)", "bị tịch thu phương tiện", "bị tạm giữ phương tiện", "buộc khôi phục lại tình trạng ban đầu" — xoá toàn bộ cụm này, không giữ lại chút nào.
   - Xoá luôn các cụm khung/chapeau lặp lại ở mọi điểm/khoản cùng Điều, không mang nội dung hành vi riêng của điều khoản này, ví dụ: "vi phạm quy tắc giao thông đường bộ", "thực hiện một trong các hành vi vi phạm sau đây".
   - Chỉ giữ lại HÀNH VI + ĐỐI TƯỢNG + ĐIỀU KIỆN áp dụng (xem mục 2). TẤT CẢ thông tin còn lại ngoài chế tài/khung câu — kể cả khi section không mô tả hành vi vi phạm cụ thể (định nghĩa, phạm vi áp dụng, thủ tục, thẩm quyền...) — vẫn PHẢI được diễn đạt lại bình thường, KHÔNG được bỏ qua hay trả về rỗng. Việc đánh giá section nào hữu ích cho đồ thị tri thức sẽ làm ở bước khác sau này, không phải ở bước này.
   - Tham chiếu chéo tới điều/khoản/điểm khác (vd "quy định tại điểm e khoản 5 Điều này", "theo khoản 1 Điều này") — lược bỏ cụm tham chiếu này, diễn đạt lại hành vi một cách tự nhiên mà không cần trỏ tới điều/khoản/điểm khác, nhưng PHẢI giữ lại phần nội dung hành vi/đối tượng còn lại của câu.
   - Danh sách LIỆT KÊ NHIỀU địa điểm/trường hợp cụ thể (3 trường hợp trở lên nối bằng dấu phẩy) mà đều chỉ là các biến thể tương đương của CÙNG một hành vi, trong CÙNG một điểm/khoản (cùng một địa chỉ điều khoản) — KHÔNG cần liệt kê hết từng trường hợp cụ thể, NHƯNG PHẢI khái quát hoá thành một cụm chung giữ đúng TÍNH CHẤT CÓ ĐIỀU KIỆN của hành vi, KHÔNG ĐƯỢC xoá hẳn điều kiện đó (xoá hẳn sẽ biến một hành vi CÓ ĐIỀU KIỆN mới là vi phạm thành một hành vi luôn luôn vi phạm — SAI lệch nội dung pháp lý). Vd "Quay đầu xe ở phần đường dành cho người đi bộ qua đường, trên cầu, đầu cầu, gầm cầu vượt, ngầm, tại nơi đường bộ giao nhau cùng mức với đường sắt, đường hẹp, đường dốc, đoạn đường cong tầm nhìn bị che khuất, trên đường một chiều" → khái quát thành "quay đầu xe tại nơi cấm quay đầu xe" (giữ được rằng đây là hành vi có điều kiện — chỉ vi phạm khi ở nơi cấm — mà không cần liệt kê hết 10 vị trí). Lý do: các địa điểm cụ thể đều thuộc cùng một địa chỉ điều khoản nên liệt kê hết không có giá trị điều hướng, nhưng việc hành vi này CÓ ĐIỀU KIỆN (chỉ vi phạm ở nơi cấm, không phải luôn vi phạm) là nội dung pháp lý cốt lõi PHẢI giữ lại.
   ⚠️ QUY TẮC NÀY KHÔNG ÁP DỤNG cho danh sách LOẠI CHỦ THỂ/LOẠI PHƯƠNG TIỆN cụ thể (xem mục 3 bên dưới) — chỉ áp dụng cho địa điểm/trường hợp xảy ra hành vi.
2. GIỮ NGUYÊN ý nghĩa hành vi/quan hệ pháp lý cốt lõi: chủ thể, hành động, đối tượng. KHÔNG giữ hình thức xử phạt/biện pháp khắc phục dưới bất kỳ hình thức nào (xem mục 1). Chỉ giữ lại hoàn cảnh/điều kiện khi đó là một ĐIỀU KIỆN ÁP DỤNG DUY NHẤT, bắt buộc, làm thay đổi việc hành vi có bị coi là vi phạm hay không (vd "khi xe đang chạy" trong "không thắt dây đai an toàn khi xe đang chạy" — nếu xe không chạy thì không vi phạm) — phân biệt với trường hợp liệt kê nhiều địa điểm/trường hợp tương đương ở mục 1 phía trên (trường hợp đó nên lược bỏ).
3. KHÔNG cần tách thành nhiều câu đơn nhất (atomic) như liệt kê nhiều đối tượng — giữ nguyên cấu trúc liệt kê bằng dấu phẩy/"và"/"hoặc" trong CÙNG MỘT câu (vd "xe ô tô, xe chở người bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô" giữ nguyên trong 1 câu, không tách thành 4 câu riêng).
   ⚠️ TUYỆT ĐỐI KHÔNG được rút gọn/khái quát hoá danh sách LOẠI CHỦ THỂ hoặc LOẠI PHƯƠNG TIỆN cụ thể xuống thành một từ chung chung (vd KHÔNG được viết "xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô" thành chỉ còn "xe" — phải giữ NGUYÊN VẸN toàn bộ danh sách liệt kê loại phương tiện/chủ thể y như câu mở đầu Điều/Khoản đã nêu, mỗi khi câu hành vi nhắc tới chủ thể đó). Đây là lỗi hay gặp — luôn kiểm tra lại câu output có giữ đủ danh sách loại xe/chủ thể như bản gốc hay không trước khi trả lời.
4. CHỈ tách thành nhiều câu khi text_content chứa các quy định/hành vi THỰC SỰ độc lập, không thể gộp tự nhiên thành một câu liệt kê (vd 2 hình thức xử phạt bổ sung khác nhau áp dụng cho 2 nhóm hành vi khác nhau). Trong đa số trường hợp, kết quả chỉ nên có 1 câu duy nhất.
5. Giải quyết các trường hợp ẩn chủ ngữ/hành động/đối tượng bằng cách làm rõ chủ thể chính, không suy diễn thêm nội dung ngoài văn bản gốc.
6. Diễn đạt bằng tiếng Việt chuẩn xác, mạch lạc, tự nhiên.
7. KHÔNG viết hai hành vi nối tiếp theo kiểu mệnh đề rút gọn dùng CHUNG chủ ngữ ngầm nếu chủ ngữ thực sự của chúng khác nhau. Phải viết RÕ chủ ngữ cho từng hành vi:
   - Nếu cùng một chủ thể A thực hiện cả hai hành vi → lặp lại chủ ngữ, nối bằng "và": "A làm X và A gây Y".
   - Nếu hành vi thứ hai (Y) là hậu quả do chính đối tượng/sự việc X gây ra → tách thành mệnh đề riêng: "Người điều khiển xe mở cửa xe; việc mở cửa xe gây tai nạn giao thông thì bị phạt tiền."
   Tránh đại từ mơ hồ ("hành vi này", "việc đó") — luôn dùng cụm danh từ cụ thể.
8. Khi CHỦ THỂ là danh sách liệt kê nhiều thực thể (vd "cá nhân, tổ chức"), PHẢI nối vế CUỐI CÙNG bằng "và" để tránh VnCoreNLP hiểu lầm vế cuối thành động từ.
   Quy tắc này CHỈ áp dụng cho danh sách CHỦ THỂ, KHÔNG áp dụng cho danh sách ĐỐI TƯỢNG.
9. Nếu câu mở đầu Điều/Khoản đã nêu rõ chủ thể CỤ THỂ, PHẢI dùng chủ thể đó thay cho cụm chung (vd "người điều khiển phương tiện") xuất hiện trong câu hành vi. Lý do: cụm chung làm mất khả năng phân biệt điều khoản.

Ví dụ few-shot:
Input:  "Điều 21. ... người điều khiển xe ô tô tải, máy kéo ... 5. Phạt tiền ... người điều khiển phương tiện chở hàng vượt quá khối lượng cho phép."
Output: {"results": {"<id>": ["Người điều khiển xe ô tô tải, máy kéo và các loại xe tương tự xe ô tô vận chuyển hàng hóa chở hàng vượt quá khối lượng cho phép."]}}
(chú ý: "bị phạt tiền" ở cuối câu đã bị xoá — đây là chế tài. Danh sách loại xe "xe ô tô tải, máy kéo và các loại xe tương tự xe ô tô" được GIỮ NGUYÊN VẸN, không rút gọn thành "xe".)

Input:  "Điều 6. Xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm quy tắc giao thông đường bộ. 9. Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;"
Output: {"results": {"<id>": ["Người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông."]}}
(chú ý: chủ thể PHẢI lấy đầy đủ danh sách loại xe cụ thể nêu ở đầu Điều — "xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô" — TUYỆT ĐỐI không được rút gọn thành "người điều khiển xe" chung chung. Phủ định "không chấp hành" giữ nguyên. Chế tài "phạt tiền", khung câu "vi phạm quy tắc giao thông đường bộ" bị xoá.)

Input:  "Điều 6. Xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm quy tắc giao thông đường bộ. 5. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: l) Chuyển hướng không nhường quyền đi trước cho: người đi bộ, xe lăn của người khuyết tật qua đường tại nơi có vạch kẻ đường dành cho người đi bộ; xe thô sơ đang đi trên phần đường dành cho xe thô sơ;"
Output: {"results": {"<id>": ["Người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô chuyển hướng không nhường quyền đi trước cho người đi bộ, xe lăn của người khuyết tật qua đường tại nơi có vạch kẻ đường dành cho người đi bộ; xe thô sơ đang đi trên phần đường dành cho xe thô sơ."]}}
(chú ý: "vi phạm quy tắc giao thông đường bộ", "bị trừ điểm giấy phép lái xe", "bị phạt tiền" biến mất hoàn toàn; danh sách loại xe ở đầu câu vẫn giữ nguyên vẹn.)

Bạn nhận DANH SÁCH NHIỀU đoạn văn, mỗi đoạn có "id" riêng. Xử lý ĐỘC LẬP từng đoạn.
Trả về JSON duy nhất:
{
  "results": {
    "<id_1>": ["câu diễn đạt lại 1", "câu 2 (chỉ khi thực sự cần tách)"],
    "<id_2>": ["câu diễn đạt lại"]
  }
}
PHẢI trả về đầy đủ TẤT CẢ id có trong input."""


def llm_call(messages: list, max_retries: int = 5) -> dict:
    backoff = 2
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2


def rewrite_batch(batch: list[dict]) -> dict[str, list[str]]:
    """One LLM call for a batch of sections. Returns {id: [propositions]}."""
    parts = [f'--- id: {s["id"]} ---\n{s["text_content"]}' for s in batch]
    user_msg = "Các đoạn văn bản cần xử lý:\n\n" + "\n\n".join(parts)
    result = llm_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    return result.get("results", {})


def save(path: str, data: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--threads",    type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit",      type=int, default=0)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    if args.limit:
        sections = sections[:args.limit]

    # Resume: load already-processed sections
    done: dict[str, dict] = {}
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for item in json.load(f):
                if "rewritten_propositions" in item:
                    done[item["id"]] = item

    to_process = [s for s in sections if s["id"] not in done]
    print(f"Total: {len(sections)} | Done: {len(done)} | Remaining: {len(to_process)}")
    if not to_process:
        return

    batches = [to_process[i:i + args.batch_size] for i in range(0, len(to_process), args.batch_size)]
    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}

    def process(batch):
        try:
            results = rewrite_batch(batch)
        except Exception as e:
            print(f"\n[FAIL] batch {[s['id'] for s in batch]}: {e}")
            results = {}
        output = {}
        for s in batch:
            props = results.get(s["id"])
            enriched = {**s, "rewritten_propositions": props if isinstance(props, list) else []}
            output[s["id"]] = enriched
        return output

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(process, b): b for b in batches}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            result = fut.result()
            with lock:
                for sid, item in result.items():
                    done[sid] = item
                    if item["rewritten_propositions"]:
                        counts["ok"] += 1
                    else:
                        counts["fail"] += 1
                ordered = [done[s["id"]] for s in sections if s["id"] in done]
                save(args.output, ordered)
            print(f"[{completed}/{len(batches)}] ok={counts['ok']} fail={counts['fail']}", end="\r")

    print(f"\nDone. ok={counts['ok']} fail={counts['fail']} → {args.output}")


if __name__ == "__main__":
    main()
