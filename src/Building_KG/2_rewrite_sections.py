import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

# Load env variables from .env
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in env variables or .env file.")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """Bạn là một trợ lý phân tích ngôn ngữ pháp lý Việt Nam chuyên nghiệp.
Nhiệm vụ của bạn là đọc đoạn văn bản pháp luật (text_content) được cung cấp và DIỄN ĐẠT LẠI thành câu văn gọn gàng hơn, phục vụ việc điều hướng câu hỏi người dùng tới đúng điều khoản (không phải để trả lời trực tiếp câu hỏi).

Mục tiêu là giữ lại đúng phần nội dung mô tả CHỦ THỂ - HÀNH VI/QUAN HỆ - ĐỐI TƯỢNG, loại bỏ phần thông tin không cần thiết cho việc điều hướng. Cụ thể:

1. LƯỢC BỎ CHỈ PHẦN GIÁ TRỊ ĐỊNH LƯỢNG CỤ THỂ (số tiền, số tháng/ngày, số điểm) — đây là hậu quả/mức độ, không phải nội dung hành vi:
   - Số tiền phạt cụ thể (vd "phạt tiền từ 200.000 đồng đến 400.000 đồng" → chỉ còn "bị phạt tiền").
   - Số tháng/ngày tước quyền sử dụng cụ thể (vd "tước quyền sử dụng giấy phép lái xe từ 10 tháng đến 12 tháng" → giữ lại "bị tước quyền sử dụng giấy phép lái xe", CHỈ bỏ "từ 10 tháng đến 12 tháng").
   - Số điểm trừ cụ thể (vd "trừ 2 điểm giấy phép lái xe" → giữ lại "bị trừ điểm giấy phép lái xe", chỉ bỏ số "2").
   ⚠️ TUYỆT ĐỐI KHÔNG được lược bỏ LOẠI hình thức xử phạt/biện pháp (vd "tước quyền sử dụng giấy phép lái xe", "tịch thu phương tiện", "trừ điểm giấy phép lái xe", "tạm giữ phương tiện") — đây là nội dung quan hệ pháp lý cốt lõi cần giữ, chỉ bỏ phần SỐ LIỆU đi kèm. Không gộp/khái quát hoá các loại hình thức xử phạt khác nhau thành câu chung mơ hồ như "bị áp dụng hình thức xử phạt bổ sung" nếu văn bản gốc nêu rõ loại hình thức cụ thể.
   - Tham chiếu chéo tới điều/khoản/điểm khác (vd "quy định tại điểm e khoản 5 Điều này", "theo khoản 1 Điều này") — lược bỏ cụm tham chiếu này, diễn đạt lại hành vi một cách tự nhiên mà không cần trỏ tới điều/khoản/điểm khác, nhưng PHẢI giữ lại phần nội dung hành vi/đối tượng còn lại của câu.
   - Danh sách LIỆT KÊ NHIỀU địa điểm/trường hợp cụ thể (3 trường hợp trở lên nối bằng dấu phẩy) mà đều chỉ là các biến thể tương đương của CÙNG một hành vi, trong CÙNG một điểm/khoản (cùng một địa chỉ điều khoản) — KHÔNG cần liệt kê hết từng trường hợp cụ thể, NHƯNG PHẢI khái quát hoá thành một cụm chung giữ đúng TÍNH CHẤT CÓ ĐIỀU KIỆN của hành vi, KHÔNG ĐƯỢC xoá hẳn điều kiện đó (xoá hẳn sẽ biến một hành vi CÓ ĐIỀU KIỆN mới là vi phạm thành một hành vi luôn luôn vi phạm — SAI lệch nội dung pháp lý). Vd "Quay đầu xe ở phần đường dành cho người đi bộ qua đường, trên cầu, đầu cầu, gầm cầu vượt, ngầm, tại nơi đường bộ giao nhau cùng mức với đường sắt, đường hẹp, đường dốc, đoạn đường cong tầm nhìn bị che khuất, trên đường một chiều" → khái quát thành "quay đầu xe tại nơi cấm quay đầu xe" (giữ được rằng đây là hành vi có điều kiện — chỉ vi phạm khi ở nơi cấm — mà không cần liệt kê hết 10 vị trí). Lý do: các địa điểm cụ thể đều thuộc cùng một địa chỉ điều khoản nên liệt kê hết không có giá trị điều hướng, nhưng việc hành vi này CÓ ĐIỀU KIỆN (chỉ vi phạm ở nơi cấm, không phải luôn vi phạm) là nội dung pháp lý cốt lõi PHẢI giữ lại.
2. GIỮ NGUYÊN ý nghĩa hành vi/quan hệ pháp lý cốt lõi: chủ thể, hành động, đối tượng, LOẠI hình thức xử phạt/biện pháp khắc phục. Chỉ giữ lại hoàn cảnh/điều kiện khi đó là một ĐIỀU KIỆN ÁP DỤNG DUY NHẤT, bắt buộc, làm thay đổi việc hành vi có bị coi là vi phạm hay không (vd "khi xe đang chạy" trong "không thắt dây đai an toàn khi xe đang chạy" — nếu xe không chạy thì không vi phạm) — phân biệt với trường hợp liệt kê nhiều địa điểm/trường hợp tương đương ở mục 1 phía trên (trường hợp đó nên lược bỏ).
3. KHÔNG cần tách thành nhiều câu đơn nhất (atomic) như liệt kê nhiều đối tượng — giữ nguyên cấu trúc liệt kê bằng dấu phẩy/"và"/"hoặc" trong CÙNG MỘT câu (vd "xe ô tô, xe chở người bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô" giữ nguyên trong 1 câu, không tách thành 4 câu riêng).
4. CHỈ tách thành nhiều câu khi text_content chứa các quy định/hành vi THỰC SỰ độc lập, không thể gộp tự nhiên thành một câu liệt kê (vd 2 hình thức xử phạt bổ sung khác nhau áp dụng cho 2 nhóm hành vi khác nhau). Trong đa số trường hợp, kết quả chỉ nên có 1 câu duy nhất.
5. Giải quyết các trường hợp ẩn chủ ngữ/hành động/đối tượng bằng cách làm rõ chủ thể chính, không suy diễn thêm nội dung ngoài văn bản gốc.
6. Diễn đạt bằng tiếng Việt chuẩn xác, mạch lạc, tự nhiên.
7. KHÔNG viết hai hành vi nối tiếp theo kiểu mệnh đề rút gọn dùng CHUNG chủ ngữ ngầm nếu chủ ngữ thực sự của chúng khác nhau (vd "A làm X gây Y" — chủ ngữ ngầm của "gây" rất dễ bị công cụ xử lý ngôn ngữ phía sau hiểu lầm là A, trong khi ngữ nghĩa thật là X gây ra Y, không phải A trực tiếp gây ra). Phải viết RÕ chủ ngữ cho từng hành vi:
   - Nếu cùng một chủ thể A thực hiện cả hai hành vi → lặp lại chủ ngữ, nối bằng "và": "A làm X và A gây Y".
   - Nếu hành vi thứ hai (Y) là hậu quả do chính đối tượng/sự việc X gây ra (không phải A trực tiếp) → tách thành mệnh đề/câu riêng có chủ ngữ là X, hoặc dùng cụm danh từ hoá "việc A làm X" làm chủ ngữ cho "gây", để chủ ngữ của "gây" không lẫn với A.
   Tránh dùng đại từ/cụm chỉ định mơ hồ ("hành vi này", "việc đó", "điều trên") làm chủ ngữ — luôn dùng cụm danh từ cụ thể.
8. Khi CHỦ THỂ của câu là một danh sách liệt kê nhiều thực thể (vd "cá nhân, tổ chức", "cá nhân, tổ chức, hộ gia đình"), PHẢI nối vế CUỐI CÙNG của danh sách bằng "và" (không dùng dấu phẩy thuần cho tới hết danh sách), NGAY CẢ KHI văn bản gốc chỉ dùng dấu phẩy. Lý do: công cụ xử lý ngôn ngữ phía sau (VnCoreNLP) hay hiểu lầm vế cuối của danh sách CHỦ THỂ — nếu nối bằng dấu phẩy và từ đó vốn đa nghĩa danh từ/động từ (vd "tổ chức", "quản lý", "đào tạo", "kinh doanh") đứng ngay trước động từ chính — thành chính động từ đó, làm sai toàn bộ cấu trúc câu (vd "Cá nhân, tổ chức vi phạm..." bị hiểu lầm "tổ chức" là động từ thay vì 1 chủ thể). Chỉ cần nối "và" trước vế cuối là tránh được lỗi này. Quy tắc này CHỈ áp dụng cho danh sách CHỦ THỂ (đứng ngay trước động từ chính của câu) — KHÔNG áp dụng cho danh sách ĐỐI TƯỢNG (xem mục 3, giữ nguyên cấu trúc liệt kê gốc cho đối tượng).
9. text_content thường gồm nhiều cấp lồng nhau (vd câu mở đầu của Điều/Khoản, rồi mới tới câu chứa hành vi cụ thể ở Khoản/Điểm). Nếu câu mở đầu (Điều/Khoản) đã NÊU RÕ một chủ thể CỤ THỂ (vd một loại phương tiện/nhóm người cụ thể), còn câu chứa hành vi chính lại dùng một cụm CHUNG hơn để chỉ lại chủ thể đó (vd "người điều khiển phương tiện", "người vi phạm"), PHẢI dùng chủ thể CỤ THỂ đã nêu ở câu mở đầu cho proposition, KHÔNG dùng lại cụm chung. Lý do: cụm chung sẽ làm mất khả năng phân biệt với các Điều/Khoản khác áp dụng cho chủ thể khác (vd Điều này chỉ áp dụng cho "người điều khiển xe ô tô tải, máy kéo", nếu viết chung thành "người điều khiển phương tiện" sẽ lẫn với Điều khác áp dụng cho xe máy, xe ô tô con).

Ví dụ minh họa (few-shot) cho việc nối "và" ở vế cuối danh sách chủ thể (mục 8):

Input:
"Cá nhân, tổ chức vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ phải chịu hình thức xử phạt chính là phạt tiền."

Output ĐÚNG (nối "và" trước vế cuối "tổ chức" trong danh sách chủ thể để tránh bị hiểu lầm thành động từ):
{"propositions": ["Cá nhân và tổ chức vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ phải chịu hình thức xử phạt chính là phạt tiền."]}

Output SAI (giữ dấu phẩy thuần giữa "Cá nhân" và "tổ chức" như văn bản gốc — dễ bị tagger hiểu lầm "tổ chức" thành động từ, làm sai cấu trúc câu):
{"propositions": ["Cá nhân, tổ chức vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ phải chịu hình thức xử phạt chính là phạt tiền."]}

Ví dụ minh họa (few-shot) cho việc viết rõ chủ ngữ khi có hai hành vi nối tiếp (mục 7):

Input:
"Người điều khiển xe mở cửa xe gây tai nạn giao thông thì bị phạt tiền từ 800.000 đồng đến 1.000.000 đồng."

Output ĐÚNG (chủ ngữ của "gây" được viết rõ là "việc mở cửa xe", không lẫn với "Người điều khiển xe"):
{"propositions": ["Người điều khiển xe mở cửa xe; việc mở cửa xe gây tai nạn giao thông thì bị phạt tiền."]}

Output SAI (mệnh đề "gây tai nạn giao thông" nối ngay sau "mở cửa xe" không có chủ ngữ riêng, dễ bị hiểu lầm chủ ngữ là "Người điều khiển xe" thay vì "việc mở cửa xe"):
{"propositions": ["Người điều khiển xe mở cửa xe gây tai nạn giao thông thì bị phạt tiền."]}

Ví dụ minh họa (few-shot) cho việc lược bỏ danh sách liệt kê địa điểm/trường hợp (mục 1, ý cuối):

Input:
"Điều 6. ... 4. Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: i) Quay đầu xe ở phần đường dành cho người đi bộ qua đường, trên cầu, đầu cầu, gầm cầu vượt, ngầm, tại nơi đường bộ giao nhau cùng mức với đường sắt, đường hẹp, đường dốc, đoạn đường cong tầm nhìn bị che khuất, trên đường một chiều, trừ khi có hiệu lệnh của người điều khiển giao thông hoặc chỉ dẫn của biển báo hiệu tạm thời hoặc tổ chức giao thông tại những khu vực này có bố trí nơi quay đầu xe;"

Output ĐÚNG (đã cắt bỏ danh sách 10 địa điểm cụ thể và mệnh đề ngoại lệ "trừ khi...", nhưng khái quát hoá thành cụm "tại nơi cấm quay đầu xe" để GIỮ ĐÚNG tính chất có điều kiện của hành vi — không biến nó thành hành vi luôn luôn vi phạm):
{"propositions": ["Người điều khiển xe thực hiện hành vi quay đầu xe tại nơi cấm quay đầu xe."]}

Output SAI #1 (liệt kê hết cả danh sách 10 địa điểm — không cần thiết cho điều hướng vì cùng 1 địa chỉ điều khoản):
{"propositions": ["Người điều khiển xe thực hiện hành vi quay đầu xe ở phần đường dành cho người đi bộ qua đường, trên cầu, đầu cầu, gầm cầu vượt, ngầm, tại nơi đường bộ giao nhau cùng mức với đường sắt, đường hẹp, đường dốc, đoạn đường cong tầm nhìn bị che khuất, trên đường một chiều."]}

Output SAI #2 (xoá hẳn điều kiện địa điểm, biến hành vi có điều kiện thành hành vi luôn luôn vi phạm — SAI nội dung pháp lý vì quay đầu xe ở nơi cho phép không phải là vi phạm):
{"propositions": ["Người điều khiển xe thực hiện hành vi quay đầu xe."]}

Ví dụ minh họa (few-shot) cho việc giữ chủ thể cụ thể từ câu mở đầu Điều, không dùng lại cụm chung (mục 9):

Input:
"Điều 21. Xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô tải, máy kéo và các loại xe tương tự xe ô tô vận chuyển hàng hóa. 5. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển phương tiện chở hàng vượt quá khối lượng cho phép."

Output ĐÚNG (dùng chủ thể cụ thể "người điều khiển xe ô tô tải, máy kéo và các loại xe tương tự xe ô tô vận chuyển hàng hóa" đã nêu ở câu mở đầu Điều 21, thay vì cụm chung "người điều khiển phương tiện" trong câu chứa hành vi):
{"propositions": ["Người điều khiển xe ô tô tải, máy kéo và các loại xe tương tự xe ô tô vận chuyển hàng hóa chở hàng vượt quá khối lượng cho phép bị phạt tiền."]}

Output SAI (dùng lại cụm chung "người điều khiển phương tiện" từ câu chứa hành vi — mất khả năng phân biệt với các Điều khác áp dụng cho phương tiện khác như xe máy, xe ô tô con):
{"propositions": ["Người điều khiển phương tiện chở hàng vượt quá khối lượng cho phép bị phạt tiền."]}

Bạn sẽ nhận một DANH SÁCH NHIỀU đoạn văn bản trong một lượt, mỗi đoạn có một "id" riêng đi kèm. Hãy xử lý ĐỘC LẬP từng đoạn theo đúng các nguyên tắc ở trên (không lấy nội dung của đoạn này để diễn giải cho đoạn khác, dù chúng đứng cạnh nhau trong input).

Hãy trả về kết quả dưới định dạng JSON DUY NHẤT với cấu trúc sau, trong đó mỗi khóa của "results" là đúng "id" của đoạn tương ứng (giữ nguyên id, không đổi định dạng) và giá trị là danh sách câu diễn đạt lại của riêng đoạn đó:
{
  "results": {
    "<id_1>": ["Câu diễn đạt lại thứ nhất của đoạn id_1", "Câu diễn đạt lại thứ hai (chỉ khi thực sự cần tách)"],
    "<id_2>": ["Câu diễn đạt lại thứ nhất của đoạn id_2"]
  }
}

PHẢI trả về đầy đủ kết quả cho TẤT CẢ id có trong input, không được bỏ sót id nào."""

def build_batch_user_content(batch):
    """
    Build a single user message containing multiple sections, each tagged with its id.
    """
    parts = [f'--- id: {sec["id"]} ---\n{sec["text_content"]}' for sec in batch]
    return "Các đoạn văn bản cần xử lý (mỗi đoạn có id riêng, xử lý độc lập):\n\n" + "\n\n".join(parts)

def call_openai_batch_with_retry(batch, max_retries=5, initial_backoff=2):
    """
    Call OpenAI API to rewrite a batch of sections in a single request, with retry logic and exponential backoff.
    """
    user_content = build_batch_user_content(batch)
    retries = 0
    backoff = initial_backoff
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            return result.get("results", {})
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"\nError calling OpenAI API after {max_retries} attempts: {e}")
                raise e
            print(f"\nAPI call failed: {e}. Retrying in {backoff} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("max_retries exhausted without a successful response")

def process_batch(batch):
    """
    Worker function to process a batch of sections in one API call.
    Returns a dict mapping section id -> propositions (or None on failure for that id).
    """
    try:
        results = call_openai_batch_with_retry(batch)
    except Exception:
        results = {}
    output = {}
    for sec in batch:
        propositions = results.get(sec["id"])
        output[sec["id"]] = propositions if isinstance(propositions, list) else None
    return output

def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def main():
    parser = argparse.ArgumentParser(description="Rewrite legal sections into atomic propositions using GPT-4o-mini.")
    parser.add_argument("--input", default="src/Building_KG/material_for_triplets/1_sections_nghi_dinh_168_2024_1.json", help="Path to input json file")
    parser.add_argument("--output", default="src/Building_KG/material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json", help="Path to output json file")
    parser.add_argument("--threads", type=int, default=10, help="Number of concurrent threads")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of sections to send per API request")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of sections to process (for testing)")
    args = parser.parse_args()

    # Load input data
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        sections = json.load(f)

    if args.limit > 0:
        sections = sections[:args.limit]
        print(f"Limiting to first {args.limit} sections.")

    print(f"Total sections to process: {len(sections)}")

    # Load existing results if file exists for resuming
    processed_results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    # Only keep successfully processed ones (which have 'rewritten_propositions')
                    if "rewritten_propositions" in item:
                        processed_results[item["id"]] = item
            print(f"Resuming progress. Already processed {len(processed_results)} sections.")
        except Exception as e:
            print(f"Error reading existing output file: {e}. Starting fresh.")

    # Filter out already processed sections
    to_process = [s for s in sections if s["id"] not in processed_results]
    print(f"Remaining sections to process: {len(to_process)}")

    if not to_process:
        print("All sections have already been processed.")
        return

    # Sections with empty text_content don't need an API call
    empty_sections = [s for s in to_process if not s.get("text_content", "").strip()]
    for s in empty_sections:
        enriched_sec = s.copy()
        enriched_sec["rewritten_propositions"] = []
        processed_results[s["id"]] = enriched_sec

    nonempty_sections = [s for s in to_process if s.get("text_content", "").strip()]
    batches = list(chunked(nonempty_sections, args.batch_size))

    # Track results
    success_count = len(empty_sections)
    fail_count = 0
    total_to_process = len(to_process)
    processed_count = len(empty_sections)

    print(f"Starting processing using ThreadPoolExecutor with {args.threads} threads, "
          f"{len(batches)} batches of up to {args.batch_size} sections each...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit tasks
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            results = future.result()

            for sec in batch:
                propositions = results.get(sec["id"])
                processed_count += 1
                if propositions is not None:
                    success_count += 1
                    # Merge original section dict with rewritten propositions
                    enriched_sec = sec.copy()
                    enriched_sec["rewritten_propositions"] = propositions
                    processed_results[sec["id"]] = enriched_sec
                else:
                    fail_count += 1
                    print(f"\nFailed to process section {sec['id']}")

            # Print status update
            progress_percent = (processed_count / total_to_process) * 100
            print(f"Progress: {processed_count}/{total_to_process} ({progress_percent:.2f}%) | Success: {success_count} | Failed: {fail_count}", end="\r")

            # Save progress after every batch
            ordered_output = []
            for s in sections:
                if s["id"] in processed_results:
                    ordered_output.append(processed_results[s["id"]])

            # Write to temp file first to prevent corruption
            temp_output_path = args.output + ".tmp"
            with open(temp_output_path, "w", encoding="utf-8") as f:
                json.dump(ordered_output, f, ensure_ascii=False, indent=4)
            os.replace(temp_output_path, args.output)
            print(f"\nSaved progress. Total processed: {len(processed_results)}")

    print(f"\nDone processing. Success: {success_count}, Failed: {fail_count}.")
    print(f"Final results saved to {args.output}")

if __name__ == "__main__":
    main()
