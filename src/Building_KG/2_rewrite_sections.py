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

Ví dụ minh họa (few-shot) cho việc lược bỏ danh sách liệt kê địa điểm/trường hợp (mục 1, ý cuối):

Input:
"Điều 6. ... 4. Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: i) Quay đầu xe ở phần đường dành cho người đi bộ qua đường, trên cầu, đầu cầu, gầm cầu vượt, ngầm, tại nơi đường bộ giao nhau cùng mức với đường sắt, đường hẹp, đường dốc, đoạn đường cong tầm nhìn bị che khuất, trên đường một chiều, trừ khi có hiệu lệnh của người điều khiển giao thông hoặc chỉ dẫn của biển báo hiệu tạm thời hoặc tổ chức giao thông tại những khu vực này có bố trí nơi quay đầu xe;"

Output ĐÚNG (đã cắt bỏ danh sách 10 địa điểm cụ thể và mệnh đề ngoại lệ "trừ khi...", nhưng khái quát hoá thành cụm "tại nơi cấm quay đầu xe" để GIỮ ĐÚNG tính chất có điều kiện của hành vi — không biến nó thành hành vi luôn luôn vi phạm):
{"propositions": ["Người điều khiển xe thực hiện hành vi quay đầu xe tại nơi cấm quay đầu xe."]}

Output SAI #1 (liệt kê hết cả danh sách 10 địa điểm — không cần thiết cho điều hướng vì cùng 1 địa chỉ điều khoản):
{"propositions": ["Người điều khiển xe thực hiện hành vi quay đầu xe ở phần đường dành cho người đi bộ qua đường, trên cầu, đầu cầu, gầm cầu vượt, ngầm, tại nơi đường bộ giao nhau cùng mức với đường sắt, đường hẹp, đường dốc, đoạn đường cong tầm nhìn bị che khuất, trên đường một chiều."]}

Output SAI #2 (xoá hẳn điều kiện địa điểm, biến hành vi có điều kiện thành hành vi luôn luôn vi phạm — SAI nội dung pháp lý vì quay đầu xe ở nơi cho phép không phải là vi phạm):
{"propositions": ["Người điều khiển xe thực hiện hành vi quay đầu xe."]}

Hãy trả về kết quả dưới định dạng JSON với cấu trúc sau:
{
  "propositions": [
    "Câu diễn đạt lại thứ nhất",
    "Câu diễn đạt lại thứ hai (chỉ khi thực sự cần tách)",
    ...
  ]
}"""

def call_openai_with_retry(text_content, max_retries=5, initial_backoff=2):
    """
    Call OpenAI API to rewrite text_content with retry logic and exponential backoff.
    """
    retries = 0
    backoff = initial_backoff
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Văn bản cần xử lý:\n{text_content}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            return result.get("propositions", [])
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"\nError calling OpenAI API after {max_retries} attempts: {e}")
                raise e
            print(f"\nAPI call failed: {e}. Retrying in {backoff} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2

def process_section(section):
    """
    Worker function to process a single section.
    """
    text_content = section.get("text_content", "")
    if not text_content.strip():
        return section["id"], []
    try:
        propositions = call_openai_with_retry(text_content)
        return section["id"], propositions
    except Exception:
        return section["id"], None

def main():
    parser = argparse.ArgumentParser(description="Rewrite legal sections into atomic propositions using GPT-4o-mini.")
    parser.add_argument("--input", default="src/Building_KG/material_for_triplets/1_sections_nghi_dinh_168_2024_1.json", help="Path to input json file")
    parser.add_argument("--output", default="src/Building_KG/material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json", help="Path to output json file")
    parser.add_argument("--threads", type=int, default=10, help="Number of concurrent threads")
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

    # Track results
    success_count = 0
    fail_count = 0
    total_to_process = len(to_process)

    # We will save progress periodically (every 10 items processed)
    save_interval = 10

    # Initialize the output list with already processed items
    final_output = list(processed_results.values())

    print(f"Starting processing using ThreadPoolExecutor with {args.threads} threads...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit tasks
        future_to_section = {executor.submit(process_section, sec): sec for sec in to_process}

        for i, future in enumerate(as_completed(future_to_section)):
            sec = future_to_section[future]
            sec_id, propositions = future.result()

            if propositions is not None:
                success_count += 1
                # Merge original section dict with rewritten propositions
                enriched_sec = sec.copy()
                enriched_sec["rewritten_propositions"] = propositions
                processed_results[sec_id] = enriched_sec
            else:
                fail_count += 1
                print(f"\nFailed to process section {sec_id}")

            # Print status update
            progress_percent = ((i + 1) / total_to_process) * 100
            print(f"Progress: {i+1}/{total_to_process} ({progress_percent:.2f}%) | Success: {success_count} | Failed: {fail_count}", end="\r")

            # Periodically save progress
            if (i + 1) % save_interval == 0 or (i + 1) == total_to_process:
                # Build complete list in order of original sections
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
