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
Nhiệm vụ của bạn là đọc và hiểu đoạn văn bản pháp luật (text_content) được cung cấp, sau đó phân tách và viết lại văn bản đó thành một danh sách các "ý đơn nhất" (atomic propositions).

Mỗi "ý đơn nhất" phải thỏa mãn các tiêu chí sau:
1. Đầy đủ các thành phần: Chủ thể (Subject), Hành động/Trạng thái/Quan hệ (Action/Relation), và Đối tượng (Object).
2. Đơn trị (atomic): Mỗi ý chỉ chứa duy nhất một mệnh đề cốt lõi, không chứa liên từ nối nhiều hành động hoặc nhiều đối tượng phức tạp.
3. Đầy đủ ngữ cảnh: Giải quyết triệt để các trường hợp ẩn chủ ngữ, ẩn hành động, ẩn đối tượng bằng cách lặp lại hoặc làm rõ chủ thể chính. Ví dụ: Thay vì ghi "đỗ xe trên miệng cống", hãy viết đầy đủ "Người điều khiển phương tiện đỗ xe trên miệng cống thoát nước".
4. Không suy diễn: Chỉ viết lại các ý dựa trên đúng nội dung thực tế của văn bản gốc, không tự ý thêm các giả định ngoài văn bản.
5. Rõ ràng và tự nhiên: Diễn đạt bằng tiếng Việt chuẩn xác, mạch lạc, dễ hiểu.

Hãy trả về kết quả dưới định dạng JSON với cấu trúc sau:
{
  "propositions": [
    "Ý đơn nhất thứ nhất",
    "Ý đơn nhất thứ hai",
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
    parser.add_argument("--input", default="src/Building_KG/material_for_triplets/sections_nghi_dinh_168_2024_1.json", help="Path to input json file")
    parser.add_argument("--output", default="src/Building_KG/material_for_triplets/sections_rewritten_nghi_dinh_168_2024_1.json", help="Path to output json file")
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
