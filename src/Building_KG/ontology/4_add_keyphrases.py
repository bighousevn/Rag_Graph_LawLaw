"""
Bổ sung s_keyphrases / v_keyphrases / o_keyphrases cho các file JSON triplet
bằng GPT-4o-mini, xử lý song song hàng loạt.

Cài đặt thư viện (đã có sẵn trong requirements.txt của repo):
    pip install "openai>=2.30.0" "python-dotenv>=1.2.1" "tqdm>=4.67.3"

Cách chạy:
    export OPENAI_API_KEY="sk-..."          # hoặc đặt trong file .env cùng thư mục
    python 4_add_keyphrases.py
    python 4_add_keyphrases.py --input-dir ~/subagen_output --workers 8

Thiết kế:
    Mỗi file .json là một danh sách section, mỗi section có "triplets" gồm
    các cặp (s, v, o) + 3 mảng keyphrase rỗng/có sẵn. Với mỗi file, thay vì
    yêu cầu LLM trả lại NGUYÊN file JSON đã sửa (rủi ro làm hỏng/đổi các
    trường khác), script chỉ gửi lên danh sách CÁC TỪ s/v/o DUY NHẤT xuất
    hiện trong file (đã khử trùng lặp để giảm token) và yêu cầu LLM trả về
    2-3 từ đồng nghĩa cho mỗi từ. Việc merge kết quả vào đúng vị trí trong
    JSON gốc được thực hiện hoàn toàn bằng Python, đảm bảo các trường
    id/path/document_name/propositions/s/v/o không bao giờ bị LLM chạm vào.
"""

import os
import re
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

MODEL = "gpt-4o-mini"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "output", "subagent_output")
DEFAULT_WORKERS = 8
MAX_RETRIES = 5

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "4_add_keyphrases_failures.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """Bạn là chuyên gia thuật ngữ pháp luật giao thông đường bộ Việt Nam.
Bạn sẽ nhận 3 danh sách từ/cụm từ được trích từ các triplet (Chủ thể, Hành vi, Đối tượng)
của văn bản luật giao thông:
- "s_terms": các cụm danh từ đóng vai trò CHỦ THỂ
- "v_terms": các cụm động từ/hành vi
- "o_terms": các cụm danh từ đóng vai trò ĐỐI TƯỢNG

Với MỖI từ/cụm từ trong cả 3 danh sách, hãy đề xuất tối đa 2-3 từ đồng nghĩa hoặc
cách gọi khác THÔNG DỤNG NHẤT trong ngữ cảnh luật giao thông Việt Nam (ví dụ cách
người dân, báo chí, văn bản pháp luật khác hay gọi cùng một khái niệm đó).
Nếu một từ không có từ đồng nghĩa/cách gọi khác nào thực sự phù hợp và tự nhiên,
trả về mảng rỗng [] cho từ đó — TUYỆT ĐỐI không bịa từ gượng ép, không lặp lại
chính từ gốc, không thêm từ khác nghĩa.

Trả về DUY NHẤT một JSON object dạng:
{
  "s_synonyms": {"<từ 1>": ["...", "..."], "<từ 2>": []},
  "v_synonyms": {"<từ 1>": ["...", "..."]},
  "o_synonyms": {"<từ 1>": ["...", "..."]}
}
PHẢI có đầy đủ mọi từ được cung cấp trong cả 3 danh sách làm key, kể cả khi giá trị là mảng rỗng."""


def collect_unique_terms(sections: list) -> dict:
    s_terms, v_terms, o_terms = set(), set(), set()
    for section in sections:
        for t in section.get("triplets", []):
            if t.get("s"):
                s_terms.add(t["s"])
            if t.get("v"):
                v_terms.add(t["v"])
            if t.get("o"):
                o_terms.add(t["o"])
    return {
        "s_terms": sorted(s_terms),
        "v_terms": sorted(v_terms),
        "o_terms": sorted(o_terms),
    }


def call_llm_for_synonyms(terms: dict) -> dict:
    """Gọi GPT-4o-mini một lần cho toàn bộ từ duy nhất của một file. Có retry."""
    user_msg = json.dumps(terms, ensure_ascii=False)
    backoff = 2
    last_err: Exception = RuntimeError("LLM call failed with no exception captured")
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from OpenAI API")
            return json.loads(content)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
    raise last_err


def _clean_synonyms(candidates, original: str, existing: list) -> list:
    """Lọc trùng với từ gốc / mảng hiện có, chuẩn hoá khoảng trắng, giới hạn 3 từ mới."""
    if not isinstance(candidates, list):
        return []
    seen_lower = {original.strip().lower()} | {e.strip().lower() for e in existing if isinstance(e, str)}
    cleaned = []
    for c in candidates:
        if not isinstance(c, str):
            continue
        c = re.sub(r"\s+", " ", c).strip()
        if not c or c.lower() in seen_lower:
            continue
        seen_lower.add(c.lower())
        cleaned.append(c)
        if len(cleaned) == 3:
            break
    return cleaned


def merge_synonyms_into_data(sections: list, synonyms: dict) -> None:
    """Mutates `sections` in-place: chỉ nối thêm vào 3 mảng keyphrase, không đổi trường nào khác."""
    s_map = synonyms.get("s_synonyms", {}) or {}
    v_map = synonyms.get("v_synonyms", {}) or {}
    o_map = synonyms.get("o_synonyms", {}) or {}

    for section in sections:
        for t in section.get("triplets", []):
            for key, syn_map in (("s", s_map), ("v", v_map), ("o", o_map)):
                term = t.get(key)
                if not term:
                    continue
                kp_key = f"{key}_keyphrases"
                existing = t.get(kp_key)
                if not isinstance(existing, list):
                    existing = []
                new_terms = _clean_synonyms(syn_map.get(term), term, existing)
                t[kp_key] = existing + new_terms


def process_file(path: str) -> tuple[str, bool, str]:
    """Trả về (path, success, message)."""
    try:
        with open(path, encoding="utf-8") as f:
            sections = json.load(f)

        terms = collect_unique_terms(sections)
        if not any(terms.values()):
            return path, True, "no triplets, skipped"

        synonyms = call_llm_for_synonyms(terms)
        merge_synonyms_into_data(sections, synonyms)

        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

        return path, True, "ok"
    except Exception as e:
        logging.error("FAILED %s: %s", path, e, exc_info=True)
        return path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                         help=f"Thư mục chứa các file .json cần xử lý (mặc định: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help="Số luồng chạy song song (5-10 khuyến nghị)")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Không tìm thấy thư mục: {input_dir}")

    files = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.endswith(".json")
    )
    if not files:
        raise SystemExit(f"Không có file .json nào trong: {input_dir}")

    print(f"Tìm thấy {len(files)} file trong {input_dir}, chạy với {args.workers} worker...")

    ok_count, fail_count = 0, 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, path): path for path in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Xử lý file"):
            path, success, message = fut.result()
            if success:
                ok_count += 1
            else:
                fail_count += 1
                failed_files.append((path, message))

    print(f"\nHoàn tất: {ok_count} thành công, {fail_count} thất bại.")
    if failed_files:
        print(f"Chi tiết lỗi đã ghi vào: {LOG_PATH}")
        for path, message in failed_files:
            print(f"  [FAIL] {os.path.basename(path)}: {message}")


if __name__ == "__main__":
    main()
