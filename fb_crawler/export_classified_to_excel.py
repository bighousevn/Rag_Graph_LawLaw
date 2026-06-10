"""Export classified posts JSON to Excel (.xlsx).

Usage: run from the workspace root or directly:
  python fb_crawler/export_classified_to_excel.py
"""
import json
import os
import sys
from pathlib import Path

try:
    from openpyxl import Workbook
except Exception:
    print("Missing dependency 'openpyxl'. Install with: pip install openpyxl")
    raise


INPUT_FILE_NAME = "classified_posts.json"


def find_input_file():
    base = Path(__file__).parent
    return base / INPUT_FILE_NAME


def load_payload(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_xlsx(rows, out_path: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "classified_posts"

    headers = ["id", "content", "domain", "suitable"]
    ws.append(headers)

    for r in rows:
        ws.append([
            r.get("id"),
            r.get("content"),
            r.get("domain"),
            bool(r.get("suitable")),
        ])

    wb.save(str(out_path))


def main():
    input_arg = None
    if len(sys.argv) > 1:
        input_arg = Path(sys.argv[1])

    input_file = Path(input_arg) if input_arg else find_input_file()
    if not input_file or not input_file.exists():
        print(f"❌ Không tìm thấy file {INPUT_FILE_NAME}. Vui lòng chạy classify_posts.py trước hoặc cung cấp đường dẫn.")
        return 1

    payload = load_payload(input_file)
    rows = payload.get("results") or []

    out_file = input_file.with_suffix(".xlsx")
    write_xlsx(rows, out_file)

    print(f"✅ Đã xuất {len(rows)} hàng vào: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
