"""Chuyển emlaw_qa.json sang file Excel (.xlsx)."""
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

INPUT_PATH = Path(__file__).parent / "emlaw_qa.json"
OUTPUT_PATH = Path(__file__).parent / "emlaw_qa.xlsx"

COLUMNS = ["stt", "question", "answer"]
COLUMN_WIDTHS = {"stt": 8, "question": 50, "answer": 100}


def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    wb = Workbook()
    ws = wb.active
    ws.title = "QA"

    ws.append(COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="center")

    wrap = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        ws.append([row.get(col, "") for col in COLUMNS])

    for row_cells in ws.iter_rows(min_row=2):
        for cell in row_cells:
            cell.alignment = wrap

    for idx, col in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = COLUMN_WIDTHS[col]

    ws.freeze_panes = "A2"

    wb.save(OUTPUT_PATH)
    print(f"Đã ghi {len(rows)} dòng vào {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
