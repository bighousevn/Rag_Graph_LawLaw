"""
Bước 0 (chuẩn bị cho subagent): chia 1 khoảng section (theo id dạng s<N>) thành các file batch
nhỏ trong output/subagent_input/, mỗi file BATCH_SIZE section, để giao cho 1 subagent xử lý.

Đặt tên file theo id đầu-cuối của batch (vd s160_s169.json) — KHÔNG dùng số thứ tự batch_01,
batch_02... để không đụng tên giữa các đợt chạy khác nhau (mỗi đợt xử lý 1 khoảng id khác nhau).

Tự động bỏ qua section đã có trong output/triplets_raw.json (đã xử lý ở đợt trước, đã gộp bởi
2_build_ontology.py) — trừ khi truyền --force.

Dùng:
    python3 1_split_batches.py --start 160 --end 259 --batch-size 10
    → sinh output/subagent_input/s160_s169.json ... s250_s259.json (10 file, 10 section/file)

Sau khi chạy xong: đọc rule tại EXTRACTION_RULES.md, giao mỗi file cho 1 subagent (Agent tool),
subagent Read file input + EXTRACTION_RULES.md, Write kết quả ra output/subagent_output/<tên file
giống input>.json theo đúng schema trong EXTRACTION_RULES.md. Xong hết thì chạy
`python3 2_build_ontology.py` để gộp + build ontology.
"""

import os
import re
import json
import argparse

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR       = os.path.join(BASE_DIR, "..", "material_for_triplets")
DEFAULT_INPUT       = os.path.join(MATERIALS_DIR, "2_sections_rewritten_nghi_dinh_168_2024_1.json")
SUBAGENT_INPUT_DIR  = os.path.join(BASE_DIR, "output", "subagent_input")
TRIPLETS_RAW        = os.path.join(BASE_DIR, "output", "triplets_raw.json")

_ID_NUM_RE = re.compile(r"\d+")


def id_num(sid: str):
    m = _ID_NUM_RE.search(sid or "")
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser(description="Chia 1 khoảng section thành batch input cho subagent")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--start", type=int, required=True, help="id bắt đầu, bao gồm cả (vd 160 = s160)")
    ap.add_argument("--end",   type=int, required=True, help="id kết thúc, bao gồm cả (vd 259 = s259)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="Không bỏ qua section đã có trong triplets_raw.json")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)

    done_ids = set()
    if not args.force and os.path.exists(TRIPLETS_RAW):
        with open(TRIPLETS_RAW, encoding="utf-8") as f:
            done_ids = {s["id"] for s in json.load(f) if s.get("id")}

    selected = [
        s for s in sections
        if s.get("id")
        and args.start <= (id_num(s["id"]) or -1) <= args.end
        and s.get("rewritten_propositions")
        and s["id"] not in done_ids
    ]
    selected.sort(key=lambda s: id_num(s["id"]))

    skipped_done = sum(
        1 for s in sections
        if s.get("id") and args.start <= (id_num(s["id"]) or -1) <= args.end and s["id"] in done_ids
    )

    if not selected:
        print(f"Không có section mới trong khoảng s{args.start}-s{args.end} "
              f"(đã xử lý: {skipped_done}). Dùng --force nếu muốn xử lý lại.")
        return

    os.makedirs(SUBAGENT_INPUT_DIR, exist_ok=True)
    written = []
    for i in range(0, len(selected), args.batch_size):
        chunk = selected[i:i + args.batch_size]
        out = [{
            "id": s["id"],
            "path": s.get("path"),
            "document_name": s.get("document_name"),
            "rewritten_propositions": s.get("rewritten_propositions", []),
        } for s in chunk]
        fname = f"{chunk[0]['id']}_{chunk[-1]['id']}.json"
        path = os.path.join(SUBAGENT_INPUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        written.append((fname, len(chunk)))

    print(f"Đã sinh {len(written)} file batch trong {SUBAGENT_INPUT_DIR} "
          f"(tổng {len(selected)} section mới, bỏ qua {skipped_done} section đã xử lý):")
    for fname, n in written:
        print(f"  {fname}  ({n} section)")


if __name__ == "__main__":
    main()
