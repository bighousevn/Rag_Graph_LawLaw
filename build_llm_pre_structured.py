#!/usr/bin/env python3
"""Build structured JSON input for step-3 from llm_pre.txt and context metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_lines(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = normalize_spaces(line)
        if s:
            out.append(s)
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            out.append(json.loads(s))
    return out


def extract_number(prefix: str, text: Optional[str]) -> str:
    if not text:
        return "0"
    if prefix == "dieu":
        m = re.match(r"^Điều\s+(\d+)\.", text)
    elif prefix == "khoan":
        m = re.match(r"^Khoản\s+(\d+)", text)
    else:
        m = None
    return m.group(1) if m else "0"


def extract_diem_code(diem: Optional[str]) -> str:
    if not diem:
        return ""
    m = re.match(r"^Điểm\s+([a-zđ])$", diem, flags=re.IGNORECASE)
    if not m:
        return ""
    return "D" + m.group(1).lower()


def build_id(ctx: Dict[str, Any], idx: int) -> str:
    d = extract_number("dieu", ctx.get("dieu"))
    k = extract_number("khoan", ctx.get("khoan"))
    diem_code = extract_diem_code(ctx.get("diem"))
    if diem_code:
        return f"D{d}_K{k}_{diem_code}"
    return f"D{d}_K{k}_I{idx:03d}"


def get_dieu_obj(legal: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    chapter = ctx["chapter"]
    dieu = ctx["dieu"]
    muc = ctx.get("muc")

    chapter_obj = legal[chapter]
    if muc:
        return chapter_obj[muc][dieu]
    return chapter_obj[dieu]


def resolve_original_text(legal: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    dieu_obj = get_dieu_obj(legal, ctx)
    khoan = ctx.get("khoan")
    diem = ctx.get("diem")

    if khoan and diem:
        khoan_obj = dieu_obj[khoan]
        return normalize_spaces(str(khoan_obj.get(diem, "")))

    if khoan:
        khoan_obj = dieu_obj[khoan]
        goc = normalize_spaces(str(khoan_obj.get("Nội dung gốc", "")))
        bo_sung = normalize_spaces(str(khoan_obj.get("Nội dung bổ sung", "")))
        return normalize_spaces(" ".join(x for x in [goc, bo_sung] if x))

    return normalize_spaces(str(dieu_obj.get("Nội dung điều", "")))


def main() -> int:
    p = argparse.ArgumentParser(description="Build structured JSON from llm_pre and contexts")
    p.add_argument("--llm-pre", type=Path, default=Path("llm_pre.txt"))
    p.add_argument("--contexts", type=Path, default=Path("contexts_dieu_17_35_48.jsonl"))
    p.add_argument("--legal-json", type=Path, default=Path("ket_qua.json"))
    p.add_argument("--output", type=Path, default=Path("llm_pre_structured.json"))
    args = p.parse_args()

    llm_lines = read_lines(args.llm_pre)
    contexts = read_jsonl(args.contexts)
    legal = json.loads(args.legal_json.read_text(encoding="utf-8"))

    if len(llm_lines) != len(contexts):
        raise RuntimeError(f"So dong llm_pre ({len(llm_lines)}) khong khop contexts ({len(contexts)})")

    out: List[Dict[str, Any]] = []
    for idx, (ctx, llm_text) in enumerate(zip(contexts, llm_lines), start=1):
        item = {
            "id": build_id(ctx, idx),
            "metadata": {
                "chuong": ctx.get("chapter"),
                "dieu": ctx.get("dieu"),
                "khoan": ctx.get("khoan"),
                "diem": ctx.get("diem"),
            },
            "original_text": resolve_original_text(legal, ctx),
            "llm_processed_text": llm_text,
        }
        out.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Da tao file JSON cau truc: {args.output} ({len(out)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
