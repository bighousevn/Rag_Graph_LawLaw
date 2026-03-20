#!/usr/bin/env python3
"""Parse Vietnamese legal PDF text into nested JSON structure.

Usage:
  python parse_legal_pdf.py --input /path/to/input.pdf --output /path/to/output.json

The parser recognizes these levels in order:
- Chương
- Mục
- Điều
- Khoản
- Điểm

It also appends non-heading continuation lines to the nearest active node content.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


PATTERN_CHUONG = re.compile(r"^Chương\s+[IVXLCDM]+", re.IGNORECASE)
PATTERN_MUC = re.compile(r"^Mục\s+\d+", re.IGNORECASE)
PATTERN_DIEU = re.compile(r"^Điều\s+\d+\.", re.IGNORECASE)
PATTERN_KHOAN = re.compile(r"^(\d+)\.\s+(.*)")
PATTERN_DIEM = re.compile(r"^([a-zđ])\)\s+(.*)", re.IGNORECASE)


@dataclass
class DiemNode:
    title: str
    content_parts: List[str] = field(default_factory=list)

    def append(self, text: str) -> None:
        if text:
            self.content_parts.append(text)

    def to_text(self) -> str:
        return " ".join(p.strip() for p in self.content_parts if p and p.strip()).strip()


@dataclass
class KhoanNode:
    number: str
    original_line: str
    content_parts: List[str] = field(default_factory=list)
    diem: Dict[str, DiemNode] = field(default_factory=dict)

    def append(self, text: str) -> None:
        if text:
            self.content_parts.append(text)

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "Nội dung gốc": self.original_line,
        }

        joined = " ".join(p.strip() for p in self.content_parts if p and p.strip()).strip()
        if joined:
            out["Nội dung bổ sung"] = joined

        for diem_key, diem_node in self.diem.items():
            out[diem_key] = diem_node.to_text()
        return out


@dataclass
class DieuNode:
    title: str
    content_parts: List[str] = field(default_factory=list)
    khoan: Dict[str, KhoanNode] = field(default_factory=dict)

    def append(self, text: str) -> None:
        if text:
            self.content_parts.append(text)

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        joined = " ".join(p.strip() for p in self.content_parts if p and p.strip()).strip()
        if joined:
            out["Nội dung điều"] = joined

        for khoan_key, khoan_node in self.khoan.items():
            out[khoan_key] = khoan_node.to_dict()

        return out


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def extract_lines_from_pdf(pdf_path: Path) -> List[str]:
    """Extract text lines from PDF, trying pdfplumber first then PyMuPDF fallback."""
    errors: List[str] = []

    try:
        import pdfplumber  # type: ignore

        lines: List[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines.extend(text.splitlines())
        return [normalize_line(x) for x in lines if normalize_line(x)]
    except Exception as exc:  # pragma: no cover
        errors.append(f"pdfplumber failed: {exc}")

    try:
        import fitz  # type: ignore

        lines = []
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                text = page.get_text("text") or ""
                lines.extend(text.splitlines())
        return [normalize_line(x) for x in lines if normalize_line(x)]
    except Exception as exc:  # pragma: no cover
        errors.append(f"PyMuPDF failed: {exc}")

    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = proc.stdout.splitlines()
        return [normalize_line(x) for x in lines if normalize_line(x)]
    except Exception as exc:  # pragma: no cover
        errors.append(f"pdftotext failed: {exc}")

    raise RuntimeError("Cannot extract text from PDF. Details: " + " | ".join(errors))


def parse_lines_to_tree(lines: List[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}

    current_chuong_key: Optional[str] = None
    current_muc_key: Optional[str] = None
    current_dieu_key: Optional[str] = None
    current_khoan_key: Optional[str] = None
    current_diem_key: Optional[str] = None

    for raw in lines:
        line = normalize_line(raw)
        if not line:
            continue

        if PATTERN_CHUONG.match(line):
            current_chuong_key = line
            current_muc_key = None
            current_dieu_key = None
            current_khoan_key = None
            current_diem_key = None
            result.setdefault(current_chuong_key, {})
            continue

        if PATTERN_MUC.match(line):
            if current_chuong_key is None:
                current_chuong_key = "Chương chưa xác định"
                result.setdefault(current_chuong_key, {})
            current_muc_key = line
            current_dieu_key = None
            current_khoan_key = None
            current_diem_key = None
            chuong_obj = result[current_chuong_key]
            assert isinstance(chuong_obj, dict)
            chuong_obj.setdefault(current_muc_key, {})
            continue

        if PATTERN_DIEU.match(line):
            current_dieu_key = line
            current_khoan_key = None
            current_diem_key = None

            if current_chuong_key is None:
                current_chuong_key = "Chương chưa xác định"
                result.setdefault(current_chuong_key, {})

            chuong_obj = result[current_chuong_key]
            assert isinstance(chuong_obj, dict)

            if current_muc_key is not None:
                muc_obj = chuong_obj.setdefault(current_muc_key, {})
                assert isinstance(muc_obj, dict)
                muc_obj[current_dieu_key] = DieuNode(title=current_dieu_key)
            else:
                chuong_obj[current_dieu_key] = DieuNode(title=current_dieu_key)
            continue

        khoan_match = PATTERN_KHOAN.match(line)
        if khoan_match and current_dieu_key:
            khoan_num = khoan_match.group(1)
            khoan_key = f"Khoản {khoan_num}"
            current_khoan_key = khoan_key
            current_diem_key = None

            dieu_node = get_current_dieu_node(result, current_chuong_key, current_muc_key, current_dieu_key)
            dieu_node.khoan[current_khoan_key] = KhoanNode(number=khoan_num, original_line=line)
            continue

        diem_match = PATTERN_DIEM.match(line)
        if diem_match and current_dieu_key and current_khoan_key:
            letter = diem_match.group(1).lower()
            diem_key = f"Điểm {letter}"
            current_diem_key = diem_key

            dieu_node = get_current_dieu_node(result, current_chuong_key, current_muc_key, current_dieu_key)
            khoan_node = dieu_node.khoan[current_khoan_key]
            node = khoan_node.diem.setdefault(diem_key, DiemNode(title=diem_key))
            node.append(line)
            continue

        if current_dieu_key:
            dieu_node = get_current_dieu_node(result, current_chuong_key, current_muc_key, current_dieu_key)

            if current_khoan_key and current_khoan_key in dieu_node.khoan:
                khoan_node = dieu_node.khoan[current_khoan_key]
                if current_diem_key and current_diem_key in khoan_node.diem:
                    khoan_node.diem[current_diem_key].append(line)
                else:
                    khoan_node.append(line)
            else:
                dieu_node.append(line)

    return materialize_output(result)


def get_current_dieu_node(
    result: Dict[str, object],
    current_chuong_key: Optional[str],
    current_muc_key: Optional[str],
    current_dieu_key: str,
) -> DieuNode:
    if current_chuong_key is None:
        current_chuong_key = "Chương chưa xác định"
        result.setdefault(current_chuong_key, {})

    chuong_obj = result[current_chuong_key]
    assert isinstance(chuong_obj, dict)

    if current_muc_key is not None and current_muc_key in chuong_obj:
        muc_obj = chuong_obj[current_muc_key]
        assert isinstance(muc_obj, dict)
        node = muc_obj[current_dieu_key]
    else:
        node = chuong_obj[current_dieu_key]

    assert isinstance(node, DieuNode)
    return node


def materialize_output(raw_tree: Dict[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {}

    for chuong_key, chuong_val in raw_tree.items():
        assert isinstance(chuong_val, dict)
        chuong_out: Dict[str, object] = {}

        for level2_key, level2_val in chuong_val.items():
            if isinstance(level2_val, DieuNode):
                chuong_out[level2_key] = level2_val.to_dict()
            else:
                assert isinstance(level2_val, dict)
                muc_out: Dict[str, object] = {}
                for dieu_key, dieu_node in level2_val.items():
                    assert isinstance(dieu_node, DieuNode)
                    muc_out[dieu_key] = dieu_node.to_dict()
                chuong_out[level2_key] = muc_out

        output[chuong_key] = chuong_out

    return output


def run(input_pdf: Path, output_json: Path) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {input_pdf}")

    lines = extract_lines_from_pdf(input_pdf)
    tree = parse_lines_to_tree(lines)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse legal PDF to nested JSON")
    p.add_argument("--input", required=True, type=Path, help="Path to input PDF")
    p.add_argument("--output", required=True, type=Path, help="Path to output JSON")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    run(args.input, args.output)
    print(f"Da tao file JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
