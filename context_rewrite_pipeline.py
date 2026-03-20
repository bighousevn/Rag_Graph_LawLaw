#!/usr/bin/env python3
"""Build context-concatenated legal sentences and optionally rewrite via LLM API.

Phase 1 always runs: extract leaf-level contexts from selected chapters into a JSONL file.
Phase 2 is optional (--call-llm): send each context to an LLM and save rewritten output.

Example:
  python context_rewrite_pipeline.py \
    --input-json ket_qua.json \
    --chapters "Chương I,Chương II" \
    --contexts-output contexts_chuong_I_II.jsonl \
    --final-output llm_ket_qua_chuong_I_II.json

With LLM call:
  OPENAI_API_KEY=... python context_rewrite_pipeline.py \
    --input-json ket_qua.json \
    --call-llm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_footer_noise(text: str) -> str:
    text = re.sub(r"about:blank\s+\d+\/\d+\s+\d+\/\d+\/\d+,\s*\d+:\d+\s*[AP]M\s*about:blank", "", text, flags=re.IGNORECASE)
    text = re.sub(r"about:blank", "", text, flags=re.IGNORECASE)
    return normalize_spaces(text)


def clean_dieu_title(dieu_key: str) -> str:
    m = re.match(r"^Điều\s+\d+\.\s*(.*)$", dieu_key, flags=re.IGNORECASE)
    return normalize_spaces(m.group(1) if m else dieu_key)


def extract_dieu_number(dieu_key: str) -> Optional[int]:
    m = re.match(r"^Điều\s+(\d+)\.", dieu_key, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def clean_khoan_text(text: str) -> str:
    return normalize_spaces(re.sub(r"^\d+\.\s*", "", text))


def clean_diem_text(text: str) -> str:
    return normalize_spaces(re.sub(r"^[a-zđ]\)\s*", "", text, flags=re.IGNORECASE))


def pick_chapters(data: Dict[str, Any], chapter_names: List[str]) -> Dict[str, Any]:
    if any(x.lower() in {"all", "*"} for x in chapter_names):
        return {k: v for k, v in data.items() if isinstance(v, dict) and str(k).startswith("Chương ")}

    selected: Dict[str, Any] = {}
    for key in chapter_names:
        if key in data and isinstance(data[key], dict):
            selected[key] = data[key]
    return selected


def iter_dieu_nodes(chapter_node: Dict[str, Any]) -> Iterable[tuple[Optional[str], str, Dict[str, Any]]]:
    for key, value in chapter_node.items():
        if not isinstance(value, dict):
            continue

        if str(key).startswith("Điều "):
            yield None, key, value
            continue

        if str(key).startswith("Mục "):
            muc_key = key
            for dieu_key, dieu_val in value.items():
                if isinstance(dieu_val, dict) and str(dieu_key).startswith("Điều "):
                    yield muc_key, dieu_key, dieu_val


def build_context_records(
    data: Dict[str, Any],
    chapter_names: List[str],
    only_dieu_numbers: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    selected = pick_chapters(data, chapter_names)
    records: List[Dict[str, Any]] = []
    counter = 1

    for chuong_key, chapter_node in selected.items():
        for muc_key, dieu_key, dieu_obj in iter_dieu_nodes(chapter_node):
            dieu_number = extract_dieu_number(dieu_key)
            if only_dieu_numbers and (dieu_number is None or dieu_number not in only_dieu_numbers):
                continue

            dieu_title = clean_dieu_title(dieu_key)
            dieu_text = clean_footer_noise(str(dieu_obj.get("Nội dung điều", "")))

            khoan_items = [
                (k, v)
                for k, v in dieu_obj.items()
                if isinstance(v, dict) and str(k).startswith("Khoản ")
            ]

            if not khoan_items:
                context = normalize_spaces(": ".join(x for x in [dieu_title, dieu_text] if x))
                if context:
                    records.append(
                        {
                            "id": f"ctx_{counter:06d}",
                            "chapter": chuong_key,
                            "muc": muc_key,
                            "dieu": dieu_key,
                            "khoan": None,
                            "diem": None,
                            "leaf_level": "dieu",
                            "input_text": context,
                        }
                    )
                    counter += 1
                continue

            for khoan_key, khoan_obj in khoan_items:
                original = clean_footer_noise(str(khoan_obj.get("Nội dung gốc", "")))
                supplemental = clean_footer_noise(str(khoan_obj.get("Nội dung bổ sung", "")))
                khoan_text = normalize_spaces(" ".join(x for x in [clean_khoan_text(original), supplemental] if x))

                diem_items = [
                    (k, v)
                    for k, v in khoan_obj.items()
                    if str(k).startswith("Điểm ") and isinstance(v, str)
                ]

                if not diem_items:
                    context = normalize_spaces(": ".join(x for x in [dieu_title, khoan_text] if x))
                    if context:
                        records.append(
                            {
                                "id": f"ctx_{counter:06d}",
                                "chapter": chuong_key,
                                "muc": muc_key,
                                "dieu": dieu_key,
                                "khoan": khoan_key,
                                "diem": None,
                                "leaf_level": "khoan",
                                "input_text": context,
                            }
                        )
                        counter += 1
                    continue

                for diem_key, diem_text_raw in diem_items:
                    diem_text = clean_footer_noise(clean_diem_text(diem_text_raw))
                    input_text = normalize_spaces(
                        ": ".join(x for x in [dieu_title, khoan_text, diem_text] if x)
                    )
                    if not input_text:
                        continue

                    records.append(
                        {
                            "id": f"ctx_{counter:06d}",
                            "chapter": chuong_key,
                            "muc": muc_key,
                            "dieu": dieu_key,
                            "khoan": khoan_key,
                            "diem": diem_key,
                            "leaf_level": "diem",
                            "input_text": input_text,
                        }
                    )
                    counter += 1

    return records


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_openai_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_input: str,
    timeout_sec: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Văn bản luật lồng ghép:\n{user_input}"},
        ],
        "temperature": 0.2,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response has no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response content is empty")

    return normalize_spaces(content)


def rewrite_records_with_llm(
    records: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
    timeout_sec: int,
    delay_sec: float,
    max_items: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    system_prompt = (
        "Bạn là một chuyên gia pháp lý. Dựa vào đoạn văn bản luật lồng ghép dưới đây, "
        "hãy viết lại thành MỘT câu đơn giản, độc lập, có đầy đủ chủ thể, hành động và đối tượng. "
        "Tuyệt đối không làm mất đi định lượng thời gian và ý nghĩa pháp lý gốc. "
        "Chỉ trả về duy nhất 1 câu tiếng Việt, không tiêu đề, không markdown."
    )

    limit = min(max_items, len(records)) if max_items > 0 else len(records)

    for idx, rec in enumerate(records[:limit], start=1):
        item = dict(rec)
        try:
            rewritten = call_openai_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt,
                user_input=rec["input_text"],
                timeout_sec=timeout_sec,
            )
            item["llm_output"] = rewritten
            item["status"] = "ok"
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            item["status"] = "error"
            item["error"] = f"HTTP {exc.code}: {normalize_spaces(err_body)}"
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)

        item["llm_index"] = idx
        out.append(item)

        if delay_sec > 0 and idx < limit:
            time.sleep(delay_sec)

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Context concatenation + optional LLM rewrite pipeline")
    p.add_argument("--input-json", type=Path, default=Path("ket_qua.json"), help="Input nested legal JSON")
    p.add_argument(
        "--chapters",
        type=str,
        default="Chương I,Chương II",
        help='Comma-separated chapter names, e.g. "Chương I,Chương II"; use "all" for every chapter',
    )
    p.add_argument(
        "--only-dieu",
        type=str,
        default="",
        help='Optional comma-separated Dieu numbers, e.g. "15,17,48"',
    )
    p.add_argument(
        "--contexts-output",
        type=Path,
        default=Path("contexts_chuong_I_II.jsonl"),
        help="Output JSONL file generated before any LLM calls",
    )
    p.add_argument(
        "--final-output",
        type=Path,
        default=Path("llm_ket_qua_chuong_I_II.json"),
        help="Output JSON file for LLM rewrite results",
    )
    p.add_argument("--call-llm", action="store_true", help="Call LLM API after context file is created")
    p.add_argument("--api-key", type=str, default="", help="API key (optional; env fallback if empty)")
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable name for API key")
    p.add_argument("--base-url", type=str, default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    p.add_argument("--timeout-sec", type=int, default=90, help="HTTP timeout for each request")
    p.add_argument("--delay-sec", type=float, default=0.0, help="Delay between requests (seconds)")
    p.add_argument("--max-items", type=int, default=0, help="Max records to send to LLM (0 = all)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if not args.input_json.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {args.input_json}")

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chapter_names = [normalize_spaces(x) for x in args.chapters.split(",") if normalize_spaces(x)]
    only_dieu_numbers: set[int] = set()
    if args.only_dieu.strip():
        for token in args.only_dieu.split(","):
            token = normalize_spaces(token)
            if token.isdigit():
                only_dieu_numbers.add(int(token))

    records = build_context_records(
        data,
        chapter_names,
        only_dieu_numbers=only_dieu_numbers if only_dieu_numbers else None,
    )
    write_jsonl(records, args.contexts_output)
    print(f"Da tao contexts file: {args.contexts_output} ({len(records)} dong)")

    if not args.call_llm:
        print("Bo qua goi LLM (chua bat --call-llm).")
        return 0

    api_key = args.api_key.strip() or os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            "Chua co API key. Vui long truyen --api-key hoac set bien moi truong "
            f"{args.api_key_env}."
        )

    rewritten = rewrite_records_with_llm(
        records=records,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_sec=args.timeout_sec,
        delay_sec=args.delay_sec,
        max_items=args.max_items,
    )

    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    with args.final_output.open("w", encoding="utf-8") as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for x in rewritten if x.get("status") == "ok")
    err_count = len(rewritten) - ok_count
    print(f"Da tao file ket qua LLM: {args.final_output} (ok={ok_count}, error={err_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
