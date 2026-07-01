"""
Bước 3a: Tokenize các mệnh đề bằng VnCoreNLP (wseg + pos + ner + parse).
KHÔNG gọi LLM. Output dùng làm input cho 3b_extract.py.
Input:  material_for_triplets/2_sections_rewritten_*.json
Output: material_for_triplets/3a_tokenized_*.json
"""

import os
import glob
import json
import time
import argparse

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "vncorenlp_model"))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "material_for_triplets/3a_tokenized_nghi_dinh_168_2024_1.json")


def _ensure_jvm():
    cur = os.environ.get("JVM_PATH")
    if cur and os.path.exists(cur):
        return
    home = os.path.expanduser("~")
    cands = []
    jh = os.environ.get("JAVA_HOME")
    if jh:
        cands += [
            os.path.join(jh, "lib/server/libjvm.dylib"),
            os.path.join(jh, "lib/server/libjvm.so"),
            os.path.join(jh, "jre/lib/server/libjvm.so"),
        ]
    cands += glob.glob(os.path.join(home, ".local/share/mise/installs/java/*/lib/server/libjvm.dylib"))
    cands += glob.glob(os.path.join(home, ".local/share/mise/installs/java/*/lib/server/libjvm.so"))
    cands += glob.glob("/Library/Java/JavaVirtualMachines/*/Contents/Home/lib/server/libjvm.dylib")
    cands += glob.glob("/usr/lib/jvm/*/lib/server/libjvm.so")
    for c in sorted(cands, reverse=True):
        if os.path.exists(c):
            os.environ["JVM_PATH"] = c
            print(f"[jvm] JVM_PATH = {c}")
            return
    print("⚠️  Không tự dò được JVM — export JVM_PATH thủ công.")


_ensure_jvm()
import py_vncorenlp  # noqa: E402

print(f"Loading VnCoreNLP from: {MODEL_DIR}")
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=MODEL_DIR)
print("VnCoreNLP loaded.\n")


def tokenize_proposition(text: str) -> list[list[dict]]:
    """
    Annotate một proposition, trả về list sentences.
    Mỗi sentence là list token dicts với đủ 6 attributes:
      index, wordForm, posTag, nerLabel, head, depLabel
    """
    annotated = model.annotate_text(text)
    sentences = []
    for _, sentence in annotated.items():
        tokens = [
            {
                "index":    tok["index"],
                "wordForm": tok["wordForm"],
                "posTag":   tok["posTag"],
                "nerLabel": tok.get("nerLabel", "O"),
                "head":     tok["head"],
                "depLabel": tok["depLabel"],
            }
            for tok in sentence
        ]
        sentences.append(tokens)
    return sentences


def build_section_result(section: dict) -> dict:
    base = {
        "id":                     section["id"],
        "document_name":          section.get("document_name"),
        "level":                  section.get("level"),
        "path":                   section.get("path"),
        "original_text":          section.get("text_content"),
        "rewritten_propositions": section.get("rewritten_propositions", []),
    }
    props = [p for p in section.get("rewritten_propositions", []) if p and p.strip()]
    tokenized = []
    for prop in props:
        try:
            sentences = tokenize_proposition(prop)
            tokenized.append({"text": prop, "sentences": sentences})
        except Exception as e:
            print(f"  ⚠  [{section['id']}] lỗi: {e}")
            tokenized.append({"text": prop, "sentences": []})
    return {**base, "tokenized_propositions": tokenized}


def save(path: str, data: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit",  type=int, default=0, help="0 = xử lý tất cả")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    if args.limit:
        sections = sections[:args.limit]

    done: dict[str, dict] = {}
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for item in json.load(f):
                if "tokenized_propositions" in item:
                    done[item["id"]] = item

    to_process = [s for s in sections if s["id"] not in done]
    print(f"Total: {len(sections)} | Done: {len(done)} | Remaining: {len(to_process)}")
    if not to_process:
        print("Không có section mới. Thoát.")
        return

    t0 = time.time()
    for i, section in enumerate(to_process, 1):
        result = build_section_result(section)
        done[result["id"]] = result
        n = len(result["tokenized_propositions"])
        print(f"[{i}/{len(to_process)}] {section['id']} — {n} props  ({time.time()-t0:.1f}s)")

        if i % 20 == 0 or i == len(to_process):
            ordered = [done[s["id"]] for s in sections if s["id"] in done]
            save(args.output, ordered)

    print(f"\nDone. {len(done)} sections → {args.output}")


if __name__ == "__main__":
    main()
