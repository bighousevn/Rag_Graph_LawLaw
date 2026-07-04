"""
Bước 1: Tách từ (word segmentation) các mệnh đề bằng VnCoreNLP.
CHỈ tách từ (nối từ ghép bằng "_"), KHÔNG pos/ner/parse, KHÔNG xoá bất kỳ token nào.
Input:  ../material_for_triplets/2_sections_rewritten_*.json
Output: output/segmented.json
"""

import os
import glob
import json
import time
import argparse

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "vncorenlp_model"))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "..", "material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output/segmented.json")


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

print(f"Loading VnCoreNLP (wseg only) from: {MODEL_DIR}")
model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=MODEL_DIR)
print("VnCoreNLP loaded.\n")


def segment_proposition(text: str) -> str:
    """Tách từ 1 proposition. Trả về câu đã tách từ (nối từ ghép bằng '_'), không xoá gì."""
    sentences = model.word_segment(text)
    return " ".join(sentences)


def build_section_result(section: dict) -> dict:
    props = [p for p in section.get("rewritten_propositions", []) if p and p.strip()]
    propositions = []
    for prop in props:
        try:
            segmented = segment_proposition(prop)
        except Exception as e:
            print(f"  ⚠  [{section['id']}] lỗi: {e}")
            segmented = prop
        propositions.append({"text": prop, "segmented": segmented})
    return {
        "id": section["id"],
        "path": section.get("path"),
        "propositions": propositions,
    }


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
                if "propositions" in item:
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
        n = len(result["propositions"])
        print(f"[{i}/{len(to_process)}] {section['id']} — {n} props  ({time.time()-t0:.1f}s)")

        if i % 20 == 0 or i == len(to_process):
            ordered = [done[s["id"]] for s in sections if s["id"] in done]
            save(args.output, ordered)

    print(f"\nDone. {len(done)} sections → {args.output}")


if __name__ == "__main__":
    main()
