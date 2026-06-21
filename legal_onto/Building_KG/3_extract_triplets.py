"""
Phase 2 — Tách bộ ba S-V-O THUẦN VnCoreNLP (ZERO LLM)
=====================================================
Input : data/sections_rewritten.json  (mỗi section có 'rewritten_propositions')
Output: data/triplets_raw.json

Cơ chế (dựa trên cây phụ thuộc của VnCoreNLP — head index):
  - Động từ chính (posTag 'V') làm hạt nhân quan hệ (V).
  - Chủ thể (S): tìm 'sub'/'nsubj' của động từ; nếu ẩn -> lan truyền ngược cây.
  - Đối tượng (O): tân ngữ trực tiếp 'dob'/'dobj', hoặc qua giới từ 'pob'.
  - Bị động: gắn tiền tố 'bị'/'được' vào quan hệ.
Mỗi S/O lấy CẢ head word (để tra lexicon ở Phase 3) LẪN cụm đầy đủ
(subtree — để khớp keyphrase nhiều từ + PhoBERT).

KHÔNG gọi LLM. Mọi quyết định đến từ VnCoreNLP + luật cú pháp.
"""

import os
import glob
import json
import time
import argparse


def _ensure_jvm():
    """Tự dò libjvm.so cho pyjnius (tránh lỗi 'Unable to find libjvm.so')."""
    cur = os.environ.get("JVM_PATH")
    if cur and os.path.exists(cur):
        return
    home = os.path.expanduser("~")
    cands = []
    jh = os.environ.get("JAVA_HOME")
    if jh:
        cands += [os.path.join(jh, "lib/server/libjvm.so"),
                  os.path.join(jh, "jre/lib/server/libjvm.so")]
    cands += glob.glob(os.path.join(home, ".local/share/mise/installs/java/*/lib/server/libjvm.so"))
    cands += glob.glob("/usr/lib/jvm/*/lib/server/libjvm.so")
    cands += glob.glob(os.path.join(home, ".vscode/extensions/redhat.java-*/jre/*/lib/server/libjvm.so"))
    for c in sorted(cands, reverse=True):
        if os.path.exists(c):
            os.environ["JVM_PATH"] = c
            print(f"[jvm] JVM_PATH = {c}")
            return
    print("⚠️  Không tự dò được libjvm.so — hãy export JVM_PATH thủ công.")


_ensure_jvm()
import py_vncorenlp  # noqa: E402  (import sau khi set JVM_PATH)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "vncorenlp_model"))

# Tiền tố tình thái cần loại để chuẩn hóa quan hệ về động từ gốc
INTENTIONAL_PREFIXES = {"muốn", "cần", "phải", "nên", "sẽ", "đang", "đã", "có_thể", "không"}
SUBJECT_LABELS = {"sub", "nsubj"}
DOBJ_LABELS = {"dob", "dobj"}
PREP_POS = {"E", "I"}

print(f"Loading VnCoreNLP from: {MODEL_DIR}")
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "parse"], save_dir=MODEL_DIR)


def clean(w):
    return w.replace("_", " ").strip()


def build_children(sentence):
    children = {}
    for tok in sentence:
        children.setdefault(tok["head"], []).append(tok)
    return children


def subtree_tokens(idx, children):
    """Tất cả token trong nhánh con của idx (gồm cả idx)."""
    out = []
    stack = [idx]
    seen = set()
    by_idx = None
    # gom theo BFS dựa trên children map
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for ch in children.get(cur, []):
            out.append(ch)
            stack.append(ch["index"])
    return out


def phrase_of(idx, sentence, children, tokens_by_index):
    """Ghép cụm đầy đủ = token head + toàn bộ nhánh con, sắp theo thứ tự câu."""
    head_tok = tokens_by_index.get(idx)
    if not head_tok:
        return ""
    toks = [head_tok] + subtree_tokens(idx, children)
    toks = {t["index"]: t for t in toks}.values()  # khử trùng
    ordered = sorted(toks, key=lambda t: t["index"])
    return clean(" ".join(t["wordForm"] for t in ordered))


def find_subject(verb_idx, sentence, tokens_by_index):
    # 1. chủ ngữ trực tiếp
    for tok in sentence:
        if tok["head"] == verb_idx and tok["depLabel"] in SUBJECT_LABELS:
            return tok["index"]
    # 2. lan truyền ngược lên cây để lấy chủ ngữ thừa kế
    curr = verb_idx
    visited = set()
    while curr in tokens_by_index and curr not in visited:
        visited.add(curr)
        parent = tokens_by_index[curr]["head"]
        if parent == 0:
            break
        for tok in sentence:
            if tok["head"] == parent and tok["depLabel"] in SUBJECT_LABELS:
                return tok["index"]
        curr = parent
    return None


def find_object_idx(verb_idx, sentence, tokens_by_index):
    # 1. tân ngữ trực tiếp
    for tok in sentence:
        if tok["head"] == verb_idx and tok["depLabel"] in DOBJ_LABELS:
            return tok["index"]
    # 2. tân ngữ qua giới từ (pob)
    for prep in sentence:
        if prep["head"] == verb_idx and prep["posTag"] in PREP_POS:
            for pob in sentence:
                if pob["head"] == prep["index"] and pob["depLabel"] == "pob":
                    return pob["index"]
    return None


def extract_triplets(text):
    annotated = model.annotate_text(text.replace("_", " "))
    triplets = []
    for _, sentence in annotated.items():
        tokens_by_index = {t["index"]: t for t in sentence}
        children = build_children(sentence)

        for tok in sentence:
            word = clean(tok["wordForm"]).lower()
            if tok["posTag"] != "V" or word in INTENTIONAL_PREFIXES:
                continue

            v_idx = tok["index"]
            relation = clean(tok["wordForm"])
            # bị động: gắn bị/được
            parent = tokens_by_index.get(tok["head"])
            if parent and parent["wordForm"].lower() in ("bị", "được"):
                relation = f"{clean(parent['wordForm'])} {relation}"

            s_idx = find_subject(v_idx, sentence, tokens_by_index)
            o_idx = find_object_idx(v_idx, sentence, tokens_by_index)
            if s_idx is None or o_idx is None:
                continue

            triplets.append({
                "s": {
                    "head": clean(tokens_by_index[s_idx]["wordForm"]),
                    "phrase": phrase_of(s_idx, sentence, children, tokens_by_index),
                },
                "v": {
                    "head": relation,
                    "phrase": relation,
                },
                "o": {
                    "head": clean(tokens_by_index[o_idx]["wordForm"]),
                    "phrase": phrase_of(o_idx, sentence, children, tokens_by_index),
                },
            })
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(BASE, "data", "sections_rewritten.json"))
    ap.add_argument("--output", default=os.path.join(BASE, "data", "triplets_raw.json"))
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số section (0 = tất cả)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Không tìm thấy input: {args.input}")
        return
    with open(args.input, "r", encoding="utf-8") as f:
        sections = json.load(f)
    if args.limit > 0:
        sections = sections[:args.limit]

    print(f"Tách SVO cho {len(sections)} sections...")
    results = []
    n_trip = 0
    t0 = time.time()

    for i, sec in enumerate(sections, 1):
        sec_triplets = []
        for prop in sec.get("rewritten_propositions", []):
            if not isinstance(prop, str) or not prop.strip():
                continue
            try:
                for t in extract_triplets(prop):
                    t["proposition"] = prop
                    sec_triplets.append(t)
            except Exception as e:
                print(f"❌ Lỗi section {sec.get('id')}: {e}")
        n_trip += len(sec_triplets)
        results.append({
            "id": sec.get("id"),
            "document_name": sec.get("document_name"),
            "level": sec.get("level"),
            "path": sec.get("path"),
            "triplets": sec_triplets,
        })

        if i <= 3:
            print(f"\n=== {sec.get('id')} | {sec.get('path')} ===")
            for t in sec_triplets[:6]:
                print(f"  ({t['s']['phrase']}) -[{t['v']['head']}]-> ({t['o']['phrase']})")
        if i % 100 == 0 or i == len(sections):
            print(f"-> {i}/{len(sections)} sections, {n_trip} triplets ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu {n_trip} triplets tại: {args.output}")


if __name__ == "__main__":
    main()
