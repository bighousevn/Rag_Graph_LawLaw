"""
Bước 3: Trích xuất raw entities + raw triplets từ các mệnh đề đã chuẩn hoá.
Dùng VnCoreNLP (wseg + pos + ner + parse) — KHÔNG gọi LLM.
Input:  material_for_triplets/2_sections_rewritten_*.json
Output: material_for_triplets/3_triplets_extracted_*.json
"""

import os
import glob
import json
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "vncorenlp_model"))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "material_for_triplets/3_triplets_extracted_nghi_dinh_168_2024_1.json")


def _ensure_jvm():
    cur = os.environ.get("JVM_PATH")
    if cur and os.path.exists(cur):
        return
    home = os.path.expanduser("~")
    cands = []
    jh = os.environ.get("JAVA_HOME")
    if jh:
        cands += [
            os.path.join(jh, "lib/server/libjvm.so"),
            os.path.join(jh, "jre/lib/server/libjvm.so"),
            os.path.join(jh, "lib/server/libjvm.dylib"),
        ]
    cands += glob.glob(os.path.join(home, ".local/share/mise/installs/java/*/lib/server/libjvm.so"))
    cands += glob.glob(os.path.join(home, ".local/share/mise/installs/java/*/lib/server/libjvm.dylib"))
    cands += glob.glob("/usr/lib/jvm/*/lib/server/libjvm.so")
    cands += glob.glob("/Library/Java/JavaVirtualMachines/*/Contents/Home/lib/server/libjvm.dylib")
    cands += glob.glob(os.path.join(home, ".vscode/extensions/redhat.java-*/jre/*/lib/server/libjvm.so"))
    cands += glob.glob(os.path.join(home, ".vscode/extensions/redhat.java-*/jre/*/lib/server/libjvm.dylib"))
    for c in sorted(cands, reverse=True):
        if os.path.exists(c):
            os.environ["JVM_PATH"] = c
            print(f"[jvm] JVM_PATH = {c}")
            return
    print("⚠️  Không tự dò được libjvm — hãy export JVM_PATH thủ công.")


_ensure_jvm()
import py_vncorenlp  # noqa: E402

print(f"Loading VnCoreNLP from: {MODEL_DIR}")
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=MODEL_DIR)
print("VnCoreNLP loaded.")

SUBJECT_LABELS = {"sub", "nsubj"}
DOBJ_LABELS    = {"dob", "dobj"}
PREP_POS       = {"E", "I"}
PASSIVE_MARKERS = {"bị", "được"}
MODAL_VERBS = {
    "muốn", "cần", "phải", "nên", "sẽ", "đang", "đã",
    "có_thể", "không", "hãy", "đừng", "chớ",
}


def clean(w: str) -> str:
    return w.replace("_", " ").strip()


def build_children(sentence: list) -> dict:
    children: dict = {}
    for tok in sentence:
        children.setdefault(tok["head"], []).append(tok)
    return children


def subtree_tokens(idx: int, children: dict) -> list:
    out, stack, seen = [], [idx], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for ch in children.get(cur, []):
            out.append(ch)
            stack.append(ch["index"])
    return out


def phrase_of(idx: int, children: dict, tokens_by_index: dict) -> str:
    head_tok = tokens_by_index.get(idx)
    if not head_tok:
        return ""
    toks = [head_tok] + subtree_tokens(idx, children)
    unique = {t["index"]: t for t in toks}.values()
    return clean(" ".join(t["wordForm"] for t in sorted(unique, key=lambda t: t["index"])))


def find_subject(verb_idx: int, sentence: list, tokens_by_index: dict) -> int | None:
    for tok in sentence:
        if tok["head"] == verb_idx and tok["depLabel"] in SUBJECT_LABELS:
            return tok["index"]
    curr, visited = verb_idx, set()
    while curr in tokens_by_index and curr not in visited:
        visited.add(curr)
        parent_idx = tokens_by_index[curr]["head"]
        if parent_idx == 0:
            break
        for tok in sentence:
            if tok["head"] == parent_idx and tok["depLabel"] in SUBJECT_LABELS:
                return tok["index"]
        curr = parent_idx
    return None


def find_object_idx(verb_idx: int, sentence: list) -> int | None:
    for tok in sentence:
        if tok["head"] == verb_idx and tok["depLabel"] in DOBJ_LABELS:
            return tok["index"]
    for prep in sentence:
        if prep["head"] == verb_idx and prep["posTag"] in PREP_POS:
            for pob in sentence:
                if pob["head"] == prep["index"] and pob["depLabel"] == "pob":
                    return pob["index"]
    return None


def extract_from_proposition(text: str) -> tuple[list[dict], list[dict]]:
    """Return (entities, triplets) extracted by VnCoreNLP from a single proposition."""
    annotated = model.annotate_text(text.replace("_", " "))

    seen_entity_texts: set[str] = set()
    entities: list[dict] = []
    triplets: list[dict] = []

    for _, sentence in annotated.items():
        tokens_by_index = {t["index"]: t for t in sentence}
        children = build_children(sentence)

        # ── SVO from dependency tree ─────────────────────────────────────────
        for tok in sentence:
            word = clean(tok["wordForm"]).lower()
            if tok["posTag"] != "V" or word in MODAL_VERBS:
                continue

            v_idx = tok["index"]
            relation = clean(tok["wordForm"])

            parent = tokens_by_index.get(tok["head"])
            if parent and parent["wordForm"].lower() in PASSIVE_MARKERS:
                relation = f"{clean(parent['wordForm'])} {relation}"

            s_idx = find_subject(v_idx, sentence, tokens_by_index)
            o_idx = find_object_idx(v_idx, sentence)
            if s_idx is None or o_idx is None:
                continue

            s_phrase = phrase_of(s_idx, children, tokens_by_index)
            o_phrase = phrase_of(o_idx, children, tokens_by_index)

            triplets.append({
                "subject":  s_phrase,
                "relation": relation.lower().replace(" ", "_"),
                "object":   o_phrase,
            })

            for phrase, role in [(s_phrase, "subject"), (o_phrase, "object")]:
                if phrase and phrase not in seen_entity_texts:
                    seen_entity_texts.add(phrase)
                    entities.append({"text": phrase, "role": role})

        # ── NER spans (bổ sung entity chưa được capture qua SVO) ────────────
        ner_buf: list[dict] = []
        def _flush_ner():
            if not ner_buf:
                return
            phrase = clean(" ".join(t["wordForm"] for t in ner_buf))
            if phrase and phrase not in seen_entity_texts:
                seen_entity_texts.add(phrase)
                entities.append({"text": phrase, "role": "entity"})
            ner_buf.clear()

        for tok in sentence:
            label = tok.get("nerLabel", "O")
            if label.startswith("B-"):
                _flush_ner()
                ner_buf.append(tok)
            elif label.startswith("I-") and ner_buf:
                ner_buf.append(tok)
            else:
                _flush_ner()
        _flush_ner()

    return entities, triplets


def build_section_result(section: dict) -> dict:
    base = {
        "id":                    section["id"],
        "document_name":         section.get("document_name"),
        "level":                 section.get("level"),
        "path":                  section.get("path"),
        "original_text":         section.get("text_content"),
        "rewritten_propositions": section.get("rewritten_propositions", []),
    }
    props = [p for p in section.get("rewritten_propositions", []) if p and p.strip()]
    if not props:
        return {**base, "raw_entities": [], "raw_triplets": []}

    seen_entities: set[str] = set()
    raw_entities: list[dict] = []
    raw_triplets: list[dict] = []

    for prop in props:
        try:
            ents, trips = extract_from_proposition(prop)
        except Exception as e:
            print(f"  ⚠  [{section['id']}] lỗi prop: {e}")
            ents, trips = [], []

        for e in ents:
            if e["text"] not in seen_entities:
                seen_entities.add(e["text"])
                raw_entities.append(e)
        raw_triplets.extend(trips)

    return {**base, "raw_entities": raw_entities, "raw_triplets": raw_triplets}


def save(path: str, data: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit",  type=int, default=0, help="Giới hạn số section (0 = tất cả)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    if args.limit:
        sections = sections[:args.limit]

    done: dict[str, dict] = {}
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for item in json.load(f):
                if "raw_triplets" in item:
                    done[item["id"]] = item

    to_process = [s for s in sections if s["id"] not in done]
    print(f"Total: {len(sections)} | Done: {len(done)} | Remaining: {len(to_process)}")
    if not to_process:
        return

    t0 = time.time()
    for i, section in enumerate(to_process, 1):
        result = build_section_result(section)
        done[result["id"]] = result

        n_trip = len(result["raw_triplets"])
        n_ent  = len(result["raw_entities"])
        print(f"[{i}/{len(to_process)}] {section['id']} — {n_ent} entities, {n_trip} triplets"
              f"  ({time.time()-t0:.1f}s)")

        if i % 10 == 0 or i == len(to_process):
            ordered = [done[s["id"]] for s in sections if s["id"] in done]
            save(args.output, ordered)

    ordered = [done[s["id"]] for s in sections if s["id"] in done]
    save(args.output, ordered)
    total_trips = sum(len(v.get("raw_triplets", [])) for v in done.values())
    print(f"\nDone. {len(done)} sections, {total_trips} triplets → {args.output}")


if __name__ == "__main__":
    main()
