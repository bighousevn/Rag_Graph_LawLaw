"""
Bước 3b: Trích xuất raw_entities + raw_triplets từ structured token JSON (output của 3a_tokenize.py).
Thuật toán thuần đọc cây phụ thuộc VnCoreNLP — KHÔNG dùng LLM.
Input:  material_for_triplets/3a_tokenized_*.json
Output: material_for_triplets/3_triplets_extracted_*.json

Pattern A — verb-root:
    V.depLabel ∈ {root, dep, vmod, ccomp, xcomp, conj}
    → (subject_phrase, V, object_phrase)

Pattern B — noun-root + embedded verb:
    root.posTag ∈ N/Np/Nu/Nc  AND  child V.depLabel = nmod
    → (root_noun_only, V, object_phrase)
    Ví dụ: "Người [điều_khiển] xe máy"  →  (Người, điều_khiển, xe máy)

Phụ trợ:
    Negation  — R[không/chưa/đừng] trước V  → prefix "không_"
    Passive   — parent ∈ {bị, được}          → prefix "bị_"/"được_"
    Prep-obj  — V → E/I → pob
"""

import os
import json
import time
import argparse

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "material_for_triplets/3a_tokenized_nghi_dinh_168_2024_1.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "material_for_triplets/3_triplets_extracted_nghi_dinh_168_2024_1.json")

SUBJECT_LABELS  = {"sub", "nsubj"}
DOBJ_LABELS     = {"dob", "dobj"}
PREP_POS        = {"E", "I"}
NEGATION_WORDS  = {"không", "chưa", "chớ", "đừng"}
PASSIVE_MARKERS = {"bị", "được"}

# Verb có dep label này → IS chính ngữ (Pattern A)
PRED_DEPS = {"root", "dep", "vmod", "ccomp", "xcomp", "conj"}
# Verb có dep label này và là con của N-root → Pattern B
EMBEDDED_DEPS = {"nmod"}

SKIP_VERBS = {
    "muốn", "cần", "phải", "nên", "sẽ", "đang", "đã", "hãy",
    "có_thể", "là", "bao_gồm", "gồm", "quy_định",
}

ROOT_NOUN_POS = {"N", "Np", "Nu", "Nc"}


# ── Tiện ích ──────────────────────────────────────────────────────────────────

def clean(w: str) -> str:
    return w.replace("_", " ").strip()


def build_maps(sentence: list[dict]) -> tuple[dict, dict]:
    tok_by_idx: dict[int, dict] = {t["index"]: t for t in sentence}
    children: dict[int, list[dict]] = {}
    for t in sentence:
        children.setdefault(t["head"], []).append(t)
    return tok_by_idx, children


# ── Xây dựng cụm từ ──────────────────────────────────────────────────────────

def get_phrase(head_idx: int, children: dict, tok_by_idx: dict,
               is_subject: bool = False) -> str:
    """
    Thu thập cụm danh từ gốc tại head_idx.

    is_subject=True:  bỏ qua TẤT CẢ nhánh V (không kéo predicate vào subject).
    is_subject=False: cho phép V-child ở độ sâu 0→1 (compound noun: "giấy phép lái xe").

    Dừng tại:
      - CH (dấu câu)
      - coord/conj (danh sách liệt kê — không kéo hết enumerations)
    """
    collected: dict[int, dict] = {}

    def collect(idx: int, depth: int) -> None:
        tok = tok_by_idx.get(idx)
        if not tok:
            return
        if tok["posTag"] == "CH":
            return
        collected[idx] = tok
        if depth >= 3:
            return
        for child in children.get(idx, []):
            if child["posTag"] == "CH":
                continue
            if child["depLabel"] in {"coord", "conj", "punct"}:
                continue
            if child["posTag"] == "V":
                if is_subject:
                    continue          # subject: bỏ hẳn V-branch
                if depth > 0:
                    continue          # object: chỉ cho phép V ở depth 0→1
            collect(child["index"], depth + 1)

    collect(head_idx, 0)
    if not collected:
        return ""
    ordered = sorted(collected.values(), key=lambda t: t["index"])
    return clean(" ".join(t["wordForm"] for t in ordered))


# ── Xây dựng relation ─────────────────────────────────────────────────────────

def build_relation(tok: dict, children: dict, tok_by_idx: dict) -> str:
    """Tên relation: thêm tiền tố phủ định và bị động nếu có."""
    base = clean(tok["wordForm"])

    # Phủ định: R child đứng trước verb có wordForm ∈ NEGATION_WORDS
    for child in children.get(tok["index"], []):
        if (child["posTag"] == "R"
                and child["wordForm"].lower() in NEGATION_WORDS
                and child["index"] < tok["index"]):
            base = f"{clean(child['wordForm'])} {base}"
            break

    # Bị động: cha của verb là bị/được
    parent = tok_by_idx.get(tok["head"])
    if parent and parent["wordForm"].lower() in PASSIVE_MARKERS:
        base = f"{clean(parent['wordForm'])} {base}"

    return base.lower().replace(" ", "_")


# ── Tìm subject / object ──────────────────────────────────────────────────────

def find_subject(verb_idx: int, tok_by_idx: dict, children: dict,
                 max_up: int = 3) -> int | None:
    """Tìm subject index: con trực tiếp trước, leo cây tối đa max_up bước."""
    for child in children.get(verb_idx, []):
        if child["depLabel"] in SUBJECT_LABELS:
            return child["index"]

    curr, visited, hops = verb_idx, set(), 0
    while curr in tok_by_idx and curr not in visited and hops < max_up:
        visited.add(curr)
        parent_idx = tok_by_idx[curr]["head"]
        if parent_idx == 0:
            break
        for child in children.get(parent_idx, []):
            if child["depLabel"] in SUBJECT_LABELS:
                return child["index"]
        curr = parent_idx
        hops += 1
    return None


def find_objects(verb_idx: int, children: dict) -> list[int]:
    """Trả về danh sách object index: dob/dobj trước, nếu không có thì pob qua giới từ."""
    objs: list[int] = [
        ch["index"]
        for ch in children.get(verb_idx, [])
        if ch["depLabel"] in DOBJ_LABELS
    ]

    if not objs:
        for ch in children.get(verb_idx, []):
            if ch["posTag"] in PREP_POS:
                objs += [
                    gc["index"]
                    for gc in children.get(ch["index"], [])
                    if gc["depLabel"] == "pob"
                ]

    return objs


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_from_sentence(sentence: list[dict]) -> tuple[list[dict], list[dict]]:
    tok_by_idx, children = build_maps(sentence)

    seen_keys: set[tuple] = set()
    seen_ent:  set[str]   = set()
    entities:  list[dict] = []
    triplets:  list[dict] = []

    def emit(s: str, r: str, o: str) -> None:
        key = (s.lower().strip(), r, o.lower().strip())
        if not s or not o or key in seen_keys:
            return
        seen_keys.add(key)
        triplets.append({"subject": s, "relation": r, "object": o})
        for phrase, role in [(s, "subject"), (o, "object")]:
            if phrase not in seen_ent:
                seen_ent.add(phrase)
                entities.append({"text": phrase, "role": role})

    # ── Pattern A: V là predicate chính ──────────────────────────────────────
    for tok in sentence:
        if tok["posTag"] != "V":
            continue
        if tok["wordForm"].lower() in SKIP_VERBS:
            continue
        if tok["depLabel"] not in PRED_DEPS:
            continue

        rel    = build_relation(tok, children, tok_by_idx)
        s_idx  = find_subject(tok["index"], tok_by_idx, children)
        o_idxs = find_objects(tok["index"], children)

        if s_idx is None or not o_idxs:
            continue

        s_phrase = get_phrase(s_idx, children, tok_by_idx, is_subject=True)
        for o_idx in o_idxs:
            o_phrase = get_phrase(o_idx, children, tok_by_idx, is_subject=False)
            emit(s_phrase, rel, o_phrase)

    # ── Pattern B: root là danh từ, V-nmod là predicate nhúng ────────────────
    # Ví dụ: Người(root) [điều_khiển](nmod,V) xe_máy(dob)
    for tok in sentence:
        if tok["head"] != 0:
            continue
        if tok["posTag"] not in ROOT_NOUN_POS:
            continue

        s_head = clean(tok["wordForm"])   # chỉ lấy head noun, không lấy cả cụm

        for child in children.get(tok["index"], []):
            if child["posTag"] != "V":
                continue
            if child["wordForm"].lower() in SKIP_VERBS:
                continue
            if child["depLabel"] not in EMBEDDED_DEPS:
                continue

            rel    = build_relation(child, children, tok_by_idx)
            o_idxs = find_objects(child["index"], children)
            for o_idx in o_idxs:
                o_phrase = get_phrase(o_idx, children, tok_by_idx, is_subject=False)
                emit(s_head, rel, o_phrase)

    return entities, triplets


# ── Xử lý proposition / section ──────────────────────────────────────────────

def extract_from_proposition(prop_data: dict) -> tuple[list[dict], list[dict]]:
    seen_ent: set[str] = set()
    entities: list[dict] = []
    triplets: list[dict] = []

    for sentence in prop_data.get("sentences", []):
        if not sentence:
            continue
        try:
            ents, trips = extract_from_sentence(sentence)
        except Exception as e:
            print(f"    ⚠  lỗi sentence: {e}")
            continue
        for e in ents:
            if e["text"] not in seen_ent:
                seen_ent.add(e["text"])
                entities.append(e)
        triplets.extend(trips)

    return entities, triplets


def build_section_result(section: dict) -> dict:
    base = {
        "id":                     section["id"],
        "document_name":          section.get("document_name"),
        "level":                  section.get("level"),
        "path":                   section.get("path"),
        "original_text":          section.get("original_text"),
        "rewritten_propositions": section.get("rewritten_propositions", []),
    }
    props = section.get("tokenized_propositions", [])
    if not props:
        return {**base, "raw_entities": [], "raw_triplets": []}

    seen_ent:    set[str]   = set()
    raw_entities: list[dict] = []
    raw_triplets: list[dict] = []

    for prop in props:
        ents, trips = extract_from_proposition(prop)
        for e in ents:
            if e["text"] not in seen_ent:
                seen_ent.add(e["text"])
                raw_entities.append(e)
        raw_triplets.extend(trips)

    return {**base, "raw_entities": raw_entities, "raw_triplets": raw_triplets}


# ── I/O ───────────────────────────────────────────────────────────────────────

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
                if "raw_triplets" in item:
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
        print(f"[{i}/{len(to_process)}] {section['id']} — "
              f"{len(result['raw_entities'])} entities, {len(result['raw_triplets'])} triplets"
              f"  ({time.time()-t0:.1f}s)")

        if i % 20 == 0 or i == len(to_process):
            ordered = [done[s["id"]] for s in sections if s["id"] in done]
            save(args.output, ordered)

    total = sum(len(v.get("raw_triplets", [])) for v in done.values())
    print(f"\nDone. {len(done)} sections, {total} triplets → {args.output}")


if __name__ == "__main__":
    main()
