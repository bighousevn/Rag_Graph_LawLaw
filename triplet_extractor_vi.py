#!/usr/bin/env python3
"""Convert legal sentences to JSON and extract S-R-O triplets using model outputs.

Pipeline requirements:
- py_vncorenlp for Vietnamese word segmentation.
- PhoNLP for POS tagging and dependency parsing.

This script intentionally does not use heuristic fallback extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROTECTED_PHRASES = [
    "ít nhất",
    "không quá",
    "tối thiểu",
    "tối đa",
    "không cần báo trước",
    "phải báo trước",
]

SUBJECT_LABELS = {"nsubj", "nsubj:pass", "csubj", "subj", "sub"}
OBJECT_LABELS = {"obj", "dobj", "dob", "iobj", "obl", "xcomp", "ccomp", "pob"}
REL_EXPAND_LABELS = {"advmod", "adv", "aux", "cop", "neg", "mark"}
ACTION_LINK_LABELS = {"xcomp", "ccomp", "vmod", "advcl", "iob", "iobj"}
BRIDGE_OBJECT_LABELS = {"obj", "dobj", "dob", "iobj", "iob"}
ENTITY_EXPAND_LABELS = {
    "nmod",
    "amod",
    "det",
    "mnr",
    "pob",
    "iob",
    "dob",
    "conj",
    "coord",
    "punct",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_snake_phrase(text: str) -> str:
    cleaned = normalize_spaces(text)
    cleaned = re.sub(r"^[,;:.\-\s]+|[,;:.\-\s]+$", "", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned


def load_stopwords(path: Optional[Path]) -> set[str]:
    if not path or not path.exists():
        return set()
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = normalize_spaces(line).lower()
        if w:
            words.add(w)
    return words


def read_llm_lines(txt_path: Path) -> List[str]:
    lines: List[str] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        s = normalize_spaces(line)
        if s:
            lines.append(s)
    return lines


def read_context_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def read_structured_input_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Input structured JSON phai la mot mang object")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        raw_processed = item.get("llm_processed_text", "")
        processed_sentences: List[str] = []

        if isinstance(raw_processed, list):
            for sentence in raw_processed:
                text = normalize_spaces(str(sentence))
                if text:
                    processed_sentences.append(text)
        else:
            text = normalize_spaces(str(raw_processed))
            if text:
                processed_sentences.append(text)

        if not processed_sentences:
            continue

        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_parts = [
            metadata.get("chuong"),
            metadata.get("dieu"),
            metadata.get("khoan"),
            metadata.get("diem"),
        ]
        source = " | ".join(str(x) for x in source_parts if x)
        base_id = str(item.get("id") or f"sent_{idx:06d}")

        for atomic_idx, sentence_text in enumerate(processed_sentences, start=1):
            # Keep IDs stable for single-sentence records, expand IDs for atomic lists.
            if len(processed_sentences) == 1:
                record_id = base_id
            else:
                record_id = f"{base_id}_{atomic_idx:02d}"

            out.append(
                {
                    "id": record_id,
                    "parent_id": base_id,
                    "atomic_index": atomic_idx,
                    "text": sentence_text,
                    "source": source or f"index_{idx}",
                    "metadata": metadata,
                    "original_text": normalize_spaces(str(item.get("original_text", ""))),
                    "llm_processed_text": sentence_text,
                }
            )

    return out


def build_source_string(ctx: Dict[str, Any]) -> str:
    parts = [ctx.get("chapter"), ctx.get("muc"), ctx.get("dieu"), ctx.get("khoan"), ctx.get("diem")]
    return " | ".join(str(x) for x in parts if x)


def build_sentence_records(lines: Sequence[str], contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, sentence in enumerate(lines, start=1):
        rec: Dict[str, Any] = {
            "id": f"sent_{i:06d}",
            "text": sentence,
        }
        if i - 1 < len(contexts):
            rec["source"] = build_source_string(contexts[i - 1])
        else:
            rec["source"] = f"index_{i}"
        out.append(rec)
    return out


def remove_stopwords_phrase(text: str, stopwords: set[str]) -> str:
    if not text:
        return text

    protected_ranges: List[tuple[int, int]] = []
    lower = text.lower()
    for phrase in PROTECTED_PHRASES:
        start = 0
        pl = phrase.lower()
        while True:
            idx = lower.find(pl, start)
            if idx == -1:
                break
            protected_ranges.append((idx, idx + len(pl)))
            start = idx + len(pl)

    tokens = text.split()
    rebuilt: List[str] = []
    cursor = 0
    for tok in tokens:
        idx = lower.find(tok.lower(), cursor)
        if idx == -1:
            idx = cursor
        cursor = idx + len(tok)
        in_protected = any(idx >= s and idx < e for s, e in protected_ranges)
        if in_protected:
            rebuilt.append(tok)
            continue

        raw = re.sub(r"^[^\w%]+|[^\w%]+$", "", tok, flags=re.UNICODE).lower()
        if raw and raw in stopwords:
            continue
        rebuilt.append(tok)
    return normalize_spaces(" ".join(rebuilt))


def ensure_models_available() -> None:
    if shutil.which("java") is None:
        raise RuntimeError("Khong tim thay Java. VnCoreNLP can Java runtime de chay.")


def init_vncorenlp(model_dir: Path):
    import py_vncorenlp  # type: ignore

    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    jar_path = model_dir / "VnCoreNLP-1.2.jar"
    models_dir = model_dir / "models"
    if (not jar_path.exists()) or (not models_dir.exists()):
        py_vncorenlp.download_model(save_dir=str(model_dir))
    cwd = os.getcwd()
    model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(model_dir))
    os.chdir(cwd)
    return model


def init_phonlp(model_dir: Path):
    import phonlp  # type: ignore

    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    if not any(model_dir.iterdir()):
        phonlp.download(save_dir=str(model_dir))
    return phonlp.load(save_dir=str(model_dir))


def vncore_tokenize(vncore, sentence: str) -> List[str]:
    ann = vncore.annotate_text(sentence)
    if not isinstance(ann, dict) or not ann:
        raise RuntimeError("VnCoreNLP khong tra ve token hop le")

    first_sent = ann[min(ann.keys())]
    tokens: List[str] = []
    for item in first_sent:
        word = item.get("wordForm") if isinstance(item, dict) else None
        if isinstance(word, str) and word:
            tokens.append(word)
    if not tokens:
        raise RuntimeError("VnCoreNLP tra ve danh sach token rong")
    return tokens


def phonnlp_parse(ph_model, segmented_text: str) -> Dict[str, Any]:
    result = ph_model.annotate(text=segmented_text)
    if isinstance(result, tuple) and len(result) >= 4:
        words_raw, pos_raw, _ner_raw, dep_raw = result[:4]

        if not (isinstance(words_raw, list) and words_raw):
            raise RuntimeError("PhoNLP output tuple khong co words")

        words = words_raw[0]
        pos_sent = pos_raw[0] if isinstance(pos_raw, list) and pos_raw else []
        dep_sent = dep_raw[0] if isinstance(dep_raw, list) and dep_raw else []

        if not (isinstance(words, list) and isinstance(pos_sent, list) and isinstance(dep_sent, list)):
            raise RuntimeError("PhoNLP tuple output co cau truc khong hop le")

        pos: List[str] = []
        for p in pos_sent:
            if isinstance(p, list) and p:
                pos.append(str(p[0]))
            else:
                pos.append(str(p))

        head: List[int] = []
        dep: List[str] = []
        for item in dep_sent:
            if isinstance(item, list) and len(item) >= 2:
                head.append(int(item[0]))
                dep.append(str(item[1]))
            else:
                head.append(0)
                dep.append("dep")

        if not (len(words) == len(pos) == len(dep) == len(head)):
            raise RuntimeError("PhoNLP tuple output do dai khong khop")

        return {
            "tokens": [str(x) for x in words],
            "pos": pos,
            "dep": dep,
            "head": head,
        }

    if isinstance(result, dict):
        # Fallback for other PhoNLP return schemas.
        words = result.get("word") or result.get("words")
        pos = result.get("pos") or result.get("upos")
        dep = result.get("dep") or result.get("deprel")
        head = result.get("head") or result.get("heads")

        if isinstance(words, list) and words and isinstance(words[0], list):
            words = words[0]
        if isinstance(pos, list) and pos and isinstance(pos[0], list):
            pos = pos[0]
        if isinstance(dep, list) and dep and isinstance(dep[0], list):
            dep = dep[0]
        if isinstance(head, list) and head and isinstance(head[0], list):
            head = head[0]

        if not all(isinstance(x, list) for x in [words, pos, dep, head]):
            raise RuntimeError("PhoNLP dict output khong co du word/pos/dep/head")

        if not (len(words) == len(pos) == len(dep) == len(head)):
            raise RuntimeError("PhoNLP dict output do dai khong khop")

        return {
            "tokens": [str(x) for x in words],
            "pos": [str(x) for x in pos],
            "dep": [str(x) for x in dep],
            "head": [int(x) for x in head],
        }

    raise RuntimeError("PhoNLP khong tra ve schema duoc ho tro")


def gather_phrase(indices: List[int], tokens: List[str]) -> str:
    if not indices:
        return ""
    uniq = sorted(set(indices))
    return normalize_spaces(" ".join(tokens[i - 1] for i in uniq if 1 <= i <= len(tokens))).replace("_", " ")


def expand_entity_indices(seed_idxs: List[int], dep: List[str], head: List[int]) -> List[int]:
    if not seed_idxs:
        return []

    collected = set(seed_idxs)
    changed = True
    while changed:
        changed = False
        for i, d in enumerate(dep, start=1):
            parent = head[i - 1]
            if parent in collected and d.lower() in ENTITY_EXPAND_LABELS and i not in collected:
                collected.add(i)
                changed = True
    return sorted(collected)


def expand_relation_indices(root_idx: int, dep: List[str], head: List[int], tokens: List[str]) -> List[int]:
    idxs = [root_idx]
    for i, d in enumerate(dep, start=1):
        if head[i - 1] == root_idx and d.lower() in REL_EXPAND_LABELS:
            idxs.append(i)

    relation_text = gather_phrase(idxs, tokens).lower()
    for phrase in PROTECTED_PHRASES:
        if phrase in relation_text:
            continue
        if phrase in normalize_spaces(" ".join(tokens)).replace("_", " ").lower() and phrase in {"ít nhất", "không quá", "tối thiểu", "tối đa"}:
            idxs.insert(0, root_idx)
    return sorted(set(idxs))


def is_verb_pos(tag: str) -> bool:
    upper = tag.upper()
    return upper.startswith("V") or upper == "VERB"


def is_descendant(node_idx: int, ancestor_idx: int, head: List[int]) -> bool:
    current = node_idx
    seen = set()
    while 1 <= current <= len(head):
        parent = head[current - 1]
        if parent == 0:
            return False
        if parent == ancestor_idx:
            return True
        if parent in seen:
            return False
        seen.add(parent)
        current = parent
    return False


def path_to_ancestor(node_idx: int, ancestor_idx: int, head: List[int]) -> List[int]:
    path: List[int] = []
    current = node_idx
    seen = set()
    while 1 <= current <= len(head) and current not in seen:
        path.append(current)
        if current == ancestor_idx:
            return path
        seen.add(current)
        current = head[current - 1]
    return []


def pick_controlled_action_idx(root_idx: int, dep: List[str], head: List[int], pos: List[str]) -> int:
    candidates = [
        i
        for i, d in enumerate(dep, start=1)
        if head[i - 1] == root_idx and d.lower() in ACTION_LINK_LABELS and is_verb_pos(pos[i - 1])
    ]
    if not candidates:
        return -1
    # Prefer right-side verbal complements because they usually encode the required action.
    right_side = [i for i in candidates if i > root_idx]
    return right_side[0] if right_side else candidates[0]


def descend_action_verb(action_idx: int, dep: List[str], head: List[int], pos: List[str]) -> int:
    if action_idx == -1:
        return -1

    current = action_idx
    seen = set()
    while current not in seen:
        seen.add(current)
        children = [
            i
            for i, d in enumerate(dep, start=1)
            if head[i - 1] == current and d.lower() in ACTION_LINK_LABELS and is_verb_pos(pos[i - 1])
        ]
        if not children:
            break
        right_side = [i for i in children if i > current]
        current = right_side[0] if right_side else children[0]
    return current


def pick_action_via_object_bridge(root_idx: int, dep: List[str], head: List[int], pos: List[str]) -> int:
    bridge_idxs = [
        i
        for i, d in enumerate(dep, start=1)
        if head[i - 1] == root_idx and d.lower() in BRIDGE_OBJECT_LABELS
    ]
    if not bridge_idxs:
        return -1

    candidates: List[int] = []
    for i, d in enumerate(dep, start=1):
        if d.lower() not in ACTION_LINK_LABELS or not is_verb_pos(pos[i - 1]):
            continue
        if any(is_descendant(i, bridge, head) for bridge in bridge_idxs):
            candidates.append(i)

    if not candidates:
        return -1

    right_side = [i for i in sorted(candidates) if i > root_idx]
    return right_side[0] if right_side else sorted(candidates)[0]


def collect_relation_bridge_indices(root_idx: int, action_idx: int, dep: List[str], head: List[int]) -> List[int]:
    if action_idx == -1:
        return []

    idxs: set[int] = set()

    path = path_to_ancestor(action_idx, root_idx, head)
    idxs.update(path)

    bridges = [
        i
        for i, d in enumerate(dep, start=1)
        if head[i - 1] == root_idx and d.lower() in BRIDGE_OBJECT_LABELS and (i < action_idx or action_idx == -1)
    ]
    for b in bridges:
        idxs.update(expand_entity_indices([b], dep, head))

    return sorted(idxs)


def pick_action_object_indices(action_idx: int, dep: List[str], head: List[int], pos: List[str]) -> List[int]:
    direct_labels = {"obj", "dobj", "dob", "iobj", "obl", "pob"}
    direct = [
        i
        for i, d in enumerate(dep, start=1)
        if head[i - 1] == action_idx and d.lower() in direct_labels
    ]
    if direct:
        return direct

    # Fallback: take nominal descendants on the right side of the action verb.
    fallback = [
        i
        for i in range(action_idx + 1, len(dep) + 1)
        if is_descendant(i, action_idx, head)
        and (pos[i - 1].upper().startswith("N") or dep[i - 1].lower() in {"nmod", "obj", "obl", "pob", "dob"})
    ]
    return fallback


def extract_triplet_from_dependencies(parsed: Dict[str, Any], stopwords: set[str]) -> Dict[str, Any]:
    tokens: List[str] = parsed["tokens"]
    pos: List[str] = parsed["pos"]
    dep: List[str] = parsed["dep"]
    head: List[int] = parsed["head"]

    root_idx = -1
    for i, d in enumerate(dep, start=1):
        if d.lower() == "root" or head[i - 1] == 0:
            root_idx = i
            break
    if root_idx == -1:
        raise RuntimeError("Khong tim thay ROOT tu PhoNLP")

    subj_idxs = [i for i, d in enumerate(dep, start=1) if head[i - 1] == root_idx and d.lower() in SUBJECT_LABELS]
    obj_idxs = [i for i, d in enumerate(dep, start=1) if head[i - 1] == root_idx and d.lower() in OBJECT_LABELS]

    # Some PhoNLP parses use compact tags and may miss direct sub/obj edges.
    if not subj_idxs:
        subj_idxs = [
            i
            for i in range(1, root_idx)
            if pos[i - 1].upper().startswith("N") or dep[i - 1].lower() in SUBJECT_LABELS
        ]
        if subj_idxs:
            subj_idxs = [subj_idxs[-1]]

    if not obj_idxs:
        obj_idxs = [
            i
            for i in range(root_idx + 1, len(tokens) + 1)
            if pos[i - 1].upper().startswith("N") or dep[i - 1].lower() in OBJECT_LABELS
        ]

    action_idx = pick_controlled_action_idx(root_idx, dep, head, pos)
    if action_idx == -1:
        action_idx = pick_action_via_object_bridge(root_idx, dep, head, pos)
    action_idx = descend_action_verb(action_idx, dep, head, pos)

    # For causative structures (e.g., "yeu cau ... thuc hien ..."), pull object from the embedded action.
    if action_idx != -1:
        action_obj_idxs = pick_action_object_indices(action_idx, dep, head, pos)
        if action_obj_idxs:
            obj_idxs = action_obj_idxs

    subj_full_idxs = expand_entity_indices(subj_idxs, dep, head)
    obj_full_idxs = expand_entity_indices(obj_idxs, dep, head)

    relation_idxs = expand_relation_indices(root_idx, dep, head, tokens)
    if action_idx != -1:
        action_relation = expand_relation_indices(action_idx, dep, head, tokens)
        bridge_relation = collect_relation_bridge_indices(root_idx, action_idx, dep, head)
        relation_idxs = sorted(set(relation_idxs + action_relation + bridge_relation))

    relation_raw = gather_phrase(relation_idxs, tokens)
    subject_raw = gather_phrase(subj_full_idxs, tokens)
    object_raw = gather_phrase(obj_full_idxs, tokens)

    # If dependency object is empty, include right-side tokens of ROOT that are content words.
    if not object_raw:
        right = [
            i
            for i in range(root_idx + 1, len(tokens) + 1)
            if pos[i - 1].upper() not in {"PUNCT", "SCONJ"}
        ]
        object_raw = gather_phrase(right, tokens)

    subject_clean = to_snake_phrase(remove_stopwords_phrase(subject_raw.replace("_", " "), stopwords))
    relation_clean = to_snake_phrase(remove_stopwords_phrase(relation_raw.replace("_", " "), stopwords))
    object_clean = to_snake_phrase(remove_stopwords_phrase(object_raw.replace("_", " "), stopwords))

    if not subject_clean:
        raise RuntimeError("Khong trich xuat duoc subject tu parse model")
    if not relation_clean:
        raise RuntimeError("Khong trich xuat duoc relation tu parse model")

    return {
        "triplet": {
            "subject": subject_clean,
            "relation": relation_clean,
            "object": object_clean,
        },
        "dependency": {
            "root_index": root_idx,
            "tokens": tokens,
            "pos": pos,
            "head": head,
            "dep": dep,
        },
    }


def extract_triplets_with_models(
    records: Sequence[Dict[str, Any]],
    stopwords: set[str],
    vncore_model_dir: Path,
    phonlp_model_dir: Path,
) -> List[Dict[str, Any]]:
    ensure_models_available()
    vncore = init_vncorenlp(vncore_model_dir)
    ph_model = init_phonlp(phonlp_model_dir)

    out: List[Dict[str, Any]] = []
    for rec in records:
        sentence = rec["text"]
        wseg_tokens = vncore_tokenize(vncore, sentence)
        segmented_text = " ".join(wseg_tokens)
        parsed = phonnlp_parse(ph_model, segmented_text)
        extracted = extract_triplet_from_dependencies(parsed, stopwords)

        out.append(
            {
                "id": rec["id"],
                "source": rec.get("source", ""),
                "metadata": rec.get("metadata", {}),
                "original_text": rec.get("original_text", ""),
                "llm_processed_text": rec.get("llm_processed_text", sentence),
                "text": sentence,
                "tokenization": wseg_tokens,
                "triplet": extracted["triplet"],
                "dependency": extracted["dependency"],
                "method": "model_phonnlp_vncorenlp",
            }
        )
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert LLM text to JSON and extract legal triplets with models")
    p.add_argument("--input-structured-json", type=Path, default=None, help="Structured JSON input with fields: id, metadata, original_text, llm_processed_text")
    p.add_argument("--input-txt", type=Path, default=Path("llm_pre.txt"), help="Input text file with one normalized sentence per line")
    p.add_argument("--contexts-jsonl", type=Path, default=Path("contexts_dieu_17_35_48.jsonl"), help="Optional context JSONL to map source metadata by line index")
    p.add_argument("--sentences-output", type=Path, default=Path("llm_pre_sentences_model.json"), help="Output JSON containing sentence objects")
    p.add_argument("--triplets-output", type=Path, default=Path("llm_pre_triplets_model.json"), help="Output JSON containing extracted triplets")
    p.add_argument("--stopwords-file", type=Path, default=Path("vietnamese_stopwords_legal.txt"), help="Vietnamese stopwords file")
    p.add_argument("--vncore-model-dir", type=Path, default=Path(".models/vncorenlp"), help="Directory for py_vncorenlp model files")
    p.add_argument("--phonlp-model-dir", type=Path, default=Path(".models/phonlp"), help="Directory for PhoNLP model files")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.input_structured_json is not None:
        if not args.input_structured_json.exists():
            raise FileNotFoundError(f"Khong tim thay input structured JSON: {args.input_structured_json}")
        records = read_structured_input_json(args.input_structured_json)
        if not records:
            raise RuntimeError("Input structured JSON khong co ban ghi hop le de xu ly.")
    else:
        if not args.input_txt.exists():
            raise FileNotFoundError(f"Khong tim thay input text: {args.input_txt}")

        lines = read_llm_lines(args.input_txt)
        if not lines:
            raise RuntimeError("Input text file khong co dong nao de xu ly.")

        contexts = read_context_jsonl(args.contexts_jsonl)
        records = build_sentence_records(lines, contexts)

    write_json(args.sentences_output, records)
    print(f"Da tao file JSON cau: {args.sentences_output} ({len(records)} cau)")

    stopwords = load_stopwords(args.stopwords_file)
    triplets = extract_triplets_with_models(
        records=records,
        stopwords=stopwords,
        vncore_model_dir=args.vncore_model_dir,
        phonlp_model_dir=args.phonlp_model_dir,
    )
    write_json(args.triplets_output, triplets)
    print(f"Da tao file JSON triplet: {args.triplets_output} ({len(triplets)} bo ba)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
