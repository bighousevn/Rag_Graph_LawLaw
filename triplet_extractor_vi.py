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
REL_EXPAND_LABELS = {"advmod", "adv", "aux", "cop", "neg", "mark", "vmod"}
ACTION_LINK_LABELS = {"xcomp", "ccomp", "vmod", "advcl", "iob", "iobj"}
BRIDGE_OBJECT_LABELS = {"obj", "dobj", "dob", "iobj", "iob"}
CONDITION_MARKERS = {
    "nếu",
    "khi",
    "trong trường hợp",
    "trường hợp",
}
ENTITY_EXPAND_LABELS = {
    "nmod",
    "amod",
    "det",
    "mnr",
    "loc",
    "tmp",
    "tmod",
    "vmod",
    "pob",
    "iob",
    "dob",
    "conj",
    "coord",
    "punct",
}

OBJECT_SPLIT_PATTERNS = [
    r"\s*,\s*",
    r"\s+và\s+",
    r"\s+hoặc\s+",
    r"\s+hay\s+",
    r"\s+cùng\s+",
]

MICRO_SPLIT_PATTERNS = [
    r"\s*[;:.!?]+\s*",
    r"\s*,\s*(?=(nếu|khi|do|vì|để|trừ khi|trong trường hợp)\b)",
    r"\s+và\s+(?=(phải|được|không|có|bị|trách nhiệm|nghĩa vụ|quyền)\b)",
    r"\s+hoặc\s+",
    r"\s+hay\s+",
    r"\s+đồng thời\s+",
]

FILLER_WORDS = {
    "thì",
    "là",
    "việc",
    "được",
    "sẽ",
    "đang",
    "này",
    "kia",
    "ấy",
    "đó",
    "các",
    "những",
}

CORE_SINGLE_WORD_KEEP = {
    "người",
    "sử",
    "dụng",
    "lao",
    "động",
    "hợp",
    "đồng",
    "lương",
    "quyền",
    "nghĩa",
    "vụ",
    "bảo",
    "hiểm",
    "kỷ",
    "luật",
    "đình",
    "công",
    "tranh",
    "chấp",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_format_word_keywords(path: Optional[Path]) -> set[str]:
    if not path or not path.exists():
        return set()

    data = json.loads(path.read_text(encoding="utf-8"))
    out: set[str] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            normalized = normalize_spaces(obj).lower()
            if normalized:
                out.add(normalized)
            return
        if isinstance(obj, list):
            for item in obj:
                _walk(item)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                _walk(value)

    _walk(data)
    return out


def prune_fragment_text(text: str, stopwords: set[str]) -> str:
    # Keep protected phrases intact while removing low-value tokens.
    lowered = text.lower()
    protected_ranges: List[tuple[int, int]] = []
    for phrase in PROTECTED_PHRASES:
        start = 0
        p = phrase.lower()
        while True:
            idx = lowered.find(p, start)
            if idx == -1:
                break
            protected_ranges.append((idx, idx + len(p)))
            start = idx + len(p)

    tokens = text.split()
    cursor = 0
    kept: List[str] = []
    for tok in tokens:
        idx = lowered.find(tok.lower(), cursor)
        if idx == -1:
            idx = cursor
        cursor = idx + len(tok)

        in_protected = any(idx >= s and idx < e for s, e in protected_ranges)
        if in_protected:
            kept.append(tok)
            continue

        raw = re.sub(r"^[^\w%]+|[^\w%]+$", "", tok, flags=re.UNICODE).lower()
        if not raw:
            continue
        if raw in FILLER_WORDS:
            continue
        if raw in stopwords and raw not in {"không", "phải", "được", "cần"}:
            continue
        kept.append(tok)

    compact = normalize_spaces(" ".join(kept))
    return compact if compact else normalize_spaces(text)


def extract_keyword_windows(fragment: str, keyword_lexicon: set[str], max_words: int = 18) -> List[str]:
    if not keyword_lexicon:
        return []

    frag = normalize_spaces(fragment)
    lower = frag.lower()
    windows: List[str] = []

    # Use direct string matching; lexicon size is small enough for legal keyword lists.
    hits = [kw for kw in keyword_lexicon if kw in lower]
    if not hits:
        return []

    words = frag.split()
    lower_words = [w.lower() for w in words]
    for kw in sorted(hits, key=len, reverse=True)[:6]:
        kw_words = kw.split()
        k = len(kw_words)
        if k == 0 or k > len(words):
            continue
        for i in range(0, len(words) - k + 1):
            if lower_words[i : i + k] != kw_words:
                continue
            left = max(0, i - max_words // 2)
            right = min(len(words), i + k + max_words // 2)
            candidate = normalize_spaces(" ".join(words[left:right]))
            if candidate:
                windows.append(candidate)

    return windows


def build_keyword_token_lexicon(keyword_lexicon: set[str]) -> set[str]:
    tokens: set[str] = set()
    for phrase in keyword_lexicon:
        for w in normalize_spaces(phrase).split():
            lw = w.lower()
            if len(lw) >= 2:
                tokens.add(lw)
    return tokens


def extract_micro_tokens(fragment: str, stopwords: set[str], keyword_token_lexicon: set[str]) -> List[str]:
    text = normalize_spaces(fragment)
    if not text:
        return []

    words = [
        re.sub(r"^[^\w%]+|[^\w%]+$", "", w, flags=re.UNICODE).lower()
        for w in text.split()
    ]

    out: List[str] = []
    seen: set[str] = set()
    for w in words:
        if not w or len(w) < 2:
            continue
        if w in FILLER_WORDS:
            continue

        is_keyword_token = w in keyword_token_lexicon or w in CORE_SINGLE_WORD_KEEP
        if w in stopwords and not is_keyword_token:
            continue

        if w in seen:
            continue
        seen.add(w)
        out.append(w)

    return out


def split_sentence_super_micro(sentence: str, keyword_lexicon: set[str], stopwords: set[str]) -> List[str]:
    base = normalize_spaces(sentence)
    if not base:
        return []

    fragments = [base]
    for pattern in MICRO_SPLIT_PATTERNS:
        next_fragments: List[str] = []
        for frag in fragments:
            parts = [normalize_spaces(x) for x in re.split(pattern, frag, flags=re.IGNORECASE) if normalize_spaces(x)]
            if parts:
                next_fragments.extend(parts)
        fragments = next_fragments if next_fragments else fragments

    keyword_token_lexicon = build_keyword_token_lexicon(keyword_lexicon)

    expanded: List[str] = []
    for frag in fragments:
        expanded.append(frag)
        expanded.extend(extract_keyword_windows(frag, keyword_lexicon))
        expanded.extend(extract_micro_tokens(frag, stopwords, keyword_token_lexicon))

    deduped: List[str] = []
    seen: set[str] = set()
    for frag in expanded:
        # Keep raw fragment text, do not aggressively prune filler words.
        compact = normalize_spaces(frag)
        compact = re.sub(r"^[,;:.\-\s]+|[,;:.\-\s]+$", "", compact)
        compact = normalize_spaces(compact)
        # Keep 1-word fragments only when they are legally meaningful tokens.
        if len(compact.split()) < 2 and compact.lower() not in keyword_token_lexicon and compact.lower() not in CORE_SINGLE_WORD_KEEP:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(compact)

    return deduped if deduped else [base]


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


def read_structured_input_json(path: Path, keyword_lexicon: set[str], stopwords: set[str]) -> List[Dict[str, Any]]:
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

        if processed_sentences:
            # Preserve order while removing duplicate atomic sentences.
            uniq: List[str] = []
            seen: set[str] = set()
            for s in processed_sentences:
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(s)
            processed_sentences = uniq

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


def build_sentence_records(
    lines: Sequence[str],
    contexts: Sequence[Dict[str, Any]],
    keyword_lexicon: set[str],
    stopwords: set[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, sentence in enumerate(lines, start=1):
        chunks = split_sentence_super_micro(sentence, keyword_lexicon, stopwords)
        if not chunks:
            chunks = [sentence]

        for atomic_idx, chunk in enumerate(chunks, start=1):
            rec: Dict[str, Any] = {
                "id": f"sent_{i:06d}_{atomic_idx:02d}" if len(chunks) > 1 else f"sent_{i:06d}",
                "parent_id": f"sent_{i:06d}",
                "atomic_index": atomic_idx,
                "text": chunk,
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
    try:
        model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(model_dir))
        os.chdir(cwd)
        return model
    except Exception:
        # Fallback path: initialize VnCoreNLP directly via pyjnius when wrapper setup fails.
        os.chdir(cwd)
        import jnius_config  # type: ignore

        jnius_config.add_options("-Xmx2g")
        jnius_config.set_classpath(str(jar_path))

        from jnius import autoclass  # type: ignore

        java_vncorenlp = autoclass("vn.pipeline.VnCoreNLP")
        java_annotation = autoclass("vn.pipeline.Annotation")
        java_string = autoclass("java.lang.String")

        class _VnCoreNLPWseg:
            def __init__(self):
                self._model = java_vncorenlp(["wseg"])

            def annotate_text(self, sentence: str) -> Dict[int, List[Dict[str, Any]]]:
                ann = java_annotation(java_string(sentence))
                self._model.annotate(ann)
                out: Dict[int, List[Dict[str, Any]]] = {}
                sent_blocks = ann.toString().split("\n\n")
                idx = 0
                for block in sent_blocks:
                    if not block.strip():
                        continue
                    words: List[Dict[str, Any]] = []
                    for line in block.split("\n"):
                        line = line.replace("\t\t", "\t")
                        parts = line.split("\t")
                        if len(parts) < 2:
                            continue
                        word_idx = int(parts[0]) if parts[0].isdigit() else len(words) + 1
                        words.append({"index": word_idx, "wordForm": parts[1]})
                    if words:
                        out[idx] = words
                        idx += 1
                return out

        return _VnCoreNLPWseg()


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


def token_to_plain(token: str) -> str:
    return normalize_spaces(token.replace("_", " ").lower())


def collect_subtree_indices(seed_idx: int, head: List[int]) -> List[int]:
    collected = {seed_idx}
    changed = True
    while changed:
        changed = False
        for i, parent in enumerate(head, start=1):
            if parent in collected and i not in collected:
                collected.add(i)
                changed = True
    return sorted(collected)


def find_condition_clause_indices(tokens: List[str], dep: List[str], head: List[int], root_idx: int) -> List[int]:
    candidate_idxs: List[int] = []
    for i, tok in enumerate(tokens, start=1):
        plain = token_to_plain(tok)
        if plain in CONDITION_MARKERS and i > root_idx:
            candidate_idxs.append(i)

    if not candidate_idxs:
        return []

    # Prefer explicit conditional connectors attached as clause links.
    candidate_idxs.sort(key=lambda x: (dep[x - 1].lower() not in {"coord", "conj", "mark", "advcl"}, x))
    seed = candidate_idxs[0]
    return collect_subtree_indices(seed, head)


def reduce_object_relation_overlap(relation_raw: str, object_raw: str) -> str:
    rel_parts = relation_raw.split()
    obj_parts = object_raw.split()
    if not rel_parts or not obj_parts:
        return object_raw

    # Remove repeated prefix from object when it is already fully encoded in relation.
    max_k = min(len(rel_parts), len(obj_parts))
    overlap = 0
    for k in range(max_k, 1, -1):
        if obj_parts[:k] == rel_parts[-k:]:
            overlap = k
            break

    if overlap >= 3 and overlap < len(obj_parts):
        return " ".join(obj_parts[overlap:])
    return object_raw


def trim_relation_suffix_object(relation_raw: str, object_raw: str) -> str:
    rel_parts = normalize_spaces(relation_raw).split()
    obj_parts = normalize_spaces(object_raw).split()
    if not rel_parts or not obj_parts:
        return relation_raw
    if len(rel_parts) <= len(obj_parts) + 1:
        return relation_raw

    # If relation already ends with full object phrase, cut that suffix to keep relation action-centric.
    if rel_parts[-len(obj_parts) :] == obj_parts:
        return " ".join(rel_parts[: -len(obj_parts)])
    return relation_raw


def refine_triplet_semantics(subject: str, relation: str, obj: str) -> Dict[str, str]:
    r = relation
    o = obj

    # Case: "co quyen don phuong cham dut hop dong lao dong" duplicated in object.
    if r.startswith("có_quyền_đơn_phương_chấm_dứt_hợp_đồng_lao_động") and o.startswith("quyền_đơn_phương_chấm_dứt_hợp_đồng_lao_động"):
        r = "có_quyền_đơn_phương_chấm_dứt"
        o = "hợp_đồng_lao_động"

    # Case: purpose relation contains full purpose phrase while object starts with "muc dich ...".
    if r.startswith("nhằm_mục_đích_") and o.startswith("mục_đích_"):
        r = "nhằm_mục_đích"
        o = o.replace("mục_đích_", "", 1)

    # Case: "... khong can bao truoc" got split into relation + object="khong".
    if o in {"không", "khong"} and "cần_báo_trước" in r:
        r = r.replace("_cần_báo_trước", "")
        o = "không_cần_báo_trước"

    # Case: relation keeps "can_bao_truoc" while object already starts with condition.
    if "cần_báo_trước" in r and o.startswith("nếu_"):
        r = r.replace("_cần_báo_trước", "")
        o = "không_cần_báo_trước_" + o

    # Case: object accidentally repeats the same "quyen don phuong cham dut" phrase.
    if o.startswith("quyền_đơn_phương_chấm_dứt_hợp_đồng_lao_động") and "cần_báo_trước" in r:
        o = "không_cần_báo_trước"

    # Case: purpose phrase collapsed as relation with very short object.
    if r.startswith("nhằm_mục_đích_") and len(o.split("_")) <= 1:
        tail = r[len("nhằm_mục_đích_") :]
        if tail:
            r = "nhằm_mục_đích"
            o = tail

    return {
        "subject": subject,
        "relation": r,
        "object": o,
    }


def clean_triplet_parts(subject_raw: str, relation_raw: str, object_raw: str, stopwords: set[str]) -> Dict[str, str]:
    relation_raw = trim_relation_suffix_object(relation_raw, object_raw)
    object_raw = reduce_object_relation_overlap(relation_raw, object_raw)
    subject_clean = to_snake_phrase(remove_stopwords_phrase(subject_raw.replace("_", " "), stopwords))
    relation_clean = to_snake_phrase(remove_stopwords_phrase(relation_raw.replace("_", " "), stopwords))
    object_clean = to_snake_phrase(remove_stopwords_phrase(object_raw.replace("_", " "), stopwords))
    refined = refine_triplet_semantics(subject_clean, relation_clean, object_clean)
    return {
        "subject": refined["subject"],
        "relation": refined["relation"],
        "object": refined["object"],
    }


def split_compound_object_raw(object_raw: str) -> List[str]:
    text = normalize_spaces(object_raw)
    if not text:
        return []
    return [text]


def build_triplets_from_parts(subject_raw: str, relation_raw: str, object_raw: str, stopwords: set[str]) -> List[Dict[str, str]]:
    object_candidates = split_compound_object_raw(object_raw)
    if not object_candidates:
        object_candidates = [object_raw]

    triplets: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for obj_raw in object_candidates:
        tri = clean_triplet_parts(subject_raw, relation_raw, obj_raw, stopwords)
        if not tri["subject"] or not tri["relation"]:
            continue
        key = (tri["subject"], tri["relation"], tri["object"])
        if key in seen:
            continue
        seen.add(key)
        triplets.append(tri)

    return triplets


def build_full_object_indices(
    root_idx: int,
    action_idx: int,
    relation_idxs: List[int],
    subj_full_idxs: List[int],
    dep: List[str],
    pos: List[str],
    head: List[int],
) -> List[int]:
    anchor = action_idx if action_idx != -1 else root_idx
    excluded = set(relation_idxs) | set(subj_full_idxs)

    # Prefer full subtree on the right of predicate to keep legal condition/detail intact.
    subtree = [
        i
        for i in range(anchor + 1, len(dep) + 1)
        if is_descendant(i, anchor, head)
        and i not in excluded
        and dep[i - 1].lower() not in {"punct"}
    ]
    if subtree:
        return sorted(set(subtree))

    # Fallback: take right-side content words from root.
    return [
        i
        for i in range(root_idx + 1, len(dep) + 1)
        if i not in excluded and pos[i - 1].upper() not in {"PUNCT", "SCONJ"}
    ]


def pick_subject_for_predicate(
    pred_idx: int,
    dep: List[str],
    head: List[int],
    fallback_subj_idxs: List[int],
) -> List[int]:
    subj_idxs = [
        i
        for i, d in enumerate(dep, start=1)
        if head[i - 1] == pred_idx and d.lower() in SUBJECT_LABELS
    ]
    if subj_idxs:
        return subj_idxs

    # In causative chains, predicate subject often comes from bridge object of ancestor.
    current = pred_idx
    visited = set()
    while 1 <= current <= len(head) and current not in visited:
        visited.add(current)
        parent = head[current - 1]
        if parent <= 0:
            break
        bridge = [
            i
            for i, d in enumerate(dep, start=1)
            if head[i - 1] == parent and d.lower() in BRIDGE_OBJECT_LABELS and i < current
        ]
        if bridge:
            return [bridge[-1]]
        current = parent

    return fallback_subj_idxs


def extract_triplet_for_predicate(
    pred_idx: int,
    tokens: List[str],
    pos: List[str],
    dep: List[str],
    head: List[int],
    fallback_subj_idxs: List[int],
    stopwords: set[str],
) -> List[Dict[str, str]]:
    subj_idxs = pick_subject_for_predicate(pred_idx, dep, head, fallback_subj_idxs)
    obj_idxs = [
        i
        for i, d in enumerate(dep, start=1)
        if head[i - 1] == pred_idx and d.lower() in OBJECT_LABELS
    ]
    if not obj_idxs:
        obj_idxs = pick_action_object_indices(pred_idx, dep, head, pos)

    subj_full = expand_entity_indices(subj_idxs, dep, head)
    obj_full = expand_entity_indices(obj_idxs, dep, head)
    rel_idxs = expand_relation_indices(pred_idx, dep, head, tokens)

    subject_raw = gather_phrase(subj_full, tokens)
    relation_raw = gather_phrase(rel_idxs, tokens)
    object_raw = gather_phrase(obj_full, tokens)

    if not object_raw:
        local_right = [
            i
            for i in range(pred_idx + 1, len(tokens) + 1)
            if is_descendant(i, pred_idx, head) and dep[i - 1].lower() not in {"punct"}
        ]
        object_raw = gather_phrase(local_right, tokens)

    triplets = build_triplets_from_parts(subject_raw, relation_raw, object_raw, stopwords)
    return triplets


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
        # Accept wider descendant verbs under bridge objects because legal patterns
        # like "co quyen ... cham dut ..." often attach action as nmod/vmod.
        if not is_verb_pos(pos[i - 1]) or d.lower() in {"aux", "cop", "punct"}:
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


def extract_ultra_micro_triplets(parsed: Dict[str, Any], stopwords: set[str]) -> List[Dict[str, str]]:
    tokens: List[str] = parsed['tokens']
    pos: List[str] = parsed['pos']
    dep: List[str] = parsed['dep']
    head: List[int] = parsed['head']

    function_words = {
        'của', 'khi', 'và', 'hoặc', 'hay', 'là', 'thì', 'mà', 'nếu', 'do', 'vì',
        'để', 'với', 'tại', 'ở', 'theo', 'cho', 'từ', 'đến'
    }

    source_parts = {p.lower() for tok in tokens for p in tok.split('_') if p}

    children = {i: [] for i in range(1, len(tokens) + 1)}
    for i, p in enumerate(head, start=1):
        if p > 0:
            children[p].append(i)

    def is_verb(p: str) -> bool:
        return p.startswith('V')

    def is_noun(p: str) -> bool:
        return p.startswith('N') or p.startswith('P')

    def token_text(idx: int) -> str:
        if idx <= 0:
            return ""
        return tokens[idx - 1].lower()

    def is_content_text(text: str) -> bool:
        if not text:
            return False
        parts = [x for x in text.split('_') if x]
        if not parts:
            return False
        if any(p in function_words for p in parts):
            return False
        return all(p in source_parts for p in parts)

    def verb_relation(idx: int) -> str:
        parts: List[str] = []
        for c in sorted(children.get(idx, [])):
            if dep[c - 1].lower() in {'adv', 'neg', 'aux'}:
                t = token_text(c)
                if is_content_text(t):
                    parts.append(t)
        base = token_text(idx)
        if is_content_text(base):
            parts.append(base)
        return "_".join(parts)

    triplets = []

    for i in range(1, len(tokens) + 1):
        p = pos[i-1]
        d = dep[i-1]
        token_i = token_text(i)
        
        if d.lower() in {'punct', 'case', 'mark', 'det', 'cc', 'conj'}:
            continue

        if token_i in function_words:
            continue

        node_text = token_i
        if not is_content_text(node_text):
            continue

        sub_idxs = [c for c in children.get(i, []) if dep[c-1].lower() in {'sub', 'nsubj', 'nsubj:pass', 'csubj'}]
        obj_idxs = [c for c in children.get(i, []) if dep[c-1].lower() in {'obj', 'dobj', 'dob', 'iobj', 'pob', 'obl'}]
        sub_texts = [token_text(c) for c in sub_idxs if is_content_text(token_text(c))]
        obj_texts = [token_text(c) for c in obj_idxs if is_content_text(token_text(c))]
        
        # 1. Trích xuất lõi Chủ ngữ - Động từ/Từ chính - Tân ngữ thành từng Node tách biệt
        if is_verb(p) and sub_texts and obj_texts:
            rel = verb_relation(i)
            if not is_content_text(rel):
                rel = node_text
            for s in sub_texts:
                for o in obj_texts:
                    triplets.append({'subject': s, 'relation': rel, 'object': o})
        
        # Tu dong che token ghep dai theo dung tu trong cau.
        # Vi du "người_sử_dụng_lao_động" -> người -sử_dụng-> lao_động
        raw_token = tokens[i-1].lower()
        parts = raw_token.split('_')
        if len(parts) >= 3:
            s_tok = parts[0]
            r_tok = parts[1]
            o_tok = "_".join(parts[2:])
            if is_content_text(s_tok) and is_content_text(r_tok) and is_content_text(o_tok):
                triplets.append({'subject': s_tok, 'relation': r_tok, 'object': o_tok})

        # 2. Chuoi danh tu nmod theo tu noi dung (khong dung gioi tu/hư từ).
        # Dùng parent-content để tạo thêm triplet cho cụm dài như
        # giữ - bản_chính - giấy_tờ, giao_kết - hợp_đồng - lao_động.
        if is_noun(p):
            noun_mod_children = [
                c
                for c in sorted(children.get(i, []))
                if dep[c - 1].lower() in {'nmod'} and is_content_text(token_text(c))
            ]
            noun_mod_texts = [token_text(c) for c in noun_mod_children if is_content_text(token_text(c))]
            if len(noun_mod_texts) >= 2:
                triplets.append({'subject': node_text, 'relation': noun_mod_texts[0], 'object': noun_mod_texts[1]})
            elif len(noun_mod_texts) == 1:
                parent_idx = head[i - 1]
                parent_text = token_text(parent_idx)
                if parent_idx > 0 and is_content_text(parent_text):
                    triplets.append({'subject': parent_text, 'relation': node_text, 'object': noun_mod_texts[0]})

    def all_parts_in_source(text: str) -> bool:
        parts = [x for x in text.split('_') if x]
        return bool(parts) and all(p.lower() in source_parts for p in parts)

    final = []
    seen = set()
    for t in triplets:
        s = t['subject'].replace('_', ' ').strip()
        r = t['relation'].replace('_', ' ').strip()
        o = t['object'].replace('_', ' ').strip()
        
        # Dọn dẹp lại cho đẹp và tránh node trùng ý
        if s and r and o and s != r and r != o and s != o:
           if not (all_parts_in_source(s.replace(' ', '_')) and all_parts_in_source(r.replace(' ', '_')) and all_parts_in_source(o.replace(' ', '_'))):
               continue
           key = (s.lower(), r.lower(), o.lower())
           if key not in seen:
               seen.add(key)
               final.append({'subject': s.replace(' ', '_'), 'relation': r.replace(' ', '_'), 'object': o.replace(' ', '_')})

    return final

def build_linked_triplet_view(sentence_id: str, triplets: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, str]] = []
    sentence_node = {
        "sentence_id": sentence_id,
        "node_type": "sentence",
    }

    for idx, tri in enumerate(triplets, start=1):
        triplet_id = f"{sentence_id}_t{idx:02d}"
        nodes.append(
            {
                "triplet_id": triplet_id,
                "sentence_id": sentence_id,
                "order": idx,
                "subject": tri["subject"],
                "relation": tri["relation"],
                "object": tri["object"],
            }
        )
        links.append(
            {
                "from": sentence_id,
                "to": triplet_id,
                "type": "belongs_to_sentence",
            }
        )

    # Always link triplets in sentence order so they form one connected chain.
    for i in range(len(nodes) - 1):
        links.append(
            {
                "from": nodes[i]["triplet_id"],
                "to": nodes[i + 1]["triplet_id"],
                "type": "sequence",
            }
        )

    # Add semantic bridge links when object of one triplet matches subject of another.
    subject_index: Dict[str, List[str]] = {}
    for node in nodes:
        subject_index.setdefault(node["subject"], []).append(node["triplet_id"])

    seen_semantic: set[tuple[str, str]] = set()
    for node in nodes:
        for target in subject_index.get(node["object"], []):
            if target == node["triplet_id"]:
                continue
            key = (node["triplet_id"], target)
            if key in seen_semantic:
                continue
            seen_semantic.add(key)
            links.append(
                {
                    "from": node["triplet_id"],
                    "to": target,
                    "type": "semantic_bridge",
                }
            )

    return {
        "sentence_node": sentence_node,
        "triplet_nodes": nodes,
        "triplet_links": links,
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
        try:
            wseg_tokens = vncore_tokenize(vncore, sentence)
            segmented_text = " ".join(wseg_tokens)
            parsed = phonnlp_parse(ph_model, segmented_text)
            
            # Extract multiple, ultra-micro triplets
            micro_triplets = extract_ultra_micro_triplets(parsed, stopwords)

            out.append(
                {
                    "id": rec["id"],
                    "source": rec.get("source", ""),
                    "text": sentence,
                    "triplets": micro_triplets,
                }
            )
        except Exception as exc:
            out.append(
                {
                    "id": rec["id"],
                    "source": rec.get("source", ""),
                    "text": sentence,
                    "triplets": [],
                    "extraction_error": str(exc),
                }
            )
    return out


def score_triplet_quality(tri: Dict[str, str]) -> int:
    s = tri.get("subject", "")
    r = tri.get("relation", "")
    o = tri.get("object", "")

    s_parts = [x for x in s.split("_") if x]
    r_parts = [x for x in r.split("_") if x]
    o_parts = [x for x in o.split("_") if x]

    score = 0
    score += min(len(s_parts), 12)
    score += min(len(r_parts), 10)
    score += min(len(o_parts), 14)

    if 2 <= len(r_parts) <= 10:
        score += 3
    if len(r_parts) > 14:
        score -= 4

    if len(o_parts) <= 1:
        score -= 8

    if o.startswith(("khi_", "nếu_", "trong_", "vì_", "do_")) and len(o_parts) <= 5:
        score -= 4

    # Penalize circular relation/object duplication.
    overlap = set(r_parts) & set(o_parts)
    if len(overlap) >= max(4, int(0.6 * min(len(r_parts), len(o_parts)))):
        score -= 6

    if o.startswith("quyền_đơn_phương_chấm_dứt_hợp_đồng_lao_động") and "cần_báo_trước" in r:
        score -= 8

    return score


def merge_with_previous_results(current: List[Dict[str, Any]], previous_path: Path) -> List[Dict[str, Any]]:
    if not previous_path.exists():
        return current

    prev_data = json.loads(previous_path.read_text(encoding="utf-8"))
    if not isinstance(prev_data, list):
        return current

    prev_map: Dict[str, Dict[str, str]] = {}
    for item in prev_data:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("triplet"), dict):
            tri = item["triplet"]
            if all(isinstance(tri.get(k), str) for k in ["subject", "relation", "object"]):
                prev_map[item["id"]] = {
                    "subject": tri["subject"],
                    "relation": tri["relation"],
                    "object": tri["object"],
                }

    merged: List[Dict[str, Any]] = []
    for rec in current:
        rec_out = dict(rec)
        rid = rec.get("id")
        if isinstance(rid, str) and rid in prev_map and isinstance(rec.get("triplet"), dict):
            cur_tri = rec["triplet"]
            prev_tri = prev_map[rid]
            if score_triplet_quality(prev_tri) > score_triplet_quality(cur_tri):
                rec_out["triplet"] = prev_tri
        merged.append(rec_out)

    return merged


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert LLM text to JSON and extract legal triplets with models")
    p.add_argument("--input-structured-json", type=Path, default=Path("output/2_llm_pre_structured.json"), help="Structured JSON input with fields: id, metadata, original_text, llm_processed_text")
    p.add_argument("--input-txt", type=Path, default=Path("llm_pre.txt"), help="Input text file with one normalized sentence per line")
    p.add_argument("--contexts-jsonl", type=Path, default=Path("contexts_dieu_17_35_48.jsonl"), help="Optional context JSONL to map source metadata by line index")
    p.add_argument("--sentences-output", type=Path, default=Path("output/03_micro_sentences.json"), help="Output JSON containing sentence objects")
    p.add_argument("--triplets-output", type=Path, default=Path("output/04_micro_triplets.json"), help="Output JSON containing extracted triplets")
    p.add_argument("--stopwords-file", type=Path, default=Path("vietnamese_stopwords_legal.txt"), help="Vietnamese stopwords file")
    p.add_argument("--format-word-json", type=Path, default=Path("input/format_word.json"), help="Optional legal keyword JSON used for super-micro sentence splitting")
    p.add_argument("--disable-super-micro", action="store_true", help="Disable aggressive super-micro sentence splitting")
    p.add_argument("--vncore-model-dir", type=Path, default=Path(".models/VnCoreNLP"), help="Directory for py_vncorenlp model files")
    p.add_argument("--phonlp-model-dir", type=Path, default=Path(".models/phonlp"), help="Directory for PhoNLP model files")
    p.add_argument("--merge-with-prev", type=Path, default=Path("output/04_micro_triplets.prev.json"), help="Optional previous triplet JSON to blend with current output")
    return p


def main() -> int:
    args = build_parser().parse_args()

    stopwords = load_stopwords(args.stopwords_file)
    keyword_lexicon = set() if args.disable_super_micro else load_format_word_keywords(args.format_word_json)

    if not args.input_structured_json.exists():
        raise FileNotFoundError(f"Khong tim thay input structured JSON: {args.input_structured_json}")

    records = read_structured_input_json(args.input_structured_json, keyword_lexicon, stopwords)
    if not records:
        raise RuntimeError("Input structured JSON khong co ban ghi hop le de xu ly.")

    triplets = extract_triplets_with_models(
        records=records,
        stopwords=stopwords,
        vncore_model_dir=args.vncore_model_dir,
        phonlp_model_dir=args.phonlp_model_dir,
    )
    triplets = merge_with_previous_results(triplets, args.merge_with_prev)
    write_json(args.triplets_output, triplets)
    print(f"Da tao file JSON triplet: {args.triplets_output} ({len(triplets)} bo ba)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
