"""
Phase 3 — Ép triplet thô vào ONTOLOGY có ràng buộc (ZERO LLM)
============================================================
Input : data/triplets_raw.json  +  ontology/ontology.json
Output: data/triplets_mapped.json   (Tier 1-2: đã chuẩn hóa Conc-Rel-Conc)
        data/triplets_flagged.json  (Tier 3-4 / NO_REL: chờ duyệt, làm giàu ontology)

Cơ chế (đúng 4 yêu cầu đã chốt):
  ② Thuần lexicon tất định — KHÔNG LLM. (PhoBERT để dành Phase 5 đúng như paper.)
  ③ Map V trước -> lấy chữ ký (ConcKeyS_cat / ConcKeyO_cat) -> map S,O CÓ RÀNG BUỘC.
  ④ Ràng buộc ở mức NHÓM (category) + soft-tier: lệch thì GIỮ + gắn cờ, không vứt.
     Tier1: O đúng concept liệt kê cụ thể        conf 1.0
     Tier2: O cùng NHÓM với chữ ký               conf 0.8
     Tier3: map được nhưng lệch nhóm             conf 0.4  -> flagged
     Tier4: thiếu concept (placeholder)          conf 0.2  -> flagged
  + Tự ĐẢO CHIỀU nếu S/O ngược so với chữ ký (câu bị động / parse ngược).
"""

import os
import re
import json

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_IN = os.path.join(BASE, "data", "triplets_raw.json")
ONTO_IN = os.path.join(BASE, "ontology", "ontology.json")
MAPPED_OUT = os.path.join(BASE, "data", "triplets_mapped.json")
FLAGGED_OUT = os.path.join(BASE, "data", "triplets_flagged.json")


def norm(s):
    return re.sub(r"\s+", " ", str(s).replace("_", " ").strip().lower())


def toks(s):
    return norm(s).split()


def is_subseq(kp_toks, text_toks):
    """kp_toks có phải dãy con LIÊN TIẾP của text_toks? (khớp cụm trọn vẹn,
    tránh 'ô' khớp vào 'không')."""
    n = len(kp_toks)
    if n == 0 or n > len(text_toks):
        return False
    for i in range(len(text_toks) - n + 1):
        if text_toks[i:i + n] == kp_toks:
            return True
    return False


# ---------- nạp ontology & dựng index tra cứu ----------
def load_ontology():
    with open(ONTO_IN, "r", encoding="utf-8") as f:
        onto = json.load(f)

    # index concept: (kp_norm, kp_tokens, concept_dict)
    concept_kps = []
    concept_by_name = {}
    for c in onto["concepts"]:
        concept_by_name[norm(c["name"])] = c
        for kp in [c["name"]] + c["keyphrases"]:
            n = norm(kp)
            if n:
                concept_kps.append((n, n.split(), c))
    concept_kps.sort(key=lambda x: len(x[0]), reverse=True)

    # index relation
    rel_kps = []
    for r in onto["relations"]:
        for kp in [r["name"]] + r["keyphrases"]:
            n = norm(kp)
            if n:
                rel_kps.append((n, n.split(), r))
    rel_kps.sort(key=lambda x: len(x[0]), reverse=True)

    return onto, concept_kps, concept_by_name, rel_kps


def match_relation(v_head, rel_kps):
    vt = toks(v_head)
    if not vt:
        return None
    for kp, kp_toks, r in rel_kps:  # đã sort dài -> ngắn
        if is_subseq(kp_toks, vt):
            return r
    return None


def match_concept(head, phrase, concept_kps, prefer_cats=None):
    """Trả (concept, score). Khớp cụm trọn vẹn (token). Ưu tiên keyphrase dài +
    đúng nhóm chữ ký (ràng buộc ③)."""
    h_toks = toks(head)
    p_toks = toks(phrase)
    best, best_score = None, -1
    for kp, kp_toks, c in concept_kps:
        is_head = kp_toks == h_toks
        if not (is_head or is_subseq(kp_toks, p_toks)):
            continue
        score = len(kp.replace(" ", ""))
        if is_head:
            score += 5
        if prefer_cats and (c["category"] in prefer_cats or "ANY" in prefer_cats):
            score += 1000  # ràng buộc nhóm dẫn dắt lựa chọn
        if score > best_score:
            best, best_score = c, score
    return best, best_score


def tier_of(s_c, o_c, rel, concrete_names):
    """Xác định tier + (có cần đảo chiều?)."""
    sS = rel["ConcKeyS_cat"]
    aO = set(rel["ConcKeyO_cat"])
    any_s = sS == ["ANY"]

    def s_ok(c):
        return c and (any_s or c["category"] in sS)

    def o_concrete(c):
        return c and norm(c["name"]) in concrete_names

    def o_cat(c):
        return c and c["category"] in aO

    if s_c and o_c:
        if s_ok(s_c) and o_concrete(o_c):
            return 1, 1.0, False
        if s_ok(s_c) and o_cat(o_c):
            return 2, 0.8, False
        # thử đảo chiều: S đáng lẽ là O, O đáng lẽ là S
        if (s_c["category"] in aO) and (any_s or o_c["category"] in sS):
            # đảo: subject<-o_c, object<-s_c
            if o_concrete(s_c):
                return 1, 1.0, True
            return 2, 0.8, True
        return 3, 0.4, False
    return 4, 0.2, False


def main():
    onto, concept_kps, concept_by_name, rel_kps = load_ontology()
    concrete_by_rel = {
        r["id"]: {norm(x) for x in r.get("ConcKeyO_concrete", [])}
        for r in onto["relations"]
    }

    with open(RAW_IN, "r", encoding="utf-8") as f:
        sections = json.load(f)

    mapped, flagged = [], []
    counts = {1: 0, 2: 0, 3: 0, 4: 0, "NO_REL": 0}

    for sec in sections:
        sid = sec.get("id")
        for t in sec.get("triplets", []):
            rel = match_relation(t["v"]["head"], rel_kps)
            if rel is None:
                counts["NO_REL"] += 1
                flagged.append({
                    "section_id": sid, "tier": "NO_REL", "confidence": 0.0,
                    "raw": {"s": t["s"]["phrase"], "v": t["v"]["head"], "o": t["o"]["phrase"]},
                })
                continue

            prefer_s = rel["ConcKeyS_cat"]
            prefer_o = rel["ConcKeyO_cat"]
            s_c, _ = match_concept(t["s"]["head"], t["s"]["phrase"], concept_kps, prefer_s)
            o_c, _ = match_concept(t["o"]["head"], t["o"]["phrase"], concept_kps, prefer_o)

            tier, conf, swap = tier_of(s_c, o_c, rel, concrete_by_rel[rel["id"]])
            if swap:
                s_c, o_c = o_c, s_c

            def concept_out(c, raw_phrase):
                if c:
                    return {"id": c["id"], "name": c["name"], "category": c["category"]}
                return {"id": None, "name": raw_phrase, "category": "UNMAPPED"}

            rec = {
                "section_id": sid,
                "s": concept_out(s_c, t["s"]["phrase"]),
                "v": {"id": rel["id"], "name": rel["name"]},
                "o": concept_out(o_c, t["o"]["phrase"]),
                "tier": tier,
                "confidence": conf,
                "raw": {"s": t["s"]["phrase"], "v": t["v"]["head"], "o": t["o"]["phrase"]},
            }
            counts[tier] += 1
            if tier in (1, 2):
                mapped.append(rec)
            else:
                flagged.append(rec)

    with open(MAPPED_OUT, "w", encoding="utf-8") as f:
        json.dump(mapped, f, ensure_ascii=False, indent=2)
    with open(FLAGGED_OUT, "w", encoding="utf-8") as f:
        json.dump(flagged, f, ensure_ascii=False, indent=2)

    total = sum(counts.values())
    print("=" * 55)
    print(f"Tổng triplet thô: {total}")
    print(f"  Tier1 (concept khớp cụ thể) : {counts[1]:>6}  conf 1.0")
    print(f"  Tier2 (cùng nhóm chữ ký)    : {counts[2]:>6}  conf 0.8")
    print(f"  Tier3 (lệch nhóm)           : {counts[3]:>6}  -> flagged")
    print(f"  Tier4 (thiếu concept)       : {counts[4]:>6}  -> flagged")
    print(f"  NO_REL (V ngoài 24 quan hệ) : {counts['NO_REL']:>6}  -> flagged")
    print("=" * 55)
    print(f"✅ mapped (Tier1+2): {len(mapped)} -> {MAPPED_OUT}")
    print(f"⚑ flagged          : {len(flagged)} -> {FLAGGED_OUT}")

    # Thống kê quan hệ đã dùng + ví dụ PHÂN BIỆT
    from collections import Counter
    rel_count = Counter(r["v"]["name"] for r in mapped)
    print("\n--- Phân bố quan hệ trong mapped (top) ---")
    for name, cnt in rel_count.most_common(12):
        print(f"  {name:<22}: {cnt}")

    seen = set()
    print("\n--- 15 triple PHÂN BIỆT (Tier1/2) ---")
    for r in mapped:
        key = (r["s"]["name"], r["v"]["name"], r["o"]["name"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  T{r['tier']} ({r['s']['name']} :{r['s']['category']}) "
              f"-[{r['v']['name']}]-> ({r['o']['name']} :{r['o']['category']})")
        if len(seen) >= 15:
            break


if __name__ == "__main__":
    main()
