"""
Phase 0 — Parse ontology_168.md -> ontology.json
=================================================
Chuyển bộ ontology viết tay (Markdown) thành JSON mà các phase sau dùng được.

Bổ sung so với file .md gốc:
  - Mỗi Concept được gán 1 `category` (suy từ header nhóm A.1 .. A.7).
  - Mỗi Relation được suy ra "chữ ký nhóm":
        ConcKeyS_cat  : (các) nhóm hợp lệ cho CHỦ THỂ
        ConcKeyO_cat  : (các) nhóm hợp lệ cho ĐỐI TƯỢNG
    -> dùng cho ràng buộc MỀM theo nhóm ở Phase 3 (fix vụ ConcKeyO liệt kê thiếu).
  - Giữ luôn ConcKeyS_concrete / ConcKeyO_concrete để chấm Tier-1 (tin cậy cao).
  - Liệt kê ConcKeyO_unresolved: các đối tượng được tham chiếu nhưng CHƯA có Concept
    tương ứng -> danh sách việc để làm giàu ontology (active learning).

KHÔNG dùng LLM. Chỉ parse text thuần.
"""

import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_MD = os.path.join(BASE, "ontology", "ontology_168.md")
PATCH_JSON = os.path.join(BASE, "ontology", "ontology_patch.json")
OUT_JSON = os.path.join(BASE, "ontology", "ontology.json")

# A.1 .. A.7 -> mã nhóm (category)
CATEGORY_BY_SECTION = {
    "1": "PHUONG_TIEN",       # A.1 Phương tiện giao thông
    "2": "CHU_THE",           # A.2 Chủ thể
    "3": "HANH_VI_CHE_TAI",   # A.3 Hành vi & chế tài xử phạt
    "4": "GIAY_TO",           # A.4 Giấy tờ, giấy phép
    "5": "HA_TANG",           # A.5 Hạ tầng & báo hiệu
    "6": "CO_QUAN",           # A.6 Cơ quan, người có thẩm quyền
    "7": "HOAT_DONG",         # A.7 Hoạt động / lĩnh vực
}

RE_SECTION = re.compile(r"^##\s+A\.(\d)\.")
RE_PART_B = re.compile(r"^#\s+PHẦN B")
RE_CONCEPT = re.compile(r'^###\s+Bảng\s+C(\d+):\s*"(.+?)"')
RE_RELATION = re.compile(r'^###\s+Bảng\s+R(\d+):\s*"(.+?)"')


def norm(s: str) -> str:
    """Chuẩn hóa: bỏ khoảng trắng thừa, gạch dưới, về chữ thường."""
    return re.sub(r"\s+", " ", s.replace("_", " ").strip().lower())


def strip_paren(s: str) -> str:
    return re.sub(r"\(.*?\)", " ", s)


def split_list(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def table_cells(line: str):
    """| a | b | c |  ->  ['a','b','c']"""
    return [p.strip() for p in line.strip().strip("|").split("|")]


def parse_md(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    concepts = []
    relations = []
    mode = "A"
    current_category = None
    block = None          # dict đang gom dở
    block_kind = None     # 'concept' | 'relation'

    def flush():
        nonlocal block, block_kind
        if block is None:
            return
        if block_kind == "concept":
            concepts.append(block)
        elif block_kind == "relation":
            relations.append(block)
        block = None
        block_kind = None

    for line in lines:
        m_sec = RE_SECTION.match(line)
        if m_sec:
            current_category = CATEGORY_BY_SECTION.get(m_sec.group(1))
            continue

        if RE_PART_B.match(line):
            flush()
            mode = "B"
            continue

        m_c = RE_CONCEPT.match(line)
        if m_c:
            flush()
            block_kind = "concept"
            block = {
                "id": f"C{m_c.group(1)}",
                "name": m_c.group(2).strip(),
                "category": current_category,
                "keyphrases": [],
            }
            continue

        m_r = RE_RELATION.match(line)
        if m_r:
            flush()
            block_kind = "relation"
            block = {
                "id": f"R{m_r.group(1)}",
                "name": m_r.group(2).strip(),
                "keyphrases": [],
                "ConcKeyS_raw": "",
                "ConcKeyO_concrete": [],
            }
            continue

        # Dòng bảng dữ liệu
        if block is not None and line.lstrip().startswith("|"):
            cells = table_cells(line)
            if len(cells) < 3:
                continue
            field = cells[0].lower()
            value = cells[2]
            # bỏ qua dòng tiêu đề bảng / dòng phân cách
            if field in ("thành phần", "") or set(value) <= {"-"}:
                continue

            if field == "name":
                # giữ name từ header (đã có), bỏ qua
                pass
            elif field == "keyphrases":
                block["keyphrases"] = split_list(value)
            elif field == "conckeys":
                block["ConcKeyS_raw"] = value
            elif field == "conckeyo":
                block["ConcKeyO_concrete"] = split_list(value)

    flush()
    return concepts, relations


def build_resolver(concepts):
    """
    Tạo danh sách (cụm_chuẩn_hóa, category, concept_name) để dò.
    Gồm cả Name và Keyphrases. Sắp theo độ dài giảm dần để ưu tiên khớp dài.
    """
    phrases = []
    exact = {}  # norm -> (category, name)
    for c in concepts:
        cat = c["category"]
        for text in [c["name"]] + c["keyphrases"]:
            n = norm(text)
            if not n:
                continue
            exact.setdefault(n, (cat, c["name"]))
            if len(n.replace(" ", "")) >= 3:
                phrases.append((n, cat, c["name"]))
    phrases.sort(key=lambda x: len(x[0]), reverse=True)
    return exact, phrases


def resolve_one(item, exact, phrases):
    """Trả về (category, concept_name) tốt nhất cho 1 cụm, hoặc (None, None)."""
    t = norm(strip_paren(item))
    if not t:
        return None, None
    if t in exact:
        return exact[t]
    # khớp bao hàm: cụm concept nằm trong item, hoặc item nằm trong cụm concept
    for p, cat, cname in phrases:
        if p in t or t in p:
            return cat, cname
    return None, None


def resolve_all(text, exact, phrases):
    """Quét toàn chuỗi (kể cả phần trong ngoặc) -> tập category xuất hiện."""
    t = norm(text)
    cats = set()
    if not t:
        return cats
    for p, cat, _ in phrases:
        if len(p.replace(" ", "")) < 4:
            continue
        if p in t:
            cats.add(cat)
    # bổ sung khớp chính xác
    if t in exact:
        cats.add(exact[t][0])
    return cats


def apply_patch(concepts, patch):
    """Áp chỉnh sửa tay: thêm concept mới + bổ sung keyphrase. (Override chữ ký
    quan hệ làm sau khi đã auto-derive.)"""
    # 1. Thêm concept mới
    existing_ids = {c["id"] for c in concepts}
    for nc in patch.get("new_concepts", []):
        if nc["id"] in existing_ids:
            print(f"⚠️  Patch: concept {nc['id']} đã tồn tại, bỏ qua.")
            continue
        concepts.append({
            "id": nc["id"],
            "name": nc["name"],
            "category": nc["category"],
            "keyphrases": nc.get("keyphrases", []),
        })
    # 2. Bổ sung keyphrase cho concept có sẵn
    by_id = {c["id"]: c for c in concepts}
    for cid, extra in patch.get("concept_keyphrase_additions", {}).items():
        if cid in by_id:
            for kp in extra:
                if kp not in by_id[cid]["keyphrases"]:
                    by_id[cid]["keyphrases"].append(kp)
        else:
            print(f"⚠️  Patch: không thấy concept {cid} để thêm keyphrase.")


def main():
    if not os.path.exists(SRC_MD):
        print(f"❌ Không tìm thấy {SRC_MD}")
        return

    concepts, relations = parse_md(SRC_MD)

    # Nạp & áp patch (concept mới + keyphrase) TRƯỚC khi dựng resolver,
    # để các concept mới cũng tham gia khử nhập nhằng.
    patch = {}
    if os.path.exists(PATCH_JSON):
        with open(PATCH_JSON, "r", encoding="utf-8") as f:
            patch = json.load(f)
        apply_patch(concepts, patch)
        print(f"🩹 Đã áp patch: +{len(patch.get('new_concepts', []))} concept, "
              f"override {len(patch.get('signature_overrides', {}))} quan hệ.")

    exact, phrases = build_resolver(concepts)

    review = []  # các quan hệ cần người xem lại chữ ký nhóm

    for r in relations:
        # ConcKeyS: có thể ra >1 nhóm (vì kèm chú thích cơ quan trong ngoặc) -> để review
        s_cats = sorted(resolve_all(r["ConcKeyS_raw"], exact, phrases))

        o_cats = set()
        unresolved = []
        for item in r["ConcKeyO_concrete"]:
            cat, _ = resolve_one(item, exact, phrases)
            if cat:
                o_cats.add(cat)
            else:
                unresolved.append(item)

        # Áp override thủ công (thắng auto-derive)
        ov = patch.get("signature_overrides", {}).get(r["id"], {})
        if "ConcKeyS_cat" in ov:
            s_cats = list(ov["ConcKeyS_cat"])
        if "ConcKeyO_cat" in ov:
            o_cats = set(ov["ConcKeyO_cat"])

        r["ConcKeyS_cat"] = s_cats
        r["ConcKeyO_cat"] = sorted(o_cats)
        r["ConcKeyO_unresolved"] = unresolved
        # dọn field tạm
        r["ConcKeyS_concrete"] = r.pop("ConcKeyS_raw")

        # "ANY" = không khóa nhóm chủ thể (vd quan hệ is-a)
        s_ok = (s_cats == ["ANY"]) or (len(s_cats) == 1)
        if not s_ok or not o_cats:
            review.append({
                "id": r["id"], "name": r["name"],
                "ConcKeyS_cat": s_cats, "ConcKeyO_cat": sorted(o_cats),
                "ConcKeyO_unresolved": unresolved,
            })

    ontology = {
        "meta": {
            "source": os.path.basename(SRC_MD),
            "num_concepts": len(concepts),
            "num_relations": len(relations),
            "categories": sorted(set(CATEGORY_BY_SECTION.values()) | set(patch.get("new_categories", []))),
        },
        "concepts": concepts,
        "relations": relations,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã ghi {OUT_JSON}")
    print(f"   Concepts: {len(concepts)} | Relations: {len(relations)}")
    # cảnh báo concept thiếu category
    no_cat = [c["id"] for c in concepts if not c["category"]]
    if no_cat:
        print(f"⚠️  Concept thiếu category: {no_cat}")

    print(f"\n--- {len(review)} quan hệ cần REVIEW chữ ký nhóm (làm giàu ontology) ---")
    for r in review:
        print(f"  {r['id']} \"{r['name']}\"")
        print(f"     ConcKeyS_cat = {r['ConcKeyS_cat']}  (nên còn đúng 1 nhóm)")
        print(f"     ConcKeyO_cat = {r['ConcKeyO_cat']}")
        if r["ConcKeyO_unresolved"]:
            print(f"     ⤷ chưa có Concept cho: {r['ConcKeyO_unresolved']}")


if __name__ == "__main__":
    main()
