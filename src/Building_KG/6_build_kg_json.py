"""
Bước 6: Xây dựng KG JSON từ mapped triplets.
- Deduplicate: (ConceptS, Relation, ConceptO) trùng nhau → gộp listSectionId
- Build nodes (unique Concepts) và edges (Relations)
- Format tương thích với 3_query_flow.py
Input:  material_for_triplets/4_mapped_triplets_nghi_dinh_168.json
Output: triplets/triplets_nghi_dinh_168_2024_1.json
"""

import json
import os

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "material_for_triplets/4_mapped_triplets_nghi_dinh_168.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "triplets/triplets_nghi_dinh_168_2024_1.json")


def build_kg(mapped_triplets: list[dict]) -> dict:
    """
    Returns {"nodes": [...], "relationships": [...]}.
    Node: {id, name, listSectionId, addresses}
    Edge: {id, name, source, target, listSectionId, addresses}
    """
    # concept_name → {section_ids, addresses}
    concept_meta: dict[str, dict] = {}
    # (concept_s, relation, concept_o) → {section_ids, addresses}
    edge_meta: dict[tuple, dict] = {}

    for t in mapped_triplets:
        cs  = t["concept_s"]
        rel = t["relation"]
        co  = t["concept_o"]
        sid = t["section_id"]
        addr = t.get("address", {})

        for concept in (cs, co):
            if concept not in concept_meta:
                concept_meta[concept] = {"section_ids": set(), "addresses": []}
            concept_meta[concept]["section_ids"].add(sid)
            # Thêm address nếu chưa có
            if addr and addr not in concept_meta[concept]["addresses"]:
                concept_meta[concept]["addresses"].append(addr)

        key = (cs, rel, co)
        if key not in edge_meta:
            edge_meta[key] = {"section_ids": set(), "addresses": []}
        edge_meta[key]["section_ids"].add(sid)
        if addr and addr not in edge_meta[key]["addresses"]:
            edge_meta[key]["addresses"].append(addr)

    # Assign IDs
    concept_to_id: dict[str, str] = {
        name: f"N{i+1:04d}" for i, name in enumerate(sorted(concept_meta))
    }

    nodes = [
        {
            "id":            concept_to_id[name],
            "name":          name,
            "listSectionId": sorted(meta["section_ids"]),
            "addresses":     sorted(meta["addresses"], key=lambda a: (a.get("article") or 0, a.get("clause") or 0)),
        }
        for name, meta in sorted(concept_meta.items())
    ]

    edges = []
    for i, (key, meta) in enumerate(sorted(edge_meta.items())):
        cs, rel, co = key
        edges.append({
            "id":            f"E{i+1:04d}",
            "name":          rel,
            "source":        concept_to_id[cs],
            "target":        concept_to_id[co],
            "listSectionId": sorted(meta["section_ids"]),
            "addresses":     sorted(meta["addresses"], key=lambda a: (a.get("article") or 0, a.get("clause") or 0)),
        })

    return {"nodes": nodes, "relationships": edges}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        mapped = json.load(f)

    print(f"Loaded {len(mapped)} mapped triplets.")
    kg = build_kg(mapped)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([kg], f, ensure_ascii=False, indent=2)

    print(f"KG: {len(kg['nodes'])} nodes, {len(kg['relationships'])} edges → {args.output}")


if __name__ == "__main__":
    main()
