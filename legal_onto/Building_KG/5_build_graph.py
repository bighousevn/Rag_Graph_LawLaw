"""
Phase 4 — Gộp triplet đã map thành KNOWLEDGE GRAPH + Rules
=========================================================
Input : data/triplets_mapped.json  (Tier1+2, đã chuẩn hóa Conc-Rel-Conc)
Output: data/master_graph.json      (nodes + relationships, format nạp Neo4j)
        ontology/rules.json         (luật suy diễn — thành phần Rules của Legal-Onto)

Gộp:
  - Node = Concept (đóng). Gộp theo concept id, cộng dồn listSectionId.
  - Edge = (source_concept, relation, target_concept). Gộp trùng, cộng dồn
    listSectionId, đếm 'count' (tần suất — dùng cho TF-IDF ở Phase 5).
  - Bỏ self-loop (source == target).
"""

import os
import json
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
MAPPED_IN = os.path.join(BASE, "data", "triplets_mapped.json")
GRAPH_OUT = os.path.join(BASE, "data", "master_graph.json")
RULES_OUT = os.path.join(BASE, "ontology", "rules.json")


def build_graph():
    with open(MAPPED_IN, "r", encoding="utf-8") as f:
        triples = json.load(f)

    nodes = {}              # concept_id -> node
    edges = {}             # (s_id, rel_id, o_id) -> edge
    self_loops = 0

    def touch_node(c):
        cid = c["id"]
        if cid not in nodes:
            nodes[cid] = {
                "id": cid,
                "name": c["name"],
                "label": c["category"],
                "listSectionId": set(),
            }
        return nodes[cid]

    for t in triples:
        s, v, o = t["s"], t["v"], t["o"]
        if not s["id"] or not o["id"]:
            continue
        if s["id"] == o["id"]:
            self_loops += 1
            continue

        sec = t.get("section_id")
        sn = touch_node(s)
        on = touch_node(o)
        if sec:
            sn["listSectionId"].add(sec)
            on["listSectionId"].add(sec)

        key = (s["id"], v["id"], o["id"])
        if key not in edges:
            edges[key] = {
                "id": f"E{len(edges)+1:04d}",
                "name": v["name"],
                "relId": v["id"],
                "source": s["id"],
                "target": o["id"],
                "listSectionId": set(),
                "count": 0,
                "confidence_sum": 0.0,
            }
        e = edges[key]
        e["count"] += 1
        e["confidence_sum"] += t.get("confidence", 0)
        if sec:
            e["listSectionId"].add(sec)

    node_list = [
        {**n, "listSectionId": sorted(n["listSectionId"])}
        for n in nodes.values()
    ]
    edge_list = []
    for e in edges.values():
        e = dict(e)
        e["avg_confidence"] = round(e.pop("confidence_sum") / max(e["count"], 1), 3)
        e["listSectionId"] = sorted(e["listSectionId"])
        edge_list.append(e)

    graph = {"nodes": node_list, "relationships": edge_list}
    with open(GRAPH_OUT, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print("=" * 55)
    print(f"Triple Tier1/2 đầu vào : {len(triples)}")
    print(f"Self-loop đã bỏ        : {self_loops}")
    print(f"Nodes (concept)        : {len(node_list)}")
    print(f"Edges (gộp trùng)      : {len(edge_list)}")
    print("=" * 55)
    print(f"✅ Đã ghi graph: {GRAPH_OUT}")

    top = sorted(edge_list, key=lambda x: x["count"], reverse=True)[:12]
    id2name = {n["id"]: n["name"] for n in node_list}
    print("\n--- 12 cạnh phổ biến nhất ---")
    for e in top:
        print(f"  [{e['count']:>3}] ({id2name[e['source']]}) -[{e['name']}]-> "
              f"({id2name[e['target']]})  | {len(e['listSectionId'])} điều")
    return graph


def write_rules():
    """Rules — thành phần thứ 3 của Legal-Onto: {hypothesis} -> {goal}.
    Dạng khai báo để Phase truy vấn suy luận multi-hop."""
    rules = [
        {
            "id": "RULE_ISA_TRANSITIVE",
            "desc": "Phân loại bắc cầu: is-a có tính bắc cầu.",
            "if": [["?x", "Là loại của", "?y"], ["?y", "Là loại của", "?z"]],
            "then": [["?x", "Là loại của", "?z"]],
        },
        {
            "id": "RULE_SANCTION_AUTHORITY",
            "desc": "Cơ quan có thẩm quyền xử phạt thì xử phạt chủ thể thực hiện VPHC.",
            "if": [
                ["?cq", "Có thẩm quyền xử phạt", "?vp"],
                ["?ct", "Thực hiện", "?vp"],
            ],
            "then": [["?cq", "Xử phạt", "?ct"]],
        },
        {
            "id": "RULE_DRIVER_NEEDS_LICENSE",
            "desc": "Người điều khiển phương tiện thì phải có giấy phép lái xe.",
            "if": [["?ng", "Điều khiển", "?xe"]],
            "then": [["?ng", "Có", "Giấy phép lái xe"]],
        },
        {
            "id": "RULE_VIOLATION_TO_PENALTY",
            "desc": "Thực hiện VPHC thì bị áp dụng hình thức xử phạt.",
            "if": [["?ct", "Thực hiện", "?vp"]],
            "then": [["Phạt tiền", "Áp dụng", "?ct"]],
        },
    ]
    with open(RULES_OUT, "w", encoding="utf-8") as f:
        json.dump({"rules": rules}, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã ghi {len(rules)} rules: {RULES_OUT}")


if __name__ == "__main__":
    build_graph()
    write_rules()
