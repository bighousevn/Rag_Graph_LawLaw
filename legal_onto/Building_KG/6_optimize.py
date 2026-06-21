"""
Phase 5 — Tối ưu graph (Algorithm 1 của paper)
==============================================
Input : data/master_graph.json  (+ data/sections.json để biết tổng số điều N)
Output: data/master_graph_optimized.json

Stage 1 — Đánh dấu cạnh vô nghĩa qua TF-IDF (ngưỡng α = 0.15):
   IDF(e) = log( N / (1 + df(e)) )      ; df = số điều chứa cạnh e
   TF(e)  = c + (1-c) * count(e)/max_count   ; c = 0.15 (sàn của paper)
   weight = chuẩn hóa(TF*IDF) về [0,1]
   weight ≤ α  ->  meaningless = true   (Step 1.3: người duyệt mới xóa thật;
                   quan hệ cấu trúc 'Là loại của' luôn giữ).

Stage 2 — Gộp quan hệ đồng nghĩa:
   Nhờ ontology ĐÓNG (24 quan hệ canonical), các biến thể đồng nghĩa
   ("lái" / "điều khiển" / "vận hành") ĐÃ được gộp ngay lúc map (Phase 3).
   Phase này chỉ kiểm chứng: 0 cặp cạnh trùng (source,relation,target) còn sót.
"""

import os
import json
import math
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
GRAPH_IN = os.path.join(BASE, "data", "master_graph.json")
SECTIONS_IN = os.path.join(BASE, "data", "sections.json")
GRAPH_OUT = os.path.join(BASE, "data", "master_graph_optimized.json")

ALPHA = 0.15           # ngưỡng vô nghĩa
C_FLOOR = 0.15         # sàn TF của paper
STRUCTURAL_RELS = {"Là loại của"}   # luôn giữ dù IDF thấp


def main():
    with open(GRAPH_IN, "r", encoding="utf-8") as f:
        graph = json.load(f)
    nodes, edges = graph["nodes"], graph["relationships"]

    # N = tổng số điều/khoản/điểm trong văn bản
    N = 1
    if os.path.exists(SECTIONS_IN):
        with open(SECTIONS_IN, "r", encoding="utf-8") as f:
            N = max(len(json.load(f)), 1)

    max_count = max((e["count"] for e in edges), default=1)

    # Tính TF, IDF, weight thô
    raw_w = []
    for e in edges:
        df = max(len(e["listSectionId"]), 1)
        idf = math.log(N / (1 + df))
        idf = max(idf, 0.0)  # tránh âm khi df rất lớn
        tf = C_FLOOR + (1 - C_FLOOR) * (e["count"] / max_count)
        raw_w.append(tf * idf)

    max_w = max(raw_w, default=1) or 1
    flagged = 0
    for e, rw in zip(edges, raw_w):
        w = round(rw / max_w, 4)
        e["weight"] = w
        meaningless = (w <= ALPHA) and (e["name"] not in STRUCTURAL_RELS)
        e["meaningless"] = meaningless
        if meaningless:
            flagged += 1

    # Stage 2 — kiểm chứng không còn cạnh trùng
    dup = [k for k, c in Counter(
        (e["source"], e["name"], e["target"]) for e in edges).items() if c > 1]

    with open(GRAPH_OUT, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    id2name = {n["id"]: n["name"] for n in nodes}
    print("=" * 55)
    print(f"N (tổng số điều)            : {N}")
    print(f"Edges                       : {len(edges)}")
    print(f"  -> đánh dấu meaningless   : {flagged} (α={ALPHA})")
    print(f"Stage 2: cặp cạnh trùng sót : {len(dup)} (0 = ontology đóng đã gộp xong)")
    print("=" * 55)
    print(f"✅ Đã ghi: {GRAPH_OUT}")

    hi = sorted(edges, key=lambda x: x["weight"], reverse=True)[:6]
    lo = sorted(edges, key=lambda x: x["weight"])[:6]
    print("\n--- weight CAO (đặc trưng, giữ) ---")
    for e in hi:
        print(f"  w={e['weight']:.3f} ({id2name[e['source']]}) -[{e['name']}]-> ({id2name[e['target']]})")
    print("\n--- weight THẤP (nghi vô nghĩa, chờ duyệt) ---")
    for e in lo:
        tag = "✗" if e["meaningless"] else "•"
        print(f"  {tag} w={e['weight']:.3f} ({id2name[e['source']]}) -[{e['name']}]-> ({id2name[e['target']]})")


if __name__ == "__main__":
    main()
