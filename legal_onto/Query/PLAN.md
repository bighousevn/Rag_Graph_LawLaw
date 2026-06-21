# PLAN — Phase Truy vấn (Legal-Onto, bám sát paper INFOR617 §4.1, §4.4, Algorithm 2)

> Mục tiêu: trả lời câu hỏi pháp lý bằng cách **biểu diễn câu hỏi thành đồ thị**, rồi
> **phân rã star-graph → so khớp → giao tập** với knowledge base đã xây ở `Building_KG/`.
> **Phải đúng Algorithm 2 của paper**, không chế thêm bước ngoài luồng.

## 0. Công thức gốc (paper)

**Algorithm 2 — Finding relevant knowledge**
- **Input:** knowledge graph biểu diễn nghĩa của câu hỏi `q`.
- **Output:** nội dung trong KB `K` khớp với KG của `q`.
- **Step 1:** Phân rã đồ thị câu hỏi thành **star graph** (mỗi star = các triple **chung chủ thể**).
- **Step 2:** Với mỗi quan hệ `k ∈ Rel(S)`, tìm mọi `k' ∈ K.Rel` sao cho `similar(k', k) ≥ β`
  (β = 0.6, cosine PhoBERT). Lập `ID_k` = tập ID của `k'`. Xử lý **is-a** và **wildcard `*`**.
- **Step 3:** **Giao (intersect)** các tập kết quả của các star → tập subgraph khớp → ra `Section_ID`.

Hai loại truy vấn (paper §4.4): **conceptual query** và **violation query**.

---

## 1. Kiến trúc luồng (paper Fig 2 — QA module)

```
Câu hỏi NL
   │  Q1  Hiểu & biểu diễn câu hỏi  (DÙNG LẠI ontology + VnCoreNLP của Building_KG)
   ▼
Query Graph  =  các triple (Conc - Rel - Conc), có thể chứa "*"
   │  Q2  Phân rã STAR-GRAPH (Step 1)            ← gom theo chủ thể
   ▼
[ Star_1, Star_2, ... ]
   │  Q3  Match từng star vào KB (Step 2)        ← similar ≥ β + is-a + wildcard
   ▼
mỗi star → tập subgraph / tập Section_ID
   │  Q4  GIAO các tập (Step 3)                  ← intersection
   ▼
tập Section_ID khớp chặt
   │  Q5  Sinh câu trả lời                       ← lấy text gốc + Rules suy luận
   ▼
Câu trả lời (IRAC)
```

**Nguyên tắc giữ nguyên từ Building_KG:** LLM chỉ ở khâu *hiểu câu hỏi* (rewrite) và *sinh câu trả lời*; khâu **biểu diễn truy vấn thành triple & so khớp = ZERO LLM** (VnCoreNLP + ontology + vector cosine), hệt như lúc build.

---

## 2. Các bước chi tiết (mỗi bước = 1 script trong `Query/`)

### Q1 — Hiểu & biểu diễn câu hỏi → Query Graph  (`1_query_to_graph.py`)
*Đối chiếu paper §4.1: "QA module extracts entities and relations based on Conc/Rel of Legal-Onto; PhoBERT + VnCoreNLP support."*

- (tùy chọn) **rewrite** câu hỏi cho rõ chủ ngữ — **DÙNG LẠI** `Building_KG/2_rewrite_sections.py` (LLM, đúng 1 chỗ).
- **VnCoreNLP** tách SVO — **DÙNG LẠI** logic `Building_KG/3_extract_triplets.py`.
- **Ép ontology** — **DÙNG LẠI** `Building_KG/4_map_to_ontology.py`, nhưng:
  - Thành phần người hỏi **không nêu rõ → đặt `"*"`** (wildcard) thay vì bỏ.
  - Giữ cả triple Tier1/2/3 (truy vấn cần recall cao hơn build).
- **Phân loại truy vấn**: `conceptual` (hỏi định nghĩa/khái niệm) vs `violation` (hỏi chế tài/vi phạm) — dựa nhóm Rel xuất hiện (`Xử phạt/Phạt tiền/Trừ điểm…` → violation).
- **Output:** `query_graph.json` = danh sách triple `(s, v, o)` đã gán Conc/Rel, có `"*"`.

### Q2 — Phân rã Star-Graph  (`2_decompose_star.py`)  ← **Algorithm 2, Step 1**
- Gom các triple **chung chủ thể (s)** thành 1 star.
- Mỗi star = `{subject, [ (rel, object) ... ]}`.
- Ví dụ paper (Fig 5): *"phạt gì khi lái ô tô bằng GPLX hết hạn"* →
  - Star 1: `(Người, Điều khiển, Ô tô)` + `(Người, Có, GPLX)`
  - Star 2: `(GPLX, hết hạn, *)`
- **Output:** `query_stars.json`.

### Q3 — Match từng Star vào KB  (`3_match_star.py`)  ← **Algorithm 2, Step 2**
Với mỗi cạnh `(s, rel, o)` trong star:
1. **Vector anchor (similar ≥ β=0.6):** dùng embedding (Mongo vector search) tìm Concept/Relation trong KB gần nghĩa nhất với `s/rel/o`. Đây là `similar(k',k) ≥ β` của paper.
2. **Wildcard `*`:** thành phần là `*` → khớp **mọi** ID tương ứng (không ràng buộc), chỉ dựa các thành phần còn lại.
3. **is-a:** nếu KB có quan hệ `Là loại của` giữa concept câu hỏi và concept KB → lấy concept KB tương ứng (vd hỏi "xe" → khớp mọi loại xe con cháu).
4. **Duyệt đồ thị (Neo4j Cypher):** từ anchor, `MATCH`/`OPTIONAL MATCH` theo `rel` để lấy cạnh + `Section_ID`. `OPTIONAL MATCH` để không rỗng khi chủ thể khuyết.
- **Output mỗi star:** `ID_star` = tập `Section_ID` (và tập edge/subgraph) thỏa star đó.

### Q4 — Giao các Star  (`4_intersect.py`)  ← **Algorithm 2, Step 3**
- `result = ID_star_1 ∩ ID_star_2 ∩ ...`
- Nếu giao rỗng → nới lỏng: bỏ star yếu nhất / hạ ngưỡng β / dùng hợp (union) có xếp hạng theo `weight` (TF-IDF từ Phase 5).
- **Output:** `relevant_sections.json` = danh sách `Section_ID` + subgraph khớp + điểm.

### Q5 — Sinh câu trả lời  (`5_generate_answer.py`)
*Đối chiếu Building_KG: dùng `Section_ID` lấy nguyên văn + khung logic S-V-O + Rules.*
- Lấy **text gốc** từ `Building_KG/data/sections.json` theo `Section_ID`.
- Áp **`ontology/rules.json`** để suy luận multi-hop (vd is-a bắc cầu, thẩm quyền→xử phạt).
- Nhồi (text gốc + chuỗi S-V-O + rule kích hoạt) vào LLM đóng vai luật sư → trả lời **IRAC**, trích dẫn Điều/Khoản/Điểm.
- **Output:** `answer.json`.

---

## 3. Đối chiếu công cụ project ↔ thành phần paper

| Thành phần paper | Công cụ trong dự án |
|---|---|
| `similar()` cosine, β=0.6 (PhoBERT) | Embedding `AITeamVN/Vietnamese_Embedding_v2` + **Mongo `$vectorSearch`** (ngưỡng score ↔ β) |
| KB = ontology + KG | **Neo4j** (`Concept` + 24 quan hệ, có `weight`) đã nạp ở Phase 6 |
| Conc/Rel để hiểu câu hỏi | `Building_KG/ontology/ontology.json` (DÙNG LẠI) |
| Rules (suy diễn) | `Building_KG/ontology/rules.json` (DÙNG LẠI) |
| VnCoreNLP NLP support | `Building_KG/3_extract_triplets.py` (DÙNG LẠI) |

## 4. Tái sử dụng từ Building_KG (không viết lại)
`2_rewrite_sections.py`, `3_extract_triplets.py`, `4_map_to_ontology.py`, `ontology.json`, `rules.json`,
`sections.json`. → Đảm bảo **câu hỏi và KB nói cùng một bộ từ vựng ontology** (điều kiện để match đúng).

## 5. Checklist trung thành với paper
- [ ] Câu hỏi được map về **đúng Conc/Rel đóng** (không sinh nhãn mới).
- [ ] Có **wildcard `*`** cho thành phần khuyết.
- [ ] **Phân rã star theo chủ thể** (Step 1) — không bỏ qua, không thay bằng "match từng triple rời".
- [ ] Match dùng **`similar ≥ β`** + **is-a** (Step 2).
- [ ] **Giao tập** các star (Step 3) — không phải chỉ union.
- [ ] Phân biệt **conceptual vs violation query**.
- [ ] Trả lời bám `Section_ID` + Rules (không bịa ngoài KB).

## 6. Thứ tự chạy (dự kiến)
```
1_query_to_graph.py → 2_decompose_star.py → 3_match_star.py → 4_intersect.py → 5_generate_answer.py
```

## 7. Khác biệt với code query cũ (`src/Query/`)
Code cũ match **từng triple rồi cộng dồn/giao** — gần đúng nhưng **thiếu bước phân rã star chính thức**.
Plan này bổ sung **Step 1 (star)** + **Step 3 (intersect theo star)** đúng Algorithm 2, và tái dùng ontology/rules mới.
```
```
