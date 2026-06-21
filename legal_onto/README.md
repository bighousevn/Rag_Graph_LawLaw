# Legal-Onto KG — Pipeline xây Knowledge Graph (bám sát paper INFOR617)

Xây dựng đồ thị tri thức pháp luật theo mô hình **Legal-Onto**
`K = (Conc, Rel, Rules) ⊕ (Keyphrases, Rela, weight)`.
Toàn bộ code nằm trong folder này, **không sửa code cũ** của dự án.

## Nguyên tắc
- **LLM chỉ dùng ở 1 bước duy nhất** (`2_rewrite_sections.py` — viết lại câu cho rõ chủ ngữ).
- **Tách triplet & ép ontology: ZERO LLM** — thuần VnCoreNLP + lexicon ontology.
- Ontology **đóng** (closed-IE): mọi S/V/O phải ép về 70 Concept / 24 Relation định sẵn.
- Ràng buộc theo **chữ ký nhóm** `ConcKeyS_cat / ConcKeyO_cat` + **soft-tier** (lệch thì giữ + gắn cờ, không vứt).

## Thứ tự chạy (`Building_KG/`)
| Bước | Lệnh | Ra |
|---|---|---|
| 0 | `python 0_parse_ontology.py` | `ontology/ontology.json` (merge `ontology_patch.json`) |
| 1 | `python 1_parse_pdf_to_sections.py` | `data/sections.json` |
| 2 | `python 2_rewrite_sections.py` *(cần OPENAI_API_KEY)* | `data/sections_rewritten.json` |
| 3 | `python 3_extract_triplets.py` | `data/triplets_raw.json` |
| 4 | `python 4_map_to_ontology.py` | `data/triplets_mapped.json` + `triplets_flagged.json` |
| 5 | `python 5_build_graph.py` | `data/master_graph.json` + `ontology/rules.json` |
| 6 | `python 6_optimize.py` | `data/master_graph_optimized.json` (TF-IDF weight) |
| 7 | `python 7_import_neo4j.py` *(cần Neo4j + APOC)* | nạp Neo4j |
| 8 | `python 8_embedding.py` *(cần MongoDB)* | nạp vector |

## Lưu ý môi trường
- Chạy bằng venv có sẵn: `.venv/bin/python ...` (đã có py_vncorenlp, openai, sentence-transformers).
- VnCoreNLP cần Java; script `3_extract_triplets.py` **tự dò `libjvm.so`** (mise/redhat/system).
- Model VnCoreNLP ở `../../vncorenlp_model` (dùng lại của dự án).

## Kết quả tham chiếu (Nghị định 168/2024)
`1.085 sections → 17.469 triplet thô → 3.005 triple có kiểu → 46 node · 97 cạnh có weight`.
`data/triplets_flagged.json` = worklist làm giàu ontology (active learning).

## Còn lại (buổi sau)
Phần **truy vấn** (decompose star-graph + Cypher OPTIONAL MATCH + sinh câu trả lời) — chưa làm.
