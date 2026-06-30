# Legal-Onto: Hệ thống Truy vấn Pháp lý dựa trên Ontology và Đồ thị Tri thức

## Tổng quan

Hệ thống xây dựng knowledge base cho miền pháp lý bằng cách kết hợp **Ontology** (tầng khái niệm trừu tượng) và **Knowledge Graph** (tầng thực thể cụ thể). Mục tiêu là **điều hướng câu hỏi người dùng tới đúng điều khoản pháp luật** — không phải trả lời trực tiếp.

Tham khảo: _"Ontology-Based Knowledge Graph Approach for Legal Queries"_ — Informatica, Vol. 37, Issue 1 (2026), doi:10.15388/25-INFOR617

---

## Kiến trúc hệ thống

```
Văn bản luật
     ↓
[Ontology Layer]         ← Chuyên gia định nghĩa
  Concept + Keyphrases
  Relation + Keyphrases + ConceptS + ConceptO
     ↓
[Knowledge Graph Layer]  ← Tự động + chuyên gia kiểm tra
  Node = Concept
  Edge = (ConceptS, Relation, ConceptO)
  Metadata = [Điều, Khoản, Điểm, Văn bản]
     ↓
[Query Processing]       ← VnCoreNLP + PhoBERT
  Câu hỏi → Triplet → Tìm khớp → Trả về địa chỉ điều khoản
```

---

## Thiết kế Ontology

### Concept

Mỗi concept là một danh từ chung đại diện cho một loại thực thể trong miền pháp lý.

```python
Concept:
  - Name: str          # Tên chuẩn, ví dụ "Xe máy"
  - Keyphrases: set    # Các cách gọi trong văn bản
                       # {"xe máy", "xe mô tô", "xe gắn máy", "mô tô"}
```

**Nguyên tắc xác định concept:**

- Dùng mức đủ cụ thể để phân biệt các điều khoản khác nhau
- Tách concept riêng khi không có object nào trong triplet giúp phân biệt được hai chủ thể
- Không cần phủ hết mọi thực thể xuất hiện trong văn bản — chỉ tạo concept cho thực thể đóng vai trò CHỦ THỂ/ĐỐI TƯỢNG trong một Relation thể hiện đúng ý hành vi cốt lõi của Điều/Khoản/Điểm. Thực thể phụ không ảnh hưởng tới việc phân biệt điều khoản (vd chi tiết định lượng, liệt kê minh hoạ, thực thể chỉ xuất hiện thoáng qua) thì bỏ qua, không cần concept hóa
- Nhất quán mức độ trừu tượng giữa ConceptS và ConceptO trong cùng một Relation

**Ví dụ:**

| Concept          | Keyphrases                                         |
| ---------------- | -------------------------------------------------- |
| Người            | "người", "cá nhân", "người tham gia giao thông"    |
| Người đi bộ      | "người đi bộ", "người băng qua đường"              |
| Xe máy           | "xe máy", "xe mô tô", "xe gắn máy", "mô tô"        |
| Ô tô             | "ô tô", "xe ô tô", "xe hơi", "tài xế"              |
| Cồn              | "rượu", "bia", "cồn", "rượu bia", "đồ uống có cồn" |
| Điện thoại       | "điện thoại", "điện thoại di động", "smartphone"   |
| Giấy phép lái xe | "bằng lái", "giấy phép lái xe", "GPLX"             |

> **Lưu ý:** `Người đi bộ` là concept riêng vì hành vi vi phạm của người đi bộ (ví dụ: vượt vạch kẻ đường) không thể phân biệt với người lái xe chỉ qua object của triplet.

---

### Relation

Mỗi relation là một quan hệ cốt lõi, đủ để cùng Concept thể hiện đúng ý hành vi của một đơn vị luật — không cần phủ kín mọi biến thể hành vi nhỏ lẻ xuất hiện trong văn bản.

```python
Relation:
  - Name: str          # Tên chuẩn, ví dụ "sử_dụng"
  - Keyphrases: set    # Các động từ biểu hiện quan hệ này
                       # {"dùng", "sử dụng", "thực hiện"}
  - ConceptS: Concept  # Một concept chủ thể
  - ConceptO: [Concept] # Nhiều concept đối tượng
```

**Ví dụ:**

| Relation   | Keyphrases                        | ConceptS    | ConceptO                    |
| ---------- | --------------------------------- | ----------- | --------------------------- |
| điều_khiển | "điều khiển", "lái", "cầm lái"    | Người       | [Xe máy, Ô tô, ...]         |
| sử_dụng    | "dùng", "sử dụng", "thực hiện"    | Người       | [Cồn, Điện thoại, Tai nghe] |
| không_có   | "không có", "thiếu", "không mang" | Người       | [Giấy phép lái xe, ...]     |
| vi_phạm    | "vi phạm", "không tuân thủ"       | Người đi bộ | [Vạch kẻ đường, ...]        |

**Nguyên tắc:**

- Mỗi relation có đúng **một ConceptS**
- Nếu cùng hành vi có nhiều chủ thể khác nhau → tạo **nhiều relation riêng biệt**

---

## Thiết kế Knowledge Graph

### Node

Mỗi node là một **Concept** (đã được chuẩn hóa qua Ontology), không phải keyphrase thô.

### Edge (Triplet)

```python
Triplet:
  - ConceptS: Concept         # Node chủ thể
  - Relation: Relation        # Loại quan hệ
  - ConceptO: Concept         # Node đối tượng
  - addresses: [Address]      # Danh sách địa chỉ điều khoản
```

```python
Address:
  - article: int              # Số điều
  - clause: int               # Số khoản
  - point: str                # Điểm (a, b, c, ...)
  - document: str             # Tên văn bản (Nghị định 100/2019, ...)
```

**Ví dụ:**

```python
{
  "ConceptS": "Người",
  "Relation": "sử_dụng",
  "ConceptO": "Cồn",
  "addresses": [
    {"article": 5, "clause": 8, "point": "a", "document": "Nghị định 100/2019/NĐ-CP"},
    {"article": 6, "clause": 4, "point": "h", "document": "Nghị định 123/2021/NĐ-CP"}
  ]
}
```

**Nguyên tắc:**

- Triplet lặp lại ở nhiều điều khoản → **không tạo edge mới**, chỉ thêm address vào danh sách
- Đồ thị lưu quan hệ ngữ nghĩa, không lưu giá trị cụ thể (mức phạt, thời gian...)

---

## Format lưu trữ Ontology

Ontology được lưu dưới dạng JSON với hai danh sách `concepts` và `relations`:

```json
{
  "concepts": [
    {
      "name": "Ô tô",
      "keyphrases": ["xe ô tô", "xe chở người bốn bánh có gắn động cơ", "ô tô", "xe hơi"]
    }
  ],
  "relations": [
    {
      "name": "điều_khiển",
      "keyphrases": ["điều khiển", "lái", "cầm lái"],
      "concept_s": "Người",
      "concept_o": ["Ô tô", "Xe máy"]
    }
  ]
}
```

---

## Quy trình xây dựng

### Bước 1: Đọc toàn bộ văn bản

Đọc hết văn bản trước, liệt kê tất cả thực thể và hành vi xuất hiện.

### Bước 2: Xây dựng Ontology (GPT-4o-mini + chuyên gia duyệt)

Input: field `rewritten_propositions` từ file JSON (các mệnh đề đã chuẩn hoá Chủ thể–Hành vi–Đối tượng).

```
2a. Trích xuất entity và relation bằng GPT-4o-mini
    - Gửi từng batch rewritten_propositions cho GPT-4o-mini
    - LLM nhận diện entity (danh từ chủ thể/đối tượng) và relation (hành vi) từ mỗi mệnh đề
    - Nhóm các entity đồng nghĩa về cùng một Concept; nhóm các relation đồng nghĩa về cùng một Relation
    - Chỉ giữ Concept cho entity tham gia vào một Relation thể hiện ý hành vi cốt lõi của đơn vị luật
    - Bỏ qua entity phụ không phục vụ việc phân biệt điều khoản và relation rác (lỗi parser, không rõ nghĩa)

2b. Xác định ConceptS, ConceptO cho từng Relation
    - Dựa trên ví dụ subject/object đi kèm từng relation trong triplet thô
    - Mỗi Relation có đúng một ConceptS

2c. Kiểm tra mức đủ dùng (chuyên gia duyệt)
    - Mỗi Điều/Khoản/Điểm có ít nhất một triplet (ConceptS, Relation, ConceptO) diễn tả đúng ý hành vi cốt lõi của nó
    - Không cần ánh xạ mọi keyphrase trong văn bản về một concept — chỉ cần Concept + Relation đã đủ để phân biệt đơn vị luật này với các đơn vị luật khác
```

> **Lưu ý:** VnCoreNLP **không** dùng ở bước này — tách từ tự động hay phân tích cú pháp dependency sẽ tách thuật ngữ pháp lý quá nhỏ lẻ, không phù hợp để gom nhóm khái niệm ở mức miền. GPT-4o-mini xử lý trực tiếp `rewritten_propositions` hiệu quả hơn cho task này. VnCoreNLP chỉ dùng ở Bước 5 (xử lý câu hỏi người dùng).

**Công cụ:**

- GPT-4o-mini (OpenAI) — trích xuất entity/relation, gom nhóm thành Concept/Relation theo batch
- PhoBERT — đo độ tương đồng ngữ nghĩa để gợi ý gom nhóm đồng nghĩa (tuỳ chọn, bổ trợ)

### Bước 3: Xây dựng Knowledge Graph (GPT-4o-mini + bán tự động)

```
3a. Với mỗi rewritten_proposition, dùng GPT-4o-mini tạo triplet
    - Input: proposition + ontology đã có (Concept list + Relation list)
    - LLM ánh xạ entity → Concept, động từ → Relation
    - Sinh một hoặc nhiều triplet (ConceptS, Relation, ConceptO) theo dạng mạng lưới nối tiếp:
        không chỉ quan hệ trực tiếp giữa chủ thể và đối tượng, mà còn các quan hệ
        tiếp theo giữa đối tượng với các concept khác nếu proposition diễn đạt chuỗi
        Ví dụ: (Người, sử_dụng, Còi) → (Còi, bị_cấm_tại, Khu đông dân cư)

3b. Nếu triplet đã tồn tại → thêm address mới vào danh sách
    Nếu triplet chưa tồn tại → tạo edge mới

3c. Gắn address (Điều, Khoản, Điểm, Văn bản) cho mỗi triplet
    từ metadata của section tương ứng trong file JSON
```

### Bước 4: Tối ưu đồ thị

Loại bỏ triplet trùng lặp, hợp nhất cạnh có cùng nghĩa.

### Bước 5: Xử lý câu hỏi người dùng

```
5a. VnCoreNLP + PhoBERT phân tích câu hỏi
5b. Ánh xạ từ khóa → Concept và Relation (qua Keyphrases trong Onto)
5c. Tạo một hoặc nhiều triplet từ câu hỏi
5d. Tìm triplet khớp trong đồ thị
5e. Lấy giao các tập address → trả về danh sách điều khoản
```

---

## Công cụ và thư viện

| Công việc                              | Công cụ                                                                     |
| -------------------------------------- | --------------------------------------------------------------------------- |
| Tách từ, NER, POS tiếng Việt           | VnCoreNLP                                                                   |
| Embedding câu truy vấn                 | PhoBERT                                                                     |
| Lưu trữ Knowledge Graph               | Neo4j hoặc RDFLib                                                           |
| Tính trọng số keyphrase               | TF-IDF (scikit-learn)                                                       |
| Parse cấu trúc văn bản                | Python + regex / pdfplumber                                                 |
| Trích xuất entity/relation, tạo triplet | GPT-4o-mini (OpenAI) — $0.15/1M input tokens, $0.60/1M output tokens; rẻ hơn ~7× Claude Haiku, đủ mạnh cho task trích xuất có cấu trúc |

---
