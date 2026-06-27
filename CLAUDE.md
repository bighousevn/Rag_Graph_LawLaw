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
- Mọi keyphrase trong văn bản phải ánh xạ được về đúng một concept (độ phủ 100%)
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

Mỗi relation là một quan hệ cốt lõi đủ phủ toàn bộ hành vi trong văn bản luật.

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

## Quy trình xây dựng

### Bước 1: Đọc toàn bộ văn bản

Đọc hết văn bản trước, liệt kê tất cả thực thể và hành vi xuất hiện.

### Bước 2: Xây dựng Ontology (chuyên gia thực hiện)

```
2a. Xác định Concept
    - Nhóm các thực thể đồng nghĩa về cùng một concept
    - Kiểm tra độ phủ: mọi keyphrase trong văn bản đều ánh xạ được

2b. Xác định Relation
    - Liệt kê các quan hệ cốt lõi
    - Xác định ConceptS, ConceptO cho từng relation

2c. Kiểm tra độ phủ
    - Không có keyphrase nào bị lơ lửng (không thuộc concept nào)
```

**Công cụ hỗ trợ tìm concept và relation:**

- VnCoreNLP / spaCy — NER, POS tagging để nhận diện danh từ và động từ
- PhoBERT — đo độ tương đồng ngữ nghĩa để gợi ý nhóm đồng nghĩa
- LLM (GPT-4, Gemini) — sinh draft để chuyên gia xét duyệt
- OpenIE — trích xuất triplet thô (subject, relation, object)

### Bước 3: Xây dựng Knowledge Graph (bán tự động)

```
3a. Trích xuất keyphrases từ từng điều khoản (VnCoreNLP)
3b. Ánh xạ keyphrase → Concept (qua Keyphrases trong Onto)
3c. Ánh xạ động từ → Relation (qua Keyphrases của Relation)
3d. Tạo triplet (ConceptS, Relation, ConceptO)
3e. Nếu triplet đã tồn tại → thêm address mới
    Nếu triplet chưa tồn tại → tạo edge mới
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

| Công việc                    | Công cụ                     |
| ---------------------------- | --------------------------- |
| Tách từ, NER, POS tiếng Việt | VnCoreNLP                   |
| Embedding câu truy vấn       | PhoBERT                     |
| Lưu trữ Knowledge Graph      | Neo4j hoặc RDFLib           |
| Tính trọng số keyphrase      | TF-IDF (scikit-learn)       |
| Parse cấu trúc văn bản       | Python + regex / pdfplumber |

---
