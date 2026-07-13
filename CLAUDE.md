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

## Quy trình tách triplet atomic (đúc kết qua thực nghiệm tay trên Điều 6)

> **Bối cảnh:** Sau nhiều lần thử tự động hoá Bước 2/3 bằng GPT-4o-mini và gặp vấn đề "dilution" (LLM áp quy tắc không nhất quán khi batch nhiều pattern khác nhau), đã quyết định **xây Concept/Relation/Triplet cho Điều 6 hoàn toàn thủ công** (Claude tự đọc `rewritten_propositions` và suy luận trực tiếp, không gọi LLM API). Quy trình dưới đây là kết quả đúc kết từ quá trình đó.
>
> **Ràng buộc bắt buộc:** Việc xây ontology mới (Điều 6 trở đi) **không được tham chiếu, đối chiếu, hay kế thừa** từ `ontology_168.json` / `ontology_168.md` (bộ ontology cũ, phạm vi Điều 5-7, xây theo tiêu chí khác — ví dụ tách "Xe ô tô", "Xe chở người bốn bánh có gắn động cơ" thành các Concept riêng thay vì gộp) — Concept, Relation, Keyphrase đều phải xây mới, tối đa hoá tính atomic.

### Quy trình 9 bước (tổng quát, áp dụng cho mọi điều khoản, không riêng giao thông)

1. **Xác định chủ thể xuyên suốt** — không đổi ConceptS sang loại xe/thực thể cụ thể ngay cả sau khi đã biết loại; loại cụ thể chỉ xuất hiện làm ConceptO của quan hệ đầu tiên, việc phân biệt loại xe dựa vào giao địa chỉ lúc truy vấn (Bước 5e).
2. **Tách vị ngữ "điều khiển + loại xe" thành triplet riêng** — danh sách liệt kê các tên đồng nhóm (vd "xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô") gộp thành **1 Concept + keyphrases**, không tách Concept riêng cho từng tên trong nhóm.
3. **Chẻ câu theo mệnh đề độc lập** (nối bằng `;`, "hoặc") — mỗi mệnh đề xử lý riêng, không gộp thành một relation ghép.
4. **Xác định relation + object cốt lõi của mỗi mệnh đề:**
   - Verb **nội tại** (tự thân thay đổi trạng thái, vd quay đầu, chuyển hướng, rẽ trái/phải, dừng, đỗ, lùi, tránh) → ConceptO = chính chủ thể/xe đang thực hiện.
   - Verb **ngoại tác** (tác động lên đối tượng khác, vd chuyển làn, không nhường, vượt) → ConceptO = thực thể bên ngoài chịu tác động.
   - **Luôn giữ phủ định trong tên relation** ("Không nhường" ≠ "Nhường") — không chuẩn hoá về dạng khẳng định để tăng recall, vì sẽ làm hai điều khoản đối lập trộn lẫn triplet (xem ví dụ điểm l/m bên dưới).
5. **Xử lý cụm phụ theo mẫu cố định:**
   - `"X dành cho/của/thuộc Y"` → luôn tách `(X, Dành_cho, Y)`, **trừ khi** cả cụm nằm dưới phủ định-tồn-tại (`"không có X..."`) → bỏ hẳn, không tạo triplet, không placeholder.
   - `"tại nơi có/không có Z"` → chỉ tạo triplet khi Z thật sự tồn tại; ConceptS của relation "Tại" luôn phải là thực thể vật lý (Người/xe/biển báo), không bao giờ là một Concept-hành-động trừu tượng.
   - `"biển báo hiệu có nội dung cấm X"` → tách `(S, Tại, Biển báo hiệu)` + `(Biển báo hiệu, Cấm, X)`, dùng chung 1 Concept "Biển báo hiệu" cho mọi loại biển, X tái dùng đúng tên relation hành vi đã dùng ở nhánh chính.
   - Ngưỡng số liệu (nồng độ cồn, khung giờ, tốc độ...) → **giữ nguyên giá trị** nếu là yếu tố phân biệt với điều/khoản khác (verify bằng đối chiếu corpus), bỏ nếu chỉ là chi tiết minh hoạ không ảnh hưởng phân biệt.
6. **Lọc bỏ:** cụm tham chiếu/loại trừ điều khoản khác ("trừ trường hợp...", "theo quy định tại Điều..."); định ngữ lặp không mang thông tin phân biệt riêng ("trái quy định" đứng một mình); đồng nghĩa miền gần nghĩa với Concept đã có → gộp vào Concept đó thay vì tạo mới (vd "xe lăn của người khuyết tật" gộp vào "Người đi bộ").
7. **Đối chiếu bắt buộc với các điểm anh em cùng Khoản** trước khi chốt triplet — nếu hai điểm gần giống hệt nhau nhưng khác đúng một chi tiết, chi tiết đó phải tạo ra khác biệt rõ trong bộ triplet.
8. **Tái dùng Concept/Relation đã tạo** ở các Điều/Khoản trước đó trước khi đặt tên mới.
9. **Keyphrase chỉ ghi danh từ thuần** — không bao giờ gán kèm động từ/tính từ vào để thành cụm (vd không được ghi "người điều khiển" làm keyphrase của Concept "Người").

> **Lưu ý về tính tổng quát:** Bước 1-9 ở dạng trên là thuật toán decomposition tổng quát. Các minh hoạ cụ thể trong bước 4-5 (danh sách verb nội tại/ngoại tác, pattern "Biển báo + Cấm", pattern ngưỡng nồng độ cồn) là ví dụ áp dụng cho miền giao thông (vì Điều 6 quy định hành vi lái xe) — khi xử lý điều khoản thuộc miền nội dung khác, các bước 1-9 vẫn áp dụng nhưng danh sách verb/pattern cụ thể sẽ khác.

### Ba ví dụ vàng (dùng làm regression-check)

**1. Cặp điểm l/m — Khoản 5, Điều 6 (minh hoạ rule phủ định-tồn-tại của "dành cho"):**

- `s109` (Khoản 5 - Điểm l): "...chuyển hướng không nhường quyền đi trước cho người đi bộ, xe lăn của người khuyết tật qua đường tại nơi **có vạch kẻ đường** dành cho người đi bộ; xe thô sơ đang đi trên phần đường dành cho xe thô sơ." → 8 triplet:
  (Người, Điều khiển, Ô tô), (Người, Chuyển hướng, Ô tô), (Người, Không nhường, Người đi bộ), (Người đi bộ, Tại, Vạch kẻ đường), (Vạch kẻ đường, Dành cho, Người đi bộ), (Người, Không nhường, Xe thô sơ), (Xe thô sơ, Tại, Phần đường), (Phần đường, Dành cho, Xe thô sơ).
- `s110` (Khoản 5 - Điểm m): "...chuyển hướng không nhường đường cho các xe đi ngược chiều; người đi bộ, xe thô sơ đang qua đường tại nơi **không có vạch kẻ đường** cho người đi bộ." → 5 triplet (không có triplet nào về vạch kẻ đường, vì cụm nằm dưới phủ định-tồn-tại nên bỏ hẳn):
  (Người, Điều khiển, Ô tô), (Người, Chuyển hướng, Ô tô), (Người, Không nhường, Xe đi ngược chiều), (Người, Không nhường, Người đi bộ), (Người, Không nhường, Xe thô sơ).

**2. s97 — Khoản 4, Điểm k (minh hoạ pattern Biển báo + Cấm):**

"...quay đầu xe tại nơi có biển báo hiệu có nội dung cấm quay đầu...; điều khiển xe rẽ trái tại nơi có biển báo hiệu có nội dung cấm rẽ trái...; điều khiển xe rẽ phải tại nơi có biển báo hiệu có nội dung cấm rẽ phải..." → 8 triplet:
(Người, Điều khiển, Ô tô), (Người, Quay đầu, Ô tô), (Người, Tại, Biển báo hiệu), (Biển báo hiệu, Cấm, Quay đầu), (Người, Rẽ trái, Ô tô), (Biển báo hiệu, Cấm, Rẽ trái), (Người, Rẽ phải, Ô tô), (Biển báo hiệu, Cấm, Rẽ phải).

**3. s117/s123/s129 — 3 mốc nồng độ cồn (minh hoạ giữ ngưỡng số liệu vì phân biệt địa chỉ):**

| Section | Khoản | Triplet |
| --- | --- | --- |
| s117 | 6-c | (Người, Sử dụng, Cồn), (Cồn, Chưa vượt quá, 50mg/100ml máu), (Cồn, Chưa vượt quá, 0,25mg/1L khí thở) |
| s123 | 9-a | (Người, Sử dụng, Cồn), (Cồn, Vượt quá, 50-80mg/100ml máu), (Cồn, Vượt quá, 0,25-0,4mg/1L khí thở) |
| s129 | 11-a | (Người, Sử dụng, Cồn), (Cồn, Vượt quá, 80mg/100ml máu), (Cồn, Vượt quá, 0,4mg/1L khí thở) |

(mỗi mốc còn kèm `(Người, Điều khiển, Ô tô)` như mọi section khác trong Điều 6)

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
