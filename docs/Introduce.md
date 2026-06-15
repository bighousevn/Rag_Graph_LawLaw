# KIẾN TRÚC TỔNG THỂ HỆ THỐNG LEGAL GRAPHRAG (PHÁP LUẬT VIỆT NAM)

Tài liệu này mô tả toàn bộ vòng đời của hệ thống GraphRAG pháp lý, từ khâu định nghĩa Lược đồ (Ontology), xử lý dữ liệu thô (Ingestion) cho đến luồng truy vấn thời gian thực (Retrieval Pipeline).

## PHẦN 1: THIẾT KẾ ONTOLOGY (NGÔN NGỮ CHUNG CỦA ĐỒ THỊ)

Để tránh hiện tượng LLM sinh rác từ vựng (Open IE), hệ thống sử dụng phương pháp **Trích xuất Đóng (Closed IE)** với bộ Ontology được thiết kế theo tiêu chuẩn MECE (Không trùng lặp, Không bỏ sót).

### 1. Cấu trúc Nút (Node - Thực thể)

Được thiết kế theo kiến trúc 2 lớp:

- **Lớp Phân loại (`label`):** Đóng vai trò là "rổ" gom nhóm. Bao gồm khoảng 15 nhãn cốt lõi chia thành 4 trụ cột:
  - _Chủ thể:_ `Co_Quan_Nha_Nuoc`, `To_Chuc`, `Ca_Nhan`.
  - _Khách thể:_ `Tai_San`, `Giay_To_Phap_Ly`, `Khoan_Tien`.
  - _Nghiệp vụ:_ `Thu_Tuc_Hanh_Chinh`, `Hoat_Dong_Quan_Ly`, `Quyen_Loi`, `Nghia_Vu`, `Dieu_Kien`, `Thoi_Han`.
  - _Vi phạm:_ `Hanh_Vi_Vi_Pham_Phap_Luat`, `Hinh_Phat_Che_Tai`.
- **Lớp Định danh (`name`):** Giá trị cụ thể của thực thể, bắt buộc rút gọn thành cụm danh từ cốt lõi (5-7 từ), tuyệt đối không chứa câu định nghĩa dài dòng.

### 2. Cấu trúc Cạnh (Relationship - Tương tác)

Chỉ sử dụng 1 lớp định danh duy nhất (Type) để tối ưu tốc độ duyệt đồ thị. Bao gồm các nhóm hành động chính:

- _Ban hành & Quy định:_ `BAN_HANH`, `CAN_CU_VAO`, `AP_DUNG_CHO`, `TRU_TRUONG_HOP`.
- _Quản lý nhà nước:_ `CAP_PHAT`, `THU_HOI`, `GIAI_QUYET`, `LAP_QUAN_LY`.
- _Tương tác dân sự:_ `SO_HUU_SU_DUNG`, `CHUYEN_GIAO_CHO`, `YEU_CAU_CO`.
- _Vi phạm luật:_ `THUC_HIEN_HANH_VI`, `BI_XU_LY_BANG`.

---

## PHẦN 2: QUY TRÌNH XÂY DỰNG ĐỒ THỊ (OFFLINE INGESTION)

Quy trình này xử lý văn bản luật thô, bóc tách thành các bộ ba S-V-O và nạp vào Neo4j. Script xử lý chính sử dụng Python kết hợp OpenAI API.

### 1. Kỹ thuật chia cắt dữ liệu (Atomic Chunking)

- **Chiến lược:** `chunk_size = 1`.
- **Mục đích:** Ép LLM chỉ đọc và xử lý duy nhất 1 Điều/Khoản luật trong mỗi lượt API. Điều này triệt tiêu hoàn toàn lỗi **Ảo giác chéo (Cross-hallucination)** – hiện tượng LLM lấy chủ thể ở điều này ghép khiên cưỡng với đối tượng ở điều khác.

### 2. Trích xuất bằng LLM (`gpt-4o`)

- **Structured Outputs:** Khóa cứng định dạng trả về bằng JSON Schema (Strict: True). Ép `source` và `target` của cạnh phải là mã `id` (VD: `N001`), không được dùng chữ text để tránh đứt gãy lúc map dữ liệu.
- **Xử lý khuyết thành phần:** Sử dụng **Nút Giả lập (Placeholder Node)** như `[Chủ_thể_ngầm_định]` hoặc `[Cơ_quan_có_thẩm_quyền]` cho các điều luật ẩn chủ ngữ, đảm bảo đồ thị không bao giờ có mũi tên lơ lửng.
- **Prompt Engineering (Few-Shot & Negative Rules):**
  - Làm mẫu cụ thể cấu trúc JSON đầu ra.
  - Khóa lỗi "Vật vô tri thành tinh" (Cấm tài sản/giấy tờ làm chủ thể hành động).
  - Khóa lỗi "Cơ quan tác động vật lý" (Ép trích xuất thủ tục hành chính thay vì tác động thẳng lên cục đất).

### 3. Gộp đồ thị & Giám sát (Merging & Observability)

- Thuật toán gộp tất cả các Nút có cùng `label` + `name` thành một Node duy nhất, nối chung mảng `listSectionId`.
- Loại bỏ hoàn toàn các cạnh tự trỏ (Self-loop).
- Tự động xuất file log `svo_triplets_log.json` (Chủ thể - Hành động - Đối tượng) để con người dễ dàng nghiệm thu ngữ nghĩa pháp lý (Semantic Logic) bằng mắt thường.

---

## PHẦN 3: QUY TRÌNH TRUY VẤN (ONLINE RETRIEVAL PIPELINE)

Luồng này xử lý câu hỏi thời gian thực của người dùng. Backend Orchestrator (Golang) chịu trách nhiệm điều phối toàn bộ quy trình 4 bước:

### Bước 1: Dịch thuật Truy vấn (Query Rewriting)

- Đưa câu hỏi tự nhiên của người dùng qua LLM trung gian.
- Dịch câu hỏi sang "ngôn ngữ Đồ thị" dựa trên bộ Ontology 16 nhãn. Xác định Nút Mỏ Neo (Anchor Node) và Mũi tên mục tiêu.

### Bước 2: Tìm Nút Mỏ Neo (Vector Anchor Matching)

- Sử dụng Vector Database để so khớp ngữ nghĩa (Cosine Similarity).
- Tìm kiếm Node cụ thể trong đồ thị có trường `name` gần nghĩa nhất với từ khóa lõi trong câu hỏi (VD: "Sang tên sổ đỏ" -> Node: "Chuyển quyền sử dụng đất").

### Bước 3: Duyệt Đồ thị bằng Cypher (Graph Traversal)

- Sử dụng **`OPTIONAL MATCH`** làm vũ khí cốt lõi để đối với dữ liệu khuyết chủ thể.
- _Cơ chế:_ Bám chặt Nút Mỏ Neo (lệnh `MATCH` cứng). Dùng `OPTIONAL MATCH` để quét các râu ria xung quanh (Chủ thể, Điều kiện). Có thì lấy thêm ngữ cảnh, là nút giả lập cũng lấy, không có thì không báo lỗi, tránh tình trạng truy vấn trả về rỗng (0 records).
- Mục tiêu cuối cùng của bước này là thu thập được danh sách các `Section_ID` (Mã điều luật) liên quan chặt chẽ nhất theo đường đi logic.

### Bước 4: Tổng hợp Câu trả lời (Generation)

- Backend dùng các `Section_ID` chọc vào Document DB để lấy nguyên văn văn bản luật.
- Nhồi toàn bộ text gốc cùng chuỗi logic S-V-O tìm được vào LLM (`gpt-4o-mini` để tối ưu tốc độ/chi phí).
- LLM đóng vai trò "Luật sư", dựa vào khung xương logic cứng để diễn đạt lại một cách tự nhiên, chính xác và triệt tiêu hoàn toàn ảo giác.

---

## PHẦN 4: BÀI HỌC CỐT LÕI VÀ TRIẾT LÝ KỸ THUẬT

1.  **Dữ liệu chuẩn thì Truy vấn mới chuẩn:** Đầu tư mạnh vào mô hình lớn (`gpt-4o`) ở khâu offline ingestion để đồ thị đạt độ tinh khiết tối đa.
2.  **Schema-induced Attention Degradation:** Khi ép LLM vào cấu trúc JSON quá khắc nghiệt, nó sẽ giảm khả năng suy luận ngữ nghĩa. Bắt buộc phải có Few-shot prompting và chia nhỏ vùng nhận thức (`chunk_size=1`).
3.  **Hội chứng "Nhét bừa" (Forced Pigeonholing):** Ontology phải bao phủ mọi kịch bản. Nếu thiếu "rổ" cho hành động hợp pháp, AI sẽ nhét bừa nó vào "rổ" vi phạm.
4.  **Sức mạnh của GraphRAG:** Không phải là việc "lấy dư dữ liệu", mà là việc cung cấp cho LLM một **bản đồ tư duy pháp lý (Mindmap)** đã được đánh dấu sẵn các đường đi bắt buộc, ép LLM suy luận nhiều bước (Multi-hop) chính xác.
