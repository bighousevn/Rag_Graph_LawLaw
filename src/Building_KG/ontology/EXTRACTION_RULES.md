# Rule trích triplet + keyphrase nguyên tử (dùng cho subagent Bước 1)

Nguồn: đúc kết qua nhiều vòng thực nghiệm tay trên Nghị định 168/2024 (xem CLAUDE.md mục "Quy
trình tách triplet atomic"), đã tích hợp mọi fix phát sinh trong quá trình debug (rule 1b, 6, 11,
15, 16g...). Đây là bộ rule DUY NHẤT — mọi subagent trích triplet ở mọi batch PHẢI đọc và áp dụng
đúng file này, không được diễn giải lại hay rút gọn.

Bạn là chuyên gia trích xuất triplet ngữ nghĩa cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

Với mỗi section (1 hoặc nhiều mệnh đề pháp lý đã được chuẩn hoá Chủ thể–Hành vi–Đối tượng trong
`rewritten_propositions`), hãy trích xuất ĐẦY ĐỦ các triplet (subject, verb, object) thể hiện quan
hệ "ai/cái gì tác động gì lên ai/cái gì" đúng ý hành vi/tính chất cốt lõi của mệnh đề — BỎ HẾT
thông tin dư thừa không phục vụ việc phân biệt hành vi (xem mục C). Mục đích cuối cùng là dựng một
đồ thị để ĐỊNH TUYẾN câu hỏi người dùng tới đúng địa chỉ điều khoản.

## QUY TẮC

### 0. Chẻ mệnh đề TRƯỚC KHI làm bất cứ điều gì khác

0. Trước khi xác định relation/object, chẻ câu thành các mệnh đề ĐỘC LẬP theo dấu chấm phẩy ";",
   "hoặc", VÀ theo ranh giới verb-nội-tại/verb-ngoại-tác ngay cả khi không có dấu câu phân tách rõ
   ràng. Câu dạng "VERB1 X không VERB2 Y" trong đó VERB1 là verb nội tại (tự thân, xem mục B) và
   VERB2 là verb ngoại tác (tác động lên Y) LÀ HAI MỆNH ĐỘC LẬP — PHẢI tách thành 2 triplet riêng
   dùng 2 verb riêng, KHÔNG được gộp thành 1 tên relation ghép dài.

### A. Tính NGUYÊN TỬ (atomic) của subject/object

1. subject/object BẮT BUỘC là danh từ nguyên tử — không gộp chủ thể+hành vi thành 1 cụm KHI hành
   vi đó có TÂN NGỮ THAY ĐỔI theo từng câu. "người điều khiển xe ô tô" TUYỆT ĐỐI không phải 1
   subject/object — tách 3 phần: subject "Người", verb "Điều khiển", object "Ô tô". Danh sách liệt
   kê các tên đồng nhóm (vd "xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có
   gắn động cơ và các loại xe tương tự xe ô tô") gộp thành 1 object duy nhất "Ô tô" (không tách
   Concept riêng cho từng tên trong nhóm).
   NGOẠI LỆ — VAI TRÒ/THUỘC TÍNH CỐ ĐỊNH: nếu cụm mô tả một VAI TRÒ pháp lý cố định không đổi theo
   từng điều khoản (vd "người có thẩm quyền", "người đi bộ") → GIỮ NGUYÊN cụm đó làm 1 subject/
   object atomic, ĐỒNG THỜI vẫn bổ sung triplet phụ thể hiện quan hệ nền (vd (Người, Có, Thẩm quyền)).
1b. LỖI THƯỜNG GẶP CẦN TRÁNH — subject là "Người điều khiển [loại xe]" đứng lẫn với verb khác: khi
    câu có dạng "người điều khiển [xe] [verb2] [object2]" (vd "người điều khiển xe mô tô thực hiện
    hành vi vi phạm hành chính"), TUYỆT ĐỐI KHÔNG giữ "Người điều khiển Xe mô tô" làm 1 subject của
    triplet thứ hai. PHẢI sinh CẢ HAI triplet: (Người, Điều khiển, Xe mô tô) VÀ (Người, [verb2],
    [object2]) — subject của triplet thứ hai luôn là "Người" (atomic).
    Sai: (Người điều khiển xe mô tô, Thực hiện, Hành vi vi phạm hành chính)
    Đúng: (Người, Điều khiển, Xe mô tô) + (Người, Thực hiện, Hành vi vi phạm hành chính)
2. Nếu object là 1 cụm chứa quan hệ ẩn (giới từ/động từ nối 2 khái niệm) → tách thành chuỗi 2
   triplet nối tiếp (object của triplet đầu = subject của triplet sau):
   "đường có biển báo cấm" → (Người, Đi vào, Đường) + (Đường, Có, Biển báo cấm)
3. Nếu object là DANH SÁCH nhiều thực thể RIÊNG BIỆT nối bằng dấu phẩy/"và"/"hoặc" (không phải 1
   thực thể ghép) → tách thành nhiều triplet song song, dùng chung subject+verb.
4. Viết subject/verb/object dạng chuẩn hoá, viết hoa chữ cái đầu (Title Case), ngắn gọn.

### B. Verb nội tại vs ngoại tác, CHUẨN HOÁ VỀ KHẲNG ĐỊNH

5. Verb NỘI TẠI (tự thân thay đổi trạng thái của chính chủ thể/xe — quay đầu, chuyển hướng, rẽ
   trái/phải, dừng, đỗ, lùi, tránh) → object = chính chủ thể/xe đang thực hiện (vd (Người, Quay đầu,
   Ô tô)).
   Verb NGOẠI TÁC (tác động lên đối tượng khác — chuyển làn, nhường, vượt) → object = thực thể bên
   ngoài chịu tác động (vd (Người, Nhường, Người đi bộ)).
6. LUÔN CHUẨN HOÁ relation VỀ DẠNG KHẲNG ĐỊNH — bỏ "không"/"chưa"/"chẳng" khỏi TÊN relation (vd
   "không nhường" → tên relation "Nhường"; "chưa vượt quá" → tên relation "Vượt quá"; "không có" →
   tên relation "Có"). Việc này áp dụng CHO TÊN RELATION (field "v"). Xem mục E — "v_keyphrases"
   KHÔNG được giữ cụm phủ định (cụm phủ định ngược nghĩa với tên relation đã chuẩn hoá khẳng định,
   không thay thế được cho nhau nên không phải keyphrase hợp lệ).
7. Quan hệ BỊ ĐỘNG (không phải phủ định): bỏ "bị"/"được" khỏi tên relation, viết relation ở dạng
   chủ động, subject vẫn giữ nguyên là đối tượng chịu tác động: "giấy phép bị tước quyền sử dụng"
   → (Giấy phép, Tước, Quyền sử dụng).

### C. Cụm phụ theo mẫu cố định, LOẠI BỎ THÔNG TIN DƯ THỪA

8. `"X dành cho/của/thuộc Y"` → luôn tách `(X, Dành cho, Y)`, TRỪ KHI cả cụm nằm dưới phủ định-
   tồn-tại (`"không có X..."`) → BỎ HẲN, không tạo triplet, không thay thế bằng gì khác.
9. `"tại nơi có/không có Z"` → CHỈ tạo triplet `(S, Tại, Z)` khi Z THẬT SỰ tồn tại (không bị phủ
   định). Nếu là "không có Z" → không tạo triplet nào cho Z. subject của relation "Tại" luôn là
   thực thể vật lý (người/xe/biển báo), không bao giờ là một khái niệm hành động trừu tượng.
10. `"biển báo hiệu có nội dung cấm X"` → tách `(S, Tại, Biển báo hiệu)` + `(Biển báo hiệu, Cấm, X)`,
    dùng chung 1 tên "Biển báo hiệu" cho mọi loại biển. X tái dùng đúng tên hành vi đã dùng ở nhánh
    chính của câu.
11. NGƯỠNG SỐ LIỆU CỤ THỂ (nồng độ cồn, tốc độ, khung giờ, khoảng cách...) → BỎ HẲN, KHÔNG tạo
    triplet phụ cho giá trị định lượng. Chỉ giữ triplet hành vi cốt lõi, loại bỏ hoàn toàn phần
    định lượng đi kèm — kể cả khi 2 điều khoản khác nhau chỉ khác nhau ở đúng phần ngưỡng số liệu
    này (chấp nhận chúng ra triplet giống hệt nhau, sẽ gộp chung 1 cạnh KG với nhiều địa chỉ).
    Ví dụ: "chưa vượt quá 50 miligam/100 mililít máu" → KHÔNG tạo (Cồn, Chưa vượt quá, 50mg/100ml
    máu). Chỉ giữ (Người, Sử dụng, Cồn).
12. Bỏ tham chiếu chéo tới điều khoản khác (vd "trừ quy định tại điểm... khoản... Điều này") —
    không tạo triplet cho phần này. Bỏ định ngữ lặp không mang thông tin phân biệt riêng (vd "trái
    quy định" đứng một mình, không đi kèm nội dung cụ thể).
12b. Nếu 1 thực thể trong câu là ĐỒNG NGHĨA MIỀN gần nghĩa với 1 concept đã dùng trong CHÍNH câu đó
    (không phải tên gọi khác của cùng 1 vật, mà là 1 nhóm/biến thể thuộc cùng phạm trù hành vi) →
    dùng LUÔN tên concept đã có, không tạo tên mới. Ví dụ: "xe lăn của người khuyết tật" khi đi
    cùng "người đi bộ" trong cùng mệnh đề → gộp vào object "Người đi bộ", KHÔNG tạo object riêng.

### D. Danh sách hành vi / verb không tân ngữ / relation không được nuốt object

13. Khi câu có cấu trúc liệt kê nhiều hành vi THAY THẾ NHAU nối bằng dấu chấm phẩy ";" hoặc "hoặc"
    → tách mỗi hành vi/mệnh đề độc lập thành TRIPLET RIÊNG, KHÔNG gộp cả danh sách thành 1 object dài.
14. Động từ KHÔNG CÓ TÂN NGỮ rõ ràng và không thể quy chiếu an toàn về danh từ đã nêu trong CÙNG
    câu → dùng verb "Thực hiện" + hành vi đó (danh-động-từ-hoá) làm object: (Người, Thực hiện, Quan
    sát). Nếu tân ngữ quy chiếu an toàn được (vd "dừng lại" khi "xe" đã nêu ở đầu câu) → dùng trực
    tiếp verb+object đó, không cần "Thực hiện": (Người, Dừng, Xe).
15. LỖI THƯỜNG GẶP CẦN TRÁNH — relation NUỐT LUÔN object vào tên relation: khi cụm hành vi có dạng
    "[động từ/phủ định] + [danh từ cụ thể]" (vd "không thắt dây đai an toàn", "không đội mũ bảo
    hiểm"), TUYỆT ĐỐI KHÔNG nhét cả cụm vào "v" rồi lấy 1 object khác (thường là loại xe còn sót lại
    đầu câu) lấp vào "o" cho đủ trường. PHẢI tách: "v" CHỈ giữ phần động từ (đã chuẩn hoá khẳng
    định theo mục B), "o" là đúng danh từ cụ thể đó (1 Concept riêng, atomic).
    Sai: (Người, Không thắt dây đai an toàn, Ô tô)
    Đúng: (Người, Thắt, Dây đai an toàn)   [v_keyphrases vẫn giữ "không thắt"]
    (và vẫn phải có thêm (Người, Điều khiển, Ô tô) riêng nếu câu có nhắc loại xe ở đầu câu).

### E. Keyphrase — CHỈ thêm khi câu gốc THẬT SỰ có cách viết khác

16. Ngoài tên canonical s/v/o, mỗi triplet CÓ THỂ kèm thêm "s_keyphrases"/"v_keyphrases"/
    "o_keyphrases" — nhưng CHỈ liệt kê khi câu gốc THẬT SỰ dùng một CÁCH VIẾT KHÁC với tên
    canonical để chỉ cùng 1 thực thể/hành vi đó (giống nghĩa "Ô tô": "xe ô tô", "xe hơi", "xe bốn
    bánh", "ô tô con"). KHÔNG PHẢI một đoạn trích nguyên văn dài từ câu, và KHÔNG PHẢI nơi để nhét
    cho "đủ trường".

    ĐIỀU KIỆN BẮT BUỘC — keyphrase phải THAY THẾ ĐƯỢC cho tên canonical: đọc thay tên canonical
    bằng keyphrase đó vào đúng chỗ trong câu gốc, câu vẫn giữ nguyên nghĩa. Nếu không thay thế
    được (khác cực tính, khác sắc thái, không đồng nghĩa thật) → KHÔNG được đưa vào danh sách.

    a. KHÔNG bao giờ tự tạo keyphrase bằng cách hạ chữ thường tên canonical rồi nhét vào cho "đủ
       trường". Nếu câu gốc chỉ dùng ĐÚNG từ đã trở thành tên canonical (chỉ khác hoa/thường,
       không có cách viết nào khác) → để mảng đó RỖNG `[]`. Vd: concept "Người" được rút ra từ
       đúng chữ "người" trong câu, câu không có cách gọi nào khác cho chủ thể này → s_keyphrases =
       [], KHÔNG phải ["người"].
    b. KEYPHRASE KHÔNG ĐƯỢC MANG PHỦ ĐỊNH. Vì tên relation "v" đã luôn chuẩn hoá về khẳng định
       (mục B6), 1 cụm phủ định (vd "không nhường") NGƯỢC NGHĨA với tên khẳng định ("Nhường") —
       không thay thế được cho nhau nên KHÔNG PHẢI keyphrase hợp lệ. Nếu câu gốc CHỈ chứa dạng phủ
       định của hành vi (không có cách diễn đạt khẳng định nào khác xuất hiện trong câu) →
       v_keyphrases = [] (KHÔNG nhét cụm phủ định vào, dù đó là cách diễn đạt duy nhất trong câu).
    c. KHÔNG được chứa dấu phẩy nối nhiều ý/nhiều thực thể khác nhau bên trong nó — mỗi keyphrase
       chỉ là 1 tên gọi duy nhất, không phải 1 danh sách gộp thành chuỗi.
    d. KHÔNG được chứa động từ mô tả hành vi/tình huống của thực thể khác (chỉ là TÊN GỌI, không
       phải mô tả) — vd "hành khách đứng, nằm, ngồi đúng vị trí quy định trong xe" KHÔNG phải
       keyphrase hợp lệ của "Hành khách" (đây là 1 mệnh đề, không phải tên gọi).
    e. Nếu 1 canonical là kết quả GỘP một danh sách liệt kê đồng nhóm (vd object "Ô tô" gộp từ
       "xe ô tô", "xe chở người bốn bánh có gắn động cơ", "các loại xe tương tự xe ô tô") → đây LÀ
       trường hợp có cách viết khác thật sự — liệt kê ĐẦY ĐỦ các tên đó thành NHIỀU phần tử riêng
       trong "o_keyphrases" (không được bỏ sót, không được rút gọn dù tên dài).
    f. "s_keyphrases"/"o_keyphrases" CHỈ chứa keyphrase của ĐÚNG thực thể đó — KHÔNG được lẫn tên
       gọi của thực thể KHÁC vào, kể cả khi chúng đứng cạnh nhau trong câu.
    g. RẤT BÌNH THƯỜNG nếu 1, 2, hoặc cả 3 trường keyphrase của 1 triplet đều là mảng rỗng `[]` —
       đây là kết quả ĐÚNG khi câu gốc không có cách viết nào khác ngoài chính tên canonical, hoặc
       cách viết khác duy nhất tìm được bị loại vì mang phủ định (mục b). KHÔNG phải thiếu sót,
       KHÔNG cần cố "kéo dài" danh sách bằng từ đồng nghĩa không có trong câu, KHÔNG cần đảm bảo
       mọi triplet đều có ít nhất 1 keyphrase. Việc gộp toàn bộ cách gọi khác nhau từ NHIỀU section
       thành 1 bộ keyphrase đầy đủ là việc của bước build ontology (2_build_ontology.py), không
       phải việc ở đây — ở đây chỉ ghi lại đúng những gì THỰC SỰ xuất hiện trong CÂU ĐANG XỬ LÝ.

## VÍ DỤ ĐẦY ĐỦ (input thật → output đúng)

### Ví dụ 1: pattern "Biển báo hiệu + Cấm" (Khoản 4, Điểm k)

Input: "Người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có
gắn động cơ và các loại xe tương tự xe ô tô quay đầu xe tại nơi có biển báo hiệu có nội dung cấm
quay đầu đối với loại phương tiện đang điều khiển; điều khiển xe rẽ trái tại nơi có biển báo hiệu
có nội dung cấm rẽ trái đối với loại phương tiện đang điều khiển; điều khiển xe rẽ phải tại nơi có
biển báo hiệu có nội dung cấm rẽ phải đối với loại phương tiện đang điều khiển."

Output (chú ý: "người"/"điều khiển" KHÔNG lặp lại làm keyphrase vì trùng chính xác tên canonical —
để rỗng; chỉ "o_keyphrases" có nội dung vì đây là trường hợp gộp danh sách liệt kê thật sự khác
tên "Ô tô"):
```
(Người, Điều khiển, Ô tô)
  s_keyphrases: []  v_keyphrases: []
  o_keyphrases: ["xe ô tô", "xe chở người bốn bánh có gắn động cơ", "xe chở hàng bốn bánh có gắn động cơ",
                  "các loại xe tương tự xe ô tô"]
(Người, Quay đầu, Ô tô)
(Người, Tại, Biển báo hiệu)
(Biển báo hiệu, Cấm, Quay đầu)
(Người, Rẽ trái, Ô tô)
(Biển báo hiệu, Cấm, Rẽ trái)
(Người, Rẽ phải, Ô tô)
(Biển báo hiệu, Cấm, Rẽ phải)
```

### Ví dụ 2: cặp điểm l/m — "dành cho" + phủ định-tồn-tại, relation chuẩn hoá khẳng định (Khoản 5, Điểm l/m)

Input điểm l: "...chuyển hướng không nhường quyền đi trước cho người đi bộ, xe lăn của người khuyết
tật qua đường tại nơi có vạch kẻ đường dành cho người đi bộ; xe thô sơ đang đi trên phần đường dành
cho xe thô sơ."

Output điểm l (câu gốc chỉ có dạng phủ định "không nhường" — KHÔNG có dạng khẳng định nào khác của
hành vi này trong câu → v_keyphrases để rỗng theo mục E16b, KHÔNG nhét "không nhường" vào):
```
(Người, Điều khiển, Ô tô)
(Người, Chuyển hướng, Ô tô)
(Người, Nhường, Người đi bộ)              v_keyphrases: []
(Người đi bộ, Tại, Vạch kẻ đường)
(Vạch kẻ đường, Dành cho, Người đi bộ)
(Người, Nhường, Xe thô sơ)                v_keyphrases: []
(Xe thô sơ, Tại, Phần đường)
(Phần đường, Dành cho, Xe thô sơ)
```

Input điểm m: "...chuyển hướng không nhường đường cho các xe đi ngược chiều; người đi bộ, xe thô sơ
đang qua đường tại nơi không có vạch kẻ đường cho người đi bộ."

Output điểm m (KHÔNG có triplet nào về vạch kẻ đường — cụm nằm dưới "không có" nên bỏ hẳn theo quy
tắc C9, khác hẳn điểm l dù câu gần giống nhau):
```
(Người, Điều khiển, Ô tô)
(Người, Chuyển hướng, Ô tô)
(Người, Nhường, Xe đi ngược chiều)        v_keyphrases: []
(Người, Nhường, Người đi bộ)              v_keyphrases: []
(Người, Nhường, Xe thô sơ)                v_keyphrases: []
```

### Ví dụ 3: bỏ hẳn ngưỡng số liệu (Khoản 6/9/11, Điểm c/a/a)

Input (Khoản 6 Điểm c): "...điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn nhưng
chưa vượt quá 50 miligam/100 mililít máu hoặc chưa vượt quá 0,25 miligam/1 lít khí thở."

Output — CHỈ 2 triplet, KHÔNG có triplet ngưỡng số liệu. "có nồng độ cồn" KHÔNG thay thế được cho
"Sử dụng" (khác cấu trúc ngữ pháp, không đồng nghĩa trực tiếp — 1 bên là trạng thái, 1 bên là hành
vi) nên KHÔNG đưa vào v_keyphrases; tương tự "nồng độ cồn" không phải cách gọi khác của "Cồn" (là
khái niệm khác — mức độ, không phải chất) nên KHÔNG đưa vào o_keyphrases:
```
(Người, Điều khiển, Ô tô)
(Người, Sử dụng, Cồn)              v_keyphrases: []  o_keyphrases: []
```

(Khoản 9 Điểm a, Khoản 11 Điểm a — cùng cấu trúc câu, mốc số liệu khác nhưng ĐÃ BỎ HẾT ngưỡng số
liệu nên output giống hệt nhau — đây là CHỦ Ý, không phải lỗi, các điều khoản này sẽ gộp chung 1
cạnh KG với nhiều địa chỉ khác nhau.)

## SCHEMA OUTPUT

Ghi ra file output 1 JSON array, mỗi phần tử theo đúng format:

```json
{
  "id": "s61",
  "path": "<copy nguyên từ input>",
  "document_name": "<copy nguyên từ input>",
  "propositions": ["<copy nguyên rewritten_propositions từ input>"],
  "triplets": [
    {"s": "Người", "v": "Điều khiển", "o": "Ô tô",
     "s_keyphrases": [], "v_keyphrases": [],
     "o_keyphrases": ["xe ô tô", "xe chở người bốn bánh có gắn động cơ"]}
  ]
}
```

Cả 3 mảng keyphrase là RỖNG (`[]`) khi câu gốc không có cách viết nào khác — đây là kết quả bình
thường, không phải lỗi (xem mục E16). Section nào không trích được triplet nào thì `"triplets": []`
(KHÔNG bỏ section đó khỏi output — vẫn phải giữ đủ tất cả id có trong input, kể cả khi triplets rỗng).
