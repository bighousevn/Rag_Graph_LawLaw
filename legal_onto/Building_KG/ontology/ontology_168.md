# Bảng Khái niệm (Conc) & Quan hệ (Rel) — Nghị định 168/2024/NĐ-CP

> Nguồn: **Nghị định 168/2024/NĐ-CP** ngày 26/12/2024 — *Quy định xử phạt vi phạm
> hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ; trừ
> điểm, phục hồi điểm giấy phép lái xe.*

## Quy ước cấu trúc

**Khái niệm (Concept):** `c = (Name, Keyphrases)`

| Thành phần | Kiểu | Ý nghĩa |
|---|---|---|
| Name | Text | Nhãn chuẩn của khái niệm |
| Keyphrases | Set | Các biến thể / từ đồng nghĩa bề mặt của khái niệm (lấy từ Nghị định + mở rộng theo kiến thức) |

**Quan hệ (Relation):** `r = (Name, ConcKeyS, ConcKeyO, Keyphrases)`

| Thành phần | Kiểu | Ý nghĩa |
|---|---|---|
| Name | Text | Vị từ chuẩn của quan hệ |
| ConcKeyS | Keyphrase | Loại khái niệm chủ thể (subject) |
| ConcKeyO | List | Loại khái niệm đối tượng (object) |
| Keyphrases | Set | Các cụm từ / từ đồng nghĩa kích hoạt quan hệ |

---

# PHẦN A — BẢNG KHÁI NIỆM (Conc)

## A.1. Nhóm Phương tiện giao thông

### Bảng C1: "Xe cơ giới"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe cơ giới |
| Keyphrases | Set | xe cơ giới, phương tiện giao thông cơ giới đường bộ, phương tiện giao thông cơ giới, phương tiện cơ giới, xe có động cơ, xe gắn động cơ, xe máy móc |

### Bảng C2: "Xe ô tô"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe ô tô |
| Keyphrases | Set | xe ô tô, ô tô, ô tô tải, ô tô chở người, ô tô chở hành khách, ô tô đầu kéo, xe ô tô và các loại xe tương tự, ôtô, xe hơi, ô tô con, xe bốn bánh, xe 4 bánh, auto |

### Bảng C3: "Xe mô tô"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe mô tô |
| Keyphrases | Set | xe mô tô, mô tô, xe mô tô hai bánh, xe mô tô ba bánh, xe mô tô và các loại xe tương tự xe mô tô, xe máy, mô-tô, môtô, xe phân khối lớn, xe hai bánh, xe pkl |

### Bảng C4: "Xe gắn máy"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe gắn máy |
| Keyphrases | Set | xe gắn máy, các loại xe tương tự xe gắn máy, xe gắn máy dưới 50cc, xe dưới 50 phân khối, xe máy điện công suất nhỏ, xe số dưới 50cc |

### Bảng C5: "Xe máy chuyên dùng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe máy chuyên dùng |
| Keyphrases | Set | xe máy chuyên dùng, máy thi công, xe công trình, máy nông nghiệp, máy lâm nghiệp, xe chuyên dùng, máy ủi, máy xúc, xe lu |

### Bảng C6: "Máy kéo"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Máy kéo |
| Keyphrases | Set | máy kéo, đầu kéo, ô tô đầu kéo, xe đầu kéo, xe kéo, xe container, tractor |

### Bảng C7: "Rơ moóc, sơ mi rơ moóc"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Rơ moóc, sơ mi rơ moóc |
| Keyphrases | Set | rơ moóc, sơ mi rơ moóc, được kéo theo, moóc, mooc, rờ moóc, sơ-mi rơ-moóc, thùng moóc, trailer |

### Bảng C8: "Xe chở người bốn bánh có gắn động cơ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe chở người bốn bánh có gắn động cơ |
| Keyphrases | Set | xe chở người bốn bánh có gắn động cơ, xe bốn bánh chở người, xe điện chở khách, xe điện du lịch, xe điện 4 bánh |

### Bảng C9: "Xe chở hàng bốn bánh có gắn động cơ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe chở hàng bốn bánh có gắn động cơ |
| Keyphrases | Set | xe chở hàng bốn bánh có gắn động cơ, xe bốn bánh chở hàng, xe ba gác máy, xe tải nhỏ 4 bánh |

### Bảng C10: "Xe ô tô tải"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe ô tô tải |
| Keyphrases | Set | xe ô tô tải, ô tô tải, xe tải, xe ben, xe thùng, xe chở hàng, xe container, xe tải nặng, xe tải nhẹ |

### Bảng C11: "Xe ô tô chở hành khách"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe ô tô chở hành khách |
| Keyphrases | Set | xe ô tô chở hành khách, ô tô chở hành khách, ô tô chở người, xe ô tô chở người và các loại xe tương tự, xe khách, xe buýt, xe bus, xe đò, xe giường nằm, xe limousine, xe 16 chỗ, xe 45 chỗ |

### Bảng C12: "Xe ô tô kinh doanh vận tải"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe ô tô kinh doanh vận tải |
| Keyphrases | Set | xe ô tô kinh doanh vận tải, xe kinh doanh vận tải, xe ô tô kinh doanh vận tải chở trẻ em mầm non học sinh, xe hợp đồng, xe taxi, xe công nghệ, xe đưa đón học sinh, xe tuyến cố định, xe dịch vụ |

### Bảng C13: "Xe đạp, xe đạp máy"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe đạp, xe đạp máy |
| Keyphrases | Set | xe đạp, xe đạp máy, xe đạp điện, xe đạp thường, xe đạp trợ lực điện, xe đạp trẻ em |

### Bảng C14: "Xe thô sơ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe thô sơ |
| Keyphrases | Set | xe thô sơ, xe thô sơ khác, xe ba gác, xe xích lô, xe lôi, xe kéo tay, xe đẩy tay, xe súc vật kéo |

### Bảng C15: "Xe vật nuôi kéo"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe vật nuôi kéo |
| Keyphrases | Set | xe vật nuôi kéo, vật nuôi, súc vật, xe súc vật kéo, xe trâu kéo, xe bò kéo, xe ngựa kéo |

### Bảng C16: "Xe cứu hộ giao thông đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe cứu hộ giao thông đường bộ |
| Keyphrases | Set | xe cứu hộ giao thông đường bộ, xe cứu hộ, xe ô tô cứu hộ giao thông, xe kéo cứu hộ, xe cẩu kéo |

### Bảng C17: "Xe cứu thương"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe cứu thương |
| Keyphrases | Set | xe cứu thương, xe cấp cứu, xe y tế, xe 115, xe ambulance |

### Bảng C18: "Xe vệ sinh môi trường, xe ô tô chở phế thải"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe vệ sinh môi trường, xe ô tô chở phế thải |
| Keyphrases | Set | xe vệ sinh môi trường, xe ô tô chở phế thải, xe chở rác, xe ép rác, xe hút bể phốt, xe quét đường, xe môi trường |

### Bảng C19: "Xe quá khổ giới hạn, xe quá tải trọng, xe bánh xích"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe quá khổ giới hạn, xe quá tải trọng, xe bánh xích |
| Keyphrases | Set | xe quá khổ giới hạn, quá khổ giới hạn, xe quá tải trọng, quá tải trọng, quá tải, quá khổ, quá trọng tải, xe bánh xích, xe vượt tải, xe cơi nới thành thùng, xe chở quá tải, xe xích |

### Bảng C20: "Phương tiện gắn biển số nước ngoài"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phương tiện gắn biển số nước ngoài |
| Keyphrases | Set | phương tiện giao thông cơ giới đường bộ gắn biển số nước ngoài, phương tiện đăng ký hoạt động trong Khu kinh tế thương mại đặc biệt, Khu kinh tế cửa khẩu quốc tế, xe gắn biển số nước ngoài, xe nước ngoài, xe mang biển số nước ngoài, xe tạm nhập tái xuất |

## A.2. Nhóm Chủ thể (người tham gia giao thông và đối tượng áp dụng)

### Bảng C21: "Người điều khiển phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người điều khiển phương tiện |
| Keyphrases | Set | người điều khiển, người điều khiển phương tiện, người lái xe, người điều khiển xe, tài xế, lái xe, người cầm lái, bác tài, người lái |

### Bảng C22: "Người đi bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người đi bộ |
| Keyphrases | Set | người đi bộ, bộ hành, khách bộ hành, người đi đường |

### Bảng C23: "Người dẫn dắt vật nuôi"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người dẫn dắt vật nuôi |
| Keyphrases | Set | người điều khiển dẫn dắt vật nuôi, dẫn dắt vật nuôi, điều khiển xe vật nuôi kéo, người dắt súc vật, người chăn dắt gia súc, người dẫn súc vật |

### Bảng C24: "Hành khách"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành khách |
| Keyphrases | Set | hành khách, hành khách đi xe, khách đi xe, người đi xe, khách, hành khách trên xe |

### Bảng C25: "Chủ phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chủ phương tiện |
| Keyphrases | Set | chủ phương tiện, chủ xe, chủ sở hữu xe, người đứng tên xe, chủ sở hữu phương tiện |

### Bảng C26: "Cá nhân, tổ chức vi phạm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cá nhân, tổ chức vi phạm |
| Keyphrases | Set | cá nhân, tổ chức, cá nhân tổ chức, hộ kinh doanh, hộ gia đình, người vi phạm, đối tượng vi phạm, doanh nghiệp, đơn vị, người bị xử phạt |

### Bảng C27: "Người đua xe, tổ chức đua xe trái phép"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người đua xe, tổ chức đua xe trái phép |
| Keyphrases | Set | đua xe, đua xe trái phép, tổ chức đua xe, xúi giục, cổ vũ đua xe trái phép, quái xế, người tổ chức đua xe, người cổ vũ đua xe |

## A.3. Nhóm Hành vi & chế tài xử phạt

### Bảng C28: "Vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vi phạm hành chính |
| Keyphrases | Set | vi phạm hành chính, hành vi vi phạm hành chính, hành vi vi phạm, vi phạm, lỗi vi phạm, vi phạm giao thông, hành vi trái pháp luật, VPHC |

### Bảng C29: "Xử phạt vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xử phạt vi phạm hành chính |
| Keyphrases | Set | xử phạt, xử phạt vi phạm hành chính, phạt, xử lý vi phạm, chế tài xử phạt, xử lý hành chính |

### Bảng C30: "Cảnh cáo"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cảnh cáo |
| Keyphrases | Set | cảnh cáo, phạt cảnh cáo, nhắc nhở |

### Bảng C31: "Phạt tiền"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phạt tiền |
| Keyphrases | Set | phạt tiền, mức phạt tiền, phạt từ ... đến, nộp phạt, tiền phạt, đóng phạt, phạt hành chính |

### Bảng C32: "Tịch thu phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tịch thu phương tiện |
| Keyphrases | Set | tịch thu phương tiện, tịch thu tang vật, tịch thu, sung công, thu giữ phương tiện, tịch thu xe |

### Bảng C33: "Tước quyền sử dụng giấy phép, chứng chỉ hành nghề có thời hạn"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tước quyền sử dụng giấy phép, chứng chỉ hành nghề có thời hạn |
| Keyphrases | Set | tước quyền sử dụng, tước quyền sử dụng giấy phép, tước quyền sử dụng giấy phép chứng chỉ hành nghề, tước, tước bằng lái, treo bằng, thu bằng, tước giấy phép lái xe |

### Bảng C34: "Đình chỉ hoạt động có thời hạn"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đình chỉ hoạt động có thời hạn |
| Keyphrases | Set | đình chỉ hoạt động có thời hạn, đình chỉ hoạt động, tạm đình chỉ, ngừng hoạt động, đình chỉ kinh doanh |

### Bảng C35: "Biện pháp khắc phục hậu quả"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biện pháp khắc phục hậu quả |
| Keyphrases | Set | biện pháp khắc phục hậu quả, buộc khôi phục, buộc nộp lại, buộc tháo dỡ, khắc phục hậu quả, khắc phục, biện pháp bổ sung, buộc khắc phục |

### Bảng C36: "Điểm giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Điểm giấy phép lái xe |
| Keyphrases | Set | điểm giấy phép lái xe, điểm, điểm bằng lái, điểm GPLX, 12 điểm, điểm trên bằng lái |

### Bảng C37: "Trừ điểm giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Trừ điểm giấy phép lái xe |
| Keyphrases | Set | trừ điểm giấy phép lái xe, trừ điểm, mức trừ điểm, bị trừ điểm, trừ điểm bằng lái, trừ điểm GPLX |

### Bảng C38: "Phục hồi điểm giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phục hồi điểm giấy phép lái xe |
| Keyphrases | Set | phục hồi điểm, phục hồi điểm giấy phép lái xe, khôi phục điểm, hồi điểm |

### Bảng C39: "Thời hiệu xử phạt vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thời hiệu xử phạt vi phạm hành chính |
| Keyphrases | Set | thời hiệu xử phạt, thời hiệu, thời hạn xử phạt, thời hiệu xử lý |

### Bảng C40: "Thẩm quyền xử phạt"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thẩm quyền xử phạt |
| Keyphrases | Set | thẩm quyền xử phạt, thẩm quyền, phân định thẩm quyền, quyền xử phạt, thẩm quyền xử lý |

### Bảng C41: "Biên bản vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biên bản vi phạm hành chính |
| Keyphrases | Set | biên bản vi phạm hành chính, lập biên bản, biên bản, lập biên bản vi phạm, biên bản phạt |

### Bảng C42: "Tạm giữ phương tiện, giấy tờ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tạm giữ phương tiện, giấy tờ |
| Keyphrases | Set | tạm giữ phương tiện, tạm giữ, giấy tờ có liên quan, giữ xe, tạm giữ xe, giữ giấy tờ, tạm giữ giấy tờ |

## A.4. Nhóm Giấy tờ, giấy phép

### Bảng C43: "Giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy phép lái xe |
| Keyphrases | Set | giấy phép lái xe, giấy phép lái, giấy phép lái xe quốc tế, giấy phép lái xe quốc gia, GPLX, bằng lái, bằng lái xe, bằng, giấy phép |

### Bảng C44: "Chứng chỉ hành nghề"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chứng chỉ hành nghề |
| Keyphrases | Set | chứng chỉ hành nghề, chứng chỉ đăng kiểm viên, chứng chỉ nghề, chứng chỉ chuyên môn |

### Bảng C45: "Giấy chứng nhận đăng ký xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy chứng nhận đăng ký xe |
| Keyphrases | Set | chứng nhận đăng ký xe, giấy chứng nhận đăng ký xe, chứng nhận đăng ký xe tạm thời, chứng nhận nguồn gốc xe, cà vẹt, cà vẹt xe, cavet, đăng ký xe, giấy đăng ký xe |

### Bảng C46: "Biển số xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biển số xe |
| Keyphrases | Set | biển số xe, biển số, số biển số, biển kiểm soát, BKS, bảng số xe, biển kiểm soát xe |

### Bảng C47: "Giấy chứng nhận, tem kiểm định an toàn kỹ thuật và bảo vệ môi trường"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy chứng nhận, tem kiểm định an toàn kỹ thuật và bảo vệ môi trường |
| Keyphrases | Set | giấy chứng nhận kiểm định, chứng nhận kiểm định an toàn kỹ thuật và bảo vệ môi trường, tem kiểm định, giấy chứng nhận hoặc tem kiểm định, giấy đăng kiểm, tem đăng kiểm, sổ kiểm định, giấy chứng nhận an toàn kỹ thuật và bảo vệ môi trường |

### Bảng C48: "Phù hiệu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phù hiệu |
| Keyphrases | Set | phù hiệu, phù hiệu xe, tem phù hiệu, phù hiệu xe tải, phù hiệu xe hợp đồng, biển hiệu |

### Bảng C49: "Giấy phép đào tạo, sát hạch lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy phép đào tạo, sát hạch lái xe |
| Keyphrases | Set | giấy phép đào tạo lái xe, giấy phép sát hạch, giấy phép cơ sở đào tạo lái xe, giấy phép trung tâm sát hạch |

### Bảng C50: "Giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới |
| Keyphrases | Set | giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới, giấy chứng nhận đăng kiểm, giấy phép trung tâm đăng kiểm |

## A.5. Nhóm Hạ tầng & báo hiệu đường bộ

### Bảng C51: "Đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đường bộ |
| Keyphrases | Set | đường bộ, giao thông đường bộ, lĩnh vực giao thông đường bộ, đường giao thông, đường sá, mạng lưới đường bộ, tuyến đường |

### Bảng C52: "Đường cao tốc"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đường cao tốc |
| Keyphrases | Set | đường cao tốc, cao tốc, tuyến cao tốc, xa lộ |

### Bảng C53: "Lòng đường, vỉa hè"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Lòng đường, vỉa hè |
| Keyphrases | Set | lòng đường, vỉa hè, hè phố, lề đường, mặt đường, vệ đường |

### Bảng C54: "Biển báo hiệu đường bộ, đèn tín hiệu giao thông"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biển báo hiệu đường bộ, đèn tín hiệu giao thông |
| Keyphrases | Set | biển báo hiệu, đèn tín hiệu giao thông, vạch kẻ đường, dải phân cách, biển báo, đèn giao thông, đèn đỏ, cọc tiêu, biển chỉ dẫn, hệ thống báo hiệu đường bộ, vạch sơn |

### Bảng C55: "Quy tắc giao thông đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Quy tắc giao thông đường bộ |
| Keyphrases | Set | quy tắc giao thông đường bộ, quy tắc giao thông, luật giao thông, quy định giao thông, luật lệ giao thông, quy tắc đường bộ |

## A.6. Nhóm Cơ quan, người có thẩm quyền

### Bảng C56: "Chủ tịch Ủy ban nhân dân các cấp"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chủ tịch Ủy ban nhân dân các cấp |
| Keyphrases | Set | Chủ tịch Ủy ban nhân dân, Chủ tịch Ủy ban nhân dân các cấp, Chủ tịch UBND, Chủ tịch tỉnh, Chủ tịch huyện, Chủ tịch xã, UBND |

### Bảng C57: "Công an nhân dân"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Công an nhân dân |
| Keyphrases | Set | Công an nhân dân, Cảnh sát giao thông, CSGT, công an, cảnh sát, công an giao thông, CAND, cảnh sát cơ động |

### Bảng C58: "Thanh tra chuyên ngành"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thanh tra chuyên ngành |
| Keyphrases | Set | Thanh tra chuyên ngành, Thanh tra, thanh tra giao thông, TTGT, thanh tra đường bộ |

### Bảng C59: "Người có thẩm quyền lập biên bản"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người có thẩm quyền lập biên bản |
| Keyphrases | Set | người có thẩm quyền lập biên bản, thẩm quyền lập biên bản, người lập biên bản, người thi hành công vụ |

## A.7. Nhóm Hoạt động / lĩnh vực

### Bảng C60: "Kinh doanh vận tải đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Kinh doanh vận tải đường bộ |
| Keyphrases | Set | kinh doanh vận tải, vận tải đường bộ, dịch vụ hỗ trợ vận tải đường bộ, kinh doanh vận tải đường bộ, doanh nghiệp vận tải, đơn vị kinh doanh vận tải, hoạt động vận tải |

### Bảng C61: "Vận chuyển hàng hóa nguy hiểm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vận chuyển hàng hóa nguy hiểm |
| Keyphrases | Set | hàng hóa nguy hiểm, vận chuyển hàng hóa nguy hiểm, hàng nguy hiểm, chở hàng nguy hiểm, vận chuyển chất nguy hiểm, hóa chất nguy hiểm |

### Bảng C62: "Vận chuyển hàng siêu trường, siêu trọng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vận chuyển hàng siêu trường, siêu trọng |
| Keyphrases | Set | hàng siêu trường, hàng siêu trọng, vận chuyển hàng siêu trường siêu trọng, giấy phép lưu hành, hàng quá khổ quá tải, hàng cồng kềnh, chở hàng siêu trường siêu trọng |

### Bảng C63: "Đào tạo, sát hạch lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đào tạo, sát hạch lái xe |
| Keyphrases | Set | đào tạo lái xe, sát hạch, sát hạch lái xe, cơ sở đào tạo lái xe, trung tâm sát hạch, trường dạy lái xe, học lái xe, thi bằng lái, thi sát hạch |

### Bảng C64: "Hoạt động đăng kiểm xe cơ giới"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hoạt động đăng kiểm xe cơ giới |
| Keyphrases | Set | đăng kiểm, hoạt động đăng kiểm, kiểm định xe cơ giới, hoạt động đăng kiểm xe cơ giới, trung tâm đăng kiểm, kiểm định xe, đăng kiểm xe, kiểm tra kỹ thuật xe |

### Bảng C65: "Đua xe trái phép"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đua xe trái phép |
| Keyphrases | Set | đua xe, đua xe trái phép, tổ chức đua xe, lạng lách đánh võng, đua tốc độ, độ xe đua, bốc đầu |

### Bảng C66: "Sản xuất, lắp ráp trái phép phương tiện; mua, bán biển số xe trái phép"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Sản xuất, lắp ráp trái phép phương tiện; mua, bán biển số xe trái phép |
| Keyphrases | Set | sản xuất lắp ráp trái phép phương tiện, mua bán biển số xe trái phép, sản xuất mua bán biển số xe trái phép, sản xuất xe lậu, lắp ráp xe trái phép, làm giả biển số, mua bán biển số giả |

---

# PHẦN B — BẢNG QUAN HỆ (Rel)

### Bảng R1: "Điều khiển"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Điều khiển |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện |
| ConcKeyO | List | Xe ô tô, xe mô tô, xe gắn máy, xe máy chuyên dùng, xe đạp, xe thô sơ, máy kéo |
| Keyphrases | Set | điều khiển, lái, điều khiển xe, vận hành, cầm lái, lái xe, chạy xe, dắt xe |

### Bảng R2: "Sử dụng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Sử dụng |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện |
| ConcKeyO | List | Điện thoại, tai nghe, thiết bị, ô (dù), rượu bia, lòng đường, vỉa hè |
| Keyphrases | Set | sử dụng, dùng, làm, thực hiện, xài, vận dụng, dùng đến |

### Bảng R3: "Vi phạm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vi phạm |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm |
| ConcKeyO | List | Quy tắc giao thông đường bộ, quy định về điều kiện của phương tiện, quy định về điều kiện của người điều khiển, quy định về bảo vệ môi trường |
| Keyphrases | Set | vi phạm, không chấp hành, không tuân thủ, không nhường, không có, không mang theo, trái quy định, làm trái, phạm lỗi, không thực hiện |

### Bảng R4: "Thực hiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thực hiện |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm |
| ConcKeyO | List | Vi phạm hành chính, hành vi vi phạm, đua xe trái phép, vận chuyển hàng hóa nguy hiểm, sản xuất lắp ráp trái phép phương tiện |
| Keyphrases | Set | thực hiện, có hành vi, thực hiện một trong, tiến hành, gây ra hành vi |

### Bảng R5: "Xử phạt"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xử phạt |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt (Chủ tịch UBND / Công an nhân dân / Thanh tra chuyên ngành) |
| ConcKeyO | List | Cá nhân, tổ chức vi phạm, người điều khiển phương tiện, chủ phương tiện, hành khách, người đi bộ |
| Keyphrases | Set | xử phạt, phạt, bị xử phạt, xử lý vi phạm, ra quyết định xử phạt |

### Bảng R6: "Áp dụng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Áp dụng |
| ConcKeyS | Keyphrase | Hình thức xử phạt / biện pháp khắc phục hậu quả |
| ConcKeyO | List | Cá nhân, tổ chức vi phạm, người điều khiển phương tiện, chủ phương tiện |
| Keyphrases | Set | áp dụng, bị áp dụng, áp dụng đối với, thi hành, áp đặt |

### Bảng R7: "Phạt tiền"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phạt tiền |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt |
| ConcKeyO | List | Cá nhân, tổ chức vi phạm, người điều khiển phương tiện, chủ phương tiện |
| Keyphrases | Set | phạt tiền, phạt từ ... đến, mức phạt, nộp phạt, đóng phạt, phạt hành chính |

### Bảng R8: "Tịch thu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tịch thu |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt |
| ConcKeyO | List | Xe ô tô, xe mô tô, xe gắn máy, phương tiện, tang vật, biển số xe |
| Keyphrases | Set | tịch thu, tịch thu phương tiện, tịch thu tang vật, sung công, thu giữ |

### Bảng R9: "Tước quyền sử dụng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tước quyền sử dụng |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt |
| ConcKeyO | List | Giấy phép lái xe, chứng chỉ hành nghề, phù hiệu, giấy chứng nhận kiểm định, giấy phép đào tạo sát hạch lái xe |
| Keyphrases | Set | tước quyền sử dụng, tước, tước bằng, treo bằng, thu bằng |

### Bảng R10: "Trừ điểm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Trừ điểm |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt |
| ConcKeyO | List | Giấy phép lái xe, điểm giấy phép lái xe |
| Keyphrases | Set | trừ điểm, trừ điểm giấy phép lái xe, bị trừ điểm, trừ điểm bằng lái |

### Bảng R11: "Phục hồi điểm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phục hồi điểm |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt |
| ConcKeyO | List | Giấy phép lái xe, điểm giấy phép lái xe |
| Keyphrases | Set | phục hồi điểm, phục hồi điểm giấy phép lái xe, khôi phục điểm, hồi điểm |

### Bảng R12: "Tạm giữ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tạm giữ |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt |
| ConcKeyO | List | Xe ô tô, xe mô tô, phương tiện, giấy phép lái xe, giấy chứng nhận đăng ký xe, giấy tờ |
| Keyphrases | Set | tạm giữ, tạm giữ phương tiện, giữ xe, giữ giấy tờ |

### Bảng R13: "Là loại của" (phân loại / kế thừa)

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Là loại của |
| ConcKeyS | Keyphrase | Phương tiện giao thông (loại cụ thể) |
| ConcKeyO | List | Xe cơ giới, xe ô tô, xe thô sơ, phương tiện giao thông |
| Keyphrases | Set | và các loại xe tương tự, bao gồm, thuộc, là loại, là một dạng, trực thuộc, phân loại |

### Bảng R14: "Có"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Có |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện / phương tiện giao thông |
| ConcKeyO | List | Giấy phép lái xe, giấy chứng nhận đăng ký xe, biển số xe, giấy chứng nhận kiểm định, phù hiệu |
| Keyphrases | Set | có, mang theo, được cấp, gắn, không có, không mang theo, sở hữu, trang bị, đem theo |

### Bảng R15: "Chở / Vận chuyển"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chở / Vận chuyển |
| ConcKeyS | Keyphrase | Phương tiện giao thông |
| ConcKeyO | List | Hành khách, hàng hóa, hàng hóa nguy hiểm, hàng siêu trường siêu trọng, động vật sống, phế thải, trẻ em mầm non học sinh |
| Keyphrases | Set | chở, vận chuyển, chuyên chở, tải, đèo, mang, vận tải |

### Bảng R16: "Kéo theo"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Kéo theo |
| ConcKeyS | Keyphrase | Xe ô tô / máy kéo |
| ConcKeyO | List | Rơ moóc, sơ mi rơ moóc |
| Keyphrases | Set | kéo theo, được kéo theo, kéo, móc kéo, lai dắt |

### Bảng R17: "Có thẩm quyền xử phạt"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Có thẩm quyền xử phạt |
| ConcKeyS | Keyphrase | Chủ tịch Ủy ban nhân dân / Công an nhân dân / Thanh tra chuyên ngành |
| ConcKeyO | List | Vi phạm hành chính, cá nhân tổ chức vi phạm, biện pháp khắc phục hậu quả |
| Keyphrases | Set | có thẩm quyền, thẩm quyền xử phạt, có quyền xử phạt, được quyền xử lý |

### Bảng R18: "Lập biên bản"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Lập biên bản |
| ConcKeyS | Keyphrase | Người có thẩm quyền lập biên bản |
| ConcKeyO | List | Vi phạm hành chính, hành vi vi phạm |
| Keyphrases | Set | lập biên bản, lập biên bản vi phạm hành chính, ghi biên bản, viết biên bản |

### Bảng R19: "Cấm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cấm |
| ConcKeyS | Keyphrase | Quy tắc giao thông đường bộ / quy định |
| ConcKeyO | List | Hành vi vi phạm, đua xe trái phép, sản xuất lắp ráp trái phép phương tiện |
| Keyphrases | Set | cấm, không được, không được phép, nghiêm cấm, ngăn cấm |

### Bảng R20: "Yêu cầu / Bắt buộc"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Yêu cầu / Bắt buộc |
| ConcKeyS | Keyphrase | Quy tắc giao thông đường bộ / quy định về điều kiện |
| ConcKeyO | List | Giấy phép lái xe, giấy chứng nhận kiểm định, điều kiện của phương tiện, điều kiện của người điều khiển |
| Keyphrases | Set | phải, buộc, có trách nhiệm, bắt buộc, yêu cầu, cần phải |

### Bảng R21: "Gây ra"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Gây ra |
| ConcKeyS | Keyphrase | Vi phạm hành chính / hành vi vi phạm |
| ConcKeyO | List | Tai nạn giao thông, ô nhiễm môi trường, hậu quả, thiệt hại |
| Keyphrases | Set | gây, gây tai nạn, gây ra, để xảy ra, làm, dẫn đến, gây nên, làm phát sinh |

### Bảng R22: "Buộc khắc phục hậu quả"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Buộc khắc phục hậu quả |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm |
| ConcKeyO | List | Khôi phục tình trạng ban đầu, khắc phục ô nhiễm môi trường, nộp lại số lợi bất hợp pháp, tháo dỡ thiết bị lắp thêm |
| Keyphrases | Set | buộc, buộc khôi phục, buộc thực hiện, khắc phục hậu quả, phải khắc phục, sửa chữa hậu quả |

### Bảng R23: "Cấp"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cấp |
| ConcKeyS | Keyphrase | Cơ quan có thẩm quyền |
| ConcKeyO | List | Giấy phép lái xe, chứng chỉ hành nghề, giấy chứng nhận đăng ký xe, biển số xe, giấy chứng nhận kiểm định |
| Keyphrases | Set | cấp, cấp mới, cấp lại, do cơ quan có thẩm quyền cấp, ban hành, phát hành, cấp phát |

### Bảng R24: "Tham gia giao thông"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tham gia giao thông |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện / phương tiện giao thông |
| ConcKeyO | List | Đường bộ, đường cao tốc, lòng đường, làn đường |
| Keyphrases | Set | tham gia giao thông, lưu hành, lưu thông, di chuyển, đi lại, chạy trên đường |

---

## Tổng kết

- **Khái niệm (Conc): 66 bảng** (C1–C66), mỗi bảng gồm `Name`, `Keyphrases`.
- **Quan hệ (Rel): 24 bảng** (R1–R24), mỗi bảng gồm `Name`, `ConcKeyS`, `ConcKeyO`, `Keyphrases`.
- Đã **bỏ** trường `Similar` và **gộp** `Keyphrases` thành một danh sách duy nhất
  (không còn tách "Trong NĐ" / "Mở rộng").
