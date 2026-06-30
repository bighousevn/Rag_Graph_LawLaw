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
| Keyphrases | Set | xe ô tô, ô tô, ôtô, xe hơi, ô tô con, xe bốn bánh, xe 4 bánh, xe ô tô và các loại xe tương tự, ô tô tải, xe ô tô tải, xe tải, xe ben, xe thùng, xe chở hàng, xe tải nặng, xe tải nhẹ, ô tô chở người, ô tô chở hành khách, xe ô tô chở người, xe ô tô chở hành khách, xe ô tô chở người và các loại xe tương tự, xe khách, xe buýt, xe bus, xe đò, xe giường nằm, xe limousine, xe 16 chỗ, xe 45 chỗ, xe chở người bốn bánh có gắn động cơ, xe bốn bánh chở người, xe điện chở khách, xe điện du lịch, xe điện 4 bánh, xe chở hàng bốn bánh có gắn động cơ, xe bốn bánh chở hàng, xe tải nhỏ 4 bánh, ô tô đầu kéo, xe container, xe ô tô kinh doanh vận tải, xe ô tô kinh doanh vận tải chở trẻ em mầm non học sinh, xe kinh doanh vận tải, xe hợp đồng, xe taxi, xe công nghệ, xe đưa đón học sinh, xe tuyến cố định, xe dịch vụ, auto |

### Bảng C3: "Xe mô tô"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe mô tô |
| Keyphrases | Set | xe mô tô, mô tô, xe mô tô hai bánh, xe mô tô ba bánh, xe mô tô và các loại xe tương tự xe mô tô, xe máy, mô-tô, môtô, xe phân khối lớn, xe hai bánh, xe ba gác máy, xe pkl |

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

### Bảng C8: "Xe đạp, xe đạp máy"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe đạp, xe đạp máy |
| Keyphrases | Set | xe đạp, xe đạp máy, xe đạp điện, xe đạp thường, xe đạp trợ lực điện, xe đạp trẻ em |

### Bảng C9: "Xe thô sơ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe thô sơ |
| Keyphrases | Set | xe thô sơ, xe thô sơ khác, xe ba gác, xe xích lô, xe lôi, xe kéo tay, xe đẩy tay, xe súc vật kéo |

### Bảng C10: "Xe vật nuôi kéo"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe vật nuôi kéo |
| Keyphrases | Set | xe vật nuôi kéo, vật nuôi, súc vật, xe súc vật kéo, xe trâu kéo, xe bò kéo, xe ngựa kéo |

### Bảng C11: "Xe cứu hộ giao thông đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe cứu hộ giao thông đường bộ |
| Keyphrases | Set | xe cứu hộ giao thông đường bộ, xe cứu hộ, xe ô tô cứu hộ giao thông, xe kéo cứu hộ, xe cẩu kéo |

### Bảng C12: "Xe cứu thương"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe cứu thương |
| Keyphrases | Set | xe cứu thương, xe cấp cứu, xe y tế, xe 115, xe ambulance |

### Bảng C13: "Xe vệ sinh môi trường, xe ô tô chở phế thải"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe vệ sinh môi trường, xe ô tô chở phế thải |
| Keyphrases | Set | xe vệ sinh môi trường, xe ô tô chở phế thải, xe chở rác, xe ép rác, xe hút bể phốt, xe quét đường, xe môi trường |

### Bảng C14: "Xe quá khổ giới hạn, xe quá tải trọng, xe bánh xích"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xe quá khổ giới hạn, xe quá tải trọng, xe bánh xích |
| Keyphrases | Set | xe quá khổ giới hạn, quá khổ giới hạn, xe quá tải trọng, quá tải trọng, quá tải, quá khổ, quá trọng tải, xe bánh xích, xe vượt tải, xe cơi nới thành thùng, xe chở quá tải, xe xích |

### Bảng C15: "Phương tiện gắn biển số nước ngoài"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phương tiện gắn biển số nước ngoài |
| Keyphrases | Set | phương tiện giao thông cơ giới đường bộ gắn biển số nước ngoài, phương tiện đăng ký hoạt động trong Khu kinh tế thương mại đặc biệt, Khu kinh tế cửa khẩu quốc tế, xe gắn biển số nước ngoài, xe nước ngoài, xe mang biển số nước ngoài, xe tạm nhập tái xuất |

## A.2. Nhóm Chủ thể (người tham gia giao thông và đối tượng áp dụng)

### Bảng C16: "Người điều khiển phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người điều khiển phương tiện |
| Keyphrases | Set | người điều khiển, người điều khiển phương tiện, người lái xe, người điều khiển xe, tài xế, lái xe, người cầm lái, bác tài, người lái |

### Bảng C17: "Người đi bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người đi bộ |
| Keyphrases | Set | người đi bộ, bộ hành, khách bộ hành, người đi đường |

### Bảng C18: "Người dẫn dắt vật nuôi"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người dẫn dắt vật nuôi |
| Keyphrases | Set | người điều khiển dẫn dắt vật nuôi, dẫn dắt vật nuôi, điều khiển xe vật nuôi kéo, người dắt súc vật, người chăn dắt gia súc, người dẫn súc vật |

### Bảng C19: "Hành khách"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành khách |
| Keyphrases | Set | hành khách, hành khách đi xe, khách đi xe, người đi xe, khách, hành khách trên xe |

### Bảng C20: "Chủ phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chủ phương tiện |
| Keyphrases | Set | chủ phương tiện, chủ xe, chủ sở hữu xe, người đứng tên xe, chủ sở hữu phương tiện |

### Bảng C21: "Cá nhân, tổ chức vi phạm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cá nhân, tổ chức vi phạm |
| Keyphrases | Set | cá nhân, tổ chức, cá nhân tổ chức, hộ kinh doanh, hộ gia đình, người vi phạm, đối tượng vi phạm, doanh nghiệp, đơn vị, người bị xử phạt |

### Bảng C22: "Người đua xe, tổ chức đua xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người đua xe, tổ chức đua xe |
| Keyphrases | Set | đua xe, tổ chức đua xe, xúi giục đua xe, cổ vũ đua xe, quái xế, người tổ chức đua xe, người cổ vũ đua xe |

## A.3. Nhóm Hành vi & chế tài xử phạt

### Bảng C23: "Vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vi phạm hành chính |
| Keyphrases | Set | vi phạm hành chính, hành vi vi phạm hành chính, hành vi vi phạm, vi phạm, lỗi vi phạm, vi phạm giao thông, hành vi trái pháp luật, VPHC |

### Bảng C24: "Xử phạt vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xử phạt vi phạm hành chính |
| Keyphrases | Set | xử phạt, xử phạt vi phạm hành chính, phạt, xử lý vi phạm, chế tài xử phạt, xử lý hành chính |

### Bảng C25: "Cảnh cáo"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cảnh cáo |
| Keyphrases | Set | cảnh cáo, phạt cảnh cáo, nhắc nhở |

### Bảng C26: "Phạt tiền"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phạt tiền |
| Keyphrases | Set | phạt tiền, mức phạt tiền, phạt từ ... đến, nộp phạt, tiền phạt, đóng phạt, phạt hành chính |

### Bảng C27: "Tịch thu phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tịch thu phương tiện |
| Keyphrases | Set | tịch thu phương tiện, tịch thu tang vật, tịch thu, sung công, thu giữ phương tiện, tịch thu xe |

### Bảng C28: "Tước quyền sử dụng giấy phép, chứng chỉ hành nghề có thời hạn"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tước quyền sử dụng giấy phép, chứng chỉ hành nghề có thời hạn |
| Keyphrases | Set | tước quyền sử dụng, tước quyền sử dụng giấy phép, tước quyền sử dụng giấy phép chứng chỉ hành nghề, tước, tước bằng lái, treo bằng, thu bằng, tước giấy phép lái xe |

### Bảng C29: "Đình chỉ hoạt động có thời hạn"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đình chỉ hoạt động có thời hạn |
| Keyphrases | Set | đình chỉ hoạt động có thời hạn, đình chỉ hoạt động, tạm đình chỉ, ngừng hoạt động, đình chỉ kinh doanh |

### Bảng C30: "Biện pháp khắc phục hậu quả"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biện pháp khắc phục hậu quả |
| Keyphrases | Set | biện pháp khắc phục hậu quả, buộc khôi phục, khôi phục tình trạng ban đầu, buộc nộp lại, nộp lại số lợi bất hợp pháp, số lợi bất hợp pháp, buộc tháo dỡ, tháo dỡ thiết bị lắp thêm, khắc phục ô nhiễm môi trường, khắc phục hậu quả, khắc phục, biện pháp bổ sung, buộc khắc phục |

### Bảng C31: "Điểm giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Điểm giấy phép lái xe |
| Keyphrases | Set | điểm giấy phép lái xe, điểm, điểm bằng lái, điểm GPLX, 12 điểm, điểm trên bằng lái |

### Bảng C32: "Trừ điểm giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Trừ điểm giấy phép lái xe |
| Keyphrases | Set | trừ điểm giấy phép lái xe, trừ điểm, mức trừ điểm, bị trừ điểm, trừ điểm bằng lái, trừ điểm GPLX |

### Bảng C33: "Phục hồi điểm giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phục hồi điểm giấy phép lái xe |
| Keyphrases | Set | phục hồi điểm, phục hồi điểm giấy phép lái xe, khôi phục điểm, hồi điểm |

### Bảng C34: "Thời hiệu xử phạt vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thời hiệu xử phạt vi phạm hành chính |
| Keyphrases | Set | thời hiệu xử phạt, thời hiệu, thời hạn xử phạt, thời hiệu xử lý |

### Bảng C35: "Thẩm quyền xử phạt"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thẩm quyền xử phạt |
| Keyphrases | Set | thẩm quyền xử phạt, thẩm quyền, phân định thẩm quyền, quyền xử phạt, thẩm quyền xử lý |

### Bảng C36: "Biên bản vi phạm hành chính"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biên bản vi phạm hành chính |
| Keyphrases | Set | biên bản vi phạm hành chính, lập biên bản, biên bản, lập biên bản vi phạm, biên bản phạt |

### Bảng C37: "Tạm giữ phương tiện, giấy tờ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tạm giữ phương tiện, giấy tờ |
| Keyphrases | Set | tạm giữ phương tiện, tạm giữ, giấy tờ có liên quan, giữ xe, tạm giữ xe, giữ giấy tờ, tạm giữ giấy tờ |

## A.4. Nhóm Giấy tờ, giấy phép

### Bảng C38: "Giấy phép lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy phép lái xe |
| Keyphrases | Set | giấy phép lái xe, giấy phép lái, giấy phép lái xe quốc tế, giấy phép lái xe quốc gia, GPLX, bằng lái, bằng lái xe, bằng, giấy phép |

### Bảng C39: "Chứng chỉ hành nghề"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chứng chỉ hành nghề |
| Keyphrases | Set | chứng chỉ hành nghề, chứng chỉ đăng kiểm viên, chứng chỉ nghề, chứng chỉ chuyên môn |

### Bảng C40: "Giấy chứng nhận đăng ký xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy chứng nhận đăng ký xe |
| Keyphrases | Set | chứng nhận đăng ký xe, giấy chứng nhận đăng ký xe, chứng nhận đăng ký xe tạm thời, chứng nhận nguồn gốc xe, cà vẹt, cà vẹt xe, cavet, đăng ký xe, giấy đăng ký xe |

### Bảng C41: "Biển số xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biển số xe |
| Keyphrases | Set | biển số xe, biển số, số biển số, biển kiểm soát, BKS, bảng số xe, biển kiểm soát xe |

### Bảng C42: "Giấy chứng nhận, tem kiểm định an toàn kỹ thuật và bảo vệ môi trường"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy chứng nhận, tem kiểm định an toàn kỹ thuật và bảo vệ môi trường |
| Keyphrases | Set | giấy chứng nhận kiểm định, chứng nhận kiểm định an toàn kỹ thuật và bảo vệ môi trường, tem kiểm định, giấy chứng nhận hoặc tem kiểm định, giấy đăng kiểm, tem đăng kiểm, sổ kiểm định, giấy chứng nhận an toàn kỹ thuật và bảo vệ môi trường |

### Bảng C43: "Phù hiệu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phù hiệu |
| Keyphrases | Set | phù hiệu, phù hiệu xe, tem phù hiệu, phù hiệu xe tải, phù hiệu xe hợp đồng, biển hiệu |

### Bảng C44: "Giấy phép đào tạo, sát hạch lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy phép đào tạo, sát hạch lái xe |
| Keyphrases | Set | giấy phép đào tạo lái xe, giấy phép sát hạch, giấy phép cơ sở đào tạo lái xe, giấy phép trung tâm sát hạch |

### Bảng C45: "Giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới |
| Keyphrases | Set | giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới, giấy chứng nhận đăng kiểm, giấy phép trung tâm đăng kiểm |

## A.5. Nhóm Hạ tầng & báo hiệu đường bộ

### Bảng C46: "Đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đường bộ |
| Keyphrases | Set | đường bộ, giao thông đường bộ, lĩnh vực giao thông đường bộ, đường giao thông, đường sá, mạng lưới đường bộ, tuyến đường |

### Bảng C47: "Đường cao tốc"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đường cao tốc |
| Keyphrases | Set | đường cao tốc, cao tốc, tuyến cao tốc, xa lộ |

### Bảng C48: "Lòng đường, vỉa hè"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Lòng đường, vỉa hè |
| Keyphrases | Set | lòng đường, vỉa hè, hè phố, lề đường, mặt đường, vệ đường |

### Bảng C49: "Biển báo hiệu đường bộ, đèn tín hiệu giao thông"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Biển báo hiệu đường bộ, đèn tín hiệu giao thông |
| Keyphrases | Set | biển báo hiệu, đèn tín hiệu giao thông, vạch kẻ đường, dải phân cách, biển báo, đèn giao thông, đèn đỏ, cọc tiêu, biển chỉ dẫn, hệ thống báo hiệu đường bộ, vạch sơn |

### Bảng C50: "Quy tắc giao thông đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Quy tắc giao thông đường bộ |
| Keyphrases | Set | quy tắc giao thông đường bộ, quy tắc giao thông, luật giao thông, quy định giao thông, luật lệ giao thông, quy tắc đường bộ |

## A.6. Nhóm Cơ quan, người có thẩm quyền

### Bảng C51: "Chủ tịch Ủy ban nhân dân các cấp"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chủ tịch Ủy ban nhân dân các cấp |
| Keyphrases | Set | Chủ tịch Ủy ban nhân dân, Chủ tịch Ủy ban nhân dân các cấp, Chủ tịch UBND, Chủ tịch tỉnh, Chủ tịch huyện, Chủ tịch xã, UBND |

### Bảng C52: "Công an nhân dân"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Công an nhân dân |
| Keyphrases | Set | Công an nhân dân, Cảnh sát giao thông, CSGT, công an, cảnh sát, công an giao thông, CAND, cảnh sát cơ động |

### Bảng C53: "Thanh tra chuyên ngành"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thanh tra chuyên ngành |
| Keyphrases | Set | Thanh tra chuyên ngành, Thanh tra, thanh tra giao thông, TTGT, thanh tra đường bộ |

### Bảng C54: "Người có thẩm quyền lập biên bản"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người có thẩm quyền lập biên bản |
| Keyphrases | Set | người có thẩm quyền lập biên bản, thẩm quyền lập biên bản, người lập biên bản, người thi hành công vụ |

## A.7. Nhóm Hoạt động / lĩnh vực

### Bảng C55: "Kinh doanh vận tải đường bộ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Kinh doanh vận tải đường bộ |
| Keyphrases | Set | kinh doanh vận tải, vận tải đường bộ, dịch vụ hỗ trợ vận tải đường bộ, kinh doanh vận tải đường bộ, doanh nghiệp vận tải, đơn vị kinh doanh vận tải, hoạt động vận tải |

### Bảng C56: "Vận chuyển hàng hóa nguy hiểm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vận chuyển hàng hóa nguy hiểm |
| Keyphrases | Set | hàng hóa nguy hiểm, vận chuyển hàng hóa nguy hiểm, hàng nguy hiểm, chở hàng nguy hiểm, vận chuyển chất nguy hiểm, hóa chất nguy hiểm |

### Bảng C57: "Vận chuyển hàng siêu trường, siêu trọng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vận chuyển hàng siêu trường, siêu trọng |
| Keyphrases | Set | hàng siêu trường, hàng siêu trọng, vận chuyển hàng siêu trường siêu trọng, giấy phép lưu hành, hàng quá khổ quá tải, hàng cồng kềnh, chở hàng siêu trường siêu trọng |

### Bảng C58: "Đào tạo, sát hạch lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đào tạo, sát hạch lái xe |
| Keyphrases | Set | đào tạo lái xe, sát hạch, sát hạch lái xe, cơ sở đào tạo lái xe, trung tâm sát hạch, trường dạy lái xe, học lái xe, thi bằng lái, thi sát hạch |

### Bảng C59: "Hoạt động đăng kiểm xe cơ giới"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hoạt động đăng kiểm xe cơ giới |
| Keyphrases | Set | đăng kiểm, hoạt động đăng kiểm, kiểm định xe cơ giới, hoạt động đăng kiểm xe cơ giới, trung tâm đăng kiểm, kiểm định xe, đăng kiểm xe, kiểm tra kỹ thuật xe |

### Bảng C60: "Đua xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đua xe |
| Keyphrases | Set | đua xe, tổ chức đua xe, lạng lách đánh võng, đua tốc độ, độ xe đua, bốc đầu |

### Bảng C61: "Sản xuất, lắp ráp trái phép phương tiện; mua, bán biển số xe trái phép"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Sản xuất, lắp ráp trái phép phương tiện; mua, bán biển số xe trái phép |
| Keyphrases | Set | sản xuất lắp ráp trái phép phương tiện, mua bán biển số xe trái phép, sản xuất mua bán biển số xe trái phép, sản xuất xe lậu, lắp ráp xe trái phép, làm giả biển số, mua bán biển số giả |

## A.8. Nhóm Bổ sung từ Nghị định 168/2024/NĐ-CP

### Bảng C62: "Thiết bị, vật dụng trên phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thiết bị, vật dụng trên phương tiện |
| Keyphrases | Set | thiết bị, vật dụng, điện thoại, điện thoại di động, tai nghe, thiết bị điện tử, thiết bị âm thanh, ô, dù, còi, còi hơi, đèn chiếu sáng, đèn chiếu xa, đèn khẩn cấp, đèn cảnh báo, đèn tín hiệu, gương chiếu hậu, hệ thống hãm, phanh, bộ phận giảm thanh, bộ phận giảm khói, kính an toàn, dây đai an toàn, ghế ngồi, mũ bảo hiểm, biển báo dấu hiệu nhận biết, thiết bị phát tín hiệu ưu tiên, thiết bị thay đổi biển số, dụng cụ cứu hộ, thiết bị chuyên dùng cứu hộ, cơ cấu khóa hãm công-ten-nơ, đồng hồ báo quãng đường, thẻ đầu cuối |

### Bảng C63: "Thiết bị giám sát, ghi nhận dữ liệu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thiết bị giám sát, ghi nhận dữ liệu |
| Keyphrases | Set | thiết bị giám sát hành trình, thiết bị ghi nhận hình ảnh người lái xe, thiết bị ghi nhận hình ảnh trẻ em mầm non học sinh, thiết bị cảnh báo chống bỏ quên trẻ em, camera, camera giám sát, thiết bị chấm điểm, máy tính sát hạch, màn hình, hệ thống âm thanh thông báo lỗi, thiết bị giám sát thời gian học lý thuyết, thiết bị giám sát thời gian học thực hành, thiết bị ghi nhận dữ liệu, dữ liệu thiết bị, phương tiện thiết bị kỹ thuật nghiệp vụ |

### Bảng C64: "Nồng độ cồn, chất ma túy"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Nồng độ cồn, chất ma túy |
| Keyphrases | Set | nồng độ cồn, cồn, rượu, bia, rượu bia, hơi thở có nồng độ cồn, trong máu có nồng độ cồn, ma túy, chất ma túy, chất kích thích, chất kích thích khác mà pháp luật cấm sử dụng |

### Bảng C65: "Hàng hóa, hành lý, vật liệu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hàng hóa, hành lý, vật liệu |
| Keyphrases | Set | hàng hóa, hàng, hàng rời, hành lý, công-ten-nơ, container, hàng cấm, hàng độc hại, hàng dễ cháy, hàng dễ nổ, hóa chất độc hại, chất dễ cháy nổ, chất thải, chất phế thải, phế thải, vật liệu xây dựng, đất đá, bùn đất, cát, đá, rác, động vật sống, thực phẩm tươi sống, hàng hóa là phương tiện vận tải, hàng dạng trụ |

### Bảng C66: "Hậu quả, thiệt hại"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hậu quả, thiệt hại |
| Keyphrases | Set | hậu quả, thiệt hại, tai nạn giao thông, vụ tai nạn giao thông, gây tai nạn giao thông, ùn tắc giao thông, cản trở giao thông, ô nhiễm môi trường, hư hại cầu đường, mất an toàn giao thông, gây hậu quả nghiêm trọng |

### Bảng C67: "Dữ liệu, thông tin điện tử"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Dữ liệu, thông tin điện tử |
| Keyphrases | Set | dữ liệu, thông tin, cơ sở dữ liệu, Cơ sở dữ liệu về xử lý vi phạm hành chính, Cơ sở dữ liệu về trật tự an toàn giao thông đường bộ, dữ liệu điểm giấy phép lái xe, thông điệp dữ liệu, thông tin điện tử, tài khoản định danh điện tử, căn cước điện tử, Cổng dịch vụ công, Ứng dụng giao thông trên thiết bị di động, Trang thông tin điện tử của Cục Cảnh sát giao thông, thông báo điện tử, kết nối chia sẻ dữ liệu |

### Bảng C68: "Điều kiện phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Điều kiện phương tiện |
| Keyphrases | Set | điều kiện của phương tiện, điều kiện phương tiện, an toàn kỹ thuật, bảo vệ môi trường, vệ sinh lưu thông, khí thải, tiếng ồn, niên hạn sử dụng, kích thước thùng xe, khoang chở hành lý, khối lượng chuyên chở, tải trọng, tải trọng trục, trọng tải, khổ giới hạn, kích thước bao ngoài, tổng trọng lượng, hình dáng, kích thước, màu sơn, nhãn hiệu, số khung, số động cơ, số máy, tiêu chuẩn kỹ thuật, quy chuẩn kỹ thuật |

### Bảng C69: "Điều kiện người điều khiển phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Điều kiện người điều khiển phương tiện |
| Keyphrases | Set | điều kiện của người điều khiển, điều kiện người điều khiển, độ tuổi, tuổi lái xe, sức khỏe, đủ điều kiện điều khiển, giấy phép lái xe phù hợp, chứng chỉ bồi dưỡng kiến thức pháp luật về giao thông đường bộ, bằng điều khiển xe máy chuyên dùng, thời gian lái xe, thời gian nghỉ giữa hai lần lái xe, số năm kinh nghiệm, tập huấn nghiệp vụ, quy trình bảo đảm an toàn |

### Bảng C70: "Hồ sơ, tài liệu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hồ sơ, tài liệu |
| Keyphrases | Set | hồ sơ, tài liệu, giấy tờ, hồ sơ đăng ký xe, hồ sơ phương tiện, hồ sơ kiểm định, hồ sơ vận chuyển hàng hóa nguy hiểm, tài liệu giả, giấy tờ giả, giấy tờ bị tẩy xóa, giấy tờ bị sửa chữa, bản dịch, bản sao chứng thực, giấy biên nhận của tổ chức tín dụng, giấy biên nhận ngân hàng, tờ khai phương tiện vận tải đường bộ tạm nhập tái xuất, văn bản chấp thuận, văn bản cấp phép, giấy chứng nhận hoàn thành chương trình tập huấn |

### Bảng C71: "Giấy phép lưu hành, vận chuyển"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy phép lưu hành, vận chuyển |
| Keyphrases | Set | giấy phép lưu hành, giấy phép vận chuyển hàng hóa nguy hiểm, giấy phép liên vận, giấy phép vận tải đường bộ quốc tế, giấy phép xe tập lái, giấy phép kinh doanh vận tải, giấy phép chấp thuận cho phương tiện nước ngoài, phù hiệu kiểm soát, ký hiệu phân biệt quốc gia, biển số tạm thời |

### Bảng C72: "Chứng nhận bảo hiểm bắt buộc trách nhiệm dân sự"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chứng nhận bảo hiểm bắt buộc trách nhiệm dân sự |
| Keyphrases | Set | chứng nhận bảo hiểm bắt buộc trách nhiệm dân sự, bảo hiểm bắt buộc trách nhiệm dân sự, bảo hiểm trách nhiệm dân sự của chủ xe cơ giới, giấy chứng nhận bảo hiểm, bảo hiểm bắt buộc |

### Bảng C73: "Giấy vận tải, lệnh vận chuyển, hợp đồng vận tải"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giấy vận tải, lệnh vận chuyển, hợp đồng vận tải |
| Keyphrases | Set | giấy vận tải, lệnh vận chuyển, hợp đồng vận tải, hợp đồng vận chuyển, hợp đồng điện tử, hợp đồng đào tạo, thanh lý hợp đồng đào tạo, danh sách hành khách, thiết bị truy cập hợp đồng điện tử, sổ nhật trình, hành trình chạy xe, lịch trình, tuyến đường vận tải |

### Bảng C74: "Thẻ nhận dạng lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thẻ nhận dạng lái xe |
| Keyphrases | Set | thẻ nhận dạng lái xe, thẻ nhận dạng, thẻ đăng nhập lái xe, đăng nhập thông tin lái xe, sử dụng thẻ nhận dạng lái xe |

### Bảng C75: "Người phục vụ, quản lý, áp tải trên xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người phục vụ, quản lý, áp tải trên xe |
| Keyphrases | Set | nhân viên phục vụ trên xe, người phục vụ trên xe, người quản lý trên xe, người áp tải, người áp tải vận chuyển hàng hóa nguy hiểm, lái xe kinh doanh vận tải, giáo viên dạy thực hành, người làm công, người đại diện |

### Bảng C76: "Cơ sở đào tạo, sát hạch lái xe"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cơ sở đào tạo, sát hạch lái xe |
| Keyphrases | Set | cơ sở đào tạo lái xe, trung tâm sát hạch lái xe, giáo viên dạy lái xe, học viên lái xe, người dự sát hạch, xe tập lái, xe sát hạch, sân tập lái, phòng sát hạch, phần mềm sát hạch, phù hiệu học viên tập lái xe, phù hiệu giáo viên dạy lái xe |

### Bảng C77: "Cơ sở đăng kiểm, thử nghiệm, sản xuất, nhập khẩu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cơ sở đăng kiểm, thử nghiệm, sản xuất, nhập khẩu |
| Keyphrases | Set | cơ sở đăng kiểm, đăng kiểm viên, nhân viên nghiệp vụ đăng kiểm, cơ sở thử nghiệm, cơ sở chứng nhận, cơ sở sản xuất, cơ sở lắp ráp, cơ sở nhập khẩu, cơ sở bảo hành, cơ sở bảo dưỡng, phụ tùng xe cơ giới, chứng nhận cải tạo, kiểm định khí thải, dịch vụ kiểm định, kiểm tra chất lượng xuất xưởng |

### Bảng C78: "Cơ quan đăng ký, đăng kiểm, cấp giấy phép"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cơ quan đăng ký, đăng kiểm, cấp giấy phép |
| Keyphrases | Set | cơ quan đăng ký xe, cơ quan đăng kiểm, cơ quan cấp giấy phép lái xe, cơ quan cấp giấy phép, cơ quan cấp phù hiệu, cơ quan quản lý nhà nước có thẩm quyền, cơ quan chức năng, Cục Cảnh sát giao thông, Trưởng phòng Cảnh sát giao thông, Bộ Công an |

### Bảng C79: "Đường, làn, khu vực giao thông"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Đường, làn, khu vực giao thông |
| Keyphrases | Set | phần đường, làn đường, làn cùng chiều, làn ngược chiều, đường một chiều, đường đôi, đường ưu tiên, đường nhánh, đường chính, đường có biển cấm đi vào, khu vực cấm, đường dành riêng cho xe buýt, làn dừng xe khẩn cấp, lề đường, dải phân cách, điểm đón trả khách, bến xe, bãi đỗ xe, trạm dừng nghỉ, khu đông dân cư |

### Bảng C80: "Công trình giao thông"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Công trình giao thông |
| Keyphrases | Set | công trình giao thông, cầu, đường sắt, đường ngang, đường bộ giao nhau cùng mức với đường sắt, công trình đường sắt, phạm vi an toàn đường sắt, hầm đường bộ, trạm thu phí, thu phí điện tử tự động không dừng, làn thu phí không dừng, gầm cầu vượt, đường dốc, đoạn đường cong, nơi đường bộ giao nhau, vòng xuyến |

### Bảng C81: "Người bị nạn, người yếu thế"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Người bị nạn, người yếu thế |
| Keyphrases | Set | người bị nạn, người khuyết tật, người cao tuổi, người già yếu, phụ nữ mang thai, người bệnh, trẻ em, trẻ em dưới 10 tuổi, trẻ em dưới 12 tuổi, trẻ em mầm non, học sinh, học sinh tiểu học, sinh viên, công nhân |

### Bảng C82: "Hành vi vi phạm quy tắc giao thông"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm quy tắc giao thông |
| Keyphrases | Set | không chấp hành hiệu lệnh, không chấp hành chỉ dẫn, chạy quá tốc độ, chạy dưới tốc độ tối thiểu, chuyển làn không đúng quy định, chuyển hướng không đúng quy định, quay đầu xe, lùi xe, vượt xe, đi ngược chiều, đi sai phần đường, đi sai làn đường, không nhường đường, không giữ khoảng cách an toàn, dừng xe, đỗ xe, đi vào đường cao tốc, đi vào khu vực cấm, không bật đèn, sử dụng còi, rú ga, nẹt pô, lạng lách, đánh võng, đuổi nhau, dùng chân điều khiển vô lăng, buông cả hai tay, chạy bằng một bánh, không đội mũ bảo hiểm, không thắt dây đai an toàn, sử dụng điện thoại khi lái xe, không chấp hành yêu cầu kiểm tra |

### Bảng C83: "Hành vi vi phạm giấy tờ, biển số, kiểm định"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm giấy tờ, biển số, kiểm định |
| Keyphrases | Set | không có giấy phép lái xe, không mang theo giấy phép lái xe, giấy phép lái xe hết hạn, giấy phép lái xe không phù hợp, giấy phép lái xe không hợp lệ, giấy phép lái xe bị tẩy xóa, không có chứng nhận đăng ký xe, không mang theo chứng nhận đăng ký xe, không gắn biển số, gắn biển số không đúng, biển số bị che lấp, biển số giả, chứng nhận kiểm định hết hạn, không có chứng nhận kiểm định, không có phù hiệu, phù hiệu hết giá trị sử dụng, giấy tờ không do cơ quan có thẩm quyền cấp, không đúng số khung, không đúng số động cơ |

### Bảng C84: "Hành vi vi phạm vận tải hành khách"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm vận tải hành khách |
| Keyphrases | Set | chở quá số người, đón trả khách không đúng nơi, đón trả khách trên đường cao tốc, chuyển tải hành khách, tranh giành hành khách, lôi kéo hành khách, đe dọa hành khách, cưỡng ép hành khách, không hướng dẫn hành khách, không niêm yết hành trình, không có nhân viên phục vụ, không có danh sách hành khách, vận chuyển không đúng đối tượng, chở người trên mui xe, để người lên xuống xe khi xe đang chạy, không đóng cửa lên xuống |

### Bảng C85: "Hành vi vi phạm vận tải hàng hóa"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm vận tải hàng hóa |
| Keyphrases | Set | chở quá tải, xếp hàng vượt tải, chở hàng vượt trọng tải, chở hàng vượt khổ, chở hàng vượt kích thước, chở hàng vượt quá chiều cao, vận chuyển hàng không chằng buộc, chằng buộc hàng hóa không bảo đảm, để rơi hàng hóa, chở công-ten-nơ không bảo đảm, nhận hàng trên đường cao tốc, trả hàng trên đường cao tốc, hạ phần hàng quá tải, dỡ phần hàng quá khổ, không sử dụng cơ cấu khóa hãm công-ten-nơ |

### Bảng C86: "Hành vi vi phạm môi trường"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm môi trường |
| Keyphrases | Set | không đáp ứng yêu cầu vệ sinh lưu thông, để rơi vãi hàng hóa, làm rơi vãi hàng hóa, để nước chảy xuống mặt đường, lôi kéo bùn đất ra đường, đổ rác trái phép, đổ phế thải trái phép, đổ đất đá trái phép, gây ô nhiễm môi trường, không bảo đảm khí thải, không bảo đảm tiếng ồn |

### Bảng C87: "Hành vi vi phạm đào tạo, sát hạch"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm đào tạo, sát hạch |
| Keyphrases | Set | đào tạo vượt lưu lượng, đào tạo ngoài địa điểm, đào tạo không đúng hạng, đào tạo không đúng chương trình, tuyển sinh không đủ điều kiện, không lưu trữ hồ sơ đào tạo, không lưu trữ hồ sơ sát hạch, sử dụng xe tập lái không đúng quy định, không đủ giáo viên, không đủ sân tập lái, gian dối sát hạch, can thiệp làm sai lệch dữ liệu, tự ý thay đổi phần mềm sát hạch, không đủ camera giám sát, không đủ thiết bị chấm điểm |

### Bảng C88: "Hành vi vi phạm đăng kiểm, thử nghiệm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm đăng kiểm, thử nghiệm |
| Keyphrases | Set | kiểm định không đúng quy định, cấp giấy chứng nhận kiểm định không đúng quy định, cấp chứng nhận cải tạo không đúng quy định, từ chối kiểm định không đúng quy định, không lưu trữ hồ sơ kiểm định, thử nghiệm không đúng quy định, chứng nhận không đúng quy định, không duy trì điều kiện đăng kiểm, sử dụng thiết bị đo lường không bảo đảm, phân công người không đủ điều kiện đăng kiểm |

### Bảng C89: "Hành vi vi phạm của chủ phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm của chủ phương tiện |
| Keyphrases | Set | giao phương tiện cho người không đủ điều kiện, đưa phương tiện không đủ điều kiện tham gia giao thông, tự ý thay đổi nhãn hiệu, tự ý thay đổi màu sơn, tự ý thay đổi khung máy, cải tạo xe trái quy định, lắp thêm ghế, tháo bớt ghế, lắp thêm giường nằm, tháo bớt giường nằm, cơi nới thùng xe, thay đổi kích thước thùng xe, can thiệp đồng hồ quãng đường, không làm thủ tục đăng ký xe, không làm thủ tục thu hồi đăng ký xe, không làm thủ tục đổi đăng ký xe |

### Bảng C90: "Hành vi vi phạm dữ liệu, thiết bị giám sát"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hành vi vi phạm dữ liệu, thiết bị giám sát |
| Keyphrases | Set | không cung cấp dữ liệu, không cập nhật dữ liệu, không truyền dữ liệu, không lưu trữ dữ liệu, không quản lý dữ liệu, làm sai lệch dữ liệu, không lắp thiết bị giám sát hành trình, không lắp thiết bị ghi nhận hình ảnh, thiết bị giám sát hành trình không hoạt động, thiết bị ghi nhận hình ảnh không hoạt động, không có thiết bị cảnh báo chống bỏ quên trẻ em, không đăng nhập thẻ nhận dạng lái xe |

### Bảng C91: "Tình tiết, trạng thái vi phạm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tình tiết, trạng thái vi phạm |
| Keyphrases | Set | tái phạm, vi phạm nhiều lần, hành vi vi phạm đã kết thúc, hành vi vi phạm đang thực hiện, cố tình trốn tránh, cản trở việc xử phạt, không hợp tác, không chứng minh, không giải trình, trực tiếp điều khiển phương tiện, vượt trên 50%, vượt trên 100%, vượt trên 150%, vượt quá quy định |

### Bảng C92: "Thủ tục, quyết định xử phạt"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thủ tục, quyết định xử phạt |
| Keyphrases | Set | thủ tục xử phạt, nguyên tắc xử phạt, quyết định xử phạt, quyết định tạm giữ, thời hạn ra quyết định xử phạt, thông báo, thông báo yêu cầu, giải quyết vụ việc vi phạm, thi hành quyết định xử phạt, xác minh, căn cứ xử phạt, lập biên bản, gửi quyết định xử phạt, chuyển kết quả thu thập, cập nhật trạng thái |

### Bảng C93: "Văn bản pháp luật"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Văn bản pháp luật |
| Keyphrases | Set | văn bản pháp luật, nghị định, Nghị định 168/2024/NĐ-CP, Nghị định 100/2019/NĐ-CP, Nghị định 123/2021/NĐ-CP, Luật Trật tự an toàn giao thông đường bộ, Luật Xử lý vi phạm hành chính, điều khoản thi hành, hiệu lực thi hành, điều khoản chuyển tiếp, trách nhiệm thi hành |

### Bảng C94: "Thông số giới hạn"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thông số giới hạn |
| Keyphrases | Set | tốc độ, tốc độ tối đa, tốc độ tối thiểu, nồng độ cồn, tải trọng, trọng tải, tải trọng trục, khổ giới hạn, kích thước giới hạn, chiều cao xếp hàng, số người quy định, số người được phép chở, khối lượng hàng chuyên chở, khối lượng toàn bộ, tổng trọng lượng |

### Bảng C95: "Hiệu lệnh, yêu cầu kiểm tra"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Hiệu lệnh, yêu cầu kiểm tra |
| Keyphrases | Set | hiệu lệnh, chỉ dẫn, hiệu lệnh của đèn tín hiệu giao thông, hướng dẫn của người điều khiển giao thông, yêu cầu kiểm tra, yêu cầu kiểm tra nồng độ cồn, yêu cầu kiểm tra chất ma túy, yêu cầu kiểm tra trọng tải, người điều khiển giao thông, người kiểm soát giao thông, người thi hành công vụ |

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
| ConcKeyO | List | Vi phạm hành chính, hành vi vi phạm, đua xe, vận chuyển hàng hóa nguy hiểm, sản xuất lắp ráp trái phép phương tiện |
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
| ConcKeyO | List | Hành vi vi phạm, đua xe, sản xuất lắp ráp trái phép phương tiện |
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

### Bảng R25: "Có mức xử phạt"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Có mức xử phạt |
| ConcKeyS | Keyphrase | Vi phạm hành chính / hành vi vi phạm |
| ConcKeyO | List | Cảnh cáo, phạt tiền, tịch thu phương tiện, tước quyền sử dụng giấy phép chứng chỉ hành nghề, đình chỉ hoạt động có thời hạn, biện pháp khắc phục hậu quả |
| Keyphrases | Set | bị phạt, bị xử phạt, phạt tiền từ, phạt cảnh cáo, áp dụng hình thức xử phạt, ngoài việc bị phạt tiền, còn bị áp dụng |

### Bảng R26: "Bị trừ điểm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Bị trừ điểm |
| ConcKeyS | Keyphrase | Vi phạm hành chính / người điều khiển phương tiện |
| ConcKeyO | List | Điểm giấy phép lái xe, trừ điểm giấy phép lái xe, giấy phép lái xe |
| Keyphrases | Set | bị trừ điểm, trừ điểm giấy phép lái xe, còn bị trừ điểm, mức trừ điểm, trừ hết điểm |

### Bảng R27: "Bị tước quyền sử dụng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Bị tước quyền sử dụng |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm / người điều khiển phương tiện / cơ sở đào tạo sát hạch lái xe / cơ sở đăng kiểm |
| ConcKeyO | List | Giấy phép lái xe, phù hiệu, giấy phép đào tạo sát hạch lái xe, chứng chỉ hành nghề, giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới |
| Keyphrases | Set | bị tước quyền sử dụng, tước quyền sử dụng, trong thời gian bị tước, tước phù hiệu, tước giấy phép sát hạch, tước chứng chỉ đăng kiểm viên |

### Bảng R28: "Thu hồi"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Thu hồi |
| ConcKeyS | Keyphrase | Thẩm quyền xử phạt / cơ quan đăng ký đăng kiểm cấp giấy phép |
| ConcKeyO | List | Giấy phép lái xe, giấy chứng nhận đăng ký xe, biển số xe, giấy chứng nhận kiểm định, phù hiệu, giấy phép lưu hành, hồ sơ tài liệu |
| Keyphrases | Set | thu hồi, bị thu hồi, tiến hành thu hồi, buộc làm thủ tục thu hồi, thu hồi giấy phép, thu hồi chứng nhận đăng ký xe, thu hồi biển số xe, thu hồi phù hiệu |

### Bảng R29: "Buộc nộp lại"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Buộc nộp lại |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm |
| ConcKeyO | List | Giấy phép lái xe, giấy chứng nhận đăng ký xe, giấy chứng nhận kiểm định, giấy phép chứng chỉ hành nghề, hồ sơ tài liệu, số lợi bất hợp pháp |
| Keyphrases | Set | buộc nộp lại, nộp lại giấy phép, nộp lại chứng nhận đăng ký xe, nộp lại số lợi bất hợp pháp, nộp lại giấy phép chứng chỉ hành nghề bị tẩy xóa |

### Bảng R30: "Buộc lắp đặt, thay thế"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Buộc lắp đặt, thay thế |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm / chủ phương tiện |
| ConcKeyO | List | Thiết bị vật dụng trên phương tiện, thiết bị giám sát ghi nhận dữ liệu, dây đai an toàn, ghế ngồi, thiết bị chuyên dùng cứu hộ |
| Keyphrases | Set | buộc lắp đặt, buộc lắp đầy đủ, buộc thay thế thiết bị, khôi phục tính năng kỹ thuật, lắp thiết bị giám sát hành trình, lắp thiết bị ghi nhận hình ảnh, lắp dây đai an toàn, lắp ghế ngồi |

### Bảng R31: "Buộc tháo dỡ, dỡ hàng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Buộc tháo dỡ, dỡ hàng |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm / chủ phương tiện |
| ConcKeyO | List | Thiết bị vật dụng trên phương tiện, hàng hóa hành lý vật liệu, xe quá khổ quá tải |
| Keyphrases | Set | buộc tháo dỡ, buộc tháo bỏ, buộc hạ phần hàng quá tải, buộc dỡ phần hàng quá khổ, buộc dỡ hàng hóa, buộc tháo dỡ thiết bị âm thanh ánh sáng |

### Bảng R32: "Buộc tái xuất, quay lại"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Buộc tái xuất, quay lại |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm / người điều khiển phương tiện |
| ConcKeyO | List | Phương tiện gắn biển số nước ngoài, Khu kinh tế thương mại đặc biệt, Khu kinh tế cửa khẩu quốc tế |
| Keyphrases | Set | buộc tái xuất, tái xuất phương tiện khỏi Việt Nam, buộc đưa phương tiện quay trở lại, quay trở lại Khu kinh tế thương mại đặc biệt, quay trở lại Khu kinh tế cửa khẩu quốc tế |

### Bảng R33: "Phát hiện bằng"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Phát hiện bằng |
| ConcKeyS | Keyphrase | Vi phạm hành chính / hành vi vi phạm |
| ConcKeyO | List | Thiết bị giám sát ghi nhận dữ liệu, dữ liệu thông tin điện tử, phương tiện thiết bị kỹ thuật nghiệp vụ |
| Keyphrases | Set | phát hiện thông qua, ghi nhận hành vi vi phạm, kết quả thu thập được bằng, sử dụng phương tiện thiết bị kỹ thuật nghiệp vụ, dữ liệu thu được |

### Bảng R34: "Cập nhật, truyền dữ liệu"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Cập nhật, truyền dữ liệu |
| ConcKeyS | Keyphrase | Cơ quan đăng ký đăng kiểm cấp giấy phép / cá nhân, tổ chức vi phạm / thiết bị giám sát ghi nhận dữ liệu |
| ConcKeyO | List | Dữ liệu thông tin điện tử, Cơ sở dữ liệu về xử lý vi phạm hành chính, điểm giấy phép lái xe |
| Keyphrases | Set | cung cấp, cập nhật, truyền dẫn, lưu trữ, quản lý thông tin dữ liệu, cập nhật dữ liệu, cập nhật trạng thái, đồng bộ dữ liệu, kết nối chia sẻ dữ liệu |

### Bảng R35: "Xuất trình, mang theo"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Xuất trình, mang theo |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện |
| ConcKeyO | List | Giấy phép lái xe, giấy chứng nhận đăng ký xe, giấy chứng nhận kiểm định, chứng nhận bảo hiểm bắt buộc trách nhiệm dân sự, giấy vận tải lệnh vận chuyển hợp đồng vận tải, giấy phép lưu hành vận chuyển |
| Keyphrases | Set | xuất trình, mang theo, không mang theo, không xuất trình được, có nhưng không mang theo, cung cấp cho lực lượng chức năng khi có yêu cầu |

### Bảng R36: "Giao phương tiện"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Giao phương tiện |
| ConcKeyS | Keyphrase | Chủ phương tiện |
| ConcKeyO | List | Người điều khiển phương tiện, phương tiện giao thông |
| Keyphrases | Set | giao phương tiện, để cho người làm công điều khiển, để cho người đại diện điều khiển, giao xe, đưa phương tiện tham gia giao thông |

### Bảng R37: "Lắp đặt, gắn"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Lắp đặt, gắn |
| ConcKeyS | Keyphrase | Chủ phương tiện / phương tiện giao thông |
| ConcKeyO | List | Biển số xe, phù hiệu, thiết bị vật dụng trên phương tiện, thiết bị giám sát ghi nhận dữ liệu, biển báo dấu hiệu nhận biết |
| Keyphrases | Set | lắp đặt, gắn, không gắn, lắp thêm, trang bị, dán, kẻ, niêm yết, có gắn, không lắp |

### Bảng R38: "Chạy, lưu hành trên"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Chạy, lưu hành trên |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện / phương tiện giao thông |
| ConcKeyO | List | Đường bộ, đường cao tốc, phần đường, làn đường, khu vực cấm, công trình giao thông, Khu kinh tế thương mại đặc biệt, Khu kinh tế cửa khẩu quốc tế |
| Keyphrases | Set | chạy trên, đi vào, đi trên, lưu hành trên, hoạt động trên, tham gia giao thông tại, đi qua, vào hoặc ra đường cao tốc |

### Bảng R39: "Dừng, đỗ tại"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Dừng, đỗ tại |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện / phương tiện giao thông |
| ConcKeyO | List | Lòng đường, vỉa hè, phần đường, làn đường, khu vực cấm, công trình giao thông, đường cao tốc |
| Keyphrases | Set | dừng xe, đỗ xe, dừng đỗ, để xe, dừng xe đỗ xe tại, dừng xe đỗ xe trên, dừng xe đỗ xe trong |

### Bảng R40: "Vượt quá"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Vượt quá |
| ConcKeyS | Keyphrase | Người điều khiển phương tiện / phương tiện giao thông / hàng hóa hành lý vật liệu / nồng độ cồn chất ma túy |
| ConcKeyO | List | Tốc độ, tải trọng, khổ giới hạn, số người quy định, điều kiện phương tiện, nồng độ cồn |
| Keyphrases | Set | vượt quá, quá tốc độ, quá tải, vượt tải, vượt trọng tải, vượt khổ, vượt quá số người, vượt quá kích thước, vượt quá chiều cao, vượt quá nồng độ |

### Bảng R41: "Không chấp hành"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Không chấp hành |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm / người điều khiển phương tiện |
| ConcKeyO | List | Biển báo hiệu đường bộ đèn tín hiệu giao thông, quy tắc giao thông đường bộ, yêu cầu kiểm tra, người điều khiển giao thông, người kiểm soát giao thông |
| Keyphrases | Set | không chấp hành, không tuân thủ, không thực hiện, không giảm tốc độ, không nhường đường, không chấp hành yêu cầu kiểm tra, không chấp hành hiệu lệnh |

### Bảng R42: "Tổ chức đào tạo, sát hạch, đăng kiểm"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Tổ chức đào tạo, sát hạch, đăng kiểm |
| ConcKeyS | Keyphrase | Cơ sở đào tạo sát hạch lái xe / cơ sở đăng kiểm thử nghiệm sản xuất nhập khẩu |
| ConcKeyO | List | Đào tạo sát hạch lái xe, hoạt động đăng kiểm xe cơ giới, giấy phép đào tạo sát hạch lái xe, giấy chứng nhận đủ điều kiện hoạt động kiểm định xe cơ giới |
| Keyphrases | Set | tổ chức đào tạo, tổ chức sát hạch, thực hiện kiểm định, chứng nhận, thử nghiệm, đào tạo lái xe, sát hạch lái xe, hoạt động đăng kiểm |

### Bảng R43: "Kinh doanh, cung cấp dịch vụ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Kinh doanh, cung cấp dịch vụ |
| ConcKeyS | Keyphrase | Cá nhân, tổ chức vi phạm / đơn vị kinh doanh vận tải / cơ sở đăng kiểm thử nghiệm sản xuất nhập khẩu |
| ConcKeyO | List | Kinh doanh vận tải đường bộ, dịch vụ hỗ trợ vận tải đường bộ, hoạt động đăng kiểm xe cơ giới, thiết bị giám sát ghi nhận dữ liệu |
| Keyphrases | Set | kinh doanh, kinh doanh vận tải, dịch vụ hỗ trợ vận tải, cung cấp dịch vụ, sản xuất lắp ráp nhập khẩu, bảo hành bảo dưỡng, cung ứng dịch vụ |

### Bảng R44: "Sửa đổi, bãi bỏ"

| Thành phần | Kiểu | Giá trị |
|---|---|---|
| Name | Text | Sửa đổi, bãi bỏ |
| ConcKeyS | Keyphrase | Nghị định 168/2024/NĐ-CP / văn bản pháp luật |
| ConcKeyO | List | Nghị định 100/2019/NĐ-CP, Nghị định 123/2021/NĐ-CP, quy định xử phạt vi phạm hành chính |
| Keyphrases | Set | sửa đổi, bổ sung, bãi bỏ, bỏ cụm từ, có hiệu lực thi hành, điều khoản chuyển tiếp, trách nhiệm thi hành |

---

## Tổng kết

- **Khái niệm (Conc): 95 bảng** (C1–C95), mỗi bảng gồm `Name`, `Keyphrases`.
- **Quan hệ (Rel): 44 bảng** (R1–R44), mỗi bảng gồm `Name`, `ConcKeyS`, `ConcKeyO`, `Keyphrases`.
- Đã **bỏ** trường `Similar` và **gộp** `Keyphrases` thành một danh sách duy nhất
  (không còn tách "Trong NĐ" / "Mở rộng").
