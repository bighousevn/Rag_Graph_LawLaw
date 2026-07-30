# Quy trình đánh giá file Q&A dạng emlaw (question/answer) theo 4 tiêu chí

Dùng file này để chạy lại quy trình đánh giá chất lượng câu trả lời trong
1 file JSON/Excel dạng `[{"stt", "question", "answer"}, ...]` (vd
`emlaw_qa.json`, `answer_dat_dai.json`...) ở các phiên Claude Code sau. Vì
tiêu chí 2 cần web search thật (verify điều luật còn hiệu lực hay không, và
đọc nội dung thật để đối chiếu), quy trình này luôn cần một subagent Claude
Code có quyền WebSearch/WebFetch trong vòng lặp — không có cách nào chạy
hoàn toàn bằng script độc lập nếu không có API key của một LLM khác.

## 4 tiêu chí (boolean true/false)

1. `xac_dinh_dung_van_de` — answer có xác định đúng bản chất/vấn đề pháp lý
   cốt lõi mà câu hỏi đặt ra không (không lạc đề, không hiểu sai tình huống).
2. `trich_dung_dieu_luat` — **bắt buộc web search thật**, ưu tiên tra trên
   **thuvienphapluat.vn**, kiểm tra:
   - Văn bản/điều được trích có tồn tại không.
   - Nội dung điều luật có khớp với cách answer diễn giải/áp dụng không.
   - **Văn bản đó còn hiệu lực tại thời điểm hiện tại hay đã bị sửa
     đổi/thay thế/bãi bỏ bởi văn bản mới hơn.** Nếu đã bị thay thế →
     **luôn đánh giá `false`**, dù nội dung điều luật cũ về bản chất có
     đúng ngữ nghĩa hay không — vì model đã dùng dữ liệu cũ, lỗi thời.
   - Cách xác định hiệu lực: search các trang tra cứu luật
     (thuvienphapluat.vn, luatvietnam.vn, vanban.chinhphu.vn) — các trang
     này có nhãn "Tình trạng: Còn hiệu lực / Hết hiệu lực..."; hoặc đọc
     điều khoản thi hành của văn bản mới nhất cùng chủ đề (thường ghi rõ
     "Nghị định này thay thế Nghị định số...").
   - **Đọc trực tiếp nội dung đầy đủ của (các) điều khoản đã trích**, đối
     chiếu với CÂU HỎI GỐC của người dùng để xác định các điều khoản đó đã
     TRẢ LỜI ĐỦ câu hỏi hay chưa. Nếu chỉ trả lời được MỘT PHẦN (hoặc hoàn
     toàn không đủ căn cứ), ghi ngắn gọn vào `ghi_chu_dieu_luat`: phần nào
     của câu hỏi đã có căn cứ trong điều luật trích dẫn, phần nào KHÔNG
     được điều luật đó đề cập tới — không cần đánh giá lại đúng/sai nội
     dung answer ở đây (việc đó thuộc tiêu chí 3), chỉ ghi nhận mức độ phủ
     của điều luật so với câu hỏi.
3. `tra_loi_dung_cau_hoi` — answer có trả lời trúng điều người dùng hỏi
   không (không né tránh, không chung chung).
4. `cau_tra_loi_ro_rang` — answer có trình bày rõ ràng, dễ hiểu với người
   không rành luật không.

Kèm field `ghi_chu_dieu_luat` (string ngắn) — tóm tắt đã verify được gì:
đặc biệt ghi rõ nếu văn bản bị thay thế thì thay thế bởi văn bản nào, VÀ
ghi rõ điều luật trích dẫn đã trả lời được phần nào / chưa trả lời được
phần nào của câu hỏi (nếu chỉ trả lời một phần hoặc không đủ căn cứ).

## Cách chạy

1. Chia số câu cần đánh giá thành các batch ~10 câu (tránh 1 agent phải
   search quá nhiều, dễ timeout/nhầm lẫn).
2. Với mỗi batch, spawn 1 agent `general-purpose` chạy song song
   (nhiều lời gọi Agent trong cùng 1 message), prompt yêu cầu:
   - Đọc đúng file JSON/Excel đầu vào của lần chạy đó (vd
     `emlaw_crawler/emlaw_qa.json`, `emlaw_crawler/answer_dat_dai.json`...),
     chỉ xử lý đúng range `stt` được giao.
   - Áp dụng đúng 4 tiêu chí + quy tắc hiệu lực ở trên.
   - Ghi kết quả ra file JSON tại thư mục scratchpad của phiên hiện tại,
     đặt tên `qa_eval_batch_<n>.json`, định dạng:
     ```json
     [{"stt": 1, "xac_dinh_dung_van_de": true, "trich_dung_dieu_luat": false,
       "tra_loi_dung_cau_hoi": true, "cau_tra_loi_ro_rang": true,
       "ghi_chu_dieu_luat": "..."}]
     ```
3. Sau khi tất cả batch xong, chạy script gộp (xem `merge_eval_and_export.py`
   — cần sửa lại đường dẫn scratchpad và range stt cho đúng phiên/lần chạy
   hiện tại vì thư mục scratchpad đổi theo mỗi phiên) để tạo file JSON/Excel
   cuối cùng gồm field cũ (`stt`, `question`, `answer`) + 5 field mới.

## Lưu ý

- Thư mục scratchpad là theo phiên (session-specific), không cố định —
  luôn lấy đường dẫn scratchpad hiện tại thay vì hard-code đường dẫn cũ.
- Nếu chạy lại toàn bộ 91 câu, nên chia 6 batch (~15 câu/batch) để cân bằng
  giữa tốc độ và độ chính xác của từng agent.
- Không tái sử dụng kết quả đánh giá cũ nếu prompt tiêu chí đã thay đổi
  (ví dụ thêm rule hiệu lực văn bản) — phải chạy lại từ đầu để nhất quán.
