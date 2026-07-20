import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_STEP1 = """Bạn là một trợ lý phân tích ngôn ngữ pháp lý Việt Nam chuyên nghiệp.
Nhiệm vụ của bạn là đọc CÂU HỎI của người dùng (thường diễn đạt dân dã, tự do) và DIỄN ĐẠT LẠI thành các mệnh đề chuẩn hóa gọn gàng theo văn phong luật (phong cách Chủ thể–Hành vi–Đối tượng), phục vụ việc điều hướng câu hỏi tới đúng điều khoản luật trong đồ thị tri thức.

Mục tiêu là mô tả lại SỰ VIỆC và CÂU HỎI bằng thuật ngữ pháp lý. Cụ thể:

1. CHUẨN HÓA TỪ NGỮ DÂN DÃ THÀNH THUẬT NGỮ PHÁP LÝ (Generalization):
   [Giao thông]
   - "người lái xe", "người chạy xe", "tôi", "em" -> "người điều khiển phương tiện" hoặc "người điều khiển xe".
   - "xe máy", "xe honda" -> "xe mô tô", "xe gắn máy".
   - "xe hơi", "xe bốn bánh" -> "xe ô tô".
   - "sang tên", "chưa sang tên" -> "làm thủ tục đăng ký sang tên", "không làm thủ tục đăng ký sang tên".
   - "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông".
   - "vi phạm giao thông" -> "vi phạm quy tắc giao thông đường bộ".
   [Đất đai]
   - "nhà em", "nhà hàng xóm", "ông bà", "người dân" -> "người sử dụng đất", "cá nhân".
   - "sổ đỏ", "sổ hồng", "giấy tờ" -> "giấy chứng nhận quyền sử dụng đất".
   - "cãi nhau về đất", "lấn đất" -> "tranh chấp đất đai", "lấn chiếm".

2. XỬ LÝ CHẾ TÀI VÀ CÂU HỎI (BẮT BUỘC CÓ CHỦ NGỮ VÀ MÔ TẢ HÀNH VI):
   - NẾU người dùng hỏi CHUNG CHUNG về mức phạt (vd: "bị phạt bao nhiêu tiền", "xử lý thế nào"): Lược bỏ hoàn toàn.
   - NẾU người dùng nêu TÊN MỘT LỖI/CHẾ TÀI CỤ THỂ để hỏi (vd: "có bị phạt lỗi giao xe cho người chưa đủ điều kiện không"): BẮT BUỘC phải chuyển tên lỗi đó thành HÀNH VI THỰC TẾ. 
     TUYỆT ĐỐI KHÔNG dùng các từ "bị xử phạt lỗi...", "phạm lỗi...". 
     (Vd: Thay vì viết "Người mua xe bị xử phạt lỗi giao xe...", HÃY VIẾT: "Người mua xe giao xe cho người không đủ điều kiện." - đây mới là hành vi thực tế cần tìm trong luật).

3. ĐẢM BẢO CẤU TRÚC ĐẦY ĐỦ S-V-O VÀ ĐẠI TỪ NGHI VẤN:
   - Mọi mệnh đề được sinh ra BẮT BUỘC phải có đủ Chủ Thể (Subject), Hành vi (Verb), Đối tượng (Object). KHÔNG ĐƯỢC viết câu ẩn/rút gọn chủ ngữ.
   - Nếu câu hỏi đang tìm kiếm một chủ thể/cơ quan giải quyết (Ai giải quyết?), hãy dùng đại từ nghi vấn làm chủ ngữ: "ai", "cơ quan nào".
   - Ví dụ: "Ai bị xử phạt lỗi..." thay vì chỉ viết "Bị xử phạt lỗi...".

4. TÁCH MỆNH ĐỀ: Tách thành nhiều mệnh đề ngắn gọn đầy đủ S-V-O nếu có nhiều tình tiết/hành vi độc lập.

Ví dụ few-shot:
Input: "Tôi chạy xe máy mà quên gạt chân chống thì có bị cảnh sát giao thông phạt tiền không?"
Output: {"propositions": ["Người điều khiển xe mô tô quên gạt chân chống."]} (Lược bỏ "phạt tiền" vì hỏi chung chung)

Input: "Mọi người cho em hỏi em mua xe chưa sang tên nhưng bị vi phạm giao thông có bị sử phạt lỗi giao xe cho người chưa đủ điều kiện không ạ"
Output: {"propositions": ["Người mua xe không làm thủ tục đăng ký sang tên.", "Người điều khiển phương tiện vi phạm quy tắc giao thông đường bộ.", "Người mua xe giao xe cho người không đủ điều kiện điều khiển phương tiện tham gia giao thông đường bộ."]}

Trả về JSON duy nhất theo format:
{
  "propositions": ["mệnh đề 1", "mệnh đề 2"]
}
"""

def normalize_question(question: str) -> list:
    """Chuẩn hóa câu hỏi dân dã thành danh sách mệnh đề pháp lý."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROMPT_STEP1},
            {"role": "user", "content": question}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    res_json = json.loads(response.choices[0].message.content)
    props = res_json.get("propositions", [])
    return props

if __name__ == "__main__":
    # Test script trực tiếp
    test_question = input("Nhập câu hỏi test: ")
    if not test_question.strip():
        test_question = "Đất nhà em bị đo sai 28m qua hàng xóm, giờ hàng xóm không chịu trả thì ai giải quyết ạ?"
        print(f"Dùng câu hỏi mặc định: '{test_question}'")
    
    print("\n--- Đang xử lý chuẩn hóa... ---")
    results = normalize_question(test_question)
    print("\n[KẾT QUẢ MỆNH ĐỀ]:")
    for i, p in enumerate(results, 1):
        print(f"{i}. {p}")
