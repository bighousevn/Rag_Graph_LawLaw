import os
import json
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Đường dẫn tới file EXTRACTION_RULES.md
RULES_PATH = os.path.join(os.path.dirname(BASE_DIR), "Building_KG", "ontology", "EXTRACTION_RULES.md")

try:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        EXTRACTION_RULES = f.read()
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {RULES_PATH}")
    EXTRACTION_RULES = ""

PROMPT_STEP2_BASE = """Bạn là chuyên gia trích xuất triplet ngữ nghĩa cho lĩnh vực pháp lý.
Với danh sách các mệnh đề pháp lý đã chuẩn hoá, hãy trích xuất ĐẦY ĐỦ các triplet (subject, verb, object) nguyên tử.
Mục đích cuối cùng là dựng các triplet truy vấn để vector search vào đồ thị tri thức. 

TUYỆT ĐỐI TUÂN THỦ TÀI LIỆU EXTRACTION RULES SAU ĐÂY:
"""

PROMPT_STEP2 = PROMPT_STEP2_BASE + "\n" + EXTRACTION_RULES + """

NẾU thành phần nào là ẩn số cần tìm (đại từ nghi vấn như ai, gì, thế nào), HÃY THAY BẰNG DẤU "*".
Chú ý: Bạn KHÔNG cần phải tạo ra các trường keyphrases (s_keyphrases, v_keyphrases, o_keyphrases) cho query. 
CHỈ CẦN TRẢ VỀ s, v, o CHO MỖI TRIPLET LÀ ĐỦ (Bỏ các field keyphrases khỏi kết quả).

Trả về JSON duy nhất theo format:
{
  "triplets": [
    {"s": "Chủ thể hoặc *", "v": "Hành vi", "o": "Đối tượng hoặc *"}
  ]
}
"""

def extract_triplets(propositions: list) -> list:
    """Trích xuất triplet nguyên tử từ danh sách mệnh đề pháp lý."""
    text_content = "\n".join(propositions)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROMPT_STEP2},
            {"role": "user", "content": text_content}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    res_json = json.loads(response.choices[0].message.content)
    triplets = res_json.get("triplets", [])
    return triplets

if __name__ == "__main__":
    # Test script trực tiếp
    test_props = [
        "Người sử dụng đất đo đạc ranh giới sai lệch.",
        "Người sử dụng đất tranh chấp đất đai.",
        "Ai giải quyết tranh chấp đất đai."
    ]
    
    print("\n--- Đang xử lý trích xuất Triplets từ các mệnh đề: ---")
    for i, p in enumerate(test_props, 1):
        print(f"{i}. {p}")
        
    results = extract_triplets(test_props)
    
    print("\n[KẾT QUẢ TRIPLETS]:")
    for t in results:
        print(f"({t.get('s')}) - [{t.get('v')}] -> ({t.get('o')})")
