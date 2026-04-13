import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Bạn là chuyên gia trích xuất thực thể cho hệ thống Legal Graph về Luật Việt Nam.
Nhiệm vụ: Đọc 1 câu hỏi của người dùng và tách ra các entity/hành động ngắn gọn để phục vụ truy vấn ngữ nghĩa.
Bước đầu tiên hãy chuẩn hóa câu hỏi về dạng thuần túy, loại bỏ các thông tin thừa thãi, diễn giải dài dòng, chỉ giữ lại phần cốt lõi liên quan đến mục đích truy vấn, chuẩn hóa các từ viết tắt.

CHIẾN LƯỢC TÁCH ENTITY:
1. Tách các thực thể nguyên tử, ngắn gọn, súc tích, đơn nghĩa.
2. Giữ lại cả danh từ và động từ/hành động quan trọng nếu chúng là ý truy vấn chính.
3. Nếu nhiều cụm từ cùng bản chất, chỉ giữ 1 tên chuẩn ngắn gọn nhất.
4. Ưu tiên chữ thường, không diễn giải dài dòng, không thêm thông tin không có trong câu hỏi.
5. Không tạo entity rỗng, không lặp nghĩa, không thêm quan hệ.

ĐỊNH DẠNG JSON BẮT BUỘC:
{
    "entities": [
        {"name": "tên entity 1"},
        {"name": "tên entity 2"}
    ]
}
"""


def load_question(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        question = str(data.get("question", "")).strip()
    else:
        question = ""

    return question


def extract_entities(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    content = response.choices[0].message.content
    return json.loads(content)


def normalize_entities(raw_result):
    raw_entities = raw_result.get("entities", [])
    normalized_names = []
    seen = set()

    if not isinstance(raw_entities, list):
        raw_entities = []

    for item in raw_entities:
        name = ""

        if isinstance(item, dict):
            name = str(item.get("name", "")).strip().lower()
        elif isinstance(item, str):
            name = item.strip().lower()

        if not name or name in seen:
            continue

        seen.add(name)
        normalized_names.append(name)

    return [
        {
            "id": f"T{index:02d}",
            "name": name,
        }
        for index, name in enumerate(normalized_names, start=1)
    ]


def main():
    input_file = "./version_2/query/1_question.json"
    output_file = "./version_2/query/1_entities_question.json"

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    question = load_question(input_file)
    if not question:
        print("Lỗi: File câu hỏi không có trường 'question' hợp lệ.")
        return

    print("Đang trích xuất entities từ câu hỏi...")

    try:
        raw_result = extract_entities(question)
        final_entities = normalize_entities(raw_result)
    except Exception as e:
        print(f"Lỗi khi gọi OpenAI hoặc chuẩn hóa dữ liệu: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_entities, f, ensure_ascii=False, indent=4)

    print(f"Hoàn tất! Đã lưu {len(final_entities)} entities vào: {output_file}")


if __name__ == "__main__":
    main()
