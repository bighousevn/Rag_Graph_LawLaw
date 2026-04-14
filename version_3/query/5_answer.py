import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

QUESTION_FILE = "./version_3/query/1_question.txt"
FILTERED_TRIPLETS_FILE = "./version_3/query/5_filtered_triplets.json"
SENTENCES_FILE = "./version_3/1_sections.json"
OUTPUT_FILE = "./version_3/query/6_irac_answer.json"

SYSTEM_PROMPT = """
Bạn là chuyên gia phân tích pháp lý Việt Nam.
Nhiệm vụ: Dựa CHỈ trên câu hỏi của người dùng và các section pháp lý được cung cấp, hãy trả lời theo chuẩn IRAC.

YÊU CẦU:
1. Chỉ sử dụng thông tin nằm trong các section được cung cấp.
2. Không viện dẫn ngoài dữ liệu đầu vào.
3. Nếu dữ liệu chưa đủ để khẳng định chắc chắn, phải nói rõ giới hạn đó trong phần application hoặc conclusion.
4. Lập luận ngắn gọn, rõ ràng, dễ hiểu cho người không chuyên.
5. Ở phần rule và application, khi nhắc đến căn cứ, hãy ưu tiên dẫn `section_id`.

ĐỊNH DẠNG JSON BẮT BUỘC:
{
    "issue": "Nêu vấn đề pháp lý cốt lõi của câu hỏi.",
    "rule": "Tóm tắt quy tắc pháp lý áp dụng từ các section liên quan.",
    "application": "Áp dụng quy tắc vào tình huống người dùng nêu.",
    "conclusion": "Kết luận trực tiếp, rõ ràng cho người dùng."
}
"""


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_question(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_relevant_section_ids(file_path):
    data = load_json(file_path)
    if isinstance(data, list) and data:
        data = data[0]

    if not isinstance(data, dict):
        return []

    section_ids = data.get("relevant_section_ids", [])
    if not isinstance(section_ids, list):
        return []

    return [str(item).strip() for item in section_ids if str(item).strip()]


def build_section_lookup(sentences_data):
    lookup = {}
    for item in sentences_data:
        if not isinstance(item, dict):
            continue

        section_id = str(item.get("section_id", "")).strip()
        if not section_id:
            continue

        # Giữ bản ghi đầu tiên xuất hiện cho mỗi section_id để tránh dữ liệu lặp.
        if section_id not in lookup:
            lookup[section_id] = {
                "section_id": section_id,
                "original_text": str(item.get("original_text", "")).strip(),
                "sentences": item.get("sentences", []) if isinstance(item.get("sentences", []), list) else [],
            }

    return lookup


def collect_relevant_sections(section_ids, section_lookup):
    relevant_sections = []
    for sid in section_ids:
        section = section_lookup.get(sid)
        if section:
            relevant_sections.append(section)
    return relevant_sections


def build_user_prompt(question, relevant_sections):
    section_blocks = []

    for section in relevant_sections:
        sid = section["section_id"]
        original_text = section.get("original_text", "")
        sentences = section.get("sentences", [])
        sentence_text = " ".join(str(s).strip() for s in sentences if str(s).strip())

        block = (
            f"[SECTION_ID: {sid}]\n"
            f"ORIGINAL_TEXT: {original_text}\n"
            f"SENTENCES: {sentence_text}"
        )
        section_blocks.append(block)

    joined_sections = "\n\n".join(section_blocks)

    return (
        f"Câu hỏi người dùng:\n{question}\n\n"
        f"Các section pháp lý liên quan:\n{joined_sections}"
    )


def generate_irac_answer(question, relevant_sections):
    user_prompt = build_user_prompt(question, relevant_sections)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    return json.loads(content)


def normalize_irac(raw_result):
    return {
        "issue": str(raw_result.get("issue", "")).strip(),
        "rule": str(raw_result.get("rule", "")).strip(),
        "application": str(raw_result.get("application", "")).strip(),
        "conclusion": str(raw_result.get("conclusion", "")).strip(),
    }


def main():
    for file_path in [QUESTION_FILE, FILTERED_TRIPLETS_FILE, SENTENCES_FILE]:
        if not os.path.exists(file_path):
            print(f"Lỗi: Không tìm thấy file {file_path}")
            return

    question = load_question(QUESTION_FILE)
    if not question:
        print("Lỗi: Không đọc được câu hỏi hợp lệ từ 1_question.txt")
        return

    relevant_section_ids = load_relevant_section_ids(FILTERED_TRIPLETS_FILE)
    if not relevant_section_ids:
        print("Lỗi: Không tìm thấy relevant_section_ids trong 3_filtered_triplets.json")
        return

    sentences_data = load_json(SENTENCES_FILE)
    if not isinstance(sentences_data, list):
        print("Lỗi: 1_input_sentences.json không đúng định dạng mảng.")
        return

    section_lookup = build_section_lookup(sentences_data)
    relevant_sections = collect_relevant_sections(relevant_section_ids, section_lookup)

    if not relevant_sections:
        print("Lỗi: Không ánh xạ được section nào từ relevant_section_ids sang 1_input_sentences.json")
        return

    print(f"Đang tổng hợp {len(relevant_sections)} section liên quan và gọi OpenAI...")

    try:
        raw_irac = generate_irac_answer(question, relevant_sections)
        irac = normalize_irac(raw_irac)
    except Exception as e:
        print(f"Lỗi khi gọi OpenAI hoặc chuẩn hóa IRAC: {e}")
        return

    output_data = {
        "question": question,
        "relevant_section_ids": relevant_section_ids,
        "relevant_sections": relevant_sections,
        "irac": irac,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Hoàn tất! Đã lưu kết quả IRAC tại: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
