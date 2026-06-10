import argparse
import glob
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


SYSTEM_PROMPT = """
Bạn là bộ phân loại dữ liệu cho bộ sưu tập câu hỏi pháp lý.

Nhiệm vụ:
1. Xác định domain pháp luật mà nội dung câu hỏi thuộc về, nều câu hỏi cần luật ở nhiều domain để giải quyết thì gán là "liên miền".
2. Xác định suitable = true/false để lọc dữ liệu đầu vào.

Quy tắc gán suitable:
- true: nội dung là câu hỏi của người dùng hoặc mô tả tình huống pháp lý đủ ngữ cảnh, có thể dùng để thu thập dữ liệu hỏi đáp.
- false: nội dung không phải câu hỏi, chỉ là bình luận/trả lời, quảng cáo, tin tức, hoặc thiếu ngữ cảnh đến mức không hiểu người hỏi đang cần gì.

Quy tắc lọc:
- Chỉ giữ lại khi nội dung chính là câu hỏi của người dùng.
- Nếu chỉ có một câu nói ngắn, tiêu đề, cảm thán, từ khóa rời rạc, hoặc không có dữ kiện đủ để đặt câu hỏi pháp lý thì loại bỏ.
- Nếu câu hỏi quá chung chung, thiếu chủ thể, thiếu sự việc, thiếu thời điểm hoặc thiếu hoàn cảnh thì loại bỏ
- Nếu câu hỏi chỉ là bình luận, trả lời cho một câu hỏi khác, hoặc chỉ là quảng cáo, tin tức thì loại bỏ.

Quy tắc gán domain:
- Chọn 1 domain chính, ngắn gọn, chữ thường, bằng tiếng Việt.
- Nếu câu hỏi liên quan đến nhiều domain chọn là "liên miền"

Các domain gợi ý:
- lao động
- bảo hiểm xã hội
- hành chính
- dân sự
- hình sự
- đất đai
- hôn nhân gia đình
- giao thông
- tố tụng
- kinh tế
- tài chính - ngân hàng - thuế
- sở hữu trí tuệ
- giáo dục
- y tế
- liên miền

Chỉ trả về JSON hợp lệ đúng cấu trúc sau, không thêm giải thích:
{
  "domain": "lao động",
    "suitable": true
}
"""


def find_latest_input_file(folder_path: str) -> str | None:
    candidates = glob.glob(os.path.join(folder_path, "crawled_posts*.json"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_posts(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    posts = payload.get("posts", [])
    if not isinstance(posts, list):
        raise ValueError("Trường 'posts' phải là một danh sách.")

    return payload, posts


def classify_content(client: OpenAI, content: str, model: str):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )

    result = json.loads(response.choices[0].message.content)
    domain = str(result.get("domain", "khác")).strip().lower() or "khác"
    suitable = bool(result.get("suitable", False))

    return {
        "domain": domain,
        "suitable": suitable,
    }


def should_keep_post(classification: dict) -> bool:
    return bool(classification.get("suitable", False))


def build_output(results, source_file: str, total_posts: int, rejected_count: int):
    return {
        "source_file": source_file,
        "processed_at": datetime.now().isoformat(),
        "total_input_posts": total_posts,
        "kept_posts": len(results),
        "rejected_posts": rejected_count,
        "results": results,
    }


def main():
    # ===== PHẦN CẤU HÌNH =====
    # TODO: Cập nhật đường dẫn file input và output của bạn tại đây
    input_file = "./fb_crawler/crawled_posts.json"
    output_file = "./fb_crawler/classified_posts.json"
    # =========================

    parser = argparse.ArgumentParser(
        description="Phân loại bài đăng Facebook theo domain pháp lý và suitable."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Tên model OpenAI dùng để phân loại.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Tên biến môi trường chứa OpenAI API key.",
    )
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"❌ Không tìm thấy API key trong biến môi trường {args.api_key_env}.")
        return

    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file đầu vào: {input_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    source_payload, posts = load_posts(input_file)
    client = OpenAI(api_key=api_key)

    total_posts = len(posts)
    print(f"📥 Đã nạp {total_posts} bài viết từ: {input_file}")
    print(f"🤖 Dùng model: {args.model}")

    results = []
    rejected_count = 0

    for index, post in enumerate(posts, start=1):
        post_id = post.get("id")
        content = str(post.get("content", "")).strip()

        print(f"\n{'=' * 60}")
        print(f"▶ Đang xử lý {index}/{total_posts} | id={post_id}")

        if not content:
            rejected_count += 1
            print("⚠️ Bỏ qua vì content rỗng.")
        else:
            try:
                classification = classify_content(client, content, args.model)
                if should_keep_post(classification):
                    results.append(
                        {
                            "id": post_id,
                            "content": content,
                            "domain": classification["domain"],
                            "suitable": True,
                        }
                    )
                    print(f"✅ GIỮ | domain={classification['domain']}")
                else:
                    rejected_count += 1
                    print(f"❌ LOẠI | domain={classification['domain']}")
            except Exception as exc:
                rejected_count += 1
                print(f"❌ Lỗi khi phân loại id={post_id}: {exc}")

        output_payload = build_output(results, input_file, total_posts, rejected_count)
        with open(output_file, "w", encoding="utf-8") as file_handle:
            json.dump(output_payload, file_handle, ensure_ascii=False, indent=2)

    print(f"\n✅ Hoàn tất. Đã lưu kết quả tại: {output_file}")
    print(f"📌 Giữ lại: {len(results)} / {len(posts)}")
    print(f"📌 Loại bỏ: {rejected_count}")
    print(f"📎 Nguồn: {source_payload.get('search_query', '')}")


if __name__ == "__main__":
    main()