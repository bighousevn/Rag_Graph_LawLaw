import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Đóng vai trò: Bạn là một hệ thống Trích xuất Bộ ba Tri thức (Knowledge Triplet Extractor) SIÊU TRỪU TƯỢNG, chuyên xử lý CÂU HỎI của người dùng cho hệ thống GraphRAG.

Mục tiêu: Đưa các tình huống cụ thể của người dùng về CÙNG MỘT MẶT BẰNG TRỪU TƯỢNG với cơ sở dữ liệu đồ thị luật pháp để thuật toán so khớp (Graph Matching) hoạt động chính xác 100%.

!!! CẢNH BÁO TỐI THƯỢNG VỀ TRỪU TƯỢNG HÓA !!!
Nhiệm vụ quan trọng nhất của bạn là TRỪU TƯỢNG HÓA TỘT ĐỘ các thành phần Subject (s), Verb (v), Object (o).
- TUYỆT ĐỐI KHÔNG dùng từ ngữ cụ thể của người dùng (VD: "ông Nguyễn Văn A", "công an phường", "sổ đỏ", "xe máy", "Luật đất đai") để làm "s" hoặc "o".
- Chủ thể (s) và Đối tượng (o) CHỈ ĐƯỢC PHÉP LÀ MỘT KHÁI NIỆM SIÊU LỚP (Superclass) đại diện chung nhất (VD: cơ quan, tổ chức, người, tài sản, thủ tục, hành vi, hình phạt, giấy phép, quyền lợi...).
- BỎ QUA MỌI TỪ BỔ SUNG NGỮ NGHĨA VÀ TỪ PHỦ ĐỊNH: Chỉ giữ lại danh từ/động từ lõi. Tuyệt đối không kèm theo tính từ, trạng từ, từ phủ định, hoặc phương hướng. (VD: "địa điểm khác" -> "địa điểm"; "không vi phạm" -> "vi phạm"; "mức phạt nặng" -> "mức phạt"; "cấp lại" -> "cấp").
- Toàn bộ các giá trị s, v, o phải được viết thường (lowercase) và KHÔNG chứa dấu chấm hỏi (?).

QUY TRÌNH TƯ DUY BẮT BUỘC (Thực hiện tuần tự 2 bước):

--- BƯỚC 1: TRỪU TƯỢNG HÓA & LÀM SẠCH NGỮ NGHĨA (Tạo normalized_text) ---
1. Áp dụng Siêu lớp: Đọc từng thực thể, tự hỏi nó thuộc nhóm chung nào trong luật? Thay thế tên riêng bằng siêu lớp (VD: "UBND Quận" -> "cơ quan", "văn hóa phẩm" -> "tài sản", "M" -> "người").
2. Lọc nhiễu: Bỏ qua mọi yếu tố thời gian, địa điểm, cảm xúc.
3. Xác định Ẩn số (Target): Tự hỏi người dùng muốn tìm kiếm điều gì? (VD: tìm "mức phạt", "tính hợp pháp"). Đưa ẩn số này thành một đối tượng siêu lớp.
4. Diễn đạt lại: Viết lại thành một đoạn văn ngắn gọn, logic chỉ chứa các siêu lớp, kết thúc bằng câu hỏi tìm ẩn số.

--- BƯỚC 2: TRÍCH XUẤT TRIPLET TỪ CÂU ĐÃ CHUẨN HÓA (S-V-O) ---
Từ chính đoạn "normalized_text" vừa tạo ở Bước 1, tiến hành tách Triplet:
1. Động từ gốc và Danh từ gốc: Lấy các cụm từ hành động/thực thể, gọt sạch các đuôi bổ ngữ ("thuộc về" -> "thuộc", "cấp lại" -> "cấp", "không đồng ý" -> "đồng ý"). Điền vào "s", "v", "o".
2. Khai thác triệt để: Chẻ nhỏ văn bản thành các đơn vị ý nghĩa nhỏ nhất. Một câu dài phải tách thành nhiều mối quan hệ.
3. Lắp ghép: (s) - [v] -> (o).

--- ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC (Không giải thích thêm) ---
{
  "normalized_text": "cơ quan phát hiện người thực hiện hành vi liên quan tài sản. cơ quan lập văn bản và ban hành quyết định áp dụng biện pháp tiền và tiêu hủy tài sản. xác định tính hợp pháp của quyết định.",
  "target": "tính hợp pháp",
  "triplets": [
    { "s": "cơ quan", "v": "phát hiện", "o": "hành vi" },
    { "s": "cơ quan", "v": "lập", "o": "văn bản" },
    { "s": "văn bản", "v": "ghi nhận", "o": "hành vi" },
    { "s": "người", "v": "thực hiện", "o": "hành vi" },
    { "s": "cơ quan", "v": "ban hành", "o": "quyết định" },
    { "s": "quyết định", "v": "xử phạt", "o": "người" },
    { "s": "quyết định", "v": "áp dụng", "o": "biện pháp" },
    { "s": "biện pháp", "v": "gồm", "o": "tiền" },
    { "s": "biện pháp", "v": "gồm", "o": "tiêu hủy" },
    { "s": "tiêu hủy", "v": "tác động", "o": "tài sản" },
    { "s": "quyết định", "v": "có", "o": "tính hợp pháp" }
  ]
}
"""


def load_posts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    posts = payload.get("posts", [])
    if not isinstance(posts, list):
        raise ValueError("Trường 'posts' không đúng định dạng danh sách.")
    return payload, posts


def process_graph_extraction(question_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
        ],
    )
    return json.loads(response.choices[0].message.content)


def save_progress(output_file, meta, results, processed_count, total_count):
    payload = {
        "source_file": meta["source_file"],
        "search_query": meta.get("search_query", ""),
        "crawled_at": meta.get("crawled_at", ""),
        "total_posts_in_source": total_count,
        "processed_posts": processed_count,
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def main():
    input_file = "./fb_crawler/crawled_posts.json"
    output_file = "./fb_crawler/extracted_posts_batch.json"

    if not API_KEY:
        print("❌ Lỗi: Chưa tìm thấy OPENAI_API_KEY trong môi trường.")
        return

    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    try:
        source_payload, posts = load_posts(input_file)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file đầu vào: {e}")
        return

    total_posts = len(posts)
    print(f"📥 Đã nạp {total_posts} bài viết từ: {input_file}")
    print("🔍 Bắt đầu chuẩn hóa và trích xuất triplet hàng loạt...")

    results = []
    meta = {
        "source_file": input_file,
        "search_query": source_payload.get("search_query", ""),
        "crawled_at": source_payload.get("crawled_at", ""),
    }

    for index, post in enumerate(posts, start=1):
        post_id = post.get("id")
        content = (post.get("content") or "").strip()

        print(f"\n{'=' * 60}")
        print(f"▶ Đang xử lý bài viết {index}/{total_posts} | post_id={post_id}")

        if not content:
            print("⚠️ Bỏ qua vì content rỗng.")
            results.append({
                "post_id": post_id,
                "source_url": post.get("source_url", ""),
                "crawled_at": post.get("crawled_at", ""),
                "original_content": content,
                "normalized_text": "",
                "target": "",
                "triplets": [],
                "status": "skipped_empty_content",
            })
            save_progress(output_file, meta, results, index, total_posts)
            continue

        try:
            result = process_graph_extraction(content)
            normalized_text = result.get("normalized_text", "")
            target = result.get("target", "")
            triplets = result.get("triplets", [])

            results.append({
                "post_id": post_id,
                "source_url": post.get("source_url", ""),
                "crawled_at": post.get("crawled_at", ""),
                "original_content": content,
                "normalized_text": normalized_text,
                "target": target,
                "triplets": triplets,
                "status": "success",
            })

            print(f"✅ Xử lý thành công | target: {target}")
            print(f"   normalized_text: {normalized_text[:160]}{'...' if len(normalized_text) > 160 else ''}")
            print(f"   số lượng triplet: {len(triplets)}")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý post_id={post_id}: {e}")
            results.append({
                "post_id": post_id,
                "source_url": post.get("source_url", ""),
                "crawled_at": post.get("crawled_at", ""),
                "original_content": content,
                "normalized_text": "",
                "target": "",
                "triplets": [],
                "status": "error",
                "error": str(e),
            })

        # Ghi đè file kết quả sau mỗi bài để tiện giám sát tiến độ
        save_progress(output_file, meta, results, index, total_posts)
        print(f"💾 Đã cập nhật tiến độ vào: {output_file}")

    print(f"\n✅ Hoàn tất xử lý {len(results)} bài viết.")
    print(f"📄 File kết quả cuối cùng: {output_file}")


if __name__ == "__main__":
    main()
