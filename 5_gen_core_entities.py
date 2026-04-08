import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. CẤU HÌNH API KEY CỦA OPENAI (Bắt đầu bằng sk-...)
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("LỖI: Vui lòng thiết lập biến môi trường OPENAI_API_KEY với API key của bạn.")
    exit(1)

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Bạn là một Chuyên gia thiết kế Kiến trúc Dữ liệu Đồ thị (Knowledge Graph Engineer) và Phân tích Pháp lý.
Nhiệm vụ của bạn là đọc phần văn bản pháp luật được cung cấp và trích xuất ra một bộ "Từ điển Thực thể Lõi" (Core Entities) cực kỳ tinh gọn.

YÊU CẦU CỐT LÕI (PHẢI TUÂN THỦ TUYỆT ĐỐI):
1. Tính Nguyên tử (Atomic): Các thực thể phải được chia nhỏ thành các hạt cơ bản nhất, ngắn gọn, súc tích, đơn nghĩa.
2. Chống trùng lặp (Semantic Deduplication): Nếu nhiều cụm từ trong luật có cùng một bản chất, bạn CHỈ ĐƯỢC TẠO 1 OBJECT. Tên chuẩn ngắn gọn nhất để ở trường name, tất cả các cách gọi dài dòng, từ viết tắt, từ đồng nghĩa phải nhét hết vào mảng synonyms.
3. Phân loại Rõ ràng:
- Nodes (Nút): Chủ thể, Đối tượng, Định lượng, Nơi chốn. Mã ID: N01, N02...
- Relationships (Mối quan hệ): Động từ hành động. Mã ID: R01, R02...):

FORMAT ĐẦU RA BẮT BUỘC (ONLY JSON): Trả về JSON theo đúng cấu trúc sau:
{
    "nodes": [
        {
            "id": "N01",
            "name": "ủy ban xã",
            "synonyms": [
                "Chủ tịch Ủy ban nhân dân cấp xã",
                "Chủ tịch UBND cấp xã",
                "chủ tịch phường",
                "chủ tịch thị trấn"
            ]
        },
        {
            "id": "N02",
            "name": "công an xã",
            "synonyms": [
                "Trưởng Công an cấp xã",
                "Trưởng đồn Công an",
                "Trạm trưởng Trạm Công an cửa khẩu"
            ]
        },
        {
            "id": "N03",
            "name": "tang vật",
            "synonyms": [
                "tang vật vi phạm hành chính",
                "vật vi phạm"
            ]
        },
        {
            "id": "N04",
            "name": "5 triệu",
            "synonyms": [
                "5.000.000 đồng"
            ]
        }
    ],
    "relationships": [
        {
            "id": "R01",
            "name": "phạt",
            "synonyms": [
                "xử phạt",
                "phạt tiền",
                "áp dụng hình thức phạt"
            ]
        },
        {
            "id": "R02",
            "name": "tịch thu",
            "synonyms": [
                "thu giữ",
                "tịch thu sung quỹ"
            ]
        },
        {
            "id": "R03",
            "name": "tháo dỡ",
            "synonyms": [
                "buộc tháo dỡ"
            ]
        }
    ]
}
"""

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_data(data, chunk_size=30):
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk_items = data[i:i + chunk_size]
        text_chunk = "\n".join([item.get("original_text", "") for item in chunk_items])
        chunks.append(text_chunk)
    return chunks

def process_chunk(text_chunk, chunk_index, total_chunks):
    print(f"\n--- Đang xử lý chunk {chunk_index}/{total_chunks} bằng gpt-4o-mini ---")

    try:
        # 2. CÚ PHÁP GỌI API MỚI THEO CHUẨN OPENAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" }, # Ép OpenAI trả về JSON 100%
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"ĐÂY LÀ PHẦN VĂN BẢN CẦN TRÍCH XUẤT:\n{text_chunk}"}
            ]
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        return result

    except Exception as e:
        print(f"Lỗi ở chunk {chunk_index}: {e}")
        return {"nodes": [], "relationships": []}

def merge_graphs(all_graphs):
    master_nodes = []
    master_rels = []
    node_names_seen = set()
    rel_names_seen = set()
    node_counter = 1
    rel_counter = 1

    for graph in all_graphs:
        for node in graph.get("nodes", []):
            name = node.get("name", "").lower()
            if name and name not in node_names_seen:
                node_names_seen.add(name)
                node["id"] = f"N{node_counter:03d}"
                master_nodes.append(node)
                node_counter += 1
            elif name in node_names_seen:
                for existing_node in master_nodes:
                    if existing_node["name"].lower() == name:
                        existing_node["synonyms"] = list(set(existing_node.get("synonyms", []) + node.get("synonyms", [])))
                        break

        for rel in graph.get("relationships", []):
            name = rel.get("name", "").lower()
            if name and name not in rel_names_seen:
                rel_names_seen.add(name)
                rel["id"] = f"R{rel_counter:03d}"
                master_rels.append(rel)
                rel_counter += 1
            elif name in rel_names_seen:
                for existing_rel in master_rels:
                    if existing_rel["name"].lower() == name:
                        existing_rel["synonyms"] = list(set(existing_rel.get("synonyms", []) + rel.get("synonyms", [])))
                        break

    return {"nodes": master_nodes, "relationships": master_rels}

def main():
    input_file = './output/2_sentences.json'
    output_file = './output/master_graph.json'

    if not os.path.exists(input_file):
        print(f"LỖI: Không tìm thấy file '{input_file}'.")
        return

    print("Đang tải dữ liệu...")
    raw_data = load_data(input_file)

    chunks = chunk_data(raw_data, chunk_size=30)
    total_chunks = len(chunks)
    print(f"Đã chia thành {total_chunks} khối dữ liệu.")

    all_graphs = []
    start_time = time.time()

    for i, chunk in enumerate(chunks, 1):
        graph_data = process_chunk(chunk, i, total_chunks)
        all_graphs.append(graph_data)

        print(f"Kết quả của chunk {i}:")
        print(json.dumps(graph_data, ensure_ascii=False, indent=2))

        # Để an toàn cho tài khoản mới nạp tiền, nghỉ 1 giây giữa các request
        time.sleep(1)

    print("\nĐang hợp nhất và khử trùng lặp (Deduplication)...")
    master_graph = merge_graphs(all_graphs)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(master_graph, f, ensure_ascii=False, indent=4)

    print(f"Hoàn tất! Kết quả đã được lưu tại: {output_file}")
    print(f"Tổng số Nodes: {len(master_graph['nodes'])}")
    print(f"Tổng số Relationships: {len(master_graph['relationships'])}")
    print(f"Thời gian chạy: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()