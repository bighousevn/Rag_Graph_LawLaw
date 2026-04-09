import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 1. CẤU HÌNH API
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Bạn là một chuyên gia Kiến trúc Đồ thị Pháp lý chuyên sâu về Luật Việt Nam.
Nhiệm vụ: Trích xuất các thực thể (Nodes) và mối quan hệ (Relationships) dưới dạng bộ ba (Subject-Verb-Object).

YÊU CẦU CHIẾN THUẬT:
1. Thực thể (Nodes): Trích xuất tên ngắn gọn,súc tích, đơn nghĩa, nguyên tử (Ví dụ: "ủy ban xã", "phạt", "tang vật","tịch thu", "5 triệu","quy định","sử dụng",...).
2. Quan hệ (Relationships): Trích xuất hành động kết nối 2 thực thể.
3. Khớp SID: Văn bản sẽ có dạng [SID: s123]. Bạn PHẢI gán SID này vào trường "sids" của cả Node và Relationship. nếu đã tồn tại Node hoặc Relationship nào đó đã có SID trùng với SID này, bạn phải gán thêm SID này vào trường "sids" của Node hoặc Relationship đó. Điều này giúp đảm bảo rằng khi gộp các Node hoặc Relationship, chúng ta có thể dựa vào SID để xác định chúng có liên quan đến cùng một đoạn văn bản hay không.
4. Cấu trúc quan hệ: Phải chỉ rõ "source" (ID node nguồn) và "target" (ID node đích) dựa trên danh sách nodes bạn vừa liệt kê.
5. Chống trùng lặp (Semantic Deduplication): Nếu nhiều cụm từ trong luật có cùng một bản chất, bạn CHỈ ĐƯỢC TẠO 1 OBJECT. Tên chuẩn ngắn gọn nhất để ở trường name, tất cả các cách gọi dài dòng, từ viết tắt, từ đồng nghĩa phải nhét hết vào mảng synonyms.
6. Đảm bảo tính toàn vẹn sids: khi bạn thấy entities tương đồng, khi gộp lại bạn phải lấy hợp tất cả các SID của những entities đó để gán vào trường sids của node đã gộp. Tương tự với relationships, khi gộp phải lấy hợp tất cả SID của những relationships đó để gán vào trường sids của relationship đã gộp.
FORMAT ĐẦU RA BẮT BUỘC (JSON):
{
    "nodes": [{"id": "T01", "name": "tên chuẩn", "synonyms": [], "sids": ["s1"]}],
    "relationships": [{"name": "hành động", "source": "T01", "target": "T02", "synonyms": [], "sids": ["s1"]}]
}
"""

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_data_with_ids(data, chunk_size=15):
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk_items = data[i:i + chunk_size]
        text_chunk = ""
        for item in chunk_items:
            sid = item.get("section_id", "unknown")
            text = item.get("original_text", "")
            text_chunk += f"[SID: {sid}] {text}\n"
        chunks.append(text_chunk)
    return chunks

def process_chunk(text_chunk, chunk_index, total_chunks):
    print(f"\n{'='*30}")
    print(f"BẮT ĐẦU XỬ LÝ KHỐI {chunk_index}/{total_chunks}")
    print(f"{'='*30}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_chunk}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        print(f"✅ Đã nhận phản hồi từ AI cho khối {chunk_index}.")
        return result
    except Exception as e:
        print(f"❌ Lỗi API tại khối {chunk_index}: {e}")
        return {"nodes": [], "relationships": []}

def merge_to_data_for_graph(all_graphs):
    master_nodes = {}
    raw_edges = []
    node_counter = 1

    for graph in all_graphs:
        id_map = {}
        node_sids_in_chunk = {}
        all_sids_in_chunk = set()

        for node in graph.get("nodes", []):
            name = str(node.get("name", "")).strip().lower()
            if not name: continue

            old_id = node.get("id")
            sids = node.get("sids") if isinstance(node.get("sids"), list) else []
            syns = node.get("synonyms") if isinstance(node.get("synonyms"), list) else []

            node_sids_in_chunk[old_id] = set(sids)
            all_sids_in_chunk.update(sids)

            if name not in master_nodes:
                new_id = f"N{node_counter:03d}"
                master_nodes[name] = {
                    "id": new_id,
                    "name": name,
                    "listSectionId": set(sids),
                    "synonym": set(syns)
                }
                node_counter += 1
            else:
                master_nodes[name]["listSectionId"].update(sids)
                master_nodes[name]["synonym"].update(syns)

            # Lưu cấu trúc map để gọi lại tên node lúc ép ngược
            id_map[old_id] = {"id": master_nodes[name]["id"], "name": name}

        for rel in graph.get("relationships", []):
            old_source = rel.get("source")
            old_target = rel.get("target")

            src_info = id_map.get(old_source)
            tgt_info = id_map.get(old_target)

            if src_info and tgt_info:
                rel_sids = rel.get("sids") if isinstance(rel.get("sids"), list) else []

                # Lọc SID ảo giác từ AI
                valid_rel_sids = set(rel_sids).intersection(all_sids_in_chunk)

                if not valid_rel_sids:
                    valid_rel_sids = node_sids_in_chunk.get(old_source, set()).intersection(node_sids_in_chunk.get(old_target, set()))
                    if not valid_rel_sids:
                        valid_rel_sids = node_sids_in_chunk.get(old_source, set())

                # Ép ngược (Back-propagation): Cập nhật SID của Edge vào Nodes
                master_nodes[src_info["name"]]["listSectionId"].update(valid_rel_sids)
                master_nodes[tgt_info["name"]]["listSectionId"].update(valid_rel_sids)

                raw_edges.append({
                    "name": str(rel.get("name", "")).strip().lower(),
                    "source": src_info["id"],
                    "target": tgt_info["id"],
                    "synonym": rel.get("synonyms", []),
                    "sids": list(valid_rel_sids)
                })

    master_edges = {}
    edge_counter = 1
    for edge in raw_edges:
        key = (edge["source"], edge["target"], edge["name"])
        if key not in master_edges:
            master_edges[key] = {
                "id": f"E{edge_counter:03d}",
                "name": edge["name"],
                "source": edge["source"],
                "target": edge["target"],
                "listSectionId": set(edge["sids"]),
                "synonym": set(edge["synonym"])
            }
            edge_counter += 1
        else:
            master_edges[key]["listSectionId"].update(edge["sids"])
            master_edges[key]["synonym"].update(edge["synonym"])

    final_nodes = []
    for n in master_nodes.values():
        n["listSectionId"] = sorted(list(n["listSectionId"]))
        n["synonym"] = sorted(list(n["synonym"]))
        final_nodes.append(n)

    final_edges = []
    for e in master_edges.values():
        e["listSectionId"] = sorted(list(e["listSectionId"]))
        e["synonym"] = sorted(list(e["synonym"]))
        final_edges.append(e)

    return [{ "nodes": final_nodes, "relationships": final_edges }]

def main():
    input_file = './version_2/1_input_sentences.json'
    output_file = './version_2/1_output_data_for_graph.json'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    raw_data = load_data(input_file)
    chunks = chunk_data_with_ids(raw_data, chunk_size=15)

    all_results = []
    start_time = time.time()
    final_output = []

    for i, chunk in enumerate(chunks, 1):
        res = process_chunk(chunk, i, len(chunks))
        all_results.append(res)

        print(f"🔄 Đang gộp dữ liệu đồ thị đến khối {i}...")
        final_output = merge_to_data_for_graph(all_results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        print(f"💾 Đã Checkpoint thành công. Hiện có: {len(final_output[0]['nodes'])} Nodes | {len(final_output[0]['relationships'])} Edges")

        time.sleep(1)

    print(f"\n{'*'*40}")
    print(f"🎉 HOÀN TẤT TRÍCH XUẤT TRIPLET TOÀN BỘ FILE")
    print(f"Tổng thời gian: {time.time() - start_time:.2f} giây")
    print(f"Lưu chung cuộc tại: {output_file}")
    if final_output:
        print(f"Thành phẩm cuối cùng: {len(final_output[0]['nodes'])} Nodes | {len(final_output[0]['relationships'])} Edges")
    print(f"{'*'*40}")

if __name__ == "__main__":
    main()