import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Đóng vai trò: Bạn là một hệ thống Trích xuất Bộ ba Tri thức (Knowledge Triplet Extractor) siêu trừu tượng cho cơ sở dữ liệu đồ thị GraphRAG.

!!! CẢNH BÁO TỐI THƯỢNG VỀ TRỪU TƯỢNG HÓA !!!
Nhiệm vụ quan trọng nhất của bạn là TRỪU TƯỢNG HÓA TỘT ĐỘ trường "name" của các Node và Relationship.
- Bạn TUYỆT ĐỐI KHÔNG ĐƯỢC dùng các danh từ cụ thể trong văn bản (như "Quốc hội", "Chính phủ", "người lao động", "xe máy", "Luật số 15") để làm "name" cho Node.
- Trường "name" CHỈ ĐƯỢC PHÉP LÀ MỘT KHÁI NIỆM SIÊU LỚP (Superclass) đại diện chung nhất (Ví dụ: Cơ quan, Tổ chức, Người, Tài sản, Thủ tục, Hành vi, Hình phạt, Giấy phép, Quyền lợi...). Bạn tự do sáng tạo Siêu lớp nhưng phải tuân thủ nguyên tắc là từ khái quát cao nhất.

QUY TRÌNH TƯ DUY BẮT BUỘC MỖI KHI TẠO NODE:
1. Gặp một thực thể (VD: "Ủy ban nhân dân xã").
2. Lấy nguyên gốc từ "Ủy ban nhân dân xã" nhét vào mảng "synonym".
3. Tự hỏi: Thực thể này thuộc thể loại chung nào? -> Trả lời: "Cơ quan".
4. Điền "Cơ quan" vào trường "name".

QUY TRÌNH TƯ DUY VỚI MỐI QUAN HỆ (RELATIONSHIP):
1. Lấy từ/cụm từ hành động cụ thể (VD: "sẽ bị xử phạt", "đã ban hành") nhét vào "synonym".
2. Rút gọn về động từ gốc, loại bỏ thì/trạng thái (VD: "Xử phạt", "Ban hành").
3. Điền động từ gốc này vào trường "name" của Relationship.

QUY TẮC KHÁC:
- Khớp SID: Bắt buộc lấy mã [SID: s...] gán vào "listSectionId".
- Bỏ qua mọi yếu tố nhiễu (thời gian, địa điểm).

ĐỊNH DẠNG JSON ĐẦU RA (Không giải thích thêm):
{
    "nodes": [
        {
            "id": "T01",
            "name": "Tên Siêu Lớp (VD: Cơ quan, Luật, Người, Tài sản... TUYỆT ĐỐI KHÔNG DÙNG TỪ CỤ THỂ)",
            "synonym": ["các từ cụ thể thực tế xuất hiện trong câu"],
            "listSectionId": ["s123"]
        }
    ],
    "relationships": [
        {
            "name": "Động từ gốc (VD: Ban hành, Xử phạt, Căn cứ...)",
            "source": "T01",
            "target": "T02",
            "synonym": ["các cụm từ hành động nguyên bản trong câu"],
            "listSectionId": ["s123"]
        }
    ]
}
"""

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_data_with_ids(data, chunk_size=5):
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
    print(f"\nBẮT ĐẦU XỬ LÝ KHỐI {chunk_index}/{total_chunks}")
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
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Lỗi API tại khối {chunk_index}: {e}")
        return {"nodes": [], "relationships": []}

def merge_to_data_for_graph(all_graphs):
    master_nodes, master_edges = {}, {}
    raw_edges = []
    node_counter, edge_counter = 1, 1

    for graph in all_graphs:
        id_map = {}
        all_sids_in_chunk = set()
        node_sids_in_chunk = {}

        # 1. Gộp Nodes và ĐẢM BẢO HỢP listSectionId bằng Set
        for node in graph.get("nodes", []):
            name = str(node.get("name", "")).strip()
            if not name: continue
            name_key = name.lower()
            old_id = node.get("id")
            sids = node.get("listSectionId", [])
            syns = node.get("synonym", [])

            node_sids_in_chunk[old_id] = set(sids)
            all_sids_in_chunk.update(sids)

            if name_key not in master_nodes:
                master_nodes[name_key] = {"id": f"N{node_counter:03d}", "name": name, "listSectionId": set(sids), "synonym": set(syns)}
                node_counter += 1
            else:
                # Phép hợp tập hợp (Union) đảm bảo không sót sectionID nào
                master_nodes[name_key]["listSectionId"].update(sids)
                master_nodes[name_key]["synonym"].update(syns)

            id_map[old_id] = {"id": master_nodes[name_key]["id"], "name_key": name_key}

        # 2. Xử lý Relationships
        for rel in graph.get("relationships", []):
            src_info, tgt_info = id_map.get(rel.get("source")), id_map.get(rel.get("target"))
            if src_info and tgt_info:
                rel_sids = rel.get("listSectionId", [])
                valid_rel_sids = set(rel_sids).intersection(all_sids_in_chunk)
                if not valid_rel_sids:
                    valid_rel_sids = node_sids_in_chunk.get(rel.get("source"), set())

                master_nodes[src_info["name_key"]]["listSectionId"].update(valid_rel_sids)
                master_nodes[tgt_info["name_key"]]["listSectionId"].update(valid_rel_sids)

                raw_edges.append({
                    "name": str(rel.get("name", "")).strip(), "source": src_info["id"],
                    "target": tgt_info["id"], "synonym": rel.get("synonym", []), "listSectionId": list(valid_rel_sids)
                })

    # 3. Gộp Edges và ĐẢM BẢO HỢP listSectionId
    for edge in raw_edges:
        key = (edge["source"], edge["target"], edge["name"].lower())
        if key not in master_edges:
            master_edges[key] = {"id": f"E{edge_counter:03d}", "name": edge["name"], "source": edge["source"], "target": edge["target"], "listSectionId": set(edge["listSectionId"]), "synonym": set(edge["synonym"])}
            edge_counter += 1
        else:
            # Phép hợp tập hợp cho Relationships
            master_edges[key]["listSectionId"].update(edge["listSectionId"])
            master_edges[key]["synonym"].update(edge["synonym"])

    return [{
        "nodes": [{**n, "listSectionId": sorted(list(n["listSectionId"])), "synonym": sorted(list(n["synonym"]))} for n in master_nodes.values()],
        "relationships": [{**e, "listSectionId": sorted(list(e["listSectionId"])), "synonym": sorted(list(e["synonym"]))} for e in master_edges.values()]
    }]

def main():
    input_file = './version_3/1_sections.json'
    output_file = './version_3/2_entities_per_chunk.json'

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    raw_data = load_data(input_file)
    chunks = chunk_data_with_ids(raw_data, chunk_size=5)
    all_results = []

    for i, chunk in enumerate(chunks, 1):
        res = process_chunk(chunk, i, len(chunks))
        all_results.append(res)

        # Gộp tất cả các chunk từ đầu đến hiện tại, hợp nhất listSectionId
        final_output = merge_to_data_for_graph(all_results)

        # Ghi đè file kết quả duy nhất
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        print(f"💾 Đã ghi đè/cập nhật checkpoint tại {output_file} (Nodes: {len(final_output[0]['nodes'])} | Edges: {len(final_output[0]['relationships'])})")

        time.sleep(1)

if __name__ == "__main__":
    main()