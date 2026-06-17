import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# ==========================================
# 1. SYSTEM PROMPT (CẬP NHẬT GENERALIZATION, ATOMIC TRIPLETS & NO LABELS)
# ==========================================
SYSTEM_PROMPT = """Đóng vai trò: Bạn là hệ thống Trích xuất Bộ ba Tri thức (Knowledge Triplet Extractor) cho hệ thống Hybrid GraphRAG pháp luật.

Nhiệm vụ của bạn là phân tích văn bản pháp luật được cung cấp dưới dạng các đoạn (section) và trích xuất các thực thể (nodes) và mối quan hệ (relationships) giữa chúng.

QUY TẮC TRÍCH XUẤT VÀ CHUẨN HÓA CỦA HỆ THỐNG:

1. ĐƯA CÁC TỪ RIÊNG VỀ TỪ CHUNG (Khái quát hóa thực thể):
   - Thay vì giữ nguyên các từ hoặc cụm từ mô tả quá chi tiết, hãy quy đổi chúng về các danh từ/thực thể chung.
   - Ví dụ:
     * "người tham gia giao thông", "người điều khiển xe máy", "người lái xe", "cá nhân sử dụng đất", "hộ gia đình sử dụng đất" -> khái quát thành "người" hoặc "cá nhân" hoặc "tổ chức".
     * "xe máy", "xe mô tô hai bánh", "xe gắn máy" -> khái quát thành "xe máy" hoặc "phương tiện".
     * "Ủy ban nhân dân cấp tỉnh", "Ủy ban nhân dân cấp huyện", "Bộ Tài nguyên và Môi trường" -> khái quát thành "cơ quan nhà nước" hoặc "cơ quan".
     * "tiền phạt 500.000 đồng", "tiền phạt từ 100k đến 200k" -> khái quát thành "tiền" hoặc "tiền phạt".
     * "đất trồng cây lâu năm", "đất rừng phòng hộ", "thửa đất" -> khái quát thành "đất".

2. TÁCH THÀNH CÁC BỘ BA NGUYÊN TỬ (Atomic Triplets):
   - Chia nhỏ các câu phức, mệnh đề phức thành nhiều bộ ba chủ ngữ - động từ/quan hệ - tân ngữ (Subject - Predicate - Object) cực kỳ đơn giản và độc lập.
   - BẮT BUỘC học theo ví dụ mẫu sau:
     * Câu: "người tham gia giao thông điều khiển xe máy không đội mũ bảo hiểm bị phạt 500"
     * Trích xuất các bộ ba sau:
       - (người, tham gia, giao thông)
       - (người, sử dụng, xe máy)
       - (người, không, đội mũ bảo hiểm)
       - (người, bị phạt, tiền)

3. QUY TRÌNH TƯ DUY TẠO NODE:
   - 'name' (Tên thực thể): Là cụm danh từ cốt lõi đã được khái quát hóa (ví dụ: "người", "xe máy", "giao thông", "đội mũ bảo hiểm", "tiền").
   - KHÔNG gán nhãn hay phân loại (label) cho thực thể nữa.

4. QUY TRÌNH TẠO QUAN HỆ (Relationships):
   - 'source' và 'target' phải khớp hoàn toàn với 'id' của node tương ứng (ví dụ: "n1", "n2").
   - 'name': Là mối quan hệ ngắn gọn, mang động từ/trạng thái bằng tiếng Việt viết thường hoặc dạng tự nhiên (ví dụ: "tham gia", "sử dụng", "không", "bị phạt").
   - 'reasoning': Giải thích ngắn gọn lý do tồn tại của mối quan hệ này dựa trên văn bản gốc.

5. KHỚP MÃ ID VÀ SECTION ID:
   - Hãy chắc chắn gán đúng 'listSectionId' tương ứng của section mà bạn đang phân tích (ví dụ: ["s1"]) cho cả các node và relationship được tạo ra từ section đó.
"""

# ==========================================
# 2. HÀM GỌI API (JSON SCHEMA STRICT)
# ==========================================
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_data_with_ids(data, chunk_size=1):
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk_items = data[i:i + chunk_size]
        text_chunk = ""
        for item in chunk_items:
            sid = item.get("id") or item.get("section_id") or "unknown"
            text = item.get("text_content") or item.get("original_text") or ""
            text_chunk += f"[SID: {sid}] {text}\n"
        chunks.append(text_chunk)
    return chunks

def process_chunk(text_chunk, chunk_index, total_chunks):
    print(f"\nBẮT ĐẦU XỬ LÝ KHỐI {chunk_index}/{total_chunks}")

    triplet_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "legal_triplet_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Mã ID duy nhất, ví dụ: n1, n2"},
                                "name": {"type": "string"},
                                "listSectionId": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["id", "name", "listSectionId"],
                            "additionalProperties": False
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string", "description": "Giải thích logic quan hệ. Rỗng nếu vô lý."},
                                "name": {"type": "string"},
                                "source": {"type": "string", "description": "BẮT BUỘC TRÙNG KHỚP với 'id' của Node chủ thể (VD: 'n1')"},
                                "target": {"type": "string", "description": "BẮT BUỘC TRÙNG KHỚP với 'id' của Node đối tượng (VD: 'n2')"},
                                "listSectionId": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["reasoning", "name", "source", "target", "listSectionId"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["nodes", "relationships"],
                "additionalProperties": False
            }
        }
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format=triplet_schema,
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

# ==========================================
# 3. HÀM HỢP NHẤT ĐỒ THỊ VÀ LOẠI BỎ SELF-LOOP
# ==========================================
def merge_to_data_for_graph(all_graphs):
    master_nodes, master_edges = {}, {}
    raw_edges = []
    node_counter, edge_counter = 1, 1

    for graph in all_graphs:
        id_map = {}
        all_sids_in_chunk = set()
        node_sids_in_chunk = {}

        # Gộp Nodes dựa trên trường name (chuyển sang lowercase)
        for node in graph.get("nodes", []):
            name = str(node.get("name", "")).strip()
            if not name: continue

            name_key = name.lower()
            old_id = node.get("id")
            sids = node.get("listSectionId", [])

            node_sids_in_chunk[old_id] = set(sids)
            all_sids_in_chunk.update(sids)

            if name_key not in master_nodes:
                master_nodes[name_key] = {
                    "id": f"N{node_counter:03d}",
                    "name": name,
                    "listSectionId": set(sids)
                }
                node_counter += 1
            else:
                master_nodes[name_key]["listSectionId"].update(sids)

            id_map[old_id] = {"id": master_nodes[name_key]["id"], "name_key": name_key}

        # Gộp Edges
        for rel in graph.get("relationships", []):
            src_info = id_map.get(rel.get("source"))
            tgt_info = id_map.get(rel.get("target"))

            # Kiểm tra an toàn: Tồn tại và không tự trỏ (Self-loop)
            if src_info and tgt_info and src_info["id"] != tgt_info["id"]:
                rel_sids = rel.get("listSectionId", [])
                valid_rel_sids = set(rel_sids).intersection(all_sids_in_chunk)
                if not valid_rel_sids:
                    valid_rel_sids = node_sids_in_chunk.get(rel.get("source"), set())

                master_nodes[src_info["name_key"]]["listSectionId"].update(valid_rel_sids)
                master_nodes[tgt_info["name_key"]]["listSectionId"].update(valid_rel_sids)

                raw_edges.append({
                    "name": str(rel.get("name", "")).strip(),
                    "source": src_info["id"],
                    "target": tgt_info["id"],
                    "listSectionId": list(valid_rel_sids)
                })

    # Xử lý trùng lặp Edge dựa trên source, target, name
    for edge in raw_edges:
        key = (edge["source"], edge["target"], edge["name"].lower())
        if key not in master_edges:
            master_edges[key] = {
                "id": f"E{edge_counter:03d}",
                "name": edge["name"],
                "source": edge["source"],
                "target": edge["target"],
                "listSectionId": set(edge["listSectionId"])
            }
            edge_counter += 1
        else:
            master_edges[key]["listSectionId"].update(edge["listSectionId"])

    return [{
        "nodes": [{**n, "listSectionId": sorted(list(n["listSectionId"]))} for n in master_nodes.values()],
        "relationships": [{**e, "listSectionId": sorted(list(e["listSectionId"]))} for e in master_edges.values()]
    }]

# ==========================================
# 4. HÀM XUẤT LOG S-V-O DỄ ĐỌC
# ==========================================
def export_svo_log(final_graph, log_file):
    if not final_graph: return
    graph_data = final_graph[0]

    id_to_name = {node["id"]: node["name"] for node in graph_data.get("nodes", [])}

    svo_list = []
    for rel in graph_data.get("relationships", []):
        s_name = id_to_name.get(rel["source"], "Unknown")
        o_name = id_to_name.get(rel["target"], "Unknown")
        v_name = rel["name"]

        svo_list.append({
            "Subject": s_name,
            "Predicate": v_name,
            "Object": o_name
        })

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(svo_list, f, ensure_ascii=False, indent=4)

# ==========================================
# 5. LUỒNG CHẠY CHÍNH
# ==========================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'material_for_triplets/sections_dat_dai_2.json')
    output_file = os.path.join(base_dir, 'triplets/triplets_dat_dai_2.json')
    svo_log_file = os.path.join(base_dir, 'triplets/svo_triplets_log_dat_dai_2.json')

    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    raw_data = load_data(input_file)
    # Chia nhỏ dữ liệu theo từng section đơn lẻ để đảm bảo trích xuất chính xác
    chunks = chunk_data_with_ids(raw_data, chunk_size=1)
    all_results = []

    for i, chunk in enumerate(chunks, 1):
        res = process_chunk(chunk, i, len(chunks))
        all_results.append(res)

        # Gộp đồ thị & lưu kết quả định kỳ để chống mất dữ liệu
        final_output = merge_to_data_for_graph(all_results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        export_svo_log(final_output, svo_log_file)

        print(f"💾 Đã lưu đồ thị tại {output_file} (Nodes: {len(final_output[0]['nodes'])} | Edges: {len(final_output[0]['relationships'])})")
        print(f"📝 Đã cập nhật log S-V-O tại {svo_log_file}")
        time.sleep(0.5)

if __name__ == "__main__":
    main()
