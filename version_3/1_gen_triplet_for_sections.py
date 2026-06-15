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
# 1. BỘ TỪ ĐIỂN ONTOLOGY PHÁP LÝ
# ==========================================
LEGAL_ONTOLOGY = {
    "ALLOWED_NODE_LABELS": [
        "Co_Quan_Nha_Nuoc", "To_Chuc", "Ca_Nhan", "Chuyen_Gia",
        "Tai_San", "Giay_To_Phap_Ly", "Khoan_Tien",
        "Quyen_Loi", "Nghia_Vu", "Hanh_Vi_Vi_Pham", "Hinh_Phat_Che_Tai",
        "Thu_Tuc_Hanh_Chinh", "Dieu_Kien", "Ngoai_Le", "Thoi_Han", "Quy_Dinh_Phap_Luat"
    ],
    "ALLOWED_RELATIONSHIPS": [
        "AP_DUNG_CHO", "CHI_AP_DUNG_KHI", "TRU_TRUONG_HOP",
        "CO_QUYEN", "PHAI_THUC_HIEN", "SO_HUU_SU_DUNG", "CAP_PHAT", "THU_HOI",
        "CHUYEN_GIAO_CHO", "YEU_CAU_CO", "THUC_HIEN_HANH_VI", "BI_XU_LY_BANG",
        "CAN_CU_VAO", "LIEN_QUAN_DEN",
        "BAN_HANH", "LAP_QUAN_LY", "GIAI_QUYET"
    ]
}

# ==========================================
# 2. SYSTEM PROMPT (CẬP NHẬT KEYWORD INJECTION)
# ==========================================
SYSTEM_PROMPT = f"""
Đóng vai trò: Bạn là một hệ thống Trích xuất Bộ ba Tri thức pháp luật (Knowledge Triplet Extractor) cho hệ thống Hybrid GraphRAG.

!!! QUY TẮC ONTOLOGY TỐI THƯỢNG !!!
Hệ thống này ĐÃ CÓ SẴN một Lược đồ. Bạn KHÔNG ĐƯỢC PHÉP tự sáng tạo Label hay Relationship. Bạn chỉ được trả về các giá trị hợp lệ theo JSON Schema đã định nghĩa.

QUY TRÌNH TƯ DUY TẠO NODE (KIẾN TRÚC 3 LỚP):
1. 'label' (Khung xương): Phân loại chính xác thực thể.
2. 'name' (Da thịt): BẮT BUỘC RÚT GỌN THÀNH CỤM DANH TỪ CỐT LÕI (Tối đa 5-7 từ). CHỈ LẤY ĐÚNG THỰC THỂ, KHÔNG LẤY CÂU ĐỊNH NGHĨA.
3. 'aliases' (Từ khóa mồi Vector): Tự động suy luận 2-3 từ đồng nghĩa, từ lóng, hoặc cách gọi bình dân đời thường của thực thể đó. Ví dụ: "Cá nhân trực tiếp sản xuất nông nghiệp" -> ["nông dân", "người làm nông"]. "Bồi thường về đất" -> ["đền bù đất"]. Nếu không có từ đồng nghĩa phù hợp, để mảng rỗng [].

QUY TRÌNH XỬ LÝ KHUYẾT THÀNH PHẦN:
Nếu câu luật ẩn chủ ngữ, BẮT BUỘC sinh ra Node chủ thể ngầm định:
- Dùng name: "[Chủ_thể_ngầm_định]" (với label: Ca_Nhan hoặc To_Chuc).
- Dùng name: "[Cơ_quan_có_thẩm_quyền]" (với label: Co_Quan_Nha_Nuoc).

QUY TRÌNH TƯ DUY VỚI RELATIONSHIP:
1. LUẬT LÀM SẠCH ĐỘNG TỪ: Bắt buộc loại bỏ các tiền tố mang ý định (như "muốn", "định", "dự kiến") để đảm bảo hành động mang tính chất gốc rễ và dứt khoát.
2. VẬT VÔ TRI KHÔNG BIẾT HÀNH ĐỘNG: KHÔNG BAO GIỜ để Vật vô tri (Bản đồ, Giấy tờ, Đất) làm 'source' cho các quan hệ mang tính chủ động (CO_QUYEN, PHAI_THUC_HIEN, THUC_HIEN_HANH_VI).
3. TRỌNG TÂM THỰC THỂ: TUYỆT ĐỐI KHÔNG để Cơ quan nhà nước thực hiện hành vi trực tiếp lên Đất đai/Tài sản (Ví dụ sai: Cơ quan -> THỰC HIỆN -> Đất tranh chấp). Phải trích xuất đúng tên của THỦ TỤC (Ví dụ đúng: Cơ quan -> GIAI_QUYET -> Tranh chấp đất đai).
4. KIỂM TRA LOGIC (REASONING): Tự giải thích mũi tên bằng tiếng Việt. Nếu vô lý, HỦY BỎ triplet đó.
5. KHỚP MÃ ID: Trường 'source' và 'target' của Relationship BẮT BUỘC phải là mã 'id' của Node tương ứng (VD: 'n1', 'n2'), TUYỆT ĐỐI không được điền 'name' vào đây.

VÍ DỤ MẪU JSON (BẮT BUỘC HỌC THEO):
Input: "Thửa đất là phần diện tích đất. Người sử dụng đất có quyền được cấp Giấy chứng nhận đối với thửa đất đang sử dụng."
{{
  "nodes": [
    {{"id": "n1", "label": "Tai_San", "name": "Thửa đất", "aliases": ["mảnh đất", "lô đất"], "listSectionId": ["s1"]}},
    {{"id": "n2", "label": "Ca_Nhan", "name": "Người sử dụng đất", "aliases": ["chủ đất", "người đứng tên"], "listSectionId": ["s1"]}},
    {{"id": "n3", "label": "Giay_To_Phap_Ly", "name": "Giấy chứng nhận", "aliases": ["sổ đỏ", "sổ hồng"], "listSectionId": ["s1"]}}
  ],
  "relationships": [
    {{"reasoning": "Người SDĐ có quyền được cấp Giấy", "name": "CO_QUYEN", "source": "n2", "target": "n3", "listSectionId": ["s1"]}},
    {{"reasoning": "Người SDĐ sử dụng thửa đất", "name": "SO_HUU_SU_DUNG", "source": "n2", "target": "n1", "listSectionId": ["s1"]}}
  ]
}}
"""

# ==========================================
# 3. HÀM GỌI API (JSON SCHEMA STRICT CẬP NHẬT)
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
                                "label": {"type": "string", "enum": LEGAL_ONTOLOGY["ALLOWED_NODE_LABELS"]},
                                "name": {"type": "string"},
                                "aliases": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Các từ đồng nghĩa, từ lóng, cách gọi đời thường."
                                },
                                "listSectionId": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["id", "label", "name", "aliases", "listSectionId"],
                            "additionalProperties": False
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string", "description": "Giải thích logic quan hệ. Rỗng nếu vô lý."},
                                "name": {"type": "string", "enum": LEGAL_ONTOLOGY["ALLOWED_RELATIONSHIPS"]},
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
# 4. HÀM HỢP NHẤT ĐỒ THỊ VÀ LOẠI BỎ SELF-LOOP (CẬP NHẬT ALIASES)
# ==========================================
def merge_to_data_for_graph(all_graphs):
    master_nodes, master_edges = {}, {}
    raw_edges = []
    node_counter, edge_counter = 1, 1

    for graph in all_graphs:
        id_map = {}
        all_sids_in_chunk = set()
        node_sids_in_chunk = {}

        # Gộp Nodes
        for node in graph.get("nodes", []):
            label = str(node.get("label", "")).strip()
            name = str(node.get("name", "")).strip()
            if not name or not label: continue

            name_key = f"{label}_{name}".lower()
            old_id = node.get("id")
            sids = node.get("listSectionId", [])
            aliases = node.get("aliases", [])

            node_sids_in_chunk[old_id] = set(sids)
            all_sids_in_chunk.update(sids)

            if name_key not in master_nodes:
                master_nodes[name_key] = {
                    "id": f"N{node_counter:03d}",
                    "label": label,
                    "name": name,
                    "aliases": set(aliases), # Dùng set để lọc trùng từ khóa mồi
                    "listSectionId": set(sids)
                }
                node_counter += 1
            else:
                master_nodes[name_key]["aliases"].update(aliases)
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

    # Xử lý trùng lặp Edge
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
        # Chuyển set về list trước khi xuất JSON
        "nodes": [{**n, "aliases": sorted(list(n["aliases"])), "listSectionId": sorted(list(n["listSectionId"]))} for n in master_nodes.values()],
        "relationships": [{**e, "listSectionId": sorted(list(e["listSectionId"]))} for e in master_edges.values()]
    }]

# ==========================================
# 5. HÀM XUẤT LOG S-V-O DỄ ĐỌC
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
# 6. LUỒNG CHẠY CHÍNH
# ==========================================
def main():
    input_file = 'output/1_sections_vphc.json'
    output_file = 'version_3/2_entities_per_chunk_vphc.json'
    svo_log_file = 'version_3/svo_triplets_log_vphc.json'

    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    raw_data = load_data(input_file)
    chunks = chunk_data_with_ids(raw_data, chunk_size=1)
    all_results = []

    for i, chunk in enumerate(chunks, 1):
        res = process_chunk(chunk, i, len(chunks))
        all_results.append(res)

        final_output = merge_to_data_for_graph(all_results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        export_svo_log(final_output, svo_log_file)

        print(f"💾 Đã lưu đồ thị tại {output_file} (Nodes: {len(final_output[0]['nodes'])} | Edges: {len(final_output[0]['relationships'])})")
        print(f"📝 Đã cập nhật log S-V-O tại {svo_log_file}")
        time.sleep(1)

if __name__ == "__main__":
    main()