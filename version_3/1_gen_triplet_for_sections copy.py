import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo OpenAI Client
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Đóng vai trò: Bạn là hệ thống Trích xuất Tri thức GraphRAG chuyên xử lý văn bản luật bằng phương pháp VERB-DRIVEN.

Mục tiêu: Đọc các điều luật, hiểu sâu ngữ cảnh, khôi phục các thành phần bị khuyết và trích xuất Triplet S-V-O theo ĐÚNG CHIỀU HÀNH ĐỘNG.

!!! DANH MỤC SIÊU LỚP BẮT BUỘC CHO NODE (ONTOLOGY) !!!
Chủ thể (S) và Đối tượng (O) CHỈ ĐƯỢC PHÉP là 1 cụm từ trong danh sách sau:
[Cơ quan, Tổ chức, Người, Phương tiện, Tài sản, Thủ tục, Hành vi, Hình phạt, Biện pháp, Giấy phép, Quyền lợi, Quy định, Luật, Thời gian, Địa điểm, Mức tiền, Trống]

!!! QUY TẮC SUY LUẬN NGỮ CẢNH VÀ CHIỀU HÀNH ĐỘNG (CỰC KỲ QUAN TRỌNG) !!!
1. ĐÚNG CHIỀU (S) -> [V] -> (O):
   - (S) LUÔN LÀ KẺ THỰC HIỆN hành động.
   - (O) LUÔN LÀ KẺ/VẬT CHỊU TÁC ĐỘNG của hành động.
2. KHÔI PHỤC CÂU BỊ ĐỘNG: Văn bản luật thường viết bị động. Bạn PHẢI dịch ngược lại thành chủ động.
   - Sai: (Người) - [bị phạt] -> (Mức tiền)
   - Đúng: (Cơ quan) - [phạt] -> (Người)
   - Sai: (Giấy phép) - [bị thu hồi] -> (Trống)
   - Đúng: (Cơ quan) - [thu hồi] -> (Giấy phép)
3. SUY LUẬN CHỦ THỂ ẨN (CẤM LẠM DỤNG NODE "TRỐNG"):
   - Với các hành động mang tính quản lý nhà nước (xử phạt, cấp phép, tịch thu, bãi bỏ...), nếu luật không ghi ai làm, BẮT BUỘC tự suy luận (S) là "Cơ quan".
   - Với hành động của người dân (khiếu nại, tố cáo, nộp phạt...), (O) thường hướng tới nhà nước, BẮT BUỘC tự suy luận (O) là "Cơ quan". VD: "Cá nhân khiếu nại" -> (Người) - [khiếu nại] -> (Cơ quan).
4. KHI NÀO MỚI ĐƯỢC DÙNG "TRỐNG"?:
   - CHỈ dùng "Trống" cho các nội động từ thực sự không tác động lên ai/cái gì (VD: "Luật có hiệu lực", "Giấy phép hết hạn").
   - VD đúng: (Luật) - [có hiệu lực] -> (Trống)

QUY TRÌNH THỰC HIỆN 2 BƯỚC:
--- BƯỚC 1: CHUẨN HÓA VÀ DIỄN ĐẠT LẠI ---
- Viết lại luật thành các câu đơn CHỦ ĐỘNG, điền bù ngay các Chủ thể/Đối tượng bị ẩn dựa theo Quy tắc suy luận trên.

--- BƯỚC 2: TÁCH TRIPLET SIÊU TRỪU TƯỢNG ---
- Rút trích S-V-O từ Bước 1. Tên hành động (V) viết thường, không dấu gạch dưới (VD: "xử phạt"). Tên (S) và (O) lấy từ Danh mục.

!!! QUY TẮC BẢO TOÀN SỐ LƯỢNG ĐẦU RA !!!
Trả về ĐỦ số lượng section_id. Nếu điều luật không có hành động, trả về nodes/relationships rỗng [].

ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC:
{
    "results": [
        {
            "section_id": "s123",
            "normalized_sentences": [
                "Cơ quan thu hồi giấy phép của Người vi phạm.",
                "Người khiếu nại lên Cơ quan."
            ],
            "nodes": [
                {"id": "T01", "name": "Cơ quan"},
                {"id": "T02", "name": "Giấy phép"},
                {"id": "T03", "name": "Người"}
            ],
            "relationships": [
                {"source": "T01", "target": "T02", "name": "thu hồi"},
                {"source": "T03", "target": "T01", "name": "khiếu nại"}
            ]
        }
    ]
}
"""

def process_chunk(chunk_data):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(chunk_data, ensure_ascii=False)}
            ]
        )
        return json.loads(response.choices[0].message.content).get("results", [])
    except Exception as e:
        print(f"❌ Lỗi API: {e}")
        return []

def merge_data(all_results):
    master_nodes = {}
    master_edges = {}

    node_id_counter = 1
    edge_id_counter = 1

    for section_result in all_results:
        sid = section_result.get("section_id")
        id_map = {}

        # 1. Gộp Nodes (Vẫn giữ logic cô lập node "Trống")
        for node in section_result.get("nodes", []):
            name = node.get("name", "Trống").strip()
            name_key = name.lower()

            if name_key == "trống":
                new_id = f"N{node_id_counter:03d}"
                unique_key = f"trống_{new_id}"
                master_nodes[unique_key] = {
                    "id": new_id,
                    "name": "Trống",
                    "listSectionId": {sid} if sid else set()
                }
                node_id_counter += 1
                id_map[node.get("id")] = new_id

            else:
                if name_key not in master_nodes:
                    new_id = f"N{node_id_counter:03d}"
                    master_nodes[name_key] = {
                        "id": new_id,
                        "name": name,
                        "listSectionId": {sid} if sid else set()
                    }
                    node_id_counter += 1
                else:
                    if sid: master_nodes[name_key]["listSectionId"].add(sid)

                id_map[node.get("id")] = master_nodes[name_key]["id"]

        # 2. Gộp Relationships
        for rel in section_result.get("relationships", []):
            src_id = id_map.get(rel.get("source"))
            tgt_id = id_map.get(rel.get("target"))
            rel_name = rel.get("name", "liên quan").strip().lower()

            if src_id and tgt_id:
                edge_key = (src_id, tgt_id, rel_name)

                if edge_key not in master_edges:
                    master_edges[edge_key] = {
                        "id": f"E{edge_id_counter:03d}",
                        "name": rel_name,
                        "source": src_id,
                        "target": tgt_id,
                        "listSectionId": {sid} if sid else set()
                    }
                    edge_id_counter += 1
                else:
                    if sid: master_edges[edge_key]["listSectionId"].add(sid)

    return {
        "nodes": [
            {**n, "listSectionId": sorted(list(n["listSectionId"]))}
            for n in master_nodes.values()
        ],
        "relationships": [
            {**e, "listSectionId": sorted(list(e["listSectionId"]))}
            for e in master_edges.values()
        ]
    }

def print_quality_monitor(results):
    print("\n" + "="*60)
    print("👀 KIỂM TRA CHẤT LƯỢNG TRÍCH XUẤT (MONITORING)")
    print("="*60)

    for res in results:
        sid = res.get("section_id", "Unknown")
        normalized_sents = res.get("normalized_sentences", [])
        nodes = res.get("nodes", [])
        rels = res.get("relationships", [])

        print(f"\n📌 [SECTION: {sid}]")

        print("   📝 Văn bản chuẩn hóa:")
        for sent in normalized_sents:
            print(f"      - {sent}")

        temp_id_map = {n.get("id"): n.get("name", "Trống") for n in nodes}

        print("\n   🔗 Triplets (Đã check chiều hành động):")
        if not rels:
            print("      (Không tìm thấy hành động cốt lõi)")
        for rel in rels:
            s_name = temp_id_map.get(rel.get("source"), "?")
            o_name = temp_id_map.get(rel.get("target"), "?")
            v_name = rel.get("name", "?")

            if o_name == "Trống": o_name = "[∅ Không có]"
            if s_name == "Trống": s_name = "[∅ Không có]"

            print(f"      👉 ({s_name})  -[ {v_name} ]->  ({o_name})")
    print("="*60 + "\n")

def main():
    input_path = './version_3/1_sections.json'
    output_path = './version_3/2_final_graph.json'

    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Không tìm thấy file {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_raw_results = []
    chunk_size = 5

    for i in range(0, len(data), chunk_size):
        chunk = [{"section_id": item["section_id"], "text": item.get("original_text", "")} for item in data[i:i+chunk_size]]

        print(f"\n⏳ Đang xử lý chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1)//chunk_size}...")

        results = process_chunk(chunk)
        all_raw_results.extend(results)

        if results:
            print_quality_monitor(results)

        final_graph = merge_data(all_raw_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_graph, f, ensure_ascii=False, indent=4)

        print(f"💾 Đã lưu Checkpoint! (Tổng Nodes: {len(final_graph['nodes'])} | Tổng Edges: {len(final_graph['relationships'])})")

        time.sleep(1)

    print(f"\n✅ HOÀN THÀNH TOÀN BỘ! Dữ liệu Graph đã xuất ra: {output_path}")

if __name__ == "__main__":
    main()