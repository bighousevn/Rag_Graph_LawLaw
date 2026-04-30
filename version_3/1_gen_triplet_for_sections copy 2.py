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
Đóng vai trò: Bạn là hệ thống Trích xuất Tri thức GraphRAG chuyên sâu, kết hợp phương pháp VERB-DRIVEN và PHÂN TÍCH Ý NGHĨA HÀNH ĐỘNG CỐT LÕI.

Mục tiêu: Đọc điều luật, hiểu sâu ngữ cảnh pháp lý để xác định các hành động thực thi quan trọng, khôi phục thực thể ẩn và trích xuất Triplet S-V-O.

!!! DANH MỤC SIÊU LỚP BẮT BUỘC CHO NODE (ONTOLOGY) !!!
Chủ thể (S) và Đối tượng (O) CHỈ ĐƯỢC PHÉP là 1 cụm từ trong danh sách sau:
[Cơ quan, Tổ chức, Người, Phương tiện, Tài sản, Thủ tục, Hành vi, Hình phạt, Biện pháp, Giấy phép, Quyền lợi, Quy định, Luật, Thời gian, Địa điểm, Mức tiền, Trống]
Mỗi node là 1 khái niệm đơn nhất (không ghép nhiều khái niệm vào cùng 1 node).

!!! QUY TẮC PHÂN TÍCH VÀ SUY LUẬN !!!
1. XÁC ĐỊNH HÀNH ĐỘNG CỐT LÕI:
   - Tập trung vào các hành động mang tính thực thi pháp lý (xử phạt, tịch thu, khiếu nại, cấp phép...).
   - Loại bỏ các trạng từ hoặc từ bổ trợ không mang tính hành động hạt nhân.
2. ĐÚNG CHIỀU (S) -> [V] -> (O):
   - (S) LUÔN LÀ KẺ THỰC HIỆN hành động. (O) LUÔN LÀ KẺ/VẬT CHỊU TÁC ĐỘNG.
3. KHÔI PHỤC CÂU BỊ ĐỘNG VÀ CHỦ THỂ ẨN:
   - Dịch ngược bị động thành chủ động.
   - Tự suy luận (S) là "Cơ quan" cho các hành động quản lý nhà nước (tịch thu, thu hồi...).
   - Tự suy luận (O) là "Cơ quan" cho các hành động của dân hướng về nhà nước (nộp phạt, tố cáo...).
4. CHIẾN LƯỢC NODE "TRỐNG":
   - Dùng "Trống" cho nội động từ không có đối tượng chịu tác động (VD: "Luật có hiệu lực").
   - Node "Trống" đóng vai trò duy trì cấu trúc triplet 3 thành phần để không gãy luồng truy vấn.

QUY TRÌNH THỰC HIỆN:
Bước 1 (Chuẩn hóa): Diễn đạt lại luật thành câu đơn CHỦ ĐỘNG, điền bù thực thể ẩn.
Bước 2 (Triplet): Trích xuất S-V-O. Động từ (V) viết thường, gọt sạch trạng thái (đã, đang, sẽ, có thể). Nếu 1 triplet có nhiều chủ thể hoặc đối tượng, tách thành nhiều triplet riêng biệt.

ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC:
{
    "results": [
        {
            "section_id": "s123",
            "normalized_sentences": ["Cơ quan tịch thu tài sản của Người."],
            "nodes": [
                {"id": "T01", "name": "Cơ quan"},
                {"id": "T02", "name": "Tài sản"}
            ],
            "relationships": [
                {"source": "T01", "target": "T02", "name": "tịch thu"}
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

        for node in section_result.get("nodes", []):
            name = node.get("name", "Trống").strip()
            name_key = name.lower()

            # CHIẾN LƯỢC CẢI TIẾN: Node "Trống" được định danh duy nhất theo ngữ cảnh
            if name_key == "trống":
                new_id = f"N{node_id_counter:03d}"
                unique_key = f"trống_{new_id}" # Tạo key duy nhất để không bị gộp chung các node Trống
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
        "nodes": [{**n, "listSectionId": sorted(list(n["listSectionId"]))} for n in master_nodes.values()],
        "relationships": [{**e, "listSectionId": sorted(list(e["listSectionId"]))} for e in master_edges.values()]
    }

def print_quality_monitor(results):
    print("\n" + "="*60)
    print("⚖️ KIỂM TRA HÀNH ĐỘNG CỐT LÕI VÀ TRIỀU HÀNH ĐỘNG")
    print("="*60)
    for res in results:
        sid = res.get("section_id", "Unknown")
        print(f"\n📌 [SECTION: {sid}]")
        temp_id_map = {n.get("id"): n.get("name", "Trống") for n in res.get("nodes", [])}
        for rel in res.get("relationships", []):
            s_name = temp_id_map.get(rel.get("source"), "?")
            o_name = temp_id_map.get(rel.get("target"), "?")
            v_name = rel.get("name", "?")
            print(f"      👉 ({s_name})  -[ {v_name} ]->  ({o_name if o_name != 'Trống' else '[∅ Trống]'})")

def main():
    input_path = './version_3/1_sections.json'
    output_path = './version_3/2_final_graph_2.json'
    if not os.path.exists(input_path): return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_raw_results = []
    chunk_size = 5

    for i in range(0, len(data), chunk_size):
        chunk = [{"section_id": item["section_id"], "text": item.get("original_text", "")} for item in data[i:i+chunk_size]]
        print(f"\n⏳ Đang phân tích chunk {i//chunk_size + 1}...")
        results = process_chunk(chunk)
        all_raw_results.extend(results)
        if results: print_quality_monitor(results)

        final_graph = merge_data(all_raw_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_graph, f, ensure_ascii=False, indent=4)
        time.sleep(1)

    print(f"\n✅ HOÀN THÀNH! Tổng Nodes: {len(final_graph['nodes'])}")

if __name__ == "__main__":
    main()