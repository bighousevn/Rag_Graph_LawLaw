import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. CẤU HÌNH API
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# ==============================================================================
# PROMPT: DÙNG CHO AI GOM NHÓM TỪ VỰNG
# ==============================================================================
GROUPING_PROMPT = """
Bạn là chuyên gia Ngôn ngữ học và Pháp lý Việt Nam.
Nhiệm vụ: Phân tích danh sách các từ vựng (Thực thể và Hành động) sau đó gom nhóm các từ ĐỒNG NGHĨA, VIẾT TẮT, HOẶC CÁCH DIỄN ĐẠT KHÁC NHAU về một tên đại diện (master) duy nhất.

QUY TẮC:
1. KHÔNG gom các từ trái nghĩa hoặc khác biệt về đối tượng áp dụng (VD: "cá nhân" và "tổ chức" không gộp chung).
2. Đảm bảo mọi từ trong danh sách đầu vào đều phải có mặt trong mảng "variants" của một nhóm nào đó (kể cả khi từ đó đứng 1 mình 1 nhóm).

FORMAT ĐẦU RA BẮT BUỘC (JSON):
{
    "node_groups": [
        {"master": "Ủy ban nhân dân cấp xã", "variants": ["UBND cấp xã", "Ủy ban nhân dân cấp xã", "ủy ban xã"]}
    ],
    "rel_groups": [
        {"master": "xử phạt", "variants": ["phạt tiền", "xử phạt vi phạm hành chính", "áp dụng hình thức xử phạt"]}
    ]
}
"""

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_ai_semantic_mapping(graph_data):
    """Gửi danh sách từ vựng duy nhất cho AI để lập Từ điển Gom nhóm"""
    unique_nodes = list(set([n["name"] for n in graph_data.get("nodes", [])]))
    unique_rels = list(set([e["name"] for e in graph_data.get("relationships", [])]))

    print(f"\n{'='*50}")
    print(f"🤖 BẮT ĐẦU NHỜ AI LẬP TỪ ĐIỂN ĐỒNG NGHĨA")
    print(f"  [+] Đang phân tích {len(unique_nodes)} Thực thể duy nhất...")
    print(f"  [+] Đang phân tích {len(unique_rels)} Hành động duy nhất...")
    print(f"  [+] Đang gửi Request cho GPT-4o-mini (Vui lòng đợi vài giây)...")

    prompt = f"Danh sách Thực thể (Nodes):\n{json.dumps(unique_nodes, ensure_ascii=False)}\n\nDanh sách Hành động (Relationships):\n{json.dumps(unique_rels, ensure_ascii=False)}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            temperature=0,
            messages=[
                {"role": "system", "content": GROUPING_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        print("✅ AI đã lập Từ điển Gom nhóm thành công!")
        result = json.loads(response.choices[0].message.content)

        # LOG GIÁM SÁT TỪ ĐIỂN AI
        node_groups = result.get("node_groups", [])
        rel_groups = result.get("rel_groups", [])
        print(f"\n📊 BÁO CÁO TỪ ĐIỂN AI:")
        print(f"  -> Tạo được {len(node_groups)} nhóm Thực thể (Nodes).")
        print(f"  -> Tạo được {len(rel_groups)} nhóm Hành động (Edges).")
        print(f"  -> [MẪU 3 NHÓM NODE ĐẦU TIÊN]:")
        print(json.dumps(node_groups[:3], ensure_ascii=False, indent=4))
        print(f"  -> [MẪU 3 NHÓM EDGE ĐẦU TIÊN]:")
        print(json.dumps(rel_groups[:3], ensure_ascii=False, indent=4))

        return result
    except Exception as e:
        print(f"❌ Lỗi khi gọi AI Gom nhóm: {e}")
        return {"node_groups": [], "rel_groups": []}

def final_ai_merge(graph_data, ai_mapping):
    """Sử dụng Từ điển của AI để gộp đồ thị, cộng dồn SID và Sửa lại kết nối Edge"""
    print(f"\n{'='*50}")
    print("--- ĐANG TIẾN HÀNH GỘP ĐỒ THỊ DỰA TRÊN TỪ ĐIỂN AI ---")

    # 1. Tạo bảng tra cứu nhanh từ Biến thể (variant) -> Tên Chuẩn (master)
    node_map = {}
    for group in ai_mapping.get("node_groups", []):
        master = str(group.get("master", "")).strip().lower()
        for v in group.get("variants", []):
            node_map[str(v).strip().lower()] = master

    rel_map = {}
    for group in ai_mapping.get("rel_groups", []):
        master = str(group.get("master", "")).strip().lower()
        for v in group.get("variants", []):
            rel_map[str(v).strip().lower()] = master

    print(f"  [+] Đã lập bộ nhớ Cache: Sẵn sàng tra cứu {len(node_map)} biến thể Node và {len(rel_map)} biến thể Edge.")

    master_nodes = {}
    id_translation = {} # Lưu vết ID cũ chuyển sang ID mới để nối Edge cho chuẩn
    node_counter = 1

    print("  [+] Bắt đầu quét và gộp Nodes...")
    # 2. XỬ LÝ GỘP NODES
    for n in graph_data.get("nodes", []):
        raw_name = str(n.get("name", "")).strip().lower()
        if not raw_name: continue

        # Tra từ điển AI, nếu AI sót thì giữ nguyên tên cũ
        target_master = node_map.get(raw_name, raw_name)

        sids = set(n.get("listSectionId", []))
        syns = set(n.get("synonym", []))
        syns.add(raw_name)

        if target_master not in master_nodes:
            nid = f"N{node_counter:03d}"
            master_nodes[target_master] = {
                "id": nid,
                "name": target_master,
                "listSectionId": sids,
                "synonym": syns
            }
            node_counter += 1
        else:
            master_nodes[target_master]["listSectionId"].update(sids)
            master_nodes[target_master]["synonym"].update(syns)

        id_translation[n["id"]] = master_nodes[target_master]["id"]

    print(f"      => Số Nodes thô: {len(graph_data.get('nodes', []))} --> Giảm còn {len(master_nodes)} Master Nodes.")

    print("  [+] Bắt đầu bẻ hướng kết nối và gộp Edges...")
    # 3. XỬ LÝ GỘP EDGES & CẬP NHẬT KẾT NỐI
    master_edges = {}
    edge_counter = 1
    for e in graph_data.get("relationships", []):
        # Bẻ lại hướng mũi tên theo ID mới của Node đã gộp
        src_id = id_translation.get(e.get("source"))
        tgt_id = id_translation.get(e.get("target"))

        if not src_id or not tgt_id: continue # Bỏ qua Edge rác không nối đi đâu

        raw_name = str(e.get("name", "")).strip().lower()
        target_master = rel_map.get(raw_name, raw_name)

        sids = set(e.get("listSectionId", []))
        syns = set(e.get("synonym", []))
        syns.add(raw_name)

        # Khóa gộp khắt khe: Edge chỉ gộp khi trùng Nguồn + Đích + Tên Chuẩn AI
        key = (src_id, tgt_id, target_master)
        if key not in master_edges:
            master_edges[key] = {
                "id": f"E{edge_counter:03d}",
                "name": target_master,
                "source": src_id,
                "target": tgt_id,
                "listSectionId": sids,
                "synonym": syns
            }
            edge_counter += 1
        else:
            master_edges[key]["listSectionId"].update(sids)
            master_edges[key]["synonym"].update(syns)

    print(f"      => Số Edges thô: {len(graph_data.get('relationships', []))} --> Giảm còn {len(master_edges)} Master Edges.")

    # 4. CHUYỂN SET VỀ LIST VÀ SẮP XẾP CHO ĐẸP
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

    return [{"nodes": final_nodes, "relationships": final_edges}]

def main():
    # File đầu vào (Chính là file Checkpoint sinh ra từ Bước 1)
    input_file = './version_2/1_output_data_for_graph.json'

    # File thành phẩm cuối cùng
    output_file = './version_2/1.1_output_data_for_graph.json'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}. Vui lòng chạy Bước 1 trước.")
        return

    print("Đang nạp dữ liệu đồ thị thô từ Bước 1...")
    start_time = time.time()

    raw_data = load_data(input_file)

    # Kiểm tra cấu trúc mảng ngoài cùng
    if isinstance(raw_data, list) and len(raw_data) > 0:
        graph_data = raw_data[0]
    else:
        graph_data = raw_data

    print(f"Dữ liệu ban đầu: {len(graph_data.get('nodes', []))} Nodes | {len(graph_data.get('relationships', []))} Edges")

    # BƯỚC 2.1: Gọi AI lấy bộ từ điển gom nhóm
    ai_mapping = get_ai_semantic_mapping(graph_data)

    # BƯỚC 2.2: Áp dụng từ điển để gộp toàn bộ Đồ thị
    final_output = final_ai_merge(graph_data, ai_mapping)

    # Ghi ra file đích
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"\n{'*'*50}")
    print(f"🎉 HOÀN TẤT BƯỚC 2: AI GOM NHÓM NGỮ NGHĨA!")
    print(f"⏳ Tổng thời gian: {time.time() - start_time:.2f} giây")
    print(f"💾 Lưu thành phẩm tại: {output_file}")
    if final_output:
        print(f"📊 THỐNG KÊ ĐỒ THỊ SAU GOM NHÓM:")
        print(f"  🔹 Tổng số Nodes: {len(final_output[0]['nodes'])}")
        print(f"  🔹 Tổng số Edges: {len(final_output[0]['relationships'])}")
    print(f"{'*'*50}")

if __name__ == "__main__":
    main()