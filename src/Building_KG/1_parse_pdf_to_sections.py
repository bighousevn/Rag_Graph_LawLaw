import pdfplumber
import re
import json
import os

class Node:
    def __init__(self, node_type, header, parent=None):
        self.node_type = node_type  # 'root', 'phan', 'chuong', 'muc', 'dieu', 'khoan', 'diem'
        self.header = header
        self.text_lines = []
        self.children = []
        self.parent = parent
        if parent:
            parent.children.append(self)

    def add_line(self, line):
        self.text_lines.append(line)

    @property
    def text_content(self):
        return " ".join(self.text_lines).strip()

def find_parent_for(node_type, current_node, root):
    levels = ['root', 'phan', 'chuong', 'muc', 'dieu', 'khoan', 'diem']
    try:
        target_idx = levels.index(node_type)
    except ValueError:
        return root

    temp = current_node
    while temp is not None:
        try:
            curr_idx = levels.index(temp.node_type)
        except ValueError:
            curr_idx = -1
        if curr_idx < target_idx:
            return temp
        temp = temp.parent
    return root

def get_leaf_nodes(node):
    leaves = []
    # Nếu nút root có text (ví dụ phần giới thiệu/căn cứ pháp lý ở đầu), coi nó là 1 nút nội dung lá
    if node.node_type == 'root' and node.text_content:
        leaves.append(node)

    is_legal_content = node.node_type in ('dieu', 'khoan', 'diem')
    has_legal_children = any(child.node_type in ('dieu', 'khoan', 'diem') for child in node.children)

    if is_legal_content and not has_legal_children:
        leaves.append(node)

    for child in node.children:
        leaves.extend(get_leaf_nodes(child))

    return leaves

def get_path_and_full_text(node):
    # Lấy danh sách các nút tổ tiên ngược lên root
    ancestors = []
    curr = node
    while curr is not None and curr.node_type != 'root':
        ancestors.append(curr)
        curr = curr.parent

    ancestors.reverse()

    path_parts = []
    text_parts = []

    for anc in ancestors:
        # Xây dựng đường dẫn phân cấp (path)
        level_name = ""
        if anc.node_type == 'phan':
            level_name = anc.header
        elif anc.node_type == 'chuong':
            level_name = anc.header
        elif anc.node_type == 'muc':
            level_name = anc.header
        elif anc.node_type == 'dieu':
            level_name = anc.header
        elif anc.node_type == 'khoan':
            level_name = f"Khoản {anc.header}"
        elif anc.node_type == 'diem':
            level_name = f"Điểm {anc.header}"

        if level_name:
            path_parts.append(level_name)

        # Xây dựng nội dung kết hợp ngữ cảnh cha (chỉ lấy dieu, khoan, diem)
        if anc.node_type in ('dieu', 'khoan', 'diem'):
            header_part = anc.header
            body_part = anc.text_content

            part_text = ""
            if anc.node_type == 'dieu':
                part_text = f"{header_part}. {body_part}".strip()
            elif anc.node_type == 'khoan':
                part_text = f"{header_part}. {body_part}".strip()
            elif anc.node_type == 'diem':
                part_text = f"{header_part}) {body_part}".strip()

            if part_text:
                text_parts.append(part_text)

    if node.node_type == 'root':
        full_text = node.text_content
    else:
        # Ghép các phần lại với nhau một cách tự nhiên
        cleaned_parts = []
        for part in text_parts:
            if not part:
                continue
            if cleaned_parts:
                prev = cleaned_parts[-1]
                # Nếu phần trước đó không kết thúc bằng dấu câu ngắt (., ;, :, ,), thêm dấu ". " để tách
                if prev[-1] not in ('.', ':', ';', ','):
                    cleaned_parts.append(". ")
                else:
                    cleaned_parts.append(" ")
            cleaned_parts.append(part)
        full_text = "".join(cleaned_parts)

    path = " - ".join(path_parts)
    return path, full_text

def parse_pdf(pdf_path, output_path, document_name="", prev_output_path=None):
    print(f"Reading {pdf_path}...")
    pdf = pdfplumber.open(pdf_path)
    lines = []

    for page in pdf.pages:
        text = page.extract_text()
        if text:
            for line in text.split('\n'):
                line = line.strip()
                # Bỏ qua số trang độc lập
                if re.match(r'^\d+$', line):
                    continue
                if line:
                    lines.append(line)

    # Khởi tạo cây phân cấp
    root = Node('root', 'Root')
    current_node = root

    # Các mẫu regex nhận diện cấp bậc
    re_phan = re.compile(r'^(PHẦN THỨ\s+[\w\s]+)(.*)$', re.IGNORECASE)
    re_chuong = re.compile(r'^(CHƯƠNG\s+[IVXLCDM]+)(.*)$', re.IGNORECASE)
    re_muc = re.compile(r'^(Mục\s+\d+)(.*)$', re.IGNORECASE)
    re_dieu = re.compile(r'^(Điều\s+\d+[a-z]*)\.(.*)$', re.IGNORECASE)
    re_khoan = re.compile(r'^(\d+)\.(.*)$')
    re_diem = re.compile(r'^([a-zđ])\)(.*)$')

    for line in lines:
        m_phan = re_phan.match(line)
        m_chuong = re_chuong.match(line)
        m_muc = re_muc.match(line)
        m_dieu = re_dieu.match(line)
        m_khoan = re_khoan.match(line)
        m_diem = re_diem.match(line)

        if m_phan:
            parent = find_parent_for('phan', current_node, root)
            current_node = Node('phan', m_phan.group(1).strip(), parent)
            if m_phan.group(2).strip():
                current_node.add_line(m_phan.group(2).strip())
        elif m_chuong:
            parent = find_parent_for('chuong', current_node, root)
            current_node = Node('chuong', m_chuong.group(1).strip(), parent)
            if m_chuong.group(2).strip():
                current_node.add_line(m_chuong.group(2).strip())
        elif m_muc:
            parent = find_parent_for('muc', current_node, root)
            current_node = Node('muc', m_muc.group(1).strip(), parent)
            if m_muc.group(2).strip():
                current_node.add_line(m_muc.group(2).strip())
        elif m_dieu:
            parent = find_parent_for('dieu', current_node, root)
            current_node = Node('dieu', m_dieu.group(1).strip(), parent)
            if m_dieu.group(2).strip():
                current_node.add_line(m_dieu.group(2).strip())
        elif m_khoan:
            parent = find_parent_for('khoan', current_node, root)
            current_node = Node('khoan', m_khoan.group(1).strip(), parent)
            if m_khoan.group(2).strip():
                current_node.add_line(m_khoan.group(2).strip())
        elif m_diem:
            parent = find_parent_for('diem', current_node, root)
            current_node = Node('diem', m_diem.group(1).strip(), parent)
            if m_diem.group(2).strip():
                current_node.add_line(m_diem.group(2).strip())
        else:
            current_node.add_line(line)

    # Lấy các nút lá nội dung pháp lý
    leaf_nodes = get_leaf_nodes(root)

    sections = []
    section_id_counter = 1
    if prev_output_path and os.path.exists(prev_output_path):
        try:
            with open(prev_output_path, 'r', encoding='utf-8') as f:
                prev_sections = json.load(f)
                if isinstance(prev_sections, list):
                    max_id_num = 0
                    for sec in prev_sections:
                        sec_id = sec.get("id", "")
                        match = re.match(r'^s(\d+)$', sec_id)
                        if match:
                            max_id_num = max(max_id_num, int(match.group(1)))
                    section_id_counter = max_id_num + 1
                    print(f"Loaded previous output file: {prev_output_path}. Starting section ID from: s{section_id_counter}")
        except Exception as e:
            print(f"Warning: Could not read previous output file {prev_output_path}: {e}")
    for leaf in leaf_nodes:
        path, full_text = get_path_and_full_text(leaf)
        if not full_text:
            continue
        sections.append({
            "id": f"s{section_id_counter}",
            "document_name": document_name,
            "level": leaf.node_type if leaf.node_type != 'root' else 'dieu',
            "text_content": full_text,
            "path": path
        })
        section_id_counter += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)

    print(f"Total sections extracted: {len(sections)}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parse_pdf(
        'version_4/Building_KG/law/trat_tu_atgt_1.pdf',
        'version_4/Building_KG/material_for_triplets/2_sections_trat_tu_atgt_1.json',
        document_name="Luật số: 36/2024/QH15",
        prev_output_path='version_4/Building_KG/material_for_triplets/1_sections_nghi_dinh_168_2024_1.json'
    )