import pdfplumber
import re
import json
import os

def parse_pdf(pdf_path, output_path):
    print(f"Reading {pdf_path}...")
    pdf = pdfplumber.open(pdf_path)
    lines = []
    
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            for line in text.split('\n'):
                line = line.strip()
                # Skip standalone page numbers
                if re.match(r'^\d+$', line):
                    continue
                if line:
                    lines.append(line)
                    
    sections = []
    
    current_phan = ""
    current_chuong = ""
    current_muc = ""
    current_dieu = ""
    current_khoan = ""
    current_diem = ""
    
    # regexes
    re_phan = re.compile(r'^(PHẦN THỨ\s+[\w\s]+)(.*)$', re.IGNORECASE)
    re_chuong = re.compile(r'^(CHƯƠNG\s+[IVXLCDM]+)(.*)$', re.IGNORECASE)
    re_muc = re.compile(r'^(Mục\s+\d+)(.*)$', re.IGNORECASE)
    re_dieu = re.compile(r'^(Điều\s+\d+[a-z]*)\.(.*)$', re.IGNORECASE)
    re_khoan = re.compile(r'^(\d+)\.(.*)$')
    re_diem = re.compile(r'^([a-zđ])\)(.*)$')
    
    current_text = []
    section_id_counter = 1
    
    def flush_section():
        nonlocal current_text, section_id_counter
        if not current_text:
            return
            
        # Build breadcrumb
        breadcrumb = []
        if current_phan: breadcrumb.append(current_phan)
        if current_chuong: breadcrumb.append(current_chuong)
        if current_muc: breadcrumb.append(current_muc)
        if current_dieu: breadcrumb.append(current_dieu)
        if current_khoan: breadcrumb.append(f"Khoản {current_khoan}")
        if current_diem: breadcrumb.append(f"Điểm {current_diem}")
        
        full_text = " - ".join(breadcrumb) + ". Nội dung: " + " ".join(current_text)
        
        sec_level = "diem" if current_diem else ("khoan" if current_khoan else "dieu")
        
        sections.append({
            "id": f"s{section_id_counter}",
            "level": sec_level,
            "text_content": full_text
        })
        section_id_counter += 1
        current_text = []

    for line in lines:
        m_phan = re_phan.match(line)
        m_chuong = re_chuong.match(line)
        m_muc = re_muc.match(line)
        m_dieu = re_dieu.match(line)
        m_khoan = re_khoan.match(line)
        m_diem = re_diem.match(line)
        
        if m_phan:
            flush_section()
            current_phan = m_phan.group(1).strip()
            # reset children
            current_chuong = current_muc = current_dieu = current_khoan = current_diem = ""
            if m_phan.group(2).strip():
                current_text.append(m_phan.group(2).strip())
        elif m_chuong:
            flush_section()
            current_chuong = m_chuong.group(1).strip()
            current_muc = current_dieu = current_khoan = current_diem = ""
            if m_chuong.group(2).strip():
                current_text.append(m_chuong.group(2).strip())
        elif m_muc:
            flush_section()
            current_muc = m_muc.group(1).strip()
            current_dieu = current_khoan = current_diem = ""
            if m_muc.group(2).strip():
                current_text.append(m_muc.group(2).strip())
        elif m_dieu:
            flush_section()
            current_dieu = m_dieu.group(1).strip()
            current_khoan = current_diem = ""
            if m_dieu.group(2).strip():
                current_text.append(m_dieu.group(2).strip())
        elif m_khoan:
            flush_section()
            current_khoan = m_khoan.group(1).strip()
            current_diem = ""
            if m_khoan.group(2).strip():
                current_text.append(m_khoan.group(2).strip())
        elif m_diem:
            flush_section()
            current_diem = m_diem.group(1).strip()
            if m_diem.group(2).strip():
                current_text.append(m_diem.group(2).strip())
        else:
            current_text.append(line)
            
    flush_section()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
        
    print(f"Total sections extracted: {len(sections)}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parse_pdf('input/15_vphc.pdf', 'output/1_sections.json')
