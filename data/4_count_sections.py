import json
from collections import Counter

FILE_PATH = './data/3_results.json'

def count_and_sort_sections():
    try:
        # 1. Đọc file JSON
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{FILE_PATH}'. Vui lòng kiểm tra lại!")
        return

    # 2. Khởi tạo bộ đếm
    section_counter = Counter()

    # 3. Lặp qua từng object (node/edge) để thu thập listSectionId
    for item in data:
        # Dùng .get() để tránh lỗi nếu properties hoặc listSectionId không tồn tại
        properties = item.get('properties', {})
        list_sections = properties.get('listSectionId', [])
        
        # Hàm update tự động cộng dồn số lượng cho từng phần tử trong list
        section_counter.update(list_sections)

    # 4. Sắp xếp từ cao xuống thấp (hàm most_common tự động làm việc này)
    sorted_sections = section_counter.most_common()

    # 5. In kết quả ra màn hình với format dạng bảng cho dễ nhìn
    print("="*35)
    print(f"{'Section ID':<15} | {'Số lần xuất hiện':<15}")
    print("="*35)
    
    for section_id, count in sorted_sections:
        print(f"{section_id:<15} | {count:<15}")
        
    print("="*35)
    print(f"Tổng cộng có {len(sorted_sections)} section khác nhau.")

if __name__ == "__main__":
    count_and_sort_sections()