import py_vncorenlp
import os

# 1. Đường dẫn tuyệt đối tới thư mục model vncorenlp_model ở thư mục gốc
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(base_dir, '../../vncorenlp_model'))

print(f"Loading VnCoreNLP model from: {model_dir}")
# 2. Khởi tạo model từ thư mục local đã tải đầy đủ model
model = py_vncorenlp.VnCoreNLP(save_dir=model_dir)

# 3. Chạy thử chú thích văn bản (Word Segmentation, POS Tagging, NER, Dependency Parsing)
text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
print("\n--- KẾT QUẢ PHÂN TÍCH CÚ PHÁP ---")
model.print_out(model.annotate_text(text))