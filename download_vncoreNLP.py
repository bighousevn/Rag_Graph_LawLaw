import py_vncorenlp
import os
import urllib.request

def custom_download_model(save_dir):
    if save_dir.endswith('/'):
        save_dir = save_dir[:-1]

    print(f"Creating directories in {save_dir}...")
    for sub in ["", "/models", "/models/dep", "/models/ner", "/models/postagger", "/models/wordsegmenter"]:
        os.makedirs(save_dir + sub, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master"

    files_to_download = [
        ("VnCoreNLP-1.2.jar", "/VnCoreNLP-1.2.jar"),
        ("models/wordsegmenter/vi-vocab", "/models/wordsegmenter/vi-vocab"),
        ("models/wordsegmenter/wordsegmenter.rdr", "/models/wordsegmenter/wordsegmenter.rdr"),
        ("models/postagger/vi-tagger", "/models/postagger/vi-tagger"),
        ("models/ner/vi-500brownclusters.xz", "/models/ner/vi-500brownclusters.xz"),
        ("models/ner/vi-ner.xz", "/models/ner/vi-ner.xz"),
        ("models/ner/vi-pretrainedembeddings.xz", "/models/ner/vi-pretrainedembeddings.xz"),
        ("models/dep/vi-dep.xz", "/models/dep/vi-dep.xz")
    ]

    for relative_url_path, local_relative_path in files_to_download:
        url = f"{base_url}/{relative_url_path}"
        dest = save_dir + local_relative_path

        if not os.path.exists(dest):
            print(f"Downloading {url} to {dest}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print("Download complete.")
            except Exception as e:
                print(f"❌ Failed to download {url}: {e}")
                # clean up partial download
                if os.path.exists(dest):
                    os.remove(dest)
                raise e
        else:
            print(f"File already exists: {dest}")

# 1. Đường dẫn lưu model (Nên dùng đường dẫn tuyệt đối)
model_dir = os.path.abspath('./vncorenlp_model')

# 2. Tự động tải model nếu chưa có (chỉ cần chạy 1 lần)
jar_path = os.path.join(model_dir, 'VnCoreNLP-1.2.jar')
if not os.path.exists(jar_path):
    print("VnCoreNLP model files not found. Starting custom download...")
    custom_download_model(model_dir)
else:
    print("VnCoreNLP model files already exist.")

# 3. Khởi tạo model với các tác vụ cần thiết: Word Segmentation, POS Tagging, Dependency Parsing
# Bỏ qua NER nếu bài toán của bạn chỉ tập trung vào trích xuất quan hệ (Subject-Verb-Object)
print("Initializing VnCoreNLP...")
nlp = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "parse"], save_dir=model_dir)
print("VnCoreNLP initialized successfully!")