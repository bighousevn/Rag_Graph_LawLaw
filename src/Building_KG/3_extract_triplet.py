import py_vncorenlp
import os
import sys

# Khởi tạo model với đầy đủ 3 annotator: tách từ, từ loại, và phân tích cú pháp
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(base_dir, '../../vncorenlp_model'))
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "parse"], save_dir=model_dir)

# Tập hợp các tiền tố ý định cần loại bỏ khỏi hành động cốt lõi
# Việc loại bỏ các từ này giúp chuẩn hóa quan hệ (Relation), ví dụ:
# "muốn thành lập" -> Relation chuẩn là "thành lập"
INTENTIONAL_PREFIXES = {"muốn", "cần", "phải", "được", "bị", "nên", "sẽ", "đang", "đã"}

# Từ phủ định: nếu là con trực tiếp của động từ, phải giữ lại trong Relation
# (vd. "không chấp hành" mang nghĩa NGƯỢC với "chấp hành", không thể bỏ qua).
NEGATORS = {"không", "chưa", "chẳng"}

# Quan hệ dependency biểu thị LIỆT KÊ song song (vd. "A, B và C") — mỗi vế nên tách
# thành 1 triplet riêng thay vì gộp/lẫn vào nhau.
COORD_LABELS = {"conj", "coord"}

# Dấu câu nên loại khi ghép cụm từ (giữ lại dấu phẩy để cụm liệt kê còn đọc tự nhiên)
DROP_PUNCT = {".", ";", ":"}

# Nhãn dependency biểu thị bổ ngữ mệnh đề (clausal complement) — vd: "quy định rằng..."
# Các nhãn này cũng là tân ngữ tiềm năng nếu không tìm thấy dob/pob.
CLAUSAL_OBJ_LABELS = {"comp", "vmod"}


def build_children_map(tokens_by_index):
    """Xây dựng map parent -> [children indices] từ dependency tree.
    Nên gọi MỘT LẦN cho mỗi câu, rồi truyền vào các hàm khác để tránh duyệt brute-force O(n²)."""
    children = {idx: [] for idx in tokens_by_index}
    for tok in tokens_by_index.values():
        if tok['head'] in children:
            children[tok['head']].append(tok['index'])
    return children


def find_subject_index(verb_idx, sentence, tokens_by_index, children_map, prior_object_spans=None):
    """Trả về INDEX của token chủ ngữ (không phải text), để có thể tái dựng cả cụm từ sau đó.

    `prior_object_spans`: list các (head_idx, token_indices) của OBJECT thuộc những triplet đã
    trích xuất trước đó TRONG CÙNG CÂU, theo thứ tự xuất hiện. Dùng để xử lý mệnh đề quan hệ rút
    gọn kiểu "người mở cửa xe gây tai nạn giao thông" — "gây" không có chủ ngữ riêng (gapped
    subject) và parser gắn nó làm con của "xe"/"cửa xe", chứ không phải con trực tiếp của "mở".
    Nếu chỉ đi ngược cây tìm 'sub' kế thừa từ mệnh đề ngoài (case 3), sẽ vớ nhầm "người" làm chủ
    ngữ của "gây", trong khi ngữ nghĩa thực sự là "cửa xe gây tai nạn". Vì vậy, nếu cha trực tiếp
    của động từ rơi vào đúng cụm từ vừa là OBJECT của một triplet trước, cụm đó mới là chủ ngữ
    thật của động từ này — ưu tiên kiểm tra ngay sau khi case 1 (chủ ngữ trực tiếp) không có kết
    quả, trước khi thử các fallback yếu hơn.
    """
    # 1. Tìm chủ ngữ trực tiếp của động từ — tín hiệu đáng tin cậy nhất, luôn ưu tiên trước.
    for child_idx in children_map.get(verb_idx, []):
        child = tokens_by_index[child_idx]
        if child['depLabel'] in ['sub', 'nsubj']:
            return child['index']

    # 1.5 Mệnh đề quan hệ rút gọn nối tiếp một OBJECT vừa trích xuất — xem giải thích ở docstring.
    if prior_object_spans:
        parent_idx = tokens_by_index[verb_idx]['head']
        for head_idx, span in reversed(prior_object_spans):
            if parent_idx in span:
                return head_idx

    # 2. Câu có ROOT là DANH TỪ (không phải động từ như bình thường) là dấu hiệu parser đã
    # "rối" cấu trúc của 1 câu quá dài/nhiều tầng liệt kê (vd câu mở đầu liệt kê nhiều loại xe
    # rồi mới tới nhiều vị ngữ nối tiếp bằng coord/conj) — trong các câu này, việc dò 'sub' bằng
    # cách đi ngược lên cây dễ vớ nhầm một nhánh liệt kê ở giữa đường (vd "loại" trong "các loại
    # xe tương tự xe ô tô") làm chủ ngữ, trong khi danh từ ở ROOT chính là chủ ngữ thực sự dùng
    # chung cho mọi vị ngữ trong câu. Ưu tiên kiểm tra dấu hiệu này TRƯỚC khi đi ngược lên cây.
    root_token = next((t for t in sentence if t['depLabel'] == 'root'), None)
    if root_token is not None and root_token['posTag'] in ('N', 'Np'):
        return root_token['index']

    # 3. Trường hợp còn lại (ROOT là động từ như bình thường): duyệt ngược lên cây quan hệ
    # (traversing up) để tìm chủ ngữ thừa kế từ mệnh đề chứa động từ này.
    curr_idx = verb_idx
    visited = set()
    while curr_idx in tokens_by_index and curr_idx not in visited:
        visited.add(curr_idx)
        parent_idx = tokens_by_index[curr_idx]['head']
        if parent_idx == 0:
            break
        for child_idx in children_map.get(parent_idx, []):
            child = tokens_by_index[child_idx]
            if child['depLabel'] in ['sub', 'nsubj']:
                return child['index']
        curr_idx = parent_idx
    return None


def _is_real_verb_boundary(idx, children_map, tokens_by_index):
    """True nếu idx là động từ THỰC SỰ làm ranh giới một mệnh đề khác (có ít nhất 1 token con
    không phải dấu câu — tức governs một cụm bổ ngữ nào đó), không phải một từ chỉ bị
    POS-tagger gắn nhãn 'V' nhầm trong khi thực chất là một phần của cụm danh từ cố định
    (vd "lái_xe" trong "giấy phép lái xe" — bị tag 'V' nhưng chỉ có con là dấu câu cuối câu,
    không thực sự governs một mệnh đề nào).
    """
    tok = tokens_by_index[idx]
    if tok['posTag'] != 'V':
        return False
    return any(
        tokens_by_index[child]['posTag'] != 'CH'
        for child in children_map.get(idx, [])
    )


def collect_phrase(head_idx, tokens_by_index, children_map):
    """Thu thập cụm từ (subtree trong cây dependency) gắn với head_idx, để không bị mất các
    định ngữ/bổ ngữ như trước đây (vốn chỉ lấy đúng 1 token).

    Thuật toán gồm 2 bước:
    1. Duyệt cây dependency từ head_idx, loại trừ những nhánh RÕ RÀNG không thuộc cụm từ này
       (liệt kê song song, từ nối "và"/"hoặc" trơ trọi, dấu câu, hoặc một mệnh đề khác — xem
       các điều kiện trong vòng lặp). Mỗi token bị loại trừ được ghi vào `excluded` (khác với
       token chưa từng được xét tới).
    2. Lọc theo TÍNH LIÊN TỤC VỀ VỊ TRÍ: một cụm từ hợp lệ trong tiếng Việt luôn là một đoạn
       LIÊN TỤC các từ trong câu (cho phép có khoảng trống do các nhánh ở bước 1 bị loại trừ
       một cách CÓ CHỦ ĐÍCH). Nếu cây dependency nối head_idx tới một token cách xa, xuyên qua
       một vùng văn bản hoàn toàn KHÔNG được xét tới ở bước 1 (không nằm trong cụm, cũng không
       bị loại trừ có chủ đích — tức là parser gắn lệch qua một quãng văn bản không liên quan),
       đó chính là dấu hiệu của lỗi gắn nhãn xa (long-distance mis-attachment) — cụm từ bị cắt
       ngay tại điểm đó, không đi xa hơn. Đây là cách xử lý chắc chắn hơn việc dò tìm "động từ
       bị gắn lệch ở đâu trong subtree", vì lỗi gắn lệch luôn để lại 1 khoảng trống không liên
       tục trong câu, dù động từ bị lệch nằm ở độ sâu nào trong cây.

    Trả về (text, indices) — `indices` là tập index token thực sự thuộc cụm từ, để nơi gọi có
    thể đăng ký cụm này làm prior_object_spans cho find_subject_index() (object của triplet
    trước có thể là chủ ngữ thật của triplet sau, xem giải thích ở đó).
    """
    kept = {head_idx}
    excluded = set()
    stack = [head_idx]
    while stack:
        cur = stack.pop()
        for idx in children_map.get(cur, []):
            if idx in kept or idx in excluded:
                continue
            tok = tokens_by_index[idx]
            if (
                tok['depLabel'] in COORD_LABELS
                or tok['posTag'] == 'Cc'
                or (tok['posTag'] == 'CH' and tok['wordForm'] in DROP_PUNCT)
                or _is_real_verb_boundary(idx, children_map, tokens_by_index)
            ):
                excluded.add(idx)
                continue
            kept.add(idx)
            stack.append(idx)

    # Bước 2: thu hẹp về đoạn liên tục (kept ∪ excluded) chứa head_idx — cắt bỏ mọi token nằm
    # ngoài đoạn liên tục đó dù cây dependency có nối tới (xem giải thích ở trên).
    passable = kept | excluded
    lo = hi = head_idx
    while (lo - 1) in passable:
        lo -= 1
    while (hi + 1) in passable:
        hi += 1

    ordered = sorted(i for i in kept if lo <= i <= hi)
    text = ' '.join(tokens_by_index[i]['wordForm'].replace('_', ' ') for i in ordered)
    # Dấu phẩy còn sót lại ở đầu/cuối cụm là rác do nhánh liệt kê phía sau đã bị loại trừ —
    # không còn vế nào để liệt kê cùng nữa, nên dấu phẩy trơ trọi không còn ý nghĩa.
    return text.replace(' ,', ',').strip(' ,'), set(ordered)


def get_coordinated_indices(head_idx, tokens_by_index, children_map, max_index=None):
    """Trả về list index của TẤT CẢ các vế được liệt kê song song với head_idx (kể cả chính
    head_idx) — vd câu "A, B và C" sẽ trả về [idx_A, idx_B, idx_C] để tách thành 3 triplet
    riêng thay vì 1 triplet duy nhất bị mất 2 vế sau. `max_index` (dùng cho chủ ngữ) loại các
    vế liệt kê nằm sau động từ — xem giải thích ở collect_phrase()."""
    result = [head_idx]
    for child_idx in children_map.get(head_idx, []):
        tok = tokens_by_index[child_idx]
        if max_index is not None and tok['index'] > max_index:
            continue
        if tok['depLabel'] == 'conj':
            result.append(tok['index'])
        elif tok['depLabel'] == 'coord':
            # 'coord' thường là từ nối ("và"/"hoặc") — vế liệt kê thực sự là con của từ nối đó
            for sub_idx in children_map.get(tok['index'], []):
                sub = tokens_by_index[sub_idx]
                if sub['depLabel'] == 'conj':
                    if max_index is None or sub['index'] <= max_index:
                        result.append(sub['index'])
    return result


def extract_triplets(text):
    # Dọn dẹp dấu gạch dưới có sẵn để VnCoreNLP tự phân tách lại chuẩn xác
    cleaned_text = text.replace("_", " ")
    annotated_doc = model.annotate_text(cleaned_text)
    triplets = []

    for sentence_idx, sentence in annotated_doc.items():
        # Tạo dictionary tra cứu nhanh token theo index
        tokens_by_index = {token['index']: token for token in sentence}
        # Xây children_map MỘT LẦN cho mỗi câu — tất cả các hàm con đều dùng chung map này
        # thay vì mỗi hàm tự duyệt brute-force O(n) mỗi lần tìm con.
        children_map = build_children_map(tokens_by_index)

        # Cụm từ OBJECT của những triplet đã trích trước đó TRONG CÙNG CÂU — dùng để xử lý mệnh
        # đề quan hệ rút gọn nối tiếp (vd "người mở cửa xe gây tai nạn giao thông": "cửa xe" phải
        # là chủ ngữ của "gây", không phải "người"). Xem chi tiết ở find_subject_index().
        prior_object_spans = []

        # 1. Tìm các Động từ làm hạt nhân (Relation)
        for token in sentence:
            word = token['wordForm'].lower().replace("_", " ")
            pos = token['posTag']
            idx = token['index']

            if pos != 'V' or word in INTENTIONAL_PREFIXES:
                continue

            relation = token['wordForm'].replace("_", " ")

            # Giữ nghĩa phủ định: nếu có con trực tiếp là "không/chưa/chẳng", PHẢI gắn vào
            # Relation (vd "không chấp hành" khác hẳn "chấp hành").
            for child_idx in children_map.get(idx, []):
                child = tokens_by_index[child_idx]
                if child['wordForm'].lower() in NEGATORS:
                    relation = f"{child['wordForm']} {relation}".replace("_", " ")
                    break

            # Bị động: nếu động từ chính là con của "bị"/"được", gắn tiền tố đó vào Relation
            parent_idx = token['head']
            if parent_idx in tokens_by_index:
                parent_word = tokens_by_index[parent_idx]['wordForm'].lower()
                if parent_word in ("bị", "được"):
                    relation = f"{tokens_by_index[parent_idx]['wordForm']} {relation}".replace("_", " ")

            # 2. Tìm Chủ thể (Subject)
            subject_idx = find_subject_index(idx, sentence, tokens_by_index, children_map, prior_object_spans)

            # 3. Tìm Đối tượng (Object)
            object_idx = None

            # 3.1 Tìm tân ngữ trực tiếp (dob)
            for child_idx in children_map.get(idx, []):
                child = tokens_by_index[child_idx]
                if child['depLabel'] in ['dob', 'dobj']:
                    object_idx = child['index']
                    break

            # 3.2 Nếu không có dob, tìm tân ngữ qua giới từ (pob)
            if object_idx is None:
                for child_idx in children_map.get(idx, []):
                    prep_token = tokens_by_index[child_idx]
                    if prep_token['posTag'] in ['E', 'I']:
                        for pob_idx in children_map.get(prep_token['index'], []):
                            pob_token = tokens_by_index[pob_idx]
                            if pob_token['depLabel'] == 'pob':
                                object_idx = pob_token['index']
                                break
                        if object_idx is not None:
                            break

            # 3.3 Nếu vẫn không tìm thấy, thử tìm bổ ngữ mệnh đề (clausal complement).
            # Trong văn bản pháp luật hay gặp dạng "quy định rằng...", "yêu cầu người vi phạm
            # phải nộp phạt" — phần bổ ngữ mệnh đề chứa thông tin quan trọng cần trích xuất.
            if object_idx is None:
                for child_idx in children_map.get(idx, []):
                    child = tokens_by_index[child_idx]
                    if child['depLabel'] in CLAUSAL_OBJ_LABELS:
                        # Bổ ngữ mệnh đề thường là một động từ — tìm tân ngữ THỰC SỰ của nó
                        # (danh từ) để triplet có object là danh từ, dễ dùng trong KG hơn.
                        clause_obj = None
                        for grandchild_idx in children_map.get(child['index'], []):
                            grandchild = tokens_by_index[grandchild_idx]
                            if grandchild['depLabel'] in ['dob', 'dobj', 'sub', 'nsubj']:
                                clause_obj = grandchild['index']
                                break
                        # Nếu tìm được danh từ con → dùng nó; nếu không → dùng head của mệnh đề
                        object_idx = clause_obj if clause_obj is not None else child['index']
                        break

            if subject_idx is None or object_idx is None:
                continue

            # 4. Tách liệt kê (nếu có) thành nhiều triplet, rồi tái dựng cả cụm từ cho mỗi vế
            # (subject giới hạn index < idx của động từ — chủ ngữ tiếng Việt theo cấu trúc SVO
            # luôn nằm liền trước vị ngữ, nên 1 vế liệt kê chủ ngữ thật sự không thể nằm sau)
            subject_variants = get_coordinated_indices(subject_idx, tokens_by_index, children_map, max_index=idx - 1)
            object_variants = get_coordinated_indices(object_idx, tokens_by_index, children_map)

            # Dựng cụm OBJECT trước (dùng chung cho mọi subject variant), để có thể đăng ký vào
            # prior_object_spans một lần duy nhất sau khi xử lý xong động từ này.
            object_phrases = [
                (o_idx, *collect_phrase(o_idx, tokens_by_index, children_map))
                for o_idx in object_variants
            ]

            for s_idx in subject_variants:
                subject_text, _ = collect_phrase(s_idx, tokens_by_index, children_map)
                for o_idx, object_text, object_indices in object_phrases:
                    if subject_text and relation and object_text:
                        triplets.append({
                            "subject": subject_text,
                            "relation": relation,
                            "object": object_text,
                        })

            # Đăng ký OBJECT của động từ này — nếu một động từ sau trong câu gắn (head) vào đúng
            # cụm này, nó nên kế thừa cụm này làm chủ ngữ (xem find_subject_index()).
            for o_idx, object_text, object_indices in object_phrases:
                if object_text:
                    prior_object_spans.append((o_idx, object_indices))

    return triplets


# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    import json
    import time

    input_path = os.path.join(base_dir, "material_for_triplets/2_sections_rewritten_nghi_dinh_168_2024_1.json")
    output_path = os.path.join(base_dir, "material_for_triplets/3_triplets_extracted_nghi_dinh_168_2024_1.json")

    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file input: {input_path}")
        sys.exit(1)

    print(f"Đọc dữ liệu từ: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    results = []
    total_sections = len(sections)

    print(f"Bắt đầu trích xuất triplets từ {total_sections} sections...")
    start_time = time.time()

    for idx, section in enumerate(sections, 1):
        sec_id = section.get("id")
        propositions = section.get("rewritten_propositions", [])

        section_triplets = []
        seen_triplets = set()  # Khử trùng triplet trong phạm vi section (subject, relation, object)
        prop_to_triplets = {}  # Cache to avoid duplicate extraction for print

        for prop in propositions:
            if not prop or not isinstance(prop, str) or not prop.strip():
                continue
            try:
                extracted = extract_triplets(prop)
                prop_to_triplets[prop] = extracted
                for triplet in extracted:
                    key = (triplet["subject"], triplet["relation"], triplet["object"])
                    if key not in seen_triplets:
                        seen_triplets.add(key)
                        section_triplets.append(triplet)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý mệnh đề trong section {sec_id}: {e}")

        results.append({
            "id": sec_id,
            "document_name": section.get("document_name"),
            "level": section.get("level"),
            "path": section.get("path"),
            "original_text": section.get("text_content"),
            "rewritten_propositions": propositions,
            "triplets": section_triplets
        })

        # Chỉ in ra thông tin của 5 section đầu tiên làm mẫu để tránh tràn màn hình
        if idx <= 5:
            print("\n" + "="*60)
            print(f"Section {sec_id} | Path: {section.get('path')}")
            print(f"Văn bản gốc: {section.get('text_content')}")
            print("Mệnh đề & Triplets trích xuất được:")
            for prop in propositions:
                if prop in prop_to_triplets:
                    print(f"  - Mệnh đề: {prop}")
                    for triplet in prop_to_triplets[prop]:
                        print(f"    👉 ({triplet['subject']}) -[{triplet['relation']}]-> ({triplet['object']})")

        if idx % 50 == 0 or idx == total_sections:
            elapsed = time.time() - start_time
            print(f"-> Đã xử lý {idx}/{total_sections} sections ({elapsed:.1f}s)...")

    # Lưu kết quả ra file mới
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Hoàn thành! Đã trích xuất và lưu triplets tại: {output_path}")
