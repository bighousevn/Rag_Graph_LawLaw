import json
import os
from pathlib import Path
from triplet_extractor_vi import extract_triplets_with_models

def extract_to_graph(input_path, raw_triplets_path, graph_output_path):
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    sent_to_section = {}

    for item in data:
        sec_id = item['section_id']
        sentences = item.get('sentences', [])
        for i, sent in enumerate(sentences):
            sent_id = f"{sec_id}_sent_{i}"
            records.append({
                "id": sent_id,
                "text": sent,
                "source": sec_id
            })
            sent_to_section[sent_id] = sec_id

    print(f"Total sentences to process: {len(records)}")

    stopwords_file = Path("vietnamese_stopwords_legal.txt")
    stopwords = set()
    if stopwords_file.exists():
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    stopwords.add(line.strip().lower())

    vncore_dir = Path(".models/VnCoreNLP")
    phonlp_dir = Path(".models/phonlp")

    print("Running triplet extraction models... (This may take a while)")
    extracted = extract_triplets_with_models(
        records=records,
        stopwords=stopwords,
        vncore_model_dir=vncore_dir,
        phonlp_model_dir=phonlp_dir
    )

    # Save raw triplets as requested
    os.makedirs(os.path.dirname(raw_triplets_path), exist_ok=True)
    with open(raw_triplets_path, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, ensure_ascii=False, indent=4)
    print(f"Saved raw extracted triplets to {raw_triplets_path}")

    # Group results into new GraphData format
    nodes_map = {}
    edges_map = {}

    node_counter = 1
    edge_counter = 1

    # Danh sách các từ khóa cần loại bỏ
    blacklist = {"điều", "khoản", "điểm", "phần", "chương", "mục"}

    for item in extracted:
        sent_id = item['id']
        sec_id = sent_to_section.get(sent_id)
        if not sec_id: continue

        for tri in item.get('triplets', []):
            subj = tri.get('subject', '').strip()
            rel = tri.get('relation', '').strip()
            obj = tri.get('object', '').strip()

            if not subj or not obj or not rel:
                continue

            # Bỏ qua nếu bất kỳ thành phần nào nằm trong blacklist
            if subj.lower() in blacklist or rel.lower() in blacklist or obj.lower() in blacklist:
                continue

            # Process Subject Node
            if subj not in nodes_map:
                nodes_map[subj] = {
                    "id": f"N{node_counter:02d}",
                    "name": subj,
                    "list_section_id": [sec_id]
                }
                node_counter += 1
            elif sec_id not in nodes_map[subj]["list_section_id"]:
                nodes_map[subj]["list_section_id"].append(sec_id)

            # Process Object Node
            if obj not in nodes_map:
                nodes_map[obj] = {
                    "id": f"N{node_counter:02d}",
                    "name": obj,
                    "list_section_id": [sec_id]
                }
                node_counter += 1
            elif sec_id not in nodes_map[obj]["list_section_id"]:
                nodes_map[obj]["list_section_id"].append(sec_id)

            # Process Edge
            edge_key = f"{subj}|{rel}|{obj}"
            source_id = nodes_map[subj]["id"]
            target_id = nodes_map[obj]["id"]

            if edge_key not in edges_map:
                edges_map[edge_key] = {
                    "id": f"E{edge_counter:02d}",
                    "name": rel,
                    "source": source_id,
                    "target": target_id,
                    "list_section_id": [sec_id]
                }
                edge_counter += 1
            elif sec_id not in edges_map[edge_key]["list_section_id"]:
                edges_map[edge_key]["list_section_id"].append(sec_id)

    graph_data = [
        {
            "nodes": list(nodes_map.values()),
            "relationships": list(edges_map.values())
        }
    ]

    os.makedirs(os.path.dirname(graph_output_path), exist_ok=True)
    with open(graph_output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=4)

    print(f"Extracted {len(nodes_map)} unique nodes and {len(edges_map)} unique edges.")
    print(f"Saved graph data to {graph_output_path}")

if __name__ == "__main__":
    extract_to_graph('output/2_sentences.json', 'output/3_triplets_raw.json', 'output/3_graphdata.json')
