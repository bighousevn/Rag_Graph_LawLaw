"""
Bước 4: Parse ontology_168.md → ontology_168.json
Chạy một lần duy nhất, không cần LLM.
Input:  ontology_168.md (cùng thư mục)
Output: ontology_168.json (cùng thư mục)
"""

import re
import json
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_MD    = os.path.join(BASE_DIR, "ontology_168.md")
OUTPUT_JSON = os.path.join(BASE_DIR, "ontology_168.json")


def extract_table_value(lines: list[str], key: str) -> str:
    """Lấy cột Giá trị từ dòng bảng markdown có Thành phần = key."""
    for line in lines:
        if f"| {key} |" in line:
            parts = [p.strip() for p in line.split("|")]
            # parts: ['', 'Key', 'Type', 'Value', '']
            if len(parts) >= 4:
                return parts[3]
    return ""


def parse_keyphrases(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def parse_concept_block(lines: list[str]) -> dict | None:
    name = extract_table_value(lines, "Name")
    kps  = parse_keyphrases(extract_table_value(lines, "Keyphrases"))
    if not name:
        return None
    return {"name": name, "keyphrases": kps}


def parse_relation_block(lines: list[str]) -> dict | None:
    name       = extract_table_value(lines, "Name")
    kps        = parse_keyphrases(extract_table_value(lines, "Keyphrases"))
    concept_s  = extract_table_value(lines, "ConcKeyS").strip()
    concept_o  = parse_keyphrases(extract_table_value(lines, "ConcKeyO"))
    if not name:
        return None
    return {"name": name, "keyphrases": kps, "concept_s": concept_s, "concept_o": concept_o}


def parse_ontology(md_path: str) -> dict:
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    concepts:  list[dict] = []
    relations: list[dict] = []

    current_type:  str | None = None  # "C" or "R"
    current_block: list[str]  = []

    def flush():
        if not current_block:
            return
        if current_type == "C":
            entry = parse_concept_block(current_block)
            if entry:
                concepts.append(entry)
        elif current_type == "R":
            entry = parse_relation_block(current_block)
            if entry:
                relations.append(entry)

    for line in lines:
        m = re.match(r"^### Bảng (C|R)\d+:", line)
        if m:
            flush()
            current_type  = m.group(1)
            current_block = [line]
        elif current_type:
            current_block.append(line)

    flush()
    return {"concepts": concepts, "relations": relations}


def main():
    ontology = parse_ontology(INPUT_MD)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)
    print(f"{len(ontology['concepts'])} concepts, {len(ontology['relations'])} relations → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
