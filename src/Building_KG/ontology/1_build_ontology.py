"""
Phase 1: Build draft ontology from law text using LLM.

Run ONCE per new law document. Expert must review output before using in Phase 2.

Input:  ../material_for_triplets/2_sections_rewritten_*.json
Output:
  output/discovery_raw.json   ← accumulated per-batch results (auto-saved, resumable)
  output/ontology_draft.json  ← final merged ontology

Two-pass strategy:
  Pass 1 (discovery): For EACH batch of propositions, two grounded LLM calls:
                        1a. Extract entities and group them into concepts, reusing exact
                            names from the concept list accumulated so far (injected into
                            the prompt) instead of creating duplicate concepts with new names.
                        1b. Extract relations, constrained so concept_s/concept_o MUST be
                            names that exist in the just-updated concept list.
                      Results saved after EVERY batch → resumable on crash.
  Pass 2 (merge):     Concepts/relations are already grounded per-batch; this pass only
                      mops up leftover cross-batch duplicates (same concept named
                      differently in different batches) and re-applies the rename map to
                      relations before a final relation-synonym merge.

Grounding: concept_s/concept_o can never reference a name outside the known concept set —
enforced both by prompt injection and by code-level post-validation.
"""

import os
import json
import time
import argparse
import glob
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR     = os.path.join(BASE_DIR, "..", "material_for_triplets")
OUTPUT_DIR        = os.path.join(BASE_DIR, "output")
DISCOVERY_RAW     = os.path.join(OUTPUT_DIR, "discovery_raw.json")
DEFAULT_OUTPUT    = os.path.join(OUTPUT_DIR, "ontology_draft.json")

DISCOVERY_BATCH   = 30    # propositions per discovery call
MAX_CANDIDATES    = 350   # max candidates fed to merge (stay within token budget)
MERGE_CHUNK_SIZE  = 150   # if candidates > this, merge in chunks first


# ── Prompts ───────────────────────────────────────────────────────────────────

_ATOMIC_NOUN_RULES = """\
QUAN TRỌNG — Tên concept phải là danh từ NGUYÊN TỬ, không chứa động từ:
- "Người điều khiển xe ô tô" → KHÔNG phải 1 concept. Tách thành:
    concept "Người" (subject) + relation "Điều_khiển" + concept "Ô tô" (object)
- "Người điều khiển xe mô tô" → tương tự: "Người" + "Điều_khiển" + "Xe máy"
- Tên concept đúng: "Người", "Người đi bộ", "Cá nhân", "Tổ chức"
- "tài xế", "lái xe", "người điều khiển", "người điều khiển phương tiện" là keyphrases
  của concept "Người", KHÔNG phải tên concept riêng"""


def _format_concepts_for_prompt(concepts: list[dict], max_keyphrases: int = 4) -> str:
    if not concepts:
        return "  (chưa có concept nào)"
    lines = []
    for c in sorted(concepts, key=lambda x: x["name"]):
        kps = c.get("keyphrases", [])[:max_keyphrases]
        lines.append(f"  - {c['name']}: {', '.join(kps)}")
    return "\n".join(lines)


def _build_entity_discovery_system(existing_concepts: list[dict]) -> str:
    """Entity/concept extraction for one batch, grounded against concepts accumulated so far."""
    concepts_str = _format_concepts_for_prompt(existing_concepts)
    return f"""\
Bạn là chuyên gia xây dựng ontology cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

CONCEPTS đã xác định từ các batch trước (nếu entity bạn tìm thấy trùng nghĩa với một concept
dưới đây, BẮT BUỘC dùng lại CHÍNH XÁC tên "name" đó, chỉ bổ sung keyphrase mới nếu có.
CHỈ tạo tên concept mới khi thực sự không khớp concept nào có sẵn):
{concepts_str}

Từ các mệnh đề pháp lý dưới đây, hãy xác định và GOM NHÓM thành CONCEPTS — danh từ/cụm danh từ
đóng vai trò CHỦ THỂ hoặc ĐỐI TƯỢNG trong hành vi pháp lý.
Nhóm các cách diễn đạt ĐỒNG NGHĨA về cùng một loại thực thể → một entry duy nhất.
Ví dụ: "xe ô tô", "ô tô", "xe chở người bốn bánh có gắn động cơ", "xe bốn bánh", "xe hơi"
        → TẤT CẢ là keyphrases của concept canonical "Ô tô"
Bỏ qua: con số tiền phạt, số ngày, số lần cụ thể (không phải loại thực thể).

{_ATOMIC_NOUN_RULES}

Quy tắc về keyphrases:
- Thu thập ĐẦY ĐỦ mọi cách viết xuất hiện trong văn bản, dù dài hay ngắn
- KHÔNG lọc bỏ keyphrase vì "dài dòng" — keyphrase dài vẫn cần để nhận diện văn bản luật
- Mỗi concept phải có ít nhất 2-3 keyphrases nếu có nhiều cách gọi trong đoạn văn

Trả về JSON:
{{"concepts": [
  {{"name": "Ô tô", "keyphrases": ["xe ô tô", "ô tô", "xe hơi", "xe chở người bốn bánh có gắn động cơ",
                                  "xe chở hàng bốn bánh có gắn động cơ", "xe bốn bánh", "ô tô tải", "xe khách"]}},
  {{"name": "Người", "keyphrases": ["người", "người tham gia giao thông", "tài xế", "lái xe",
                                    "người điều khiển", "người điều khiển phương tiện"]}}
]}}"""


def _build_relation_discovery_system(concept_names: list[str]) -> str:
    """Relation extraction for one batch, constrained to the just-updated concept list."""
    names_str = "\n".join(f"  - {n}" for n in sorted(concept_names))
    return f"""\
Bạn là chuyên gia xây dựng ontology pháp lý cho lĩnh vực giao thông đường bộ Việt Nam.

CONCEPTS đã xác định (CHỈ được dùng tên CHÍNH XÁC từ danh sách này làm concept_s/concept_o,
KHÔNG được bịa tên ngoài danh sách):
{names_str}

Từ các mệnh đề pháp lý dưới đây, hãy xác định RELATIONS — động từ/cụm động từ thể hiện HÀNH VI
cốt lõi giữa hai concept.

Với mỗi relation, trả về:
- name: tên chuẩn dùng dấu gạch dưới (vd "Điều_khiển", "Sử_dụng")
- keyphrases: các cách diễn đạt động từ đồng nghĩa xuất hiện trong văn bản
- concept_s: MỘT tên từ danh sách CONCEPTS ở trên (chủ thể thực hiện hành vi)
- concept_o: DANH SÁCH tên từ danh sách CONCEPTS ở trên (đối tượng của hành vi)

Nguyên tắc:
- Mỗi relation có ĐÚNG MỘT concept_s
- Nếu chủ thể hoặc đối tượng của mệnh đề KHÔNG khớp concept nào trong danh sách trên
  → BỎ QUA mệnh đề đó, KHÔNG tự tạo tên concept mới ở bước này
- Nếu cùng hành vi có nhiều chủ thể rất khác nhau → tạo 2 relations riêng (tên khác nhau)

Trả về JSON:
{{"relations": [
  {{"name": "Điều_khiển", "keyphrases": ["điều khiển", "lái", "cầm lái"],
    "concept_s": "Người", "concept_o": ["Ô tô", "Xe máy"]}}
]}}"""


def _build_recovery_concept_system(existing_concepts: list[dict], missing_phrases: list[str]) -> str:
    """
    Relation extraction surfaced concept_s/concept_o phrases that don't resolve to any known
    concept — likely because the entity-discovery step missed them. Ask the LLM to place each
    phrase into an existing concept (reused by exact name) or, if genuinely new, name a proper
    atomic-noun concept for it, instead of silently dropping the relation that referenced it.
    """
    concepts_str = _format_concepts_for_prompt(existing_concepts)
    phrases_str = "\n".join(f"  - {p}" for p in missing_phrases)
    return f"""\
Bạn là chuyên gia xây dựng ontology cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

CONCEPTS đã xác định (nếu cụm từ dưới đây trùng nghĩa với concept nào ở đây, BẮT BUỘC dùng lại
CHÍNH XÁC tên "name" đó):
{concepts_str}

Các cụm từ sau được trích ra khi phân tích relation nhưng CHƯA khớp được concept nào ở trên
(nhiều khả năng do bước trích concept trước đó bỏ sót):
{phrases_str}

Với MỖI cụm từ, hãy xác định nó thuộc CONCEPT nào — tái sử dụng tên đã có nếu trùng nghĩa,
chỉ đặt tên mới khi thực sự là một loại thực thể chưa có trong danh sách.

{_ATOMIC_NOUN_RULES}

Nếu một cụm từ KHÔNG phải là loại thực thể hợp lệ (ví dụ chỉ là số tiền, số lượng, mức độ,
trạng từ...) thì BỎ QUA cụm đó, không tạo concept cho nó.

Trả về JSON:
{{"concepts": [
  {{"name": "...", "keyphrases": ["...", "..."]}}
]}}"""


CONCEPT_MERGE_SYSTEM = """\
Bạn là chuyên gia xây dựng ontology pháp lý.

Dưới đây là danh sách concepts đã được xác định từ nhiều batch riêng lẻ. Phần lớn đã là tên
chuẩn, nhưng có thể còn sót một số concept ĐỒNG NGHĨA bị tách thành nhiều entry khác tên
(ví dụ "Ô tô" và "Xe hơi" cùng một loại thực thể, do được đặt tên ở các batch khác nhau).

Nhiệm vụ: hợp nhất các concept đồng nghĩa còn sót lại thành 1 entry — gộp toàn bộ keyphrases
và source_names (loại trùng lặp). Concept nào đã rõ ràng không trùng ai thì giữ nguyên
(source_names chỉ chứa chính tên nó).

""" + _ATOMIC_NOUN_RULES + """

Trả về JSON:
{"concepts": [
  {"name": "Ô tô", "keyphrases": ["xe ô tô", "ô tô", "xe hơi"], "source_names": ["Ô tô", "Xe hơi"]},
  {"name": "Người", "keyphrases": ["người", "tài xế"], "source_names": ["Người"]}
]}"""


def _build_relation_final_merge_system(concept_names: list[str], concept_s: str) -> str:
    """
    Final relation merge for ONE concept_s group at a time (caller partitions relations by
    concept_s before calling this — see merge_pass Step 3). This lets the prompt state as fact
    that every input relation shares the same subject, instead of relying on the LLM to notice
    and never cross-merge subjects on its own.
    """
    names_str = "\n".join(f"  - {n}" for n in sorted(concept_names))
    return f"""\
Bạn là chuyên gia xây dựng ontology pháp lý.

CONCEPTS hợp lệ (concept_o CHỈ được dùng tên chính xác từ danh sách này):
{names_str}

Dưới đây là các relation đã gán concept_s/concept_o từ nhiều batch riêng lẻ. TẤT CẢ đều có
CÙNG concept_s = "{concept_s}" — không có relation nào của concept_s khác lẫn vào đây.
Có thể có relation ĐỒNG NGHĨA bị tách thành nhiều entry khác tên (cùng bản chất hành vi,
concept_o trùng hoặc gần trùng nhau).

Nhiệm vụ: hợp nhất các relation đồng nghĩa còn sót lại thành 1 entry — gộp keyphrases,
hợp nhất concept_o (loại trùng). Relation có ý nghĩa khác nhau dù chung concept_o thì
GIỮ RIÊNG, không gộp. concept_s của mọi entry trả về PHẢI giữ nguyên là "{concept_s}".

Trả về JSON:
{{"relations": [
  {{"name": "Điều_khiển", "keyphrases": ["điều khiển", "lái", "cầm lái"],
    "concept_s": "{concept_s}", "concept_o": ["Ô tô", "Xe máy"]}}
]}}"""


# ── LLM helpers ───────────────────────────────────────────────────────────────

def llm_call(messages: list, max_retries: int = 4) -> dict:
    backoff = 2
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(backoff)
            backoff *= 2
    return {}


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ── Pass 1: Discovery ─────────────────────────────────────────────────────────

def _build_concept_lookup(concepts: list[dict]) -> dict[str, list[str]]:
    """
    Map both canonical names and every keyphrase (lowercased) → ALL concept names that
    claim it. A keyphrase can legitimately belong to more than one concept (e.g. a generic
    word shared by two sibling concepts) — every caller must handle multiple matches instead
    of assuming the first one found is the only one.
    """
    lookup: dict[str, list[str]] = {}
    for c in concepts:
        name = c["name"]
        keys = [name.strip().lower()] + [kp.strip().lower() for kp in c.get("keyphrases", []) if kp]
        for key in keys:
            names = lookup.setdefault(key, [])
            if name not in names:
                names.append(name)
    return lookup


def _merge_into_concept_index(concept_index: dict[str, dict], new_concepts: list[dict]) -> int:
    """Fold {name, keyphrases} entries into the accumulated index; returns count of brand-new concepts."""
    new_count = 0
    for c in new_concepts:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        kps = [k for k in c.get("keyphrases", []) if k]
        if name in concept_index:
            existing_kps = concept_index[name]["keyphrases"]
            for k in kps:
                if k not in existing_kps:
                    existing_kps.append(k)
        else:
            concept_index[name] = {"name": name, "keyphrases": list(dict.fromkeys(kps))}
            new_count += 1
    return new_count


def _find_unresolved_phrases(relations: list[dict], concepts: list[dict]) -> list[str]:
    """Collect distinct concept_s/concept_o phrases in raw relations that match no known concept."""
    lookup = _build_concept_lookup(concepts)
    missing: list[str] = []
    seen: set[str] = set()
    for r in relations:
        candidates = [r.get("concept_s", "")] + list(r.get("concept_o", []) or [])
        for phrase in candidates:
            phrase = (phrase or "").strip()
            key = phrase.lower()
            if phrase and key not in seen and not lookup.get(key):
                missing.append(phrase)
                seen.add(key)
    return missing


def _referenced_concept_names(relations: list[dict]) -> set[str]:
    """Names actually used as concept_s or concept_o by at least one relation."""
    used: set[str] = set()
    for r in relations:
        cs = r.get("concept_s")
        if cs:
            used.add(cs)
        used.update(r.get("concept_o", []) or [])
    return used


def _post_validate_relations(relations: list[dict], concepts: list[dict]) -> list[dict]:
    """
    concept_s/concept_o from the LLM may be a keyphrase ("xe mô tô", "tài xế") rather than
    the canonical concept name ("Xe máy", "Người") — resolve through the keyphrase lookup
    before deciding validity, instead of requiring an exact canonical-name match.

    A keyphrase can match more than one concept. concept_o already accepts multiple concepts,
    so every match is kept. concept_s must stay singular per relation (ontology rule: one
    ConceptS per relation), so an ambiguous concept_s phrase duplicates the relation once per
    matching concept instead of arbitrarily keeping only one.

    - concept_s not resolvable to any known concept → drop relation
    - concept_o items not resolvable → remove invalid entries
    - concept_o ends up empty (no valid object left, or none was ever provided) → drop relation
    """
    lookup = _build_concept_lookup(concepts)
    validated = []
    for r in relations:
        cs_raw = (r.get("concept_s") or "").strip()
        cs_names = lookup.get(cs_raw.lower()) or []
        if not cs_names:
            print(f"    [drop] '{r.get('name')}': concept_s '{cs_raw}' not in concepts")
            continue
        if len(cs_names) > 1:
            print(f"    [split] '{r.get('name')}': concept_s '{cs_raw}' matches {cs_names} → duplicating relation")

        co_raw, co_valid, co_bad = r.get("concept_o", []), [], []
        for o in co_raw:
            resolved = lookup.get((o or "").strip().lower()) or []
            if not resolved:
                co_bad.append(o)
            for name in resolved:
                if name not in co_valid:
                    co_valid.append(name)
        if co_bad:
            print(f"    [warn] '{r.get('name')}': removed invalid concept_o: {co_bad}")
        if not co_valid:
            print(f"    [drop] '{r.get('name')}' (S={cs_names}): no valid concept_o left")
            continue

        for cs in cs_names:
            validated.append(dict(r, concept_s=cs, concept_o=list(co_valid)))
    return validated


def discovery_pass(
    propositions: list[str],
    batch_size: int = DISCOVERY_BATCH,
    raw_path: str = DISCOVERY_RAW,
) -> tuple[list, list]:
    """
    For each batch: (1) extract entities, grounding them against the concept list
    accumulated so far so synonyms reuse the same name instead of forking a new concept;
    (2) extract relations, constrained so concept_s/concept_o must be names that exist in
    that just-updated concept list; (3) if the relation step still surfaces a concept_s/
    concept_o phrase that resolves to nothing — a sign Step 1 missed an entity — recover it
    with one extra LLM call instead of silently dropping the triplet.
    Saves progress after each batch → resumable on crash.

    A concept can be discovered (Step 1) in one batch but only get referenced by a relation
    several batches later, once grounding lets a later batch's relation step reuse its name.
    So the full concept set (incl. not-yet-referenced ones) is kept internally in
    `concept_index` for grounding continuity across batches/resumes, while the "concepts"
    that get saved/returned are filtered down to only those actually used by some relation
    so far — orphan concepts never leak into the exported ontology.
    """
    relations: list[dict] = []
    props_done = 0
    concept_index: dict[str, dict] = {}

    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            saved = json.load(f)
        full_concepts = saved.get("concept_index_all") or saved.get("concepts", [])
        relations     = saved.get("relations", [])
        props_done    = saved.get("props_done", 0)
        for c in full_concepts:
            c["keyphrases"] = list(dict.fromkeys(c.get("keyphrases", [])))
            concept_index[c["name"]] = c
        print(f"  [resume] loaded {len(concept_index)} concepts ({len(saved.get('concepts', []))} used), "
              f"{len(relations)} relations ({props_done}/{len(propositions)} props already done)")

    remaining   = propositions[props_done:]
    batches     = [remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
    total_batches = (len(propositions) + batch_size - 1) // batch_size
    done_batches  = props_done // batch_size

    print(f"Discovery: {len(propositions)} props total | "
          f"{len(remaining)} remaining | {len(batches)} batches left")

    for i, batch in enumerate(batches, 1):
        batch_num = done_batches + i
        numbered  = "\n".join(f"{j+1}. {p}" for j, p in enumerate(batch))

        # ── Step A: extract entities, grounded against concepts accumulated so far ──
        try:
            entity_system = _build_entity_discovery_system(list(concept_index.values()))
            entity_result = llm_call([
                {"role": "system", "content": entity_system},
                {"role": "user",   "content": f"Mệnh đề pháp lý:\n{numbered}"},
            ])
        except Exception as e:
            print(f"  [batch {batch_num}/{total_batches}] entity LLM FAIL: {e} — skipping batch")
            continue

        batch_concepts = entity_result.get("concepts", []) or []
        new_count = _merge_into_concept_index(concept_index, batch_concepts)

        # ── Step B: extract relations, constrained to the updated concept list ──────
        concept_names = list(concept_index.keys())
        try:
            relation_system = _build_relation_discovery_system(concept_names)
            relation_result = llm_call([
                {"role": "system", "content": relation_system},
                {"role": "user",   "content": f"Mệnh đề pháp lý:\n{numbered}"},
            ])
        except Exception as e:
            print(f"  [batch {batch_num}/{total_batches}] relation LLM FAIL: {e} — keeping concepts, skipping relations")
            relation_result = {}

        batch_relations = relation_result.get("relations", []) or []

        # ── Step C: recover concepts for any concept_s/concept_o phrase Step A missed ──
        missing_phrases = _find_unresolved_phrases(batch_relations, list(concept_index.values()))
        recovered_count = 0
        if missing_phrases:
            try:
                recovery_system = _build_recovery_concept_system(list(concept_index.values()), missing_phrases)
                recovery_result = llm_call([
                    {"role": "system", "content": recovery_system},
                    {"role": "user",   "content": "Cụm từ cần xác định concept:\n" + "\n".join(missing_phrases)},
                ])
                recovered_count = _merge_into_concept_index(concept_index, recovery_result.get("concepts", []) or [])
                print(f"    [recover] {len(missing_phrases)} missing phrase(s) → +{recovered_count} concepts")
            except Exception as e:
                print(f"    [recover] LLM FAIL: {e} — unresolved phrases will be dropped")
            new_count += recovered_count

        batch_relations = _post_validate_relations(batch_relations, list(concept_index.values()))
        relations.extend(batch_relations)
        props_done += len(batch)

        used_names = _referenced_concept_names(relations)
        used_concepts = [c for c in concept_index.values() if c["name"] in used_names]

        print(f"  [{batch_num}/{total_batches}] "
              f"+{new_count} new concepts (discovered {len(concept_index)}, used {len(used_concepts)}), "
              f"+{len(batch_relations)} relations (total {len(relations)})")
        if batch_concepts:
            names = [c.get("name", "?") for c in batch_concepts]
            print(f"    concepts touched: {', '.join(names)}")
        if batch_relations:
            names = [r.get("name", "?") for r in batch_relations]
            print(f"    relations: {', '.join(names)}")

        # Save after every batch
        save_json(raw_path, {
            "props_done": props_done,
            "total_props": len(propositions),
            "concepts":   used_concepts,               # exported: only concepts used by a relation
            "concept_index_all": list(concept_index.values()),  # internal: full state, for grounding continuity
            "relations":  relations,
        })

    used_names = _referenced_concept_names(relations)
    return [c for c in concept_index.values() if c["name"] in used_names], relations


# ── Pass 2: Merge (mop up cross-batch duplicates left over from Pass 1) ───────

def _prep_concepts_for_merge(concepts: list[dict]) -> list[dict]:
    return [
        {
            "name": c["name"],
            "keyphrases": c.get("keyphrases", []),
            "source_names": c.get("source_names", [c["name"]]),
        }
        for c in concepts
    ]


def _merge_concepts_call(items: list[dict]) -> list[dict]:
    payload = json.dumps({"concepts": items}, ensure_ascii=False)
    result = llm_call([
        {"role": "system", "content": CONCEPT_MERGE_SYSTEM},
        {"role": "user",   "content": f"Concepts:\n{payload}"},
    ])
    return result.get("concepts", []) or []


def _pre_merge_concepts(items: list[dict]) -> list[dict]:
    """Pre-merge concepts in chunks to stay within token budget before the final pass."""
    chunks = [items[i:i+MERGE_CHUNK_SIZE] for i in range(0, len(items), MERGE_CHUNK_SIZE)]
    merged = []
    for ci, chunk in enumerate(chunks, 1):
        chunk_out = _merge_concepts_call(chunk)
        merged.extend(chunk_out)
        print(f"    concept chunk [{ci}/{len(chunks)}]: {len(chunk)} → {len(chunk_out)}")
    return merged


def _merge_relation_group(concept_s: str, group: list[dict], concept_names: list[str]) -> list[dict]:
    """
    Merge synonymous relations that all share the same concept_s (caller guarantees this by
    partitioning before calling — see merge_pass Step 3). Chunks first if the group is large,
    then does one final merge call. concept_s is force-set back on the output regardless of
    what the LLM returns, so this can never smuggle in a different subject.
    """
    if len(group) <= 1:
        return group

    system = _build_relation_final_merge_system(concept_names, concept_s)

    if len(group) > MERGE_CHUNK_SIZE:
        chunks = [group[i:i+MERGE_CHUNK_SIZE] for i in range(0, len(group), MERGE_CHUNK_SIZE)]
        pre_merged = []
        for ci, chunk in enumerate(chunks, 1):
            payload = json.dumps({"relations": chunk}, ensure_ascii=False)
            result = llm_call([
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Relations:\n{payload}"},
            ])
            chunk_out = result.get("relations", []) or []
            pre_merged.extend(chunk_out)
            print(f"    [{concept_s}] chunk [{ci}/{len(chunks)}]: {len(chunk)} → {len(chunk_out)}")
        group = pre_merged

    if len(group) > MAX_CANDIDATES:
        print(f"  ⚠ truncating '{concept_s}' relations {len(group)} → {MAX_CANDIDATES}")
        group = group[:MAX_CANDIDATES]

    if len(group) <= 1:
        return [dict(r, concept_s=concept_s) for r in group]

    payload = json.dumps({"relations": group}, ensure_ascii=False)
    result = llm_call([
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Relations:\n{payload}"},
    ])
    merged = result.get("relations", []) or group
    return [dict(r, concept_s=concept_s) for r in merged]


def merge_pass(concepts: list[dict], relations: list[dict]) -> dict:
    """
    Concepts and relations are already grounded per-batch from Pass 1, so this pass only
    mops up leftovers:
      Step 1: merge concepts still duplicated across batches (same thing, different name)
              → build a rename map from source_names back to the final chosen name
      Step 2: apply the rename map to relations' concept_s/concept_o
      Step 3: merge relations that are still synonymous after renaming
      Step 4: post-validate (safety net against any residual bad reference)
    """
    print(f"\n{'─'*60}")
    print(f"Merge pass: {len(concepts)} concept candidates, {len(relations)} relation candidates")

    # ── Step 1: final concept dedup ───────────────────────────────────────────
    prepped = _prep_concepts_for_merge(concepts)
    if len(prepped) > MERGE_CHUNK_SIZE:
        print(f"\n  [Step 1a] pre-merging {len(prepped)} concepts in chunks...")
        prepped = _pre_merge_concepts(prepped)
        print(f"  after pre-merge: {len(prepped)} concept candidates")

    if len(prepped) > MAX_CANDIDATES:
        print(f"  ⚠ truncating concepts {len(prepped)} → {MAX_CANDIDATES}")
        prepped = prepped[:MAX_CANDIDATES]

    print(f"\n  [Step 1] final concept merge on {len(prepped)} candidates...")
    final_concepts = _merge_concepts_call(prepped)
    if not final_concepts:
        print("  ⚠ WARNING: concept merge returned 0 concepts")

    rename_map: dict[str, str] = {}
    for c in final_concepts:
        for src in c.get("source_names", [c["name"]]):
            rename_map[src] = c["name"]
    concept_set = {c["name"] for c in final_concepts}
    # safety net: any original name the LLM forgot to echo back in source_names maps to itself
    for c in concepts:
        rename_map.setdefault(c["name"], c["name"])
        concept_set.add(rename_map[c["name"]])

    print(f"  ✓ {len(final_concepts)} concepts: {', '.join(sorted(concept_set))}")

    # ── Step 2: apply rename map to relations ─────────────────────────────────
    renamed_relations = []
    for r in relations:
        cs = rename_map.get(r.get("concept_s", ""), r.get("concept_s", ""))
        co = list(dict.fromkeys(rename_map.get(o, o) for o in r.get("concept_o", [])))
        renamed_relations.append(dict(r, concept_s=cs, concept_o=co))

    # ── Step 3: final relation synonym merge, partitioned by concept_s ────────
    # Merging across different concept_s must never happen — even when told not to, an LLM
    # can still conflate same-named relations belonging to different subjects (this caused
    # real data loss before: "Không_có" for "Người" vs for "Cơ sở đào tạo lái xe" got merged
    # into one, silently dropping the latter's concept_o). Partitioning by concept_s before
    # the call makes cross-subject merging structurally impossible instead of just discouraged.
    subject_groups: dict[str, list[dict]] = {}
    for r in renamed_relations:
        cs_key = str(r.get("concept_s", ""))
        subject_groups.setdefault(cs_key, []).append(r)

    print(f"\n  [Step 3] final relation merge, partitioned into {len(subject_groups)} concept_s group(s)...")
    rels_out: list[dict] = []
    for cs, group in subject_groups.items():
        merged_group = _merge_relation_group(cs, group, list(concept_set))
        rels_out.extend(merged_group)
        if len(merged_group) != len(group):
            print(f"    [{cs}] {len(group)} → {len(merged_group)} relations")
    if not rels_out:
        print("  ⚠ WARNING: relation merge returned 0 relations")

    # ── Step 4: post-validate ─────────────────────────────────────────────────
    print(f"\n  [Step 4] validating {len(rels_out)} relations against {len(concept_set)} concepts...")
    rels_out = _post_validate_relations(rels_out, final_concepts)

    # Print results
    final_concepts_out = [{"name": c["name"], "keyphrases": c.get("keyphrases", [])} for c in final_concepts]
    print(f"\n  ✓ Final: {len(final_concepts_out)} concepts, {len(rels_out)} relations")
    print(f"\n  CONCEPTS:")
    for c in final_concepts_out:
        kps = c.get("keyphrases", [])
        print(f"    [{c.get('name', '?')}]  ({len(kps)} keyphrases)")
        print(f"      {', '.join(kps[:6])}{'...' if len(kps) > 6 else ''}")

    print(f"\n  RELATIONS:")
    for r in rels_out:
        cs  = r.get("concept_s", "?")
        co  = r.get("concept_o", [])
        kps = r.get("keyphrases", [])
        print(f"    [{r.get('name', '?')}]  S={cs}  O=[{', '.join(co[:4])}{'...' if len(co)>4 else ''}]")
        print(f"      keyphrases: {', '.join(kps[:5])}{'...' if len(kps)>5 else ''}")

    return {"concepts": final_concepts_out, "relations": rels_out}


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_propositions(input_glob: str) -> list[str]:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    props = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            sections = json.load(f)
        for sec in sections:
            for p in sec.get("rewritten_propositions", []):
                if p and p.strip():
                    props.append(p.strip())
    return props


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Build draft ontology from law text")
    parser.add_argument("--input",       default=os.path.join(MATERIALS_DIR, "2_sections_rewritten_*.json"),
                        help="Glob pattern for rewritten sections JSON files")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT,
                        help="Final merged ontology output path")
    parser.add_argument("--raw",         default=DISCOVERY_RAW,
                        help="Per-batch discovery cache (auto-saved, used for resume)")
    parser.add_argument("--batch-size",  type=int, default=DISCOVERY_BATCH)
    parser.add_argument("--limit-props", type=int, default=0,
                        help="Limit number of propositions (0 = all, useful for testing)")
    parser.add_argument("--merge-only",  action="store_true",
                        help="Skip discovery, run merge on existing --raw file")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.merge_only:
        if not os.path.exists(args.raw):
            print(f"ERROR: --merge-only requires existing raw file: {args.raw}")
            return
        with open(args.raw, encoding="utf-8") as f:
            saved = json.load(f)
        concepts  = saved.get("concepts",  [])
        relations = saved.get("relations", [])
        print(f"Loaded from {args.raw}: {len(concepts)} concepts, {len(relations)} relations")
    else:
        print(f"Collecting propositions from: {args.input}")
        propositions = collect_propositions(args.input)
        if args.limit_props:
            propositions = propositions[:args.limit_props]
        print(f"Total propositions: {len(propositions)}")
        print(f"Discovery raw cache: {args.raw}\n")

        concepts, relations = discovery_pass(
            propositions, batch_size=args.batch_size, raw_path=args.raw
        )

    print(f"\n{'─'*60}")
    print("Running merge pass...")
    ontology = merge_pass(concepts, relations)

    save_json(args.output, ontology)
    n_c = len(ontology.get("concepts", []))
    n_r = len(ontology.get("relations", []))
    print(f"\n{'─'*60}")
    print(f"Done: {n_c} concepts, {n_r} relations → {args.output}")
    print("⚠  Expert review required before using in Phase 2 (2_extract_triplets.py)")
    print(f"   Discovery raw cache kept at: {args.raw}")


if __name__ == "__main__":
    main()
