"""
Bước 0: Chạy hàng loạt 4 bước (1_extract_triplets → 2_embedding_triplets → 3_query_flow →
4_answer) cho DANH SÁCH câu hỏi, thay vì chỉ 1 câu hỏi trong input/1_question.txt.

KHÔNG đổi gì trong 4 script — với mỗi câu hỏi: ghi câu hỏi vào input/1_question.txt (đúng file
4 script vẫn đọc/ghi), gọi tuần tự 4 script bằng subprocess (script sau chỉ chạy sau khi script
trước chạy xong), rồi copy toàn bộ output sinh ra sang output/batch/<số thứ tự>/ trước khi sang
câu hỏi tiếp theo — tránh câu sau ghi đè output của câu trước.

Input:  input/questions.txt   mỗi dòng 1 câu hỏi (bỏ dòng rỗng)
Output: output/batch/0001/    (1_normalized_question.txt, question_triplets.json,
                                question_triplets_vectorized.json, query_result.json, 4_answer.txt)
        output/batch/0002/    ... (tương tự cho từng câu hỏi)
        output/batch/summary.json   tổng hợp câu hỏi gốc + câu trả lời cuối của TOÀN BỘ câu hỏi

Lưu ý: mỗi câu hỏi chạy lại từ đầu (gọi LLM 2 lần ở bước 1, nạp lại model embedding ở bước 2,
nạp lại index ở bước 3, gọi LLM 1 lần ở bước 4) — không tối ưu tái sử dụng model/index giữa các
câu hỏi. Chấp nhận được ở quy mô vài/vài chục câu hỏi; nếu sau này cần chạy hàng trăm câu hỏi thì
mới đáng để tối ưu (tái sử dụng model đã nạp thay vì gọi subprocess).
"""

import os
import sys
import json
import shutil
import subprocess

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(BASE_DIR, "input", "questions.txt")
QUESTION_FILE  = os.path.join(BASE_DIR, "input", "1_question.txt")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
BATCH_DIR      = os.path.join(OUTPUT_DIR, "batch")

STEPS = ["1_extract_triplets.py", "2_embedding_triplets.py", "3_query_flow.py", "4_answer.py"]

FILES_TO_ARCHIVE = [
    "1_normalized_question.txt",
    "question_triplets.json",
    "question_triplets_vectorized.json",
    "query_result.json",
    "4_answer.txt",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def run_step(script: str) -> None:
    result = subprocess.run([sys.executable, os.path.join(BASE_DIR, script)], cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{script} thất bại (exit code {result.returncode}) — dừng batch.")


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    log(f"Đã tải {len(questions)} câu hỏi từ {QUESTIONS_FILE}")

    os.makedirs(BATCH_DIR, exist_ok=True)
    summary = []

    for i, question in enumerate(questions, 1):
        qdir = os.path.join(BATCH_DIR, f"{i:04d}")
        os.makedirs(qdir, exist_ok=True)
        log(f"\n{'='*70}\n[{i}/{len(questions)}] {question}\n{'='*70}")

        with open(QUESTION_FILE, "w", encoding="utf-8") as f:
            f.write(question)

        for step in STEPS:
            log(f"\n--- Chạy {step} ---")
            run_step(step)

        for fname in FILES_TO_ARCHIVE:
            src = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(qdir, fname))

        answer = ""
        answer_path = os.path.join(qdir, "4_answer.txt")
        if os.path.exists(answer_path):
            with open(answer_path, encoding="utf-8") as f:
                answer = f.read().strip()

        summary.append({"index": i, "question": question, "answer": answer, "dir": f"{i:04d}"})

    summary_path = os.path.join(BATCH_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n\nHoàn tất {len(questions)} câu hỏi. Tổng hợp → {summary_path}")


if __name__ == "__main__":
    main()
