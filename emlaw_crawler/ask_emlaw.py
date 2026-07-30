"""
Tự động hỏi chatbot "Hỏi đáp" trên emlaw.vn và lưu câu hỏi–trả lời ra JSON.
Đầu vào: emlaw_crawler/question.xlsx (mỗi hàng cột A là một câu hỏi, không có header).
Chạy ẩn danh (không đăng nhập) qua Playwright, mỗi câu hỏi dùng một phiên hội thoại mới
(reload trang) để câu trả lời không bị ảnh hưởng bởi ngữ cảnh câu hỏi trước.

Lưu ý: chatbot giới hạn số lượt hỏi/ngày cho khách ẩn danh. Nếu gặp thông báo hết lượt,
script dừng lại, giữ nguyên kết quả đã thu được trong file JSON (ghi lại sau mỗi câu).
Chạy lại script vào lần sau sẽ tự động bỏ qua các câu đã có trả lời và hỏi tiếp phần còn lại.
"""

import json
import os
import time

import openpyxl
from playwright.sync_api import sync_playwright

# ==================== CẤU HÌNH ====================
CHAT_URL = "https://emlaw.vn/"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(OUTPUT_DIR, "question_dat_dai.xlsx")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "answer_dat_dai.json")

HEADLESS = True
ANSWER_TIMEOUT_S = 120          # tối đa chờ 1 câu trả lời
STABLE_CHECKS_REQUIRED = 2      # số lần poll liên tiếp text không đổi -> coi là đã xong
POLL_INTERVAL_S = 1.0
DELAY_BETWEEN_QUESTIONS_S = 3.0
# ===================================================

QUOTA_EXCEEDED_MARKER = "hết lượt"


def load_questions(path: str) -> list[str]:
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    if ws is None:
        raise ValueError(f"Không tìm thấy sheet nào trong {path}")
    return [
        str(row[0]).strip()
        for row in ws.iter_rows(values_only=True)
        if row and row[0] and str(row[0]).strip()
    ]


def load_existing_results(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_results(path: str, results: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def wait_for_answer(page):
    """Poll đến khi câu trả lời cuối cùng ổn định (không còn loading, text không đổi)."""
    deadline = time.time() + ANSWER_TIMEOUT_S
    last_text = None
    stable_count = 0

    while time.time() < deadline:
        loading = page.query_selector(".loading-shimmer-text, .loading-dot")
        prose_nodes = page.query_selector_all(
            '[data-component-name="ChatMessage"] .prose'
        )
        current_text = prose_nodes[-1].inner_text() if prose_nodes else ""

        if not loading and current_text and current_text == last_text:
            stable_count += 1
            if stable_count >= STABLE_CHECKS_REQUIRED:
                return current_text, False
        else:
            stable_count = 0

        last_text = current_text
        time.sleep(POLL_INTERVAL_S)

    return last_text or "", True  # timeout, có thể chưa hoàn chỉnh


def ask_question(page, question: str):
    page.goto(CHAT_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_selector('textarea[placeholder="Nhập câu hỏi..."]', timeout=15000)

    textarea = page.query_selector('textarea[placeholder="Nhập câu hỏi..."]')
    textarea.click()
    page.keyboard.type(question)
    page.wait_for_timeout(200)

    send_btn = page.query_selector('[data-tutorial="chat-send-button"]')
    send_btn.click()

    page.wait_for_timeout(1000)  # để bubble user + khung trả lời render trước khi poll

    body_text = page.inner_text("body")
    if QUOTA_EXCEEDED_MARKER in body_text:
        return "", True, False  # answer, quota_exceeded, timed_out

    answer, timed_out = wait_for_answer(page)

    # kiểm tra lại phòng trường hợp thông báo hết lượt xuất hiện muộn (sau khi bấm gửi)
    if QUOTA_EXCEEDED_MARKER in page.inner_text("body") and not answer:
        return "", True, False

    return answer, False, timed_out


def main():
    questions = load_questions(INPUT_FILE)
    results = load_existing_results(OUTPUT_FILE)

    already_done = len(results)
    print(f"Tổng số câu hỏi: {len(questions)}. Đã có sẵn: {already_done}.")

    if already_done >= len(questions):
        print("Không còn câu hỏi nào để hỏi thêm.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()

        for i in range(already_done, len(questions)):
            question = questions[i]
            stt = i + 1
            print(f"[{stt}/{len(questions)}] Đang hỏi: {question[:80]}")

            answer, quota_exceeded, timed_out = ask_question(page, question)

            if quota_exceeded:
                print("⚠️  Đã hết lượt hỏi ẩn danh trong ngày. Dừng script.")
                print(f"   Đã lưu {len(results)}/{len(questions)} câu vào {OUTPUT_FILE}")
                print("   Chạy lại script sau để tiếp tục các câu còn thiếu.")
                break

            if timed_out:
                print("   ⏱️  Quá thời gian chờ, lưu câu trả lời hiện có (có thể chưa đầy đủ).")

            results.append({"stt": stt, "question": question, "answer": answer})
            save_results(OUTPUT_FILE, results)
            print(f"   ✅ Đã lưu câu trả lời ({len(answer)} ký tự).")

            time.sleep(DELAY_BETWEEN_QUESTIONS_S)

        browser.close()

    print(f"\nHoàn tất. Kết quả tại: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
