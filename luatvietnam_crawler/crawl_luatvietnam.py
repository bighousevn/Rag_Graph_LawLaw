"""
LuatVietnam Q&A Crawler
Crawl câu hỏi & trả lời luật sư từ:
  https://luatvietnam.vn/luat-su-tu-van/hanh-chinh-27.html

Dùng requests + BeautifulSoup (không cần browser, không cần login).
"""

import json
import os
import time
import re
from datetime import datetime

import requests
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup

# ==================== CẤU HÌNH ====================
BASE_LIST_URL   = "https://luatvietnam.vn/luat-su-tu-van/hanh-chinh-27.html"
BASE_DOMAIN     = "https://luatvietnam.vn"
TOTAL_PAGES     = 6          # Tổng số trang danh sách (106 câu hỏi / ~20 mỗi trang)
DELAY_LIST      = 1.5        # Giây chờ giữa mỗi trang danh sách
DELAY_DETAIL    = 1.5        # Giây chờ giữa mỗi trang chi tiết
OUTPUT_DIR      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "crawled_qa.json")
# ===================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


def get_soup(url: str) -> BeautifulSoup | None:
    """Fetch URL và trả về BeautifulSoup object."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"  ❌ Lỗi fetch {url}: {e}")
        return None


def clean_text(text: str) -> str:
    """Xóa khoảng trắng thừa."""
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────
# TẦNG 1: Lấy danh sách link câu hỏi
# ─────────────────────────────────────────────
def scrape_list_page(page: int) -> list[dict]:
    """
    Scrape 1 trang danh sách, trả về list dicts:
    [{"title": ..., "url": ..., "date": ...}, ...]
    """
    url = BASE_LIST_URL if page == 1 else f"{BASE_LIST_URL}?page={page}"
    print(f"  📄 Trang danh sách {page}: {url}")

    soup = get_soup(url)
    if not soup:
        return []

    items = []
    # Các câu hỏi nằm trong thẻ <h3> bên trong main content
    for h3 in soup.select("h3"):
        a = h3.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        if "-faqs.html" not in href:   # Chỉ lấy link bài hỏi đáp
            continue

        full_url = href if href.startswith("http") else BASE_DOMAIN + href
        title    = clean_text(a.get_text())

        # Lấy ngày đăng — nằm ngay sau thẻ h3 trong cùng parent
        date_str = ""
        parent = h3.find_parent()
        if parent:
            # Tìm thẻ a có href giống href câu hỏi nhưng chứa ngày
            for sibling_a in parent.find_all("a", href=href):
                next_sib = sibling_a.find_next_sibling(string=True)
                if next_sib and re.search(r"\d{2}/\d{2}/\d{4}", next_sib):
                    date_str = next_sib.strip()
                    break

            if not date_str:
                # Fallback: tìm text có dạng dd/mm/yyyy trong parent
                raw = parent.get_text()
                m = re.search(r"\d{2}/\d{2}/\d{4}", raw)
                if m:
                    date_str = m.group()

        items.append({"title": title, "url": full_url, "date": date_str})

    return items


# ─────────────────────────────────────────────
# TẦNG 2: Lấy nội dung chi tiết từng câu hỏi
# ─────────────────────────────────────────────
def scrape_detail_page(item: dict) -> dict:
    """
    Scrape trang chi tiết của 1 câu hỏi, bổ sung:
      - question (nội dung câu hỏi đầy đủ)
      - answer   (nội dung trả lời của luật sư)
    """
    url  = item["url"]
    soup = get_soup(url)
    if not soup:
        return {**item, "question": item["title"], "answer": ""}

    # Nội dung bài nằm sau breadcrumb, trước phần liên quan
    # Tìm block chứa câu hỏi + trả lời:
    # Cấu trúc: breadcrumb → h1/title → [nội dung câu hỏi] → "Trả lời:" → [nội dung TL]
    question = ""
    answer   = ""

    # Xác định vùng nội dung chính (loại bỏ nav, footer, sidebar)
    # Tất cả nội dung chính nằm trong 1 div lớn ở giữa trang
    content_block = (
        soup.select_one("div.faq-content")
        or soup.select_one("div.detail-content")
        or soup.select_one("article")
        or soup.select_one("div#content")
    )

    if not content_block:
        # Fallback: lấy toàn bộ text vùng sau breadcrumb
        all_paragraphs = soup.select("p")
        full_text = "\n".join(p.get_text() for p in all_paragraphs)
    else:
        full_text = content_block.get_text(separator="\n")

    # Tách câu hỏi / trả lời theo marker "Trả lời:"
    if "Trả lời:" in full_text:
        parts    = full_text.split("Trả lời:", 1)
        question = clean_text(parts[0])
        answer   = clean_text("Trả lời:" + parts[1])
    else:
        # Nếu không tìm thấy marker, lấy toàn bộ là nội dung
        question = clean_text(full_text)

    # Cắt bỏ phần cuối thừa (footer, related posts) bằng marker
    for stop_marker in ["Xem thêm:", "Trên đây là nội dung tư vấn", "Để được giải đáp"]:
        if stop_marker in answer:
            answer = answer.split(stop_marker)[0].strip()

    return {**item, "question": question, "answer": answer}


# ─────────────────────────────────────────────
# Lưu file JSON
# ─────────────────────────────────────────────
def save_results(results: list):
    output = {
        "total": len(results),
        "source": BASE_LIST_URL,
        "category": "Hành chính",
        "crawled_at": datetime.now().isoformat(),
        "data": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  💾 Đã lưu {len(results)} mục → {OUTPUT_FILE}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("🚀 LuatVietnam Q&A Crawler")
    print(f"🔗 Source : {BASE_LIST_URL}")
    print(f"📄 Pages  : {TOTAL_PAGES}")
    print(f"📂 Output : {OUTPUT_FILE}\n")

    # ── Bước 1: Thu thập danh sách câu hỏi ──
    print("=" * 55)
    print("BƯỚC 1: Crawl danh sách câu hỏi")
    print("=" * 55)

    all_items: list[dict] = []
    seen_urls: set = set()

    for page in range(1, TOTAL_PAGES + 1):
        items = scrape_list_page(page)
        new = 0
        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)
                new += 1
        print(f"    ✅ Trang {page}: +{new} câu hỏi (tổng: {len(all_items)})")
        time.sleep(DELAY_LIST)

    print(f"\n📋 Tổng câu hỏi tìm được: {len(all_items)}\n")

    # ── Bước 2: Crawl từng trang chi tiết ──
    print("=" * 55)
    print("BƯỚC 2: Crawl nội dung chi tiết")
    print("=" * 55)

    results = []
    for i, item in enumerate(all_items, 1):
        print(f"  [{i}/{len(all_items)}] {item['title'][:60]}...")
        detail = scrape_detail_page(item)
        detail["id"] = i
        results.append(detail)

        # Lưu tạm mỗi 10 bài
        if i % 10 == 0:
            save_results(results)

        time.sleep(DELAY_DETAIL)

    # ── Lưu lần cuối ──
    save_results(results)

    print(f"\n{'=' * 55}")
    print(f"✅ HOÀN TẤT!")
    print(f"📊 Tổng Q&A đã crawl: {len(results)}")
    print(f"📁 File output: {OUTPUT_FILE}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
