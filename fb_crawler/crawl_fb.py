"""
Facebook Group Search Crawler
Crawl bài viết từ kết quả tìm kiếm trong các nhóm Facebook.
Sử dụng Playwright (headed mode) để user tự đăng nhập.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from playwright.async_api import async_playwright

# ==================== CẤU HÌNH ====================
URLS = [
   "https://www.facebook.com/groups/454353863660050/search/?q=lu%E1%BA%ADt%20h%C3%A0nh%20ch%C3%ADnh",
    "https://www.facebook.com/groups/1545991175927462/search/?q=lu%E1%BA%ADt%20h%C3%A0nh%20ch%C3%ADnh",
     
]

TARGET_POSTS = 200
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "crawled_posts.json")
SCROLL_PAUSE = 3          # Giây chờ sau mỗi lần scroll
MAX_NO_NEW_SCROLLS = 15   # Số lần scroll không có bài mới trước khi chuyển link
# ===================================================


async def wait_for_login(page):
    """Chờ user đăng nhập Facebook thủ công."""
    print("\n" + "=" * 60)
    print("🔐  VUI LÒNG ĐĂNG NHẬP FACEBOOK TRONG CỬA SỔ TRÌNH DUYỆT")
    print("    Script sẽ tự động tiếp tục sau khi bạn đăng nhập xong.")
    print("=" * 60 + "\n")

    # Chờ tối đa 5 phút cho user đăng nhập
    # Kiểm tra bằng nhiều selector khác nhau để phát hiện đã login
    login_selectors = [
        '[aria-label="Facebook"]',
        '[aria-label="Trang chủ"]',
        '[aria-label="Home"]',
        'svg[aria-label="Facebook"]',
        '[data-pagelet="LeftRail"]',
    ]

    for attempt in range(300):  # 5 phút (300 x 1s)
        for sel in login_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    print("✅ Đã phát hiện đăng nhập thành công!")
                    await asyncio.sleep(2)
                    return True
            except Exception:
                pass

        # Kiểm tra bằng URL - nếu đã redirect khỏi trang login
        current_url = page.url
        if "facebook.com" in current_url and "/login" not in current_url and "checkpoint" not in current_url:
            # Nếu URL không phải login page, kiểm tra thêm
            try:
                # Kiểm tra có navigation bar không
                nav = await page.query_selector('[role="navigation"]')
                if nav:
                    print("✅ Đã phát hiện đăng nhập thành công!")
                    await asyncio.sleep(2)
                    return True
            except Exception:
                pass

        if attempt % 10 == 0 and attempt > 0:
            print(f"  ⏳ Đang chờ đăng nhập... ({attempt}s)")

        await asyncio.sleep(1)

    print("⚠️  Timeout đăng nhập (5 phút). Thử tiếp tục...")
    return False


def clean_post_text(text: str) -> str:
    """Làm sạch nội dung bài viết."""
    if not text:
        return ""

    # Loại bỏ các chuỗi UI thừa
    noise_patterns = [
        r"Thích$", r"Bình luận$", r"Chia sẻ$",
        r"\d+ bình luận", r"\d+ lượt thích",
        r"Xem thêm bình luận", r"Viết bình luận\.\.\.",
        r"^\s*Thích\s*$", r"^\s*Trả lời\s*$",
        r"Xem thêm$", r"Ẩn bớt$",
        r"^\d+\s*(giờ|phút|ngày|tuần|tháng|năm)\s*$",
        r"^(Đã chỉnh sửa|Sponsored|Được gợi ý cho bạn)\s*$",
    ]

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        skip = False
        for pattern in noise_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                skip = True
                break
        if not skip:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


async def click_see_more_buttons(page):
    """Bấm tất cả nút 'Xem thêm' / 'See more' bằng Javascript thuần để mở rộng bài viết đầy đủ."""
    try:
        await page.evaluate('''() => {
            // Quét tất cả thẻ div và span có khả năng là nút Xem thêm
            const elements = document.querySelectorAll('div[role="button"], span, div');
            for (let el of elements) {
                if (el.innerText && (el.innerText.trim() === "Xem thêm" || el.innerText.trim() === "See more")) {
                    try {
                        el.click();
                    } catch (e) {}
                }
            }
        }''')
    except Exception as e:
        pass


async def extract_posts_from_page(page) -> list[str]:
    """Trích xuất nội dung bài viết từ trang hiện tại."""

    posts = await page.evaluate('''() => {
        const results = [];
        const seen = new Set();

        const autoDivs = document.querySelectorAll('div[dir="auto"]');
        autoDivs.forEach(div => {
            // Bỏ qua nếu thẻ div này nằm bên trong một thẻ div[dir="auto"] khác (tránh bị trùng lặp đoạn văn con)
            if (div.parentElement && div.parentElement.closest('div[dir="auto"]')) {
                return;
            }

            const postText = div.innerText.trim();
            // Lọc ra các text dài (khả năng cao là nội dung bài viết)
            if (postText && postText.length > 50) {
                // Rút 80 kí tự đầu tiên làm fingerprint để lọc trùng nếu quá nhiều div lồng nhau
                const fingerprint = postText.substring(0, 80);
                if (!seen.has(fingerprint)) {
                    seen.add(fingerprint);
                    results.push(postText);
                }
            }
        });

        return results;
    }''')

    return posts


async def scroll_and_collect(page, target_count: int, collected: list) -> list:
    """Cuộn trang và thu thập bài viết cho đến khi đủ số lượng."""
    seen_fingerprints = set()
    for p in collected:
        fp = p["content"][:100]
        seen_fingerprints.add(fp)

    no_new_count = 0
    scroll_count = 0

    while len(collected) < target_count:
        scroll_count += 1

        # Bấm "Xem thêm" trên các bài viết bị cắt ngắn
        await click_see_more_buttons(page)
        await asyncio.sleep(0.5)

        # Trích xuất bài viết
        raw_posts = await extract_posts_from_page(page)
        new_this_round = 0

        # Mảng tạm để đếm log
        skipped_dup = 0

        for raw_text in raw_posts:
            preview_raw = raw_text[:40].replace("\n", " ").strip()
            fp = raw_text[:80]
            
            if fp in seen_fingerprints:
                skipped_dup += 1
                continue

            cleaned = clean_post_text(raw_text)
            if not cleaned or len(cleaned) < 50:
                print(f"  ❌ Bỏ qua (quá ngắn/rác): '{preview_raw}...'")
                continue

            seen_fingerprints.add(fp)
            new_this_round += 1

            post_entry = {
                "id": len(collected) + 1,
                "content": cleaned,
                "source_url": page.url,
                "crawled_at": datetime.now().isoformat(),
            }
            collected.append(post_entry)

            print(f"  ✅ THÊM MỚI [{len(collected)}/{target_count}] {preview_raw}...")

            if len(collected) >= target_count:
                break
        
        if skipped_dup > 0:
            print(f"  🔍 Quét DOM: Thấy {len(raw_posts)} thẻ bài viết, trong đó đã bỏ qua {skipped_dup} bài cũ trùng lặp.")

        if new_this_round == 0:
            no_new_count += 1
            if no_new_count >= MAX_NO_NEW_SCROLLS:
                print(f"  ⚠️  Không có bài mới sau {MAX_NO_NEW_SCROLLS} lần cuộn. Chuyển link...")
                break
        else:
            no_new_count = 0

        # Cuộn xuống
        await page.evaluate("window.scrollBy(0, 800)")
        await asyncio.sleep(SCROLL_PAUSE)

        # Log tiến độ mỗi 5 lần scroll
        if scroll_count % 5 == 0:
            print(f"  📊 Scroll #{scroll_count} — Đã thu thập: {len(collected)}/{target_count}")

        # Lưu tạm nếu có bài mới
        if new_this_round > 0:
            save_results(collected)
            # print(f"  💾 Đã cập nhật file save với {len(collected)} bài")

    return collected


def save_results(posts: list):
    """Lưu kết quả ra file JSON."""
    output = {
        "total_posts": len(posts),
        "target": TARGET_POSTS,
        "crawled_at": datetime.now().isoformat(),
        "search_query": "luật hành chính",
        "posts": posts,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


async def main():
    print("🚀 Facebook Group Search Crawler")
    print(f"🎯 Mục tiêu: {TARGET_POSTS} bài viết")
    print(f"📂 Output: {OUTPUT_FILE}")
    print(f"🔗 Số link: {len(URLS)}\n")

    async with async_playwright() as p:
        # Mở trình duyệt ở chế độ có giao diện (headed)
        browser = await p.firefox.launch(
            headless=False,
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="vi-VN",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # Bước 1: Mở Facebook và chờ đăng nhập
        print("📌 Đang mở Facebook...")
        await page.goto("https://www.facebook.com", wait_until="domcontentloaded")
        await asyncio.sleep(2)

        logged_in = await wait_for_login(page)
        if not logged_in:
            print("❌ Không thể xác nhận đăng nhập. Vẫn thử tiếp tục...")

        await asyncio.sleep(3)

        # Bước 2: Crawl từng link
        all_posts = []

        for i, url in enumerate(URLS):
            if len(all_posts) >= TARGET_POSTS:
                print(f"\n✅ Đã đủ {TARGET_POSTS} bài viết!")
                break

            remaining = TARGET_POSTS - len(all_posts)
            print(f"\n{'='*60}")
            print(f"📌 Link {i+1}/{len(URLS)}: {url[:80]}...")
            print(f"   Cần thêm: {remaining} bài viết")
            print(f"{'='*60}")

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(5)  # Chờ trang load kết quả tìm kiếm

                # Kiểm tra xem có kết quả không
                content = await page.content()
                if "Không tìm thấy kết quả" in content or "No results found" in content:
                    print(f"  ⚠️ Không có kết quả tìm kiếm cho link này. Bỏ qua...")
                    continue

                all_posts = await scroll_and_collect(page, TARGET_POSTS, all_posts)

            except Exception as e:
                print(f"  ❌ Lỗi khi crawl link {i+1}: {e}")
                continue

        # Đóng trình duyệt
        await browser.close()

    # Lưu kết quả cuối cùng
    save_results(all_posts)

    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT!")
    print(f"📊 Tổng số bài viết: {len(all_posts)}/{TARGET_POSTS}")
    print(f"📁 File output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
