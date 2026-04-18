"""
Debug script: Mở Facebook, đăng nhập, và lấy thẻ div[dir="auto"].
"""

import asyncio
import json
import os
from playwright.async_api import async_playwright

URL = "https://www.facebook.com/groups/1545991175927462/search/?q=lu%E1%BA%ADt%20h%C3%A0nh%20ch%C3%ADnh"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

async def wait_for_login(page):
    print("🔐 Vui lòng đăng nhập Facebook...")
    login_selectors = [
        '[aria-label="Facebook"]', '[aria-label="Trang chủ"]',
        '[aria-label="Home"]', '[role="navigation"]',
    ]
    for attempt in range(300):
        for sel in login_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    print("✅ Đã đăng nhập!")
                    await asyncio.sleep(2)
                    return
            except:
                pass
        if attempt % 10 == 0 and attempt > 0:
            print(f"  ⏳ Chờ... ({attempt}s)")
        await asyncio.sleep(1)

async def main():
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="vi-VN"
        )
        page = await context.new_page()

        await page.goto("https://www.facebook.com", wait_until="domcontentloaded")
        await wait_for_login(page)
        await asyncio.sleep(3)

        print(f"\n📌 Đang mở: {URL[:80]}...")
        await page.goto(URL, wait_until="domcontentloaded")
        await asyncio.sleep(8)

        # Scroll
        for i in range(5):
            await page.evaluate("window.scrollBy(0, 800)")
            await asyncio.sleep(2)

        print("\n🔍 Đang phân tích DOM...\n")

        debug_info = await page.evaluate('''() => {
            return {
                total_articles: document.querySelectorAll('[role="article"]').length,
                url: window.location.href,
                body_text: document.body.innerText.substring(0, 15000),
                div_auto_texts: Array.from(document.querySelectorAll('div[dir="auto"]'))
                    .map(d => d.innerText.trim())
                    .filter(t => t.length > 50)
            };
        }''')

        output_file = os.path.join(OUTPUT_DIR, "debug_dom.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)

        print(f"📊 Tổng số articles: {debug_info['total_articles']}")
        print(f"📊 Thu được {len(debug_info['div_auto_texts'])} texts từ div[dir=auto]")
        for d in debug_info['div_auto_texts'][:5]:
            print(f"  - {d[:100]}...")

        print(f"\n💾 Debug data saved to: {output_file}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
