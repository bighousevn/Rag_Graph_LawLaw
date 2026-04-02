import json
import os
import asyncio
from openai import AsyncOpenAI

async def process_section(client, semaphore, sec):
    async with semaphore:
        text = sec['text_content']
        prompt = f"""
Nhiệm vụ của bạn là chia đoạn văn bản pháp luật sau thành các câu đơn giản, độc lập và trọn vẹn ngữ nghĩa.
Tuyệt đối giữ nguyên ngữ nghĩa pháp lý. Nếu đoạn văn có phần dẫn (Chương, Điều...), hãy lồng ghép nó vào câu đơn để câu thật sự có ý nghĩa độc lập.

Văn bản:
{text}

Trả về một mảng JSON hợp lệ chứa các câu, không có text nào khác. Ví dụ:
["câu 1", "câu 2"]
"""
        try:
            completion = await client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            resp = completion.choices[0].message.content.strip()
            # Clean up markdown if any
            if resp.startswith("```json"):
                resp = resp[7:]
            elif resp.startswith("```"):
                resp = resp[3:]
            if resp.endswith("```"):
                resp = resp[:-3]
                
            sentences = json.loads(resp.strip())
            
            return {
                "section_id": sec['id'],
                "original_text": text,
                "sentences": sentences
            }
        except Exception as e:
            print(f"Error processing section {sec['id']}: {e}")
            return {
                "section_id": sec['id'],
                "original_text": text,
                "sentences": [text]
            }

async def split_sentences(input_path, output_path):
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        sections = json.load(f)
        
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    
    total = len(sections)
    print(f"Loaded {total} sections. Starting API processing concurrently...")
    
    # Process up to 10 concurrent requests to respect rate limits
    semaphore = asyncio.Semaphore(10)
    
    tasks = [process_section(client, semaphore, sec) for sec in sections]
    
    # To display progress
    results = []
    completed = 0
    
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        completed += 1
        if completed % 50 == 0:
            print(f"Processed {completed}/{total} sections...")
            # Save progress periodically
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Finished splitting sentences. Saved to {output_path}")

if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Cảnh báo: Không tìm thấy OPENROUTER_API_KEY trong biến môi trường. Vui lòng thiết lập biến này.")
        exit(1)
    asyncio.run(split_sentences('output/1_sections.json', 'output/2_sentences.json'))
