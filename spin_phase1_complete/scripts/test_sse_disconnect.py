#!/usr/bin/env python3
"""
scripts/test_sse_disconnect.py

چک می‌کند سرویس واقعا وقتی کلاینت وسط استریم قطع می‌شود، درست رفتار
می‌کند: در app/api/chat.py چند جای مشخص await request.is_disconnected()
را چک می‌کنند - این اسکریپت آن ادعا را عملا امتحان می‌کند.

چیزی که این‌جا نمی‌بینید ولی باید همزمان در لاگ/متریک سرور چک کنید (این
اسکریپت فقط سمت کلاینت را می‌بیند):
  - آیا vLLM هم بلافاصله بعد از قطع کلاینت تولید توکن را متوقف کرد، یا
    ادامه داد تا آخر (که یعنی GPU/هزینه برای هیچ مصرف‌کننده‌ای هدر می‌رود)؟
  - آیا exception ناقصی (traceback) در لاگ سرور افتاد، یا تمیز return شد؟

اجرا (سرویس باید بالا باشد):
    python scripts/test_sse_disconnect.py --base-url http://localhost:8080
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import uuid

import aiohttp


async def run(base_url: str, tokens_before_disconnect: int) -> None:
    payload = {
        "request_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "user_hash": hashlib.sha256(b"disconnect-test").hexdigest(),
        "query": "یک سوال معمولی درباره درد دندان که چند ثانیه طول بکشد تا کامل جواب بدهد",
    }
    seen_tokens = 0
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/medical/chat", json=payload,
            headers={"Authorization": "Bearer disconnect-test"},
        ) as resp:
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line.startswith("data:") and '"type": "token"' in line.replace(" ", ""):
                    seen_tokens += 1
                    if seen_tokens >= tokens_before_disconnect:
                        print(f"{seen_tokens} توکن دریافت شد - الان کانکشن را قطع می‌کنم (خارج از with)")
                        break  # خروج از async with = بستن کانکشن سمت کلاینت، دقیقا شبیه رفتن کاربر
    print("کلاینت قطع شد. حالا لاگ سرور را چک کنید: آیا هیچ 500/traceback افتاد؟ "
          "آیا generator متوقف شد (نه اینکه vLLM تا آخر ادامه داد)؟")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--tokens-before-disconnect", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(run(args.base_url, args.tokens_before_disconnect))


if __name__ == "__main__":
    main()
