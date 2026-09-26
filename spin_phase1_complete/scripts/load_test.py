#!/usr/bin/env python3
"""
scripts/load_test.py

تست بار واقعی روی /v1/medical/chat در حال اجرا - نه یک درخواست تکی، بلکه
N درخواست هم‌زمان، چون سوال اصلی OpenApi_Spin.yaml این است: «آیا تریاژ
زیر ۱۰۰ms می‌ماند وقتی چند کاربر هم‌زمان هستند، نه وقتی فقط یکی هست؟»

نکته مهندسی که این تست دقیقا برایش طراحی شده: triage.is_emergency() و
router.resolve_specialty() و embedder.embed_query() هر سه synchronous/
blocking هستند و در app/api/chat.py مستقیم (بدون asyncio.to_thread) صدا
زده می‌شوند. یعنی تئوری این است که زیر بار هم‌زمان، این call ها event loop
تک‌رشته‌ای FastAPI را می‌بندند و درخواست‌ها به‌جای موازی، صف می‌شوند - حتی
اگر هرکدام به‌تنهایی سریع باشد. این اسکریپت دقیقا همین را با عدد نشان
می‌دهد؛ اگر p99 زمان رسیدن به رویداد "start" با افزایش concurrency خطی رشد
کرد (نه ثابت ماند)، این فرضیه تایید شده و باید در چت.پی async.to_thread
دور این سه فراخوانی اضافه شود.

اجرا (سرویس باید از قبل روی uvicorn بالا باشد):
    pip install aiohttp   # اگر از قبل نصب نیست
    python scripts/load_test.py --base-url http://localhost:8080 --concurrency 20 --requests 100
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import statistics
import time
import uuid

import aiohttp

SAMPLE_QUERIES = [
    "چند روزه دندون آسیام درد میکنه",
    "دو ماهه پریود نشدم و حالت تهوع دارم",
    "زانوم موقع پله رفتن صدا میده",
    "بچم دیر حرف میزنه نگرانم",
    "شکمم درد میکنه سمت راست پایین",
    "سرما خوردم نیاز به نسخه دارم",
]


async def one_request(session: aiohttp.ClientSession, base_url: str, query: str) -> dict:
    payload = {
        "request_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "user_hash": hashlib.sha256(f"loadtest-{uuid.uuid4()}".encode()).hexdigest(),
        "query": query,
    }
    t0 = time.perf_counter()
    time_to_start = None
    error = None
    try:
        async with session.post(
            f"{base_url}/v1/medical/chat", json=payload,
            headers={"Authorization": "Bearer loadtest"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line.startswith("data:") and time_to_start is None:
                    # اولین رویداد data: همان "start" است - دقیقا لحظه‌ای که
                    # تریاژ+روتر (بلاک‌کننده‌ها) تمام شده‌اند.
                    time_to_start = time.perf_counter() - t0
                if line.startswith('data:') and '"type": "done"' in line.replace(" ", ""):
                    break
    except Exception as e:  # noqa: BLE001
        error = str(e)
    total = time.perf_counter() - t0
    return {"time_to_start_ms": (time_to_start or total) * 1000, "total_ms": total * 1000, "error": error}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * p))
    return values[idx]


async def run(base_url: str, concurrency: int, total_requests: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(i: int):
        async with sem:
            async with aiohttp.ClientSession() as session:
                r = await one_request(session, base_url, SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)])
                results.append(r)

    t0 = time.perf_counter()
    await asyncio.gather(*[bounded(i) for i in range(total_requests)])
    wall = time.perf_counter() - t0

    ok = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]
    starts = [r["time_to_start_ms"] for r in ok]
    totals = [r["total_ms"] for r in ok]

    print(f"concurrency={concurrency}  requests={total_requests}  wall_time={wall:.1f}s  errors={len(errors)}")
    print(f"زمان تا رویداد start (شامل تریاژ+روتر - باید مستقل از concurrency بماند):")
    print(f"  p50={percentile(starts, 0.5):.0f}ms  p95={percentile(starts, 0.95):.0f}ms  p99={percentile(starts, 0.99):.0f}ms")
    print(f"زمان کل پاسخ:")
    print(f"  p50={percentile(totals, 0.5):.0f}ms  p95={percentile(totals, 0.95):.0f}ms  p99={percentile(totals, 0.99):.0f}ms")
    if errors:
        print(f"نمونه خطا: {errors[0]['error']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=50)
    args = ap.parse_args()
    asyncio.run(run(args.base_url, args.concurrency, args.requests))


if __name__ == "__main__":
    main()
