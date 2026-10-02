#!/usr/bin/env python3
"""
scripts/purge_cache.py

پاک‌سازی دوره‌ای نقاط منقضی‌شده کش معنایی (app/pipeline/cache.py). چون
Qdrant برخلاف Redis، TTL بومی ندارد، این اسکریپت باید دوره‌ای اجرا شود
(مثلا یک k8s CronJob یا cron ساده هر شب) - نه بخشی از مسیر داغ چت.

اجرا:
    python scripts/purge_cache.py                 # همه ۶ تخصص
    python scripts/purge_cache.py --specialty dentistry
    python scripts/purge_cache.py --dry-run        # فقط بشمار، پاک نکن
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.rag import qdrant_client  # noqa: E402

SPECIALTIES = [
    "gynecology", "orthopedics", "dentistry",
    "speech_therapy", "general_surgery", "general_practice",
]


def purge_one(specialty: str, dry_run: bool) -> tuple[int, int]:
    collection = f"{settings.QDRANT_CACHE_COLLECTION_PREFIX}_{specialty}"
    cutoff = time.time() - settings.CACHE_TTL_SECONDS
    expired_ids = []
    total = 0
    for point in qdrant_client.scroll_all(collection, with_payload=True):
        total += 1
        created_at = (point.payload or {}).get("created_at", 0)
        if created_at < cutoff:
            expired_ids.append(point.id)

    if expired_ids and not dry_run:
        qdrant_client.delete_points(collection, expired_ids)

    return total, len(expired_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialty", choices=SPECIALTIES, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = [args.specialty] if args.specialty else SPECIALTIES
    for specialty in targets:
        total, expired = purge_one(specialty, args.dry_run)
        action = "پیدا شد (dry-run، پاک نشد)" if args.dry_run else "پاک شد"
        print(f"{specialty}: {total} نقطه، {expired} منقضی {action}")


if __name__ == "__main__":
    main()
