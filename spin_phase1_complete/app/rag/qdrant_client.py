# app/rag/qdrant_client.py
"""
لایه نازک روی Qdrant - مشترک بین دو مصرف‌کننده:
  - app/pipeline/retriever.py  (RAG، کالکشن‌های spin_kb_<specialty>)
  - app/pipeline/cache.py      (کش معنایی، کالکشن‌های spin_cache_<specialty>)

به همین دلیل توابع این‌جا نام کالکشن را مستقیم می‌گیرند (نه specialty_key)؛
منطق نام‌گذاری/پیشوند در خودِ retriever.py و cache.py است، نه اینجا - این
فایل فقط یک لایه نازک و عمومی روی Qdrant است.

یک کالکشن جدا به ازای هر (تخصص × مصرف‌کننده) چون:
  - سرچ محدود به یک کالکشن کوچک از فیلتر روی یک کالکشن بزرگ سریع‌تر است
  - ایزوله‌سازی: اگر کالکشن کش یک تخصص خراب/بیش‌ازحد بزرگ شد، drop کردنش
    اثری روی knowledge-base همان تخصص یا بقیه تخصص‌ها ندارد.

فراخوان‌ها هرگز نباید کل درخواست چت را با خطا متوقف کنند - چه RAG چه کش،
هر دو بهبود کیفیت/سرعت‌اند، نه پیش‌نیاز؛ به همین دلیل search() در صورت هر
خطایی (سرویس Qdrant پایین، کالکشن هنوز ساخته نشده، ...) به‌جای raise کردن،
لیست خالی برمی‌گرداند و لاگ می‌کند.
"""
from __future__ import annotations

from threading import Lock
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_client = None
_load_lock = Lock()


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _load_lock:
        if _client is None:
            from qdrant_client import QdrantClient

            _client = QdrantClient(url=settings.QDRANT_URL, timeout=settings.QDRANT_TIMEOUT_SECONDS)
            log.info("qdrant_client_connected", url=settings.QDRANT_URL)
    return _client


def ping() -> bool:
    """برای /ready - آیا Qdrant اصلا جواب می‌دهد؟ هیچ استثنایی بیرون نمی‌دهد."""
    try:
        _get_client().get_collections()
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_collection(collection: str, vector_size: int) -> None:
    """اگر کالکشن با این نام وجود ندارد، می‌سازد. Idempotent."""
    from qdrant_client.http import models as qmodels

    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )
    log.info("qdrant_collection_created", collection=collection, vector_size=vector_size)


def upsert_points(collection: str, points: list[dict[str, Any]], vector_size: Optional[int] = None) -> int:
    """
    points: [{"id": ..., "vector": [...], "payload": {...}}, ...]
    اگر کالکشن وجود نداشت و vector_size داده شده بود، خودکار ساخته می‌شود
    (راحتی برای کش که کالکشن‌هایش lazy و روی اولین نوشتن ساخته می‌شوند).
    خروجی: تعداد نقاطی که واقعا نوشته شدند.
    """
    from qdrant_client.http import models as qmodels

    if not points:
        return 0
    if vector_size is not None:
        ensure_collection(collection, vector_size)
    client = _get_client()
    client.upsert(
        collection_name=collection,
        points=[
            qmodels.PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
            for p in points
        ],
    )
    return len(points)


def search(collection: str, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """
    نتیجه: [{"score": float(0..1 تقریبا, cosine), "payload": {...}, "id": ...}, ...]
    مرتب‌شده نزولی. در هر خطایی (کالکشن نیست، سرویس پایین) لیست خالی
    برمی‌گرداند - هرگز raise نمی‌کند.
    """
    try:
        client = _get_client()
        hits = client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
        return [{"score": float(h.score), "payload": dict(h.payload or {}), "id": h.id} for h in hits]
    except Exception:  # noqa: BLE001
        log.warning("qdrant_search_failed", collection=collection)
        return []


def delete_points(collection: str, point_ids: list) -> None:
    """برای پاک‌سازی دوره‌ای ورودی‌های منقضی‌شده کش (scripts/purge_cache.py)."""
    try:
        _get_client().delete(collection_name=collection, points_selector=point_ids)
    except Exception:  # noqa: BLE001
        log.warning("qdrant_delete_failed", collection=collection)


def scroll_all(collection: str, with_payload: bool = True, batch_size: int = 256):
    """همه نقاط یک کالکشن را صفحه‌به‌صفحه برمی‌گرداند - فقط برای
    scripts/purge_cache.py (پاک‌سازی کش‌های منقضی)، نه مسیر داغ چت."""
    client = _get_client()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=batch_size, offset=offset, with_payload=with_payload
        )
        if not points:
            break
        yield from points
        if offset is None:
            break
