# app/pipeline/cache.py
"""
مرحله ۵ از Pipeline: کش معنایی سوالات مشابه.

پیاده‌سازی روی همان Qdrant که برای RAG (app/pipeline/retriever.py) استفاده
می‌شود - یک کالکشن جدا به ازای هر تخصص با پیشوند متفاوت
(spin_cache_<specialty> در برابر spin_kb_<specialty> برای دانش تخصصی)،
نه یک سرویس جدید. دلیل انتخاب Qdrant به‌جای Redis/Redis Stack و
جزئیات کامل تصمیم را در تاریخچه گفتگو (یا هر مستندسازی جدا که نگه
می‌دارید) ببینید؛ خلاصه: صفر زیرساخت جدید، همان embedder/qdrant_client
از قبل تست‌شده.

جستجو معنایی است، نه تطابق دقیق: سوال ورودی امبد می‌شود، در کالکشن همان
تخصص جستجو می‌شود، و اگر نزدیک‌ترین نتیجه از app.core.config.settings.CACHE_SIMILARITY_THRESHOLD
بالاتر بود "hit" حساب می‌شود. آستانه عمدا بالا (پیش‌فرض ۰.۹۳) است -
توضیح کامل در همان تنظیم، در config.py.

محدودیت شناخته‌شده (صادقانه، نه پنهان): Qdrant برخلاف Redis، TTL بومی
ندارد. راه‌حل این‌جا: هر نقطه created_at را در payload نگه می‌دارد؛
get_cached ورودی‌های منقضی را نادیده می‌گیرد (miss حساب می‌کند، پاک هم
نمی‌کند - پاک‌سازی واقعی وظیفه scripts/purge_cache.py است که باید دوره‌ای
(مثلا یک cron/k8s CronJob شبانه) اجرا شود، نه در مسیر داغ چت).
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.rag import embedder, qdrant_client

log = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[!؟?.,،؛;:\"'«»]+")


def _normalize_light(text: str) -> str:
    """نرمال‌سازی سبک - فقط برای ساخت id پایدار، نه برای امبدینگ/نمایش."""
    t = _PUNCT_RE.sub(" ", text.strip())
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip().lower()


def _cache_collection(specialty_key: str) -> str:
    return f"{settings.QDRANT_CACHE_COLLECTION_PREFIX}_{specialty_key}"


def _point_id(specialty_key: str, query: str) -> str:
    """id پایدار و deterministic (UUID5) از (تخصص+متن نرمال‌شده) - یعنی
    اگر عین همان سوال دوباره ذخیره شود، همان نقطه upsert/update می‌شود
    (فقط created_at تازه می‌شود)، نه این‌که نقطه تکراری جدید بسازد."""
    raw = f"{specialty_key}::{_normalize_light(query)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


@dataclass(frozen=True, slots=True)
class CacheResult:
    hit: bool
    answer: Optional[str] = None
    structured: Optional[dict] = None
    similarity: float = 0.0


def get_cached(specialty_key: str, query: str) -> CacheResult:
    """
    جستجوی معنایی. هرگز استثنا بیرون نمی‌دهد - اگر Qdrant/embedder در
    دسترس نباشند، فقط miss حساب می‌شود (مسیر عادی تولید پاسخ طی می‌شود،
    دقیقا مثل نبودن هرگونه کش).
    """
    try:
        query_vector = embedder.embed_query(query)
    except Exception:  # noqa: BLE001
        log.warning("cache_embed_failed", specialty=specialty_key)
        return CacheResult(hit=False)

    hits = qdrant_client.search(_cache_collection(specialty_key), query_vector, top_k=1)
    if not hits:
        return CacheResult(hit=False)

    top = hits[0]
    similarity = top["score"]
    if similarity < settings.CACHE_SIMILARITY_THRESHOLD:
        return CacheResult(hit=False)

    payload = top["payload"]
    created_at = payload.get("created_at", 0)
    if time.time() - created_at > settings.CACHE_TTL_SECONDS:
        # منقضی شده - miss حساب می‌شود؛ حذف فیزیکی واقعی کار
        # scripts/purge_cache.py است، نه این مسیر داغ.
        return CacheResult(hit=False)

    return CacheResult(
        hit=True,
        answer=payload.get("answer"),
        structured=payload.get("structured"),
        similarity=similarity,
    )


def set_cached(specialty_key: str, query: str, answer: str, structured: dict) -> None:
    """
    ذخیره پاسخ نهایی. فقط باید *بعد* از عبور کامل از guard_output صدا زده
    شود - چیزی که بلاک/سانسور نشده هرگز نباید وارد کش شود. patient_record
    نباید داخل structured باشد وقتی این تابع صدا زده می‌شود (مسئولیت
    caller در app/api/chat.py - نگاه کنید به یادداشت حریم خصوصی آنجا).
    """
    if not answer or not answer.strip():
        return
    try:
        vector = embedder.embed_query(query)
    except Exception:  # noqa: BLE001
        log.warning("cache_embed_failed_on_write", specialty=specialty_key)
        return

    point = {
        "id": _point_id(specialty_key, query),
        "vector": vector,
        "payload": {
            "query_text": query,
            "answer": answer,
            "structured": structured,
            "specialty": specialty_key,
            "created_at": time.time(),
        },
    }
    qdrant_client.upsert_points(
        _cache_collection(specialty_key), [point], vector_size=len(vector)
    )


def chunk_for_replay(text: str, words_per_chunk: int = 4) -> list[str]:
    """پاسخ کش‌شده را برای بازپخش به شکل رویدادهای token (به‌جای یک تکه
    بزرگ) خرد می‌کند - رفتار کلاینت (نمایش تدریجی متن) با پاسخ تازه‌تولید
    یکسان می‌ماند، فقط بدون تاخیر واقعی تولید."""
    words = text.split(" ")
    return [
        " ".join(words[i:i + words_per_chunk]) + (" " if i + words_per_chunk < len(words) else "")
        for i in range(0, len(words), words_per_chunk)
    ]
