# app/pipeline/retriever.py
"""
بازیابی منابع مرتبط (RAG) برای یک سوال، محدود به تخصص تشخیص‌داده‌شده توسط
روتر (مرحله ۴). خروجی مستقیما به‌عنوان `sources` هم به build_prompt (برای
مبتنی‌کردن پاسخ مدل بر منابع واقعی) و هم به EventCitation/StructuredMedicalResponse
داده می‌شود.

مثل router.py، این فایل هرگز نباید استثنا بیرون بدهد: اگر Qdrant/امبدر در
دسترس نباشند یا کالکشن آن تخصص خالی باشد، فقط [] برمی‌گردد - یعنی مدل بدون
منبع بازیابی‌شده (ولی همچنان با مرزهای ایمنی SYSTEM_PROMPT) پاسخ می‌دهد،
نه اینکه کل درخواست ۵۰۰ بگیرد. RAG یک بهبود کیفیت است، نه پیش‌نیاز جریان.
"""
from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.rag import embedder, qdrant_client
from app.rag.reranker import rerank

log = get_logger(__name__)


def _kb_collection(specialty_key: str) -> str:
    return f"{settings.QDRANT_COLLECTION_PREFIX}_{specialty_key}"


def retrieve(specialty_key: str, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    خروجی: لیستی از dict سازگار با اسکیمای Source:
        {"title": str, "url": str|None, "author": str|None,
         "published_date": str|None, "relevance": float}
    اگر چیزی پیدا نشد (یا هر جزئی از RAG در دسترس نبود)، [] برمی‌گردد.
    """
    k = top_k or settings.RAG_TOP_K
    try:
        query_vector = embedder.embed_query(query)
    except Exception:  # noqa: BLE001
        log.warning("retriever_embed_failed", specialty=specialty_key)
        return []

    # کمی بیشتر از نیاز واقعی می‌گیریم تا reranker چیزی برای انتخاب داشته باشد
    candidates = qdrant_client.search(_kb_collection(specialty_key), query_vector, top_k=max(k * 3, k))
    if not candidates:
        return []

    top = rerank(query, candidates, top_k=k)

    sources: list[dict[str, Any]] = []
    for item in top:
        payload = item.get("payload", {})
        title = payload.get("title")
        if not title:
            continue  # طبق اسکیمای Source، title الزامی است - سند بی‌عنوان کنار گذاشته می‌شود
        sources.append({
            "title": title,
            "url": payload.get("url"),
            "author": payload.get("author"),
            "published_date": payload.get("published_date"),
            "relevance": max(0.0, min(1.0, float(item.get("final_score", 0.0)))),
        })
    return sources
