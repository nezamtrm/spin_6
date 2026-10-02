# app/rag/reranker.py
"""
رتبه‌بندی مجدد نتایج بازیابی‌شده از Qdrant.

عمدا یک مدل cross-encoder جدا اضافه نشده (هزینه بارگذاری/GPU دیگری برای
فاز ۱ توجیه ندارد). این یک reranker سبک و بدون وابستگی است: امتیاز نهایی،
ترکیب وزنی امتیاز شباهت برداری (از Qdrant) و همپوشانی واژگانی متن سوال با
عنوان/متن سند است. همپوشانی واژگانی مکمل خوبی برای شباهت برداری است: بردار
می‌تواند دو جمله‌ی «مرتبط از نظر موضوع ولی با کلمه کلیدی متفاوت» را نزدیک
هم ببیند؛ همپوشانی واژگانی این را با تطبیق شواهد لغوی واقعی متعادل می‌کند.

اگر بعدا یک cross-encoder فارسی مناسب پیدا شد، فقط تابع _lexical_overlap
با یک فراخوانی مدل جایگزین می‌شود؛ امضای rerank() ثابت می‌ماند.

وزن‌ها دیگر اینجا hardcode نیستند - از app.core.config.settings می‌آیند
(RERANK_VECTOR_WEIGHT/RERANK_LEXICAL_WEIGHT) تا evaluate_retrieval.py
بتواند روی یک gold set واقعی چند ترکیب را امتحان و بهترین را به‌عنوان
مقدار نهایی در config.py/`.env` ثبت کند - نه یک عدد حدسی ثابت.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.config import settings

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _lexical_overlap(query_tokens: set[str], doc_text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / len(query_tokens)


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    vector_weight: float | None = None,
    lexical_weight: float | None = None,
) -> list[dict[str, Any]]:
    """
    candidates: خروجی qdrant_client.search -> [{"score": float, "payload": {...}}]
    vector_weight/lexical_weight: اگر داده نشوند، از settings خوانده
        می‌شوند (مقدار پیش‌فرض production). evaluate_retrieval.py برای
        sweep کردن ترکیب‌های مختلف این دو را صریح پاس می‌دهد.
    خروجی: همان ساختار، رتبه‌بندی‌شده و بریده به top_k، با یک فیلد اضافه
    "final_score" (0..1 تقریبی).
    """
    if not candidates:
        return []

    vw = settings.RERANK_VECTOR_WEIGHT if vector_weight is None else vector_weight
    lw = settings.RERANK_LEXICAL_WEIGHT if lexical_weight is None else lexical_weight

    query_tokens = _tokenize(query)
    scored = []
    for c in candidates:
        payload = c.get("payload", {})
        doc_text = f"{payload.get('title', '')} {payload.get('text', '')}"
        lexical = _lexical_overlap(query_tokens, doc_text)
        vector_score = max(0.0, min(1.0, float(c.get("score", 0.0))))
        final = vw * vector_score + lw * lexical
        scored.append({**c, "final_score": round(final, 4)})

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]
