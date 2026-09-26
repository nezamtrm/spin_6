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
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+")

# وزن شباهت برداری در برابر همپوشانی واژگانی در امتیاز نهایی
_VECTOR_WEIGHT = 0.75
_LEXICAL_WEIGHT = 0.25


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


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """
    candidates: خروجی qdrant_client.search -> [{"score": float, "payload": {...}}]
    خروجی: همان ساختار، رتبه‌بندی‌شده و بریده به top_k، با یک فیلد اضافه
    "final_score" (0..1 تقریبی).
    """
    if not candidates:
        return []

    query_tokens = _tokenize(query)
    scored = []
    for c in candidates:
        payload = c.get("payload", {})
        doc_text = f"{payload.get('title', '')} {payload.get('text', '')}"
        lexical = _lexical_overlap(query_tokens, doc_text)
        vector_score = max(0.0, min(1.0, float(c.get("score", 0.0))))
        final = _VECTOR_WEIGHT * vector_score + _LEXICAL_WEIGHT * lexical
        scored.append({**c, "final_score": round(final, 4)})

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]
