# app/pipeline/intake.py
"""
مرحله ۶.۵ از Pipeline: ایجاد/تکمیل «پرونده ابتدایی» بیمار.

جایگاه در جریان: بعد از روتر (مرحله ۴، تخصص را می‌دهد) و قبل/همراه ساخت
پرامپت (مرحله ۶) - توسط app/pipeline/inference.py صدا زده می‌شود، نه
مستقیما توسط chat.py.

طراحی عمدا ساده و قطعی (deterministic) است، نه یک استخراج مبتنی بر مدل
جداگانه: رفتار «سوال تکمیلی بپرس» در prompt.py::_specialty_directive از
قبل توسط خودمان کنترل می‌شود (وقتی is_first_turn است، به مدل می‌گوییم
حداکثر ۲ سوال بپرس). پس هر پیام assistant در تاریخچه = یک سوال تکمیلی،
و پیام user بلافاصله بعدش = جواب همان سوال. یعنی زوج‌های سوال/جواب را
می‌شود بدون فراخوانی مدل اضافه و بدون هزینه/تاخیر بیشتر، مستقیم از
conversation_history استخراج کرد.

محدودیت شناخته‌شده (مستند، نه پنهان): اگر مدل به‌جای «فقط سوال»، توصیه و
سوال را با هم در یک پیام بدهد، کل همان پیام خام به‌عنوان question ذخیره
می‌شود. قابل قبول است چون این فیلد کمکی/نمایشی است برای پزشک مقصد در صورت
ارجاع - نه منبع تصمیم بالینی خودکار.

حریم خصوصی: patient_record همیشه باید تازه (per-request) ساخته شود، حتی
برای پاسخ‌های آمده از کش (app/pipeline/cache.py) - چون شامل جزئیات همین
مکالمه/همین کاربر است. جزئیات در inference.py و chat.py.
"""
from __future__ import annotations

from typing import Any


def build_patient_record(
    specialty: str,
    specialty_fa: str,
    history: list[dict],
    current_query: str,
) -> dict[str, Any]:
    """
    history: همان لیست dict تمیزشده‌ای که به prompt.build_prompt هم داده
             می‌شود (کلیدهای role/content، بدون پیام system).
    current_query: سوال فعلی کاربر (متن sanitized، بعد از گارد ورودی).

    خروجی سازگار با app.pipeline.schemas.PatientRecord.
    """
    is_first_turn = len(history) == 0
    chief_complaint = history[0]["content"] if history else current_query

    qa_pairs: list[dict[str, str]] = []
    pending_question: str | None = None
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            pending_question = content
        elif role == "user" and pending_question is not None:
            qa_pairs.append({"question": pending_question, "answer": content})
            pending_question = None

    return {
        "specialty": specialty,
        "specialty_fa": specialty_fa,
        "chief_complaint": chief_complaint,
        "qa_pairs": qa_pairs,
        # in_progress: هنوز سوال تکمیلی در جریان است (اولین پیام کاربر است)
        # completed: حداقل یک دور سوال/جواب رد و بدل شده، جمع‌بندی/ارجاع نهایی است
        "status": "in_progress" if is_first_turn else "completed",
    }
