# app/pipeline/9_structured.py
"""
مرحله ۹ از Pipeline: خروجی ساخت‌یافته.

بعد از پایان استریم توکن‌ها، این ماژول رویداد `structured` را می‌سازد؛
همان رویدادی که کلاینت برای نمایش کارت خلاصه، توصیه‌ها، سطح فوریت و
Disclaimer از آن استفاده می‌کند.

ساختار خروجی دقیقا مطابق EventStructured در OpenApi_Spin.yaml است:
    {"type": "structured", "request_id": ..., "structured": {...}}
"""
import logging
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from app.pipeline.schemas import (
    CONFIDENCE_REVIEW_THRESHOLD,
    DEFAULT_DISCLAIMER,
    REVIEW_URGENCY_LEVELS,
    EventStructured,
    PatientRecord,
    Source,
    StructuredMedicalResponse,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)


def needs_human_review(
    urgency: str,
    confidence_score: float | None,
) -> bool:
    """
    قانون قرارداد: needs_human_review اگر confidence_score < 0.6
    یا urgency یکی از [high, critical] باشد.

    نکته: اگر confidence_score اصلا داده نشده باشد (None) یعنی میزان اطمینان
    نامعلوم است؛ در یک سیستم پزشکی حالت امن این است که همان را هم نیازمند
    بازبینی انسانی بدانیم (fail-safe).
    """
    if urgency in REVIEW_URGENCY_LEVELS:
        return True
    if confidence_score is None:
        return True
    return confidence_score < CONFIDENCE_REVIEW_THRESHOLD


def _parse_sources(
    sources: Iterable[Mapping[str, Any]] | None,
) -> list[Source] | None:
    """
    منابع RAG (مرحله ۵) را به اسکیمای Source تبدیل می‌کند.

    هر منبع جداگانه اعتبارسنجی می‌شود: یک منبع خراب (بدون title، با
    relevance خارج از بازه ۰ تا ۱، یا اصلا غیر-Mapping) فقط خودش کنار
    گذاشته و لاگ می‌شود.

    دلیل: sources یک فیلد اختیاری و کمکی است. اجازه ندارد یک ردیف بدشکل از
    ایندکس، پاسخ پزشکیِ آماده را - که مدل قبلا تولیدش کرده و کاربر
    استریمش را دیده - به خطای ۵۰۰ تبدیل کند.
    """
    if not sources:
        return None

    parsed: list[Source] = []
    dropped = 0
    for item in sources:
        try:
            parsed.append(Source.model_validate(dict(item)))
        except (ValidationError, TypeError, ValueError):
            dropped += 1
            logger.warning(
                "منبع نامعتبر در ساخت رویداد structured نادیده گرفته شد: %r", item
            )

    if dropped:
        logger.warning("%d منبع از %d منبع کنار گذاشته شد.", dropped, dropped + len(parsed))

    # اگر هیچ منبع سالمی نماند، بهتر است فیلد اصلا فرستاده نشود تا کلاینت
    # لیست خالیِ گمراه‌کننده نبیند (exclude_none آن را حذف می‌کند).
    return parsed or None


def _parse_patient_record(
    patient_record: Mapping[str, Any] | None,
) -> PatientRecord | None:
    """پرونده ابتدایی (مرحله ۶.۵) را اعتبارسنجی می‌کند. یک پرونده خراب فقط
    حذف می‌شود (خود field اختیاری است)، نه اینکه کل پاسخِ آماده را با خطای
    ۵۰۰ از بین ببرد - همان دلیل _parse_sources."""
    if not patient_record:
        return None
    try:
        return PatientRecord.model_validate(dict(patient_record))
    except (ValidationError, TypeError, ValueError):
        logger.warning("پرونده ابتدایی نامعتبر نادیده گرفته شد: %r", patient_record)
        return None


def build_structured_event(
    request_id: str,
    summary: str,
    urgency: UrgencyLevel,
    recommendations: Iterable[str] | None = None,
    confidence_score: float | None = None,
    specialty: str | None = None,
    sources: Iterable[Mapping[str, Any]] | None = None,
    disclaimer: str | None = None,
    force_human_review: bool = False,
    patient_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    ساخت رویداد `structured` در انتهای استریم.

    ورودی‌ها:
      request_id: همان شناسه درخواست که در بقیه رویدادها فرستاده می‌شود
      summary: خلاصه متن تولیدشده توسط مدل
      urgency: یکی از low | medium | high | critical - الزامی است.
               عمدا مقدار پیش‌فرض ندارد: پیش‌فرضِ «low» یعنی یک فراخوانِ
               فراموش‌شده، یک مورد بحرانی را بی‌سروصدا کم‌خطر گزارش می‌کند و
               needs_human_review را هم False نگه می‌دارد. نبودن این مقدار
               باید همان‌جا خطا بدهد، نه اینکه به آرام‌ترین حالت سقوط کند.
      recommendations: فهرست توصیه‌های عملی
      confidence_score: اطمینان مدل/روتر بین ۰ تا ۱
      specialty: تخصص تشخیص‌داده‌شده توسط روتر (مرحله ۴) -> detected_specialty
      sources: منابع RAG (مرحله ۵) با ساختار اسکیمای Source؛ منابع خراب کنار
               گذاشته می‌شوند و کل درخواست را با خطا مواجه نمی‌کنند
      disclaimer: اگر داده نشود یا خالی باشد، متن پیش‌فرض فارسی گذاشته می‌شود
      force_human_review: اگر مرحله دیگری (مثلا گارد خروجی) تشخیص داده که
                          حتما باید بازبینی انسانی شود، این فلگ نتیجه قانون
                          بالا را فقط می‌تواند True کند، نه False.
      patient_record: پرونده ابتدایی از مرحله ۶.۵ (app/pipeline/intake.py).
                      یک dict خراب/ناقص فقط خودش کنار گذاشته می‌شود، نه کل
                      درخواست (همان الگوی _parse_sources).

    خروجی: dict آماده برای سریالایز شدن در SSE.
    """
    # disclaimer الزامی است - خالی یا فقط فاصله قابل قبول نیست
    final_disclaimer = disclaimer.strip() if isinstance(disclaimer, str) else ""
    if not final_disclaimer:
        final_disclaimer = DEFAULT_DISCLAIMER

    # توصیه‌های خالی حذف و فاصله‌های اضافه پاک می‌شوند
    cleaned_recommendations: list[str] = [
        item.strip()
        for item in (recommendations or [])
        if isinstance(item, str) and item.strip()
    ]

    # منبع خراب فقط خودش کنار می‌رود؛ ۵۰۰ شدن کل ریکوئست معنا ندارد
    parsed_sources = _parse_sources(sources)
    parsed_patient_record = _parse_patient_record(patient_record)

    review_flag = needs_human_review(urgency, confidence_score) or force_human_review

    structured = StructuredMedicalResponse(
        summary=summary.strip(),
        recommendations=cleaned_recommendations,
        urgency=urgency,
        sources=parsed_sources,
        disclaimer=final_disclaimer,
        needs_human_review=review_flag,
        detected_specialty=specialty,
        confidence_score=confidence_score,
        patient_record=parsed_patient_record,
    )
    event = EventStructured(request_id=request_id, structured=structured)

    # exclude_none تا فیلدهای اختیاری خالی در JSON نهایی ظاهر نشوند؛
    # فیلدهای الزامی (از جمله disclaimer) همیشه مقدار دارند و حذف نمی‌شوند.
    return event.model_dump(mode="json", exclude_none=True)
