# app/api/chat.py
import asyncio
import time
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.api.deps import verify_service_token
from app.core.sse import format_sse_event, sse_from_event
from app.observability.metrics import (
    cache_hits_total,
    cache_misses_total,
    emergency_detected_total,
    patient_record_status_total,
    rag_sources_found,
    safety_blocks_total,
    triage_duration_ms,
    vllm_stream_duration_seconds,
)
from app.pipeline import cache as response_cache
from app.pipeline import intake
from app.pipeline import inference
from app.pipeline.guard_input import check_input
from app.pipeline.guard_output import OutputBlocked, guarded_token_stream
from app.pipeline.router import resolve_specialty
from app.pipeline.schemas import SPECIALTIES
from app.pipeline.structured import build_structured_event
from app.pipeline.triage import is_emergency
from app.llm.vllm_client import stream_chat_completion

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str = Field(max_length=1000)
    timestamp: Optional[datetime] = None


class UserMetadata(BaseModel):
    age_range: Optional[str] = None
    gender: Optional[str] = None
    language: str = "fa"
    preferred_style: str = "formal"


class ChatRequest(BaseModel):
    request_id: UUID
    session_id: UUID
    user_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    query: str = Field(min_length=1, max_length=2000)
    selected_body_part: Optional[str] = None
    conversation_history: Optional[list[Message]] = Field(default=None, max_length=6)
    metadata: Optional[UserMetadata] = None
    temperature: float = Field(default=0.2, ge=0, le=0.5)

    @field_validator("selected_body_part")
    @classmethod
    def _validate_specialty(cls, value: Optional[str]) -> Optional[str]:
        """طبق قرارداد، selected_body_part باید یکی از مقادیر enum فاز ۱ باشد.
        اگر بک‌اند مقدار خارج از enum بفرستد، بهتر است همین‌جا ۴۲۲/۴۰۰ روشن
        بگیریم تا اینکه بی‌صدا به‌عنوان یک تخصص نامعتبر در باقی pipeline جاری شود."""
        if value is not None and value not in SPECIALTIES:
            raise ValueError(
                f"selected_body_part نامعتبر است: {value!r}. مقادیر مجاز: {sorted(SPECIALTIES)}"
            )
        return value


def _done_event(req_id: str, start_time: float, *, specialty: str | None,
                 total_tokens: int = 0, prompt_tokens: int = 0,
                 completion_tokens: int = 0, cached: bool = False) -> str:
    """
    رویداد done مشترک بین همهٔ مسیرها (عادی/اورژانس/safety_block/cached).
    طبق قرارداد، هر مسیر باید دقیقا با یک done تمام شود؛ استخراج این تابع
    جلوی این را می‌گیرد که یک مسیر جدید در آینده اضافه شود و done را فراموش کند.

    specialty=None برای مسیرهای safety_block/emergency که دیگر (بعد از
    اصلاح ترتیب پایپلاین) قبل از روتر کوتاه می‌شوند - یعنی اصلا تخصصی
    تشخیص داده نشده. طبق schema قرارداد، specialty در EventDone الزامی
    نیست، پس به‌جای فرستادن "specialty": null، فیلد کلا حذف می‌شود.
    """
    payload = {
        "request_id": req_id,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": int((time.perf_counter() - start_time) * 1000),
        "cached": cached,
    }
    if specialty is not None:
        payload["specialty"] = specialty
    return format_sse_event("done", payload)


async def _real_ai_stream(request: Request, chat_request: ChatRequest):
    req_id = str(chat_request.request_id)
    start_time = time.perf_counter()

    # ==========================================
    # مرحله ۱: گارد ورودی (بررسی طول و PII) - روی متن خام، قبل از هرچیز
    # ==========================================
    # عمدا اولین مرحله: ارزان‌ترین چک (regex، میکروثانیه) است؛ اگر طول
    # نامعتبر بود یا... هیچ دلیلی ندارد قبلش محاسبه‌ی گران‌تری (تریاژ/روتر)
    # انجام شود. طبق قرارداد اصلاح‌شده، safety_block دیگر منتظر "start"
    # نمی‌ماند - مستقیم اولین رویداد است.
    guard_res = check_input(chat_request.query)
    if not guard_res.is_safe:
        safety_blocks_total.labels(reason=guard_res.reason or "unknown").inc()
        yield format_sse_event("safety_block", {
            "request_id": req_id,
            "reason": guard_res.reason,
            "message": guard_res.message,
        })
        yield _done_event(req_id, start_time, specialty=None)
        return

    safe_query = guard_res.sanitized_text

    # آماده‌سازی تاریخچه یک‌بار، همین‌جا - چون هم برای پرونده ابتدایی در
    # مسیر کش‌اصابت‌شده و هم در مسیر تولید تازه لازم است.
    history_dicts = [msg.model_dump(exclude_none=True) for msg in (chat_request.conversation_history or [])]

    # ==========================================
    # مرحله ۲: تریاژ اورژانس (زیر ۵۰ میلی‌ثانیه، فقط CPU) - قبل از روتر
    # ==========================================
    # عمدا *قبل* از روتر: تریاژ خیلی ارزان‌تر از یک forward-pass کامل
    # ParsBERT است. اگر پیام اورژانسی باشد، اصلا نیازی به دانستن تخصص
    # نیست (کاربر فقط باید ۱۱۵ را بگیرد) - پس چرا محاسبه‌اش کنیم؟ این
    # ترتیب روی دقیق‌ترین مسیر latency (<100ms مستند در قرارداد) مستقیم
    # اثر مثبت دارد، بدون هیچ هزینه‌ی دقت جایی (تریاژ به specialty وابسته
    # نیست، و specialty هم به اینکه تریاژ قبلش اجرا شده یا نه وابسته نیست).
    triage_start = time.perf_counter()
    emergency_detected = await asyncio.to_thread(is_emergency, safe_query)
    triage_duration_ms.observe((time.perf_counter() - triage_start) * 1000)

    if emergency_detected:
        emergency_detected_total.inc()
        yield format_sse_event("emergency", {
            "request_id": req_id,
            "severity": "critical",
            "message": "علائم شما نیازمند بررسی فوری است. فورا با ۱۱۵ تماس بگیرید.",
            "action": "call_115",
        })
        yield _done_event(req_id, start_time, specialty=None)
        return

    # ==========================================
    # مرحله ۴: تشخیص تخصص (روتر / ParsBERT) - فقط روی مسیر عادی
    # ==========================================
    # از این‌جا به بعد مطمئنیم درخواست نه بلاک‌شدنی بود نه اورژانسی - پس
    # تنها حالا این محاسبه‌ی گران‌تر توجیه دارد. متن sanitized (نه خام)
    # به روتر داده می‌شود - چون گارد ورودی حالا از قبل اجرا شده.
    if chat_request.selected_body_part:
        specialty = chat_request.selected_body_part
        specialty_fa = None
        confidence = 1.0
    else:
        router_result = await asyncio.to_thread(resolve_specialty, safe_query)
        specialty = router_result.key
        specialty_fa = router_result.specialty_fa
        confidence = router_result.confidence

    # رویداد start حالا اینجاست: همان لحظه‌ای که specialty واقعا مشخص شده،
    # نه قبل‌ترش. طبق قرارداد اصلاح‌شده، start دیگر "همیشه اولین رویداد
    # مطلق" نیست؛ فقط برای مسیر عادی (نه emergency/safety_block) معنا دارد.
    yield format_sse_event("start", {
        "request_id": req_id,
        "specialty": specialty,
        "confidence": confidence,
    })

    # ==========================================
    # مرحله ۵: کش سوالات تکراری (بعد از رد شدن از تریاژ)
    # ==========================================
    # عمدا *بعد* از تریاژ است: حتی اگر متن با یک سوال قبلی عینا یکسان باشد،
    # تریاژ اورژانس هرگز از کش رد نمی‌شود - ایمنی روی سرعت اولویت دارد.
    cache_result = await asyncio.to_thread(response_cache.get_cached, specialty, safe_query)
    if cache_result.hit:
        cache_hits_total.inc()
        yield format_sse_event("cached", {"request_id": req_id, "similarity": cache_result.similarity})

        # منابع RAG کش‌شده درباره «موضوع»‌اند، نه درباره «این بیمار» - قابل
        # اشتراک/بازپخش امن‌اند. توجه: sources داخل ساختار تودرتوی
        # {"structured": {"sources": [...]}} است، نه سطح بالا.
        cached_sources = (cache_result.structured or {}).get("structured", {}).get("sources") or []
        yield sse_from_event({"type": "citation", "request_id": req_id, "sources": cached_sources})

        seq = 0
        for chunk in response_cache.chunk_for_replay(cache_result.answer):
            if await request.is_disconnected():
                break
            yield format_sse_event("token", {"request_id": req_id, "content": chunk, "seq": seq})
            seq += 1

        if not await request.is_disconnected():
            # پرونده ابتدایی هرگز نباید از کش بازپخش شود - مال یک کاربر/
            # مکالمه دیگر بوده. همیشه تازه، برای همین درخواست ساخته می‌شود
            # (نگاه کنید به یادداشت حریم خصوصی در intake.py).
            fresh_record = intake.build_patient_record(specialty, specialty_fa or specialty, history_dicts, safe_query)
            structured_payload = {**cache_result.structured, "request_id": req_id}
            structured_payload["structured"] = {
                **structured_payload.get("structured", {}),
                "patient_record": fresh_record,
            }
            yield sse_from_event(structured_payload)
            yield _done_event(
                req_id, start_time, specialty=specialty,
                completion_tokens=seq, total_tokens=seq, cached=True,
            )
        return

    cache_misses_total.inc()

    # ==========================================
    # مرحله ۵.۵ و ۶ و ۶.۵: RAG + پرونده ابتدایی + ساخت پرامپت
    # ==========================================
    messages, sources, patient_record = await asyncio.to_thread(
        inference.prepare,
        specialty=specialty,
        specialty_fa=specialty_fa or specialty,
        history=history_dicts,
        current_query=safe_query,
    )
    rag_sources_found.observe(len(sources))
    patient_record_status_total.labels(status=patient_record.get("status", "unknown")).inc()

    # ==========================================
    # مرحله ۷ و ۸: استریم از vLLM + گارد خروجی (پنجره تاخیر)
    # ==========================================
    completion_tokens = 0
    collected_tokens: list[str] = []
    raw_stream = stream_chat_completion(messages)
    stream_start = time.perf_counter()

    try:
        async for safe_token in guarded_token_stream(raw_stream):
            if await request.is_disconnected():
                break
            collected_tokens.append(safe_token)
            yield format_sse_event("token", {
                "request_id": req_id,
                "content": safe_token,
                "seq": completion_tokens,
            })
            completion_tokens += 1
    except OutputBlocked as e:
        # مدل سعی کرد دارو تجویز کند یا PII تولید کند - استریم قطع می‌شود.
        # پاسخ بلاک‌شده هرگز نباید وارد کش شود (چیزی برای set_cached نمی‌فرستیم).
        safety_blocks_total.labels(reason=e.result.reason or "unknown").inc()
        yield sse_from_event(e.result.to_safety_block(req_id))
        yield _done_event(req_id, start_time, specialty=specialty, completion_tokens=completion_tokens)
        return

    if await request.is_disconnected():
        return

    vllm_stream_duration_seconds.observe(time.perf_counter() - stream_start)

    # ==========================================
    # مرحله citation: منابع RAG (اگر چیزی پیدا شده بود)
    # ==========================================
    yield sse_from_event({"type": "citation", "request_id": req_id, "sources": sources})

    # ==========================================
    # مرحله ۹: خروجی ساخت‌یافته و پایان
    # ==========================================
    structured_payload = build_structured_event(
        request_id=req_id,
        summary="پاسخ با موفقیت تولید شد.",  # در فاز ۲ این باید توسط LLM خلاصه شود
        urgency="low",  # فعلاً دیفالت - اورژانس‌ها در بالا فیلتر شدند
        specialty=specialty,
        confidence_score=confidence,
        sources=sources,
        disclaimer=None,  # سیستم خودش DEFAULT_DISCLAIMER را می‌گذارد
        patient_record=patient_record,
    )
    yield sse_from_event(structured_payload)

    # فقط پاسخ سالم (رد شده از guard_output، بلاک نشده) وارد کش می‌شود -
    # ولی *بدون* patient_record (مخصوص همین بیمار است، نباید بین کاربران
    # مختلف که تصادفا سوال یکسان می‌پرسند به اشتراک برود).
    cache_structured = {
        **structured_payload,
        "structured": {k: v for k, v in structured_payload.get("structured", {}).items() if k != "patient_record"},
    }
    # asyncio.to_thread: همان دلیل بالا (embedding + upsert به Qdrant هر دو
    # blocking‌اند). این‌جا حیاتی نیست (فقط دم همین درخواست است) ولی برای
    # یکدستی و رها نکردن event loop حتی یک لحظه، رعایت شده.
    await asyncio.to_thread(
        response_cache.set_cached,
        specialty_key=specialty,
        query=safe_query,
        answer="".join(collected_tokens),
        structured=cache_structured,
    )

    prompt_tokens = len(safe_query.split())
    yield _done_event(
        req_id, start_time, specialty=specialty,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


@router.post("/chat")
async def medical_chat(chat_request: ChatRequest, request: Request, _auth=Depends(verify_service_token)):
    return StreamingResponse(
        _real_ai_stream(request, chat_request),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
