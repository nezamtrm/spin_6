# app/pipeline/inference.py
"""
مرحله ۶: آماده‌سازی تولید پاسخ.

نقطهٔ واحدی که سه چیز را قبل از فراخوانی vLLM کنار هم می‌گذارد:
  - بازیابی منابع (RAG، مرحله ۵: app/pipeline/retriever.py)
  - پرونده ابتدایی بیمار (مرحله ۶.۵: app/pipeline/intake.py)
  - ساخت پرامپت نهایی (app/pipeline/prompt.py) که منابع را هم در خودش جا می‌دهد

chat.py به‌جای صدا زدن جداگانهٔ هر سه، فقط prepare() را صدا می‌زند - اگر
ترتیب/منطق ترکیب این سه بعدا عوض شد (مثلا یک مرحله rerank دیگر اضافه شد)،
فقط همین یک‌جا تغییر می‌کند، نه chat.py.

عمدا استریم واقعی از vLLM اینجا نیست: app/llm/vllm_client.py مسئول همان
است و chat.py مستقیم صدایش می‌زند - چون آن یک HTTP client خام است، نه
منطق pipeline.
"""
from __future__ import annotations

from typing import Any

from app.pipeline import intake
from app.pipeline.prompt import build_prompt
from app.pipeline.retriever import retrieve


def prepare(
    specialty: str,
    specialty_fa: str,
    history: list[dict],
    current_query: str,
) -> tuple[list[dict], list[dict[str, Any]], dict[str, Any]]:
    """
    خروجی:
      messages       -> برای app.llm.vllm_client.stream_chat_completion
      sources        -> برای EventCitation و structured.sources (قابل کش/اشتراک)
      patient_record -> برای structured.patient_record (هرگز کش/اشتراک نشود؛
                         نگاه کنید به یادداشت حریم خصوصی در intake.py)
    """
    sources = retrieve(specialty, current_query)
    messages = build_prompt(
        history=history,
        current_query=current_query,
        specialty=specialty,
        specialty_fa=specialty_fa,
        sources=sources,
    )
    patient_record = intake.build_patient_record(specialty, specialty_fa, history, current_query)
    return messages, sources, patient_record
