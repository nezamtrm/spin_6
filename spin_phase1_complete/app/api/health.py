# app/api/health.py
import os
import subprocess

import aiohttp
from fastapi import APIRouter, Response

from app.core.config import settings
from app.rag import embedder, qdrant_client

router = APIRouter()


@router.get("/health")
async def health_check():
    """Liveness ساده - فقط یعنی «process بالاست»، نه این‌که آماده ترافیک است."""
    return {"status": "healthy", "version": settings.VERSION}


async def _check_vllm() -> bool:
    """بررسی واقعی زنده‌بودن vLLM با یک GET سبک به /models - نه فقط حدس.
    timeout کوتاه عمدی است: readiness نباید خودش کند شود."""
    try:
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"{settings.VLLM_API_BASE}/models"
            async with session.get(url) as resp:
                return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _gpu_stats() -> tuple[str | None, float | None]:
    """best-effort - فقط اگر سرویس روی همان هاست GPU اجرا شود کار می‌کند
    (nvidia-smi در PATH باشد). اگر vLLM روی سرور جدا باشد، این همیشه
    None برمی‌گرداند - آن حالت باید بعدا از متریک‌های خودِ vLLM (/metrics)
    خوانده شود، نه از این تابع."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.5, check=True,
        )
        used_mb = float(out.stdout.strip().splitlines()[0])
        return "available", round(used_mb / 1024, 2)
    except Exception:  # noqa: BLE001
        return None, None


@router.get("/ready")
async def readiness_check(response: Response):
    """
    Readiness واقعی - هر جزء واقعا بررسی می‌شود، هیچ‌کدام hardcode نیست.

    قاعده is_ready: فقط vLLM (پیش‌نیاز مطلق تولید پاسخ) باعث not_ready
    می‌شود. Qdrant/embedder نبودشان یعنی RAG خالی برمی‌گردد (نه خطا - نگاه
    کنید به app/pipeline/retriever.py) و fastText نبودنش یعنی تریاژ
    قانون‌محور fallback است (app/pipeline/triage.py) - این دو "degraded"
    هستند نه "down"، پس سرویس همچنان ترافیک می‌گیرد.
    """
    triage_model_exists = os.path.exists(settings.TRIAGE_MODEL_PATH)

    vllm_up = await _check_vllm()
    qdrant_up = qdrant_client.ping()
    embedder_up = getattr(embedder, "_model", None) is not None
    gpu_status, gpu_mem = _gpu_stats()

    components = {
        "triage_fasttext_model": "up" if triage_model_exists else "down",
        "vllm": "up" if vllm_up else "down",
        "qdrant": "up" if qdrant_up else "down",
        "embedder": "up" if embedder_up else "down",
        # کش معنایی روی Qdrant است (app/pipeline/cache.py) - پس واقعا به
        # قدرانت+امبدر وابسته است، نه یک مقدار ثابت. اگر یکی از این دو
        # پایین باشد، کش فقط یعنی همه چیز miss حساب می‌شود (مسیر عادی
        # تولید پاسخ طی می‌شود)، نه این‌که سرویس از کار بیفتد.
        "cache": "up" if (qdrant_up and embedder_up) else "down",
        "model_version": settings.VLLM_MODEL_NAME,
    }
    if gpu_status:
        components["gpu"] = gpu_status
    if gpu_mem is not None:
        components["gpu_memory_used_gb"] = gpu_mem

    # فقط vLLM پیش‌نیاز سخت است. بقیه (qdrant/embedder/fasttext) نبودشان
    # کیفیت را کم می‌کند ولی سرویس را کاملا از کار نمی‌اندازد.
    is_ready = vllm_up

    if not is_ready:
        response.status_code = 503
        return {"status": "not_ready", "components": components}
    return {"status": "ready", "components": components}
