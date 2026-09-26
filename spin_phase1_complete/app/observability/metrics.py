# app/observability/metrics.py
import time

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_COUNT = Counter(
    "spin_requests_total",
    "تعداد کل ریکوئست‌ها",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "spin_request_latency_seconds",
    "زمان پاسخگویی به ریکوئست‌ها",
    ["method", "path"],
)

ERROR_COUNT = Counter(
    "spin_errors_500_total",
    "تعداد خطاهای 500",
    ["method", "path"],
)

# --- متریک‌های اختصاصی pipeline (جدید) --------------------------------------

# هدف قرارداد: triage باید p99 زیر 100ms باشد. با این هیستوگرام می‌شود
# p50/p95/p99 واقعی را از Prometheus/Grafana بیرون کشید، نه فقط میانگین.
triage_duration_ms = Histogram(
    "spin_triage_duration_ms",
    "زمان اجرای تشخیص اورژانس (is_emergency) به میلی‌ثانیه",
    buckets=(5, 10, 20, 35, 50, 75, 100, 150, 250),
)

# فاز ۲ (وقتی cache معنایی وصل شد) از این دو برای Cache Hit Rate استفاده می‌شود؛
# از الان تعریف می‌شوند تا چیدمان Grafana بعداً نیاز به تغییر کد نداشته باشد.
cache_hits_total = Counter("spin_cache_hits_total", "تعداد پاسخ‌های برگرفته از cache معنایی")
cache_misses_total = Counter("spin_cache_misses_total", "تعداد پاسخ‌هایی که cache نداشتند")

# برای دیدن این‌که چند بار guard_input/guard_output هر کدام از دلایل را trigger
# کرده‌اند - هم برای متریک، هم برای این‌که بعداً کدام قانون trigger پرتکرار است بفهمیم.
safety_blocks_total = Counter(
    "spin_safety_blocks_total", "تعداد رویدادهای safety_block", ["reason"]
)
emergency_detected_total = Counter(
    "spin_emergency_detected_total", "تعداد تشخیص اورژانس"
)

# --- متریک‌های روتر تخصص (مرحله ۴) ------------------------------------------
specialty_requests_total = Counter(
    "spin_specialty_requests_total", "تعداد درخواست به ازای هر تخصص", ["specialty"]
)
router_confidence = Histogram(
    "spin_router_confidence",
    "اطمینان روتر (ParsBERT) به تخصص تشخیص‌داده‌شده",
    ["specialty"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
router_fallback_total = Counter(
    "spin_router_fallback_total",
    "تعداد دفعاتی که روتر بالا نیامد و fail-safe به general_practice افتاد",
)

# --- متریک‌های RAG (مرحله ۵.۵) -----------------------------------------------
rag_sources_found = Histogram(
    "spin_rag_sources_found",
    "تعداد منابع بازیابی‌شده به ازای هر درخواست (۰ یعنی RAG چیزی پیدا نکرد)",
    buckets=(0, 1, 2, 3, 5, 8),
)

# --- متریک‌های پرونده ابتدایی (مرحله ۶.۵) ------------------------------------
patient_record_status_total = Counter(
    "spin_patient_record_status_total",
    "تعداد پرونده‌های ابتدایی به تفکیک وضعیت",
    ["status"],  # in_progress | completed
)

# --- متریک زمان استریم واقعی از vLLM -----------------------------------------
vllm_stream_duration_seconds = Histogram(
    "spin_vllm_stream_duration_seconds",
    "زمان کل استریم پاسخ از vLLM (اول تا آخرین توکن)",
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 20, 30),
)

router = APIRouter()


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
        REQUEST_COUNT.labels(method=method, path=path, status_code=response.status_code).inc()

        if response.status_code == 500:
            ERROR_COUNT.labels(method=method, path=path).inc()

        return response


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

