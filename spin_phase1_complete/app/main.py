# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.observability.metrics import PrometheusMiddleware
from app.observability.metrics import router as metrics_router
from app.pipeline import router as specialty_router
from app.pipeline import triage
from app.rag import embedder

configure_logging()
log = get_logger(__name__)

# تعریف اپلیکیشن با خواندن تنظیمات
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Medical AI Streaming Service API",
)

# تنظیمات امنیتی مرورگرها
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در پروداکشن باید به دامین بک‌اند محدود شود
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قبلا این خط جا افتاده بود - بدون آن هیچ متریکی جمع نمی‌شد و /metrics
# هم هرگز register نمی‌شد (روتر import شده بود ولی هرگز include نشده بود).
app.add_middleware(PrometheusMiddleware)

app.include_router(health.router, prefix="/v1", tags=["health"])
app.include_router(chat.router, prefix="/v1/medical", tags=["chat"])
# metrics_router خودش مسیر "/metrics" را تعریف کرده؛ چون معمولا /metrics باید
# خارج از پیشوند نسخه (v1) و بدون auth عمومی باشد (برای Prometheus scrape)،
# بدون پیشوند اضافه می‌شود - همان‌طور که در docs/OpenApi_Spin.yaml هم هست.
app.include_router(metrics_router, tags=["observability"])


@app.on_event("startup")
async def on_startup():
    log.info("service_startup", env=settings.PROJECT_NAME, model=settings.VLLM_MODEL_NAME)
    # بارگذاری مدل تریاژ (fastText, اگر train شده)، روتر تخصص (ParsBERT) و
    # مدل امبدینگ RAG از همین‌جا - تا هزینه بارگذاری روی اولین ریکوئست واقعی
    # کاربر نیفتد. هر سه fail-safe هستند (اگر بارگذاری نشوند، فقط لاگ
    # می‌کنند - سرویس بالا می‌آید، همان مسیرهای fallback در خود ماژول‌ها فعال می‌شوند).
    triage.warmup()
    specialty_router.warmup()
    embedder.warmup()
    log.info("warmup_done")


# اگر سرور را مستقیم ران کنیم (برای تست)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)

