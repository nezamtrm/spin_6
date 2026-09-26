# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # تنظیمات پایه
    PROJECT_NAME: str = "SPIN"
    VERSION: str = "1.0.0"

    # تنظیمات اتصال به موتور هوش مصنوعی (vLLM)
    # در محیط لوکال: http://localhost:8000/v1
    # در محیط داکر: http://vllm:8000/v1
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"

    # تنظیمات مدل برای فاز ۱
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.2

    # تنظیمات تریاژ
    TRIAGE_MODEL_PATH: str = "app/pipeline/triage_model.bin"

    # مسیر مدل فاین‌تیون‌شده ParsBERT برای روتر تخصص (مرحله ۴).
    # خالی/None -> پیش‌فرض خودِ ParsBert/predict.py استفاده می‌شود
    # (ParsBert/models/parsbert-specialty/final)؛ اگر می‌خواهید وزن‌ها را
    # جای دیگری (مثلا یک ولوم دانلودشده در دیپلوی) نگه دارید همین را ست کنید.
    PARSBERT_MODEL_DIR: str | None = None

    # ==========================================
    # RAG: Qdrant + مدل امبدینگ (app/rag/*, app/pipeline/retriever.py)
    # ==========================================
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_TIMEOUT_SECONDS: float = 3.0
    # پیشوند نام کالکشن؛ نام نهایی هر کالکشن: f"{prefix}_{specialty_key}"
    # مثلا spin_kb_dentistry, spin_kb_gynecology, ...
    QDRANT_COLLECTION_PREFIX: str = "spin_kb"
    # None -> پیش‌فرض app/rag/embedder.py (همان بک‌بون ParsBERT) استفاده می‌شود
    EMBEDDING_MODEL_NAME: str | None = None
    # چند منبع نهایی (بعد از rerank) به prompt/citation داده شود
    RAG_TOP_K: int = 5

    # ==========================================
    # کش معنایی (app/pipeline/cache.py) - روی همان Qdrant، کالکشن جدا
    # ==========================================
    QDRANT_CACHE_COLLECTION_PREFIX: str = "spin_cache"
    # آستانه شباهت کسینوسی برای این‌که یک نتیجه "hit" حساب شود. عمدا بالا
    # (نزدیک ۱.۰) است: در یک سرویس پزشکی، یک miss اضافه (تولید دوباره پاسخ،
    # فقط کمی کندتر) بی‌خطر است؛ اما یک hit اشتباه (پاسخ یک سوال متفاوت را
    # به‌جای این یکی برگرداندن) می‌تواند خطرناک باشد. اگر بعدا دیدید نرخ
    # hit خیلی پایین است، این را کمی (نه خیلی) پایین بیاورید و با
    # evaluate روی نمونه واقعی بسنجید، حدس نزنید.
    CACHE_SIMILARITY_THRESHOLD: float = 0.93
    CACHE_TTL_SECONDS: int = 6 * 60 * 60  # ۶ ساعت

    # احراز هویت (فعلا فقط برای اعتبارسنجی حداقلی - جزئیات در app/api/deps.py)
    SERVICE_JWT_SECRET: str = "change-me-in-production"

    # سبک pydantic v2: model_config به‌جای class Config قدیمی -
    # هر دو در عمل کار می‌کنند ولی این سبک warning نمی‌دهد و مسیر ارتقای
    # آینده به pydantic v3 را هم هموارتر می‌کند.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# یک نمونه از این کلاس می‌سازیم تا همهٔ فایل‌ها از همین import کنند
settings = Settings()
