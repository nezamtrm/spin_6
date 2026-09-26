# app/pipeline/router.py
"""
مرحله ۴ از Pipeline: تشخیص تخصص (روتر).

پل بین سرویس و مدل فاین‌تیون‌شده ParsBERT (پوشه ParsBert/ در ریشه ریپو).
resolve_specialty(query) -> RouterResult تنها نقطه ورودی عمومی است.

دو حالت کاری (از خود ParsBert/predict.py به ارث می‌رسد، هیچ منطقی اینجا
تکرار نشده):
  - fine-tuned : اگر ParsBert/models/parsbert-specialty/final/ موجود باشد
                 (دقیق‌ترین حالت، بعد از اجرای ParsBert/train.py)
  - zero-shot  : وگرنه، بر پایه امبدینگ خام ParsBERT + واژگان کلیدی هر
                 تخصص در ParsBert/specialties.py (بدون نیاز به آموزش،
                 دقت کمتر ولی سرویس را کاملاً از کار نمی‌اندازد)

قید طراحی: مدل فقط یک بار (lazy) و در اولین استفاده بارگذاری می‌شود، نه در
هر ریکوئست. warmup() برای بارگذاری آن هنگام startup سرویس است تا اولین
ریکوئست واقعی هزینه بارگذاری را پرداخت نکند (شبیه triage.warmup()).
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.observability.metrics import router_confidence, router_fallback_total, specialty_requests_total

log = get_logger(__name__)

# ریشه ریپو: .../app/pipeline/router.py -> parents[2] = ریشه ریپو
_DEFAULT_PARSBERT_DIR = str(Path(__file__).resolve().parents[2] / "ParsBert")
_PARSBERT_DIR = os.environ.get("PARSBERT_DIR", _DEFAULT_PARSBERT_DIR)

# predict.py با ایمپورت‌های خام (from specialties import ...) نوشته شده،
# پس پوشه‌اش باید مستقیما به sys.path اضافه شود، نه به شکل یک پکیج.
if _PARSBERT_DIR not in sys.path:
    sys.path.insert(0, _PARSBERT_DIR)


@dataclass(frozen=True, slots=True)
class RouterResult:
    key: str              # کلید انگلیسی enum (باید عضو schemas.SPECIALTIES باشد)
    specialty_fa: str      # نام فارسی تخصص، برای پرامپت/لاگ
    confidence: float       # 0..1
    certain: bool           # آیا اطمینان بالای آستانه LOW_CONFIDENCE است؟
    mode: str               # "fine-tuned" یا "zero-shot"


_classifier = None
_load_lock = Lock()


def _get_classifier():
    """بارگذاری تنبل و thread-safe. اگر transformers/torch یا فایل مدل در
    دسترس نباشد، استثنا بالا می‌رود - فراخوان (resolve_specialty) آن را
    می‌گیرد و fail-safe روی "general_practice" می‌رود، نه اینکه کل
    درخواست با ۵۰۰ متوقف شود."""
    global _classifier
    if _classifier is not None:
        return _classifier
    with _load_lock:
        if _classifier is None:
            from predict import SpecialtyClassifier  # noqa: E402  (وابسته به sys.path بالا)

            model_dir = getattr(settings, "PARSBERT_MODEL_DIR", None) or None
            _classifier = SpecialtyClassifier(model_dir=model_dir)
            log.info("router_model_loaded", mode=_classifier.mode, parsbert_dir=_PARSBERT_DIR)
    return _classifier


# اگر روتر بالا نیامد (مدل/کتابخانه در دسترس نبود)، این مقدار fail-safe است -
# نه "other" (که دیگر در enum فاز ۱ وجود ندارد) و نه سقوط با خطای ۵۰۰.
_FALLBACK_KEY = "general_practice"
_FALLBACK_FA = "پزشک عمومی"


def resolve_specialty(query: str) -> RouterResult:
    """
    تشخیص تخصص برای یک متن. هیچ‌گاه استثنا بالا نمی‌رود؛ در بدترین حالت
    (مدل بارگذاری نشد) با اطمینان صفر روی «پزشک عمومی» fail-safe می‌شود،
    چون این سرویس روی مسیر داغ چت است و نباید کل درخواست را خراب کند.
    """
    try:
        clf = _get_classifier()
        result = clf.predict(query, top_k=1)
        top = result.top
        specialty_requests_total.labels(specialty=top.key).inc()
        router_confidence.labels(specialty=top.key).observe(top.confidence)
        return RouterResult(
            key=top.key,
            specialty_fa=top.specialty,
            confidence=top.confidence,
            certain=result.certain,
            mode=result.mode,
        )
    except Exception:  # noqa: BLE001 - مسیر fail-safe عمدی
        log.exception("router_predict_failed_fallback_general_practice")
        router_fallback_total.inc()
        specialty_requests_total.labels(specialty=_FALLBACK_KEY).inc()
        return RouterResult(
            key=_FALLBACK_KEY,
            specialty_fa=_FALLBACK_FA,
            confidence=0.0,
            certain=False,
            mode="fallback",
        )


def warmup() -> None:
    """صدا زدن هنگام startup سرویس تا هزینه بارگذاری مدل روی اولین
    ریکوئست واقعی نیفتد (مشابه app.pipeline.triage.warmup)."""
    try:
        _get_classifier()
    except Exception:  # noqa: BLE001
        log.exception("router_warmup_failed")


if __name__ == "__main__":  # pragma: no cover
    examples = [
        "دو ماهه پریود نشدم و حالت تهوع صبحگاهی دارم",
        "موقع بالا رفتن از پله زانوم صدا میده و درد میگیره",
        "چند روزه دندون آسیام تیر میکشه و به سرما حساس شده",
        "پسر سه ساله‌ام هنوز جمله‌بندی درست نمیکنه",
        "چند روزه شکمم درد میکنه، بیشتر سمت راست پایینشه",
        "سرما خوردم و میخوام یه نسخه بگیرم",
    ]
    for ex in examples:
        r = resolve_specialty(ex)
        print(f"{ex!r} -> {r.key} ({r.specialty_fa}) {r.confidence:.2f} mode={r.mode}")
