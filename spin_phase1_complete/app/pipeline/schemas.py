# app/pipeline/schemas.py
"""
مدل‌ها و ابزارهای مشترک بین مراحل مختلف Pipeline.

چون فایل‌های مرحله‌ای با عدد شروع می‌شوند (مثل `1_guard_input.py`) و با دستور
`import` معمولی قابل ایمپورت نیستند، هر چیزی که بین چند مرحله مشترک است
اینجا نگهداری می‌شود تا همه مراحل از یک منبع واحد استفاده کنند.
"""
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# ثابت‌های قرارداد (بر اساس OpenApi_Spin.yaml)
# ---------------------------------------------------------------------------

# ChatRequest.query -> minLength: 1, maxLength: 2000
MIN_QUERY_LENGTH: int = 1
MAX_QUERY_LENGTH: int = 2000

# ChatRequest.conversation_history -> maxItems: 6
MAX_HISTORY_MESSAGES: int = 6

# Message.content -> maxLength: 1000
MAX_MESSAGE_CONTENT_LENGTH: int = 1000

# StructuredMedicalResponse.needs_human_review -> قانون: confidence_score < 0.6
CONFIDENCE_REVIEW_THRESHOLD: float = 0.6

# StructuredMedicalResponse.urgency هایی که حتما باید بازبینی انسانی شوند
REVIEW_URGENCY_LEVELS: frozenset[str] = frozenset({"high", "critical"})

# متن الزامی disclaimer - طبق اسپک هرگز نباید خالی یا حذف شود
DEFAULT_DISCLAIMER: str = (
    "این مشاوره جایگزین تشخیص و ویزیت حضوری پزشک نیست. "
    "در موارد اورژانسی فورا با 115 تماس بگیرید."
)

# رشته‌ای که به جای اطلاعات هویتی کاربر جایگزین می‌شود
REDACTION_PLACEHOLDER: str = "[REDACTED]"

# مقادیر مجاز فیلد reason در EventSafetyBlock
SafetyReason = Literal[
    "unsafe_input",
    "unsafe_output",
    "off_topic",
    "pii_detected",
    "low_confidence",
]

# مقادیر مجاز فیلد urgency در StructuredMedicalResponse
UrgencyLevel = Literal["low", "medium", "high", "critical"]

# مقادیر مجاز specialty (همان enum مشترک ChatRequest.selected_body_part و
# EventStart.specialty در OpenApi_Spin.yaml) - فاز ۱: ۶ تخصص.
# این کلیدها باید دقیقا با key در ParsBert/specialties.py یکی باشند.
SPECIALTIES: frozenset[str] = frozenset({
    "gynecology", "orthopedics", "dentistry", "speech_therapy",
    "general_surgery", "general_practice",
})

# ---------------------------------------------------------------------------
# الگوهای اطلاعات محرمانه (PII)
# ---------------------------------------------------------------------------
# اینجا نگهداری می‌شوند چون هم گارد ورودی (مرحله ۱) و هم گارد خروجی
# (مرحله ۸) به همان مجموعه الگو نیاز دارند - با این تفاوت که سیاست‌شان فرق
# دارد: ورودی سانسور می‌شود، خروجی بلاک.
#
# مثل بقیه الگوها، متن قبل از تطبیق با normalize_digits لاتین می‌شود.

# جداکننده‌های رایج بین گروه‌های رقم (فاصله، خط تیره، نقطه)
_GROUP_SEP = r"[\s.\-]?"

# شماره شبا: IR + ۲۴ رقم
_SHEBA_RE = re.compile(r"(?i)(?<![A-Za-z0-9])IR[\s\-]?\d(?:[\s\-]?\d){23}(?!\d)")

# شماره کارت بانکی: ۱۶ رقم در چهار گروه چهارتایی
_BANK_CARD_RE = re.compile(
    rf"(?<![A-Za-z\d])\d{{4}}{_GROUP_SEP}\d{{4}}{_GROUP_SEP}\d{{4}}{_GROUP_SEP}\d{{4}}(?!\d)"
)

# موبایل ایران: 09xxxxxxxxx و شکل‌های +989... / 00989...
_MOBILE_RE = re.compile(
    rf"(?<![\d+])(?:(?:\+|00)98|0)9\d{{2}}{_GROUP_SEP}\d{{3}}{_GROUP_SEP}\d{{4}}(?!\d)"
)

# کد ملی ایران: دقیقا ۱۰ رقم پشت سر هم
_NATIONAL_ID_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")

# ایمیل - شکل ساده و سریع
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# ترتیب مهم است: از خاص‌ترین/طولانی‌ترین به عام‌ترین.
PII_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sheba_number", _SHEBA_RE),
    ("bank_card", _BANK_CARD_RE),
    ("mobile_number", _MOBILE_RE),
    ("national_id", _NATIONAL_ID_RE),
    ("email", _EMAIL_RE),
)

# ---------------------------------------------------------------------------
# نرمال‌سازی ارقام فارسی/عربی
# ---------------------------------------------------------------------------

# جدول تبدیل ارقام فارسی (۰-۹) و عربی-هندی (٠-٩) به ارقام لاتین.
# نکته مهم: این تبدیل «یک کاراکتر به یک کاراکتر» است، بنابراین ایندکس‌های
# متن نرمال‌شده دقیقا با متن اصلی یکی می‌ماند و می‌توان span پیدا شده روی
# متن نرمال را مستقیما روی متن اصلی سانسور کرد.
_DIGIT_TRANSLATION: dict[int, str] = {
    **{0x06F0 + i: str(i) for i in range(10)},  # ۰۱۲۳۴۵۶۷۸۹
    **{0x0660 + i: str(i) for i in range(10)},  # ٠١٢٣٤٥٦٧٨٩
}


def normalize_digits(text: str) -> str:
    """
    ارقام فارسی و عربی را به ارقام لاتین تبدیل می‌کند (طول متن تغییر نمی‌کند).
    فقط یک str.translate است، پس هزینه‌اش در حد میکروثانیه است.
    """
    return text.translate(_DIGIT_TRANSLATION)


# ---------------------------------------------------------------------------
# خروجی مرحله ۱ - گارد ورودی
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InputGuardResult:
    """
    نتیجه بررسی ورودی کاربر در مرحله ۱.

    نکته مهم درباره معنای فیلدها: پیدا شدن محتوای محرمانه باعث رد شدن درخواست
    نمی‌شود؛ متن سانسور می‌شود و جریان با sanitized_text ادامه پیدا می‌کند.
    پس تصمیم «قطع جریان» فقط با is_safe گرفته می‌شود و reason صرفا برچسب
    دلیل (هم‌ارز مقادیر EventSafetyBlock.reason) برای لاگ و متریک است.
    """

    is_safe: bool                 # False فقط وقتی ورودی اصلا قابل پردازش نیست
    reason: str | None            # یکی از مقادیر enum مربوط به EventSafetyBlock.reason
    sanitized_text: str           # متنی که باید در ادامه Pipeline استفاده شود
    pii_detected: bool = False    # فلگ مستقل برای لاگ/متریک
    matched_pii: tuple[str, ...] = ()   # نام قوانین محرمانگی که فعال شدند
    detail: str | None = None     # توضیح داخلی (مثلا length_out_of_range) - برای لاگ و پاسخ 400
    message: str | None = None    # پیام فارسی آماده برای EventSafetyBlock.message

    def to_dict(self) -> dict[str, Any]:
        """تبدیل به dict برای لاگ کردن یا ساخت رویداد SSE."""
        return asdict(self)


# ---------------------------------------------------------------------------
# خروجی مرحله ۸ - گارد خروجی
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OutputGuardResult:
    """نتیجه بررسی توکن/بافر تولیدشده توسط مدل در مرحله ۸."""

    # needs_block یعنی «این متن نباید همان‌طور که هست منتشر شود».
    # اینکه با آن چه می‌کنیم را action تعیین می‌کند:
    #   "block"  -> قطع استریم و ارسال EventSafetyBlock
    #   "redact" -> فقط همان span سانسور و بقیه پیام منتشر می‌شود
    needs_block: bool
    matched_pattern: str | None = None   # دقیقا همان متنی که با regex مچ شده
    rule: str | None = None              # نام قانون فعال‌شده (برای متریک و دیباگ)
    reason: SafetyReason | None = None   # مقدار EventSafetyBlock.reason
    message: str | None = None           # پیام فارسی آماده برای EventSafetyBlock.message
    action: str | None = None            # "block" یا "redact"
    span: tuple[int, int] | None = None  # بازه مچ در همان رشته‌ای که بررسی شد

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_safety_block(self, request_id: str) -> dict[str, Any]:
        """ساخت مستقیم رویداد EventSafetyBlock مطابق قرارداد."""
        return {
            "type": "safety_block",
            "request_id": request_id,
            "reason": self.reason,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# مدل‌های Pydantic مطابق OpenApi_Spin.yaml (مرحله ۹)
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """اسکیمای Source در قرارداد - برای citation و sources استفاده می‌شود."""

    title: str
    url: str | None = None
    author: str | None = None
    published_date: str | None = None
    relevance: float = Field(ge=0.0, le=1.0)


class PatientIntakeQA(BaseModel):
    """یک زوج سوال تکمیلی/جواب، از مرحله ۶.۵ (app/pipeline/intake.py)."""

    question: str
    answer: str


class PatientRecord(BaseModel):
    """
    «پرونده ابتدایی» بیمار - خلاصه‌ای structured از خودِ گفتگو، برای اینکه
    در صورت ارجاع، پزشک/تخصص مقصد در چند ثانیه زمینه کامل را ببیند.

    این یک پرونده پزشکی رسمی/قانونی نیست؛ صرفا بازتاب همین یک مکالمه است.
    عمدا هرگز نباید بین کاربران مختلف به اشتراک گذاشته یا کش شود (نگاه کنید
    به یادداشت حریم خصوصی در app/pipeline/intake.py).
    """

    specialty: str
    specialty_fa: str
    chief_complaint: str
    qa_pairs: list[PatientIntakeQA] = Field(default_factory=list)
    status: Literal["in_progress", "completed"]


class StructuredMedicalResponse(BaseModel):
    """
    اسکیمای StructuredMedicalResponse.
    فیلدهای الزامی: summary, recommendations, urgency, disclaimer
    """

    summary: str
    recommendations: list[str] = Field(default_factory=list)
    urgency: UrgencyLevel
    sources: list[Source] | None = None
    disclaimer: str
    needs_human_review: bool = False
    detected_specialty: str | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    patient_record: PatientRecord | None = None

    @field_validator("disclaimer")
    @classmethod
    def _disclaimer_must_not_be_empty(cls, value: str) -> str:
        """طبق قرارداد disclaimer هرگز نباید خالی باشد."""
        if not value or not value.strip():
            raise ValueError("disclaimer الزامی است و نباید خالی باشد")
        return value


class EventStructured(BaseModel):
    """اسکیمای EventStructured - رویداد SSE مرحله ۹."""

    type: Literal["structured"] = "structured"
    request_id: str
    structured: StructuredMedicalResponse
