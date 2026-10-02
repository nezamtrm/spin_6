# app/pipeline/guard_input.py
"""
مرحله ۱ از Pipeline: گارد ورودی.

نقش این مرحله «سانسور محتوای محرمانه» است، نه احراز هویت و نه رد کردن کاربر:
هویت کاربر اصلا به این سرویس نمی‌رسد؛ طبق قرارداد، بک‌اند user_id را با
SHA-256 هش می‌کند و فقط user_hash می‌فرستد. بنابراین تنها جایی که ممکن است
اطلاعات محرمانه لو برود، «متن خود سوال» است: کد ملی، شماره موبایل، شماره
کارت، شبا و ایمیلی که کاربر ناخواسته داخل سوالش می‌نویسد.

سیاست: چنین اطلاعاتی از متن حذف (سانسور) می‌شود و جریان با متن پاک‌شده
ادامه پیدا می‌کند - یعنی is_safe همچنان True می‌ماند و فقط فلگ pii_detected
ست می‌شود. تنها چیزی که واقعا ورودی را نامعتبر می‌کند، طول خارج از بازه
قرارداد است.

قید طراحی: این تابع روی مسیر داغ هر درخواست است و باید زیر ۲۰ میلی‌ثانیه
تمام شود؛ بنابراین هیچ I/O، هیچ فراخوانی مدل و هیچ کامپایل regex در زمان
اجرا انجام نمی‌شود. تمام الگوها در سطح ماژول با re.compile ساخته شده‌اند.
"""
import re
import sys
from pathlib import Path

# اجازه اجرای مستقیم فایل برای self-check پایین صفحه
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline.schemas import (
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    REDACTION_PLACEHOLDER,
    InputGuardResult,
    normalize_digits,
)

# ---------------------------------------------------------------------------
# الگوهای محتوای محرمانه - همه در زمان ایمپورت کامپایل می‌شوند
# ---------------------------------------------------------------------------
# نکته: قبل از اجرای این الگوها، متن با normalize_digits به ارقام لاتین تبدیل
# می‌شود. چون آن تبدیل «یک کاراکتر به یک کاراکتر» است، ایندکس‌های به‌دست‌آمده
# روی متن نرمال‌شده دقیقا روی متن اصلی هم معتبرند و می‌توان همان‌جا سانسور کرد.
# به همین دلیل لازم نیست داخل هر الگو کلاس [0-9۰-۹٠-٩] بنویسیم (هم خواناتر
# است و هم سریع‌تر).

# جداکننده‌های رایج بین گروه‌های رقم (فاصله، خط تیره، نقطه)
_GROUP_SEP = r"[\s.\-]?"

# شماره شبا: IR + ۲۴ رقم (با یا بدون فاصله بین گروه‌ها).
# الگو عمدا روی «رقم» تمام می‌شود تا فاصله بعد از شبا بلعیده نشود.
_SHEBA_RE = re.compile(r"(?i)(?<![A-Za-z0-9])IR[\s\-]?\d(?:[\s\-]?\d){23}(?!\d)")

# شماره کارت بانکی: ۱۶ رقم در چهار گروه چهارتایی.
# لوک‌بی‌هایند شامل حروف است تا ۱۶ رقم اولِ یک شبا را جدا مچ نکند.
_BANK_CARD_RE = re.compile(
    rf"(?<![A-Za-z\d])\d{{4}}{_GROUP_SEP}\d{{4}}{_GROUP_SEP}\d{{4}}{_GROUP_SEP}\d{{4}}(?!\d)"
)

# موبایل ایران: 09xxxxxxxxx و همچنین شکل‌های +989..., 00989... و با جداکننده
_MOBILE_RE = re.compile(
    rf"(?<![\d+])(?:(?:\+|00)98|0)9\d{{2}}{_GROUP_SEP}\d{{3}}{_GROUP_SEP}\d{{4}}(?!\d)"
)

# کد ملی ایران: دقیقا ۱۰ رقم پشت سر هم که رقم دیگری کنارش نباشد.
# لوک‌بی‌هایند/لوک‌اَهد جلوی مچ شدن وسط یک شماره ۱۱ یا ۱۶ رقمی را می‌گیرد.
_NATIONAL_ID_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")

# ایمیل - شکل ساده و سریع؛ برای سانسور کافی است
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# ترتیب مهم است: از خاص‌ترین/طولانی‌ترین به عام‌ترین.
# بازه‌های هم‌پوشان بعدا در redact_pii ادغام می‌شوند.
_PII_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sheba_number", _SHEBA_RE),
    ("bank_card", _BANK_CARD_RE),
    ("mobile_number", _MOBILE_RE),
    ("national_id", _NATIONAL_ID_RE),
    ("email", _EMAIL_RE),
)

# پیام‌های فارسی آماده برای EventSafetyBlock.message
_MSG_TOO_SHORT = "لطفا سوال خود را بنویسید."
_MSG_TOO_LONG = (
    f"سوال شما بیش از حد طولانی است. لطفا آن را در حداکثر {MAX_QUERY_LENGTH} "
    "کاراکتر خلاصه کنید."
)
_MSG_PII = (
    "برای حفظ حریم خصوصی شما، اطلاعات محرمانه (مانند کد ملی، شماره تماس یا "
    "شماره کارت) از متن حذف شد و پاسخ بدون آن‌ها آماده می‌شود."
)


def redact_pii(text: str) -> tuple[str, tuple[str, ...]]:
    """
    اطلاعات محرمانه را در متن پیدا و با REDACTION_PLACEHOLDER جایگزین می‌کند.

    خروجی: (متن سانسورشده، نام قوانینی که فعال شدند)
    اگر چیزی پیدا نشود، خود متن ورودی بدون تغییر برگردانده می‌شود.
    """
    normalized = normalize_digits(text)

    # جمع‌آوری همه بازه‌های مچ‌شده به همراه نام قانون
    spans: list[tuple[int, int, str]] = []
    for rule_name, pattern in _PII_RULES:
        for match in pattern.finditer(normalized):
            spans.append((match.start(), match.end(), rule_name))

    if not spans:
        return text, ()

    # مرتب‌سازی و ادغام بازه‌های هم‌پوشان تا یک بازه دوبار سانسور نشود
    # (مثلا ۱۶ رقم اولِ شبا نباید جدا به عنوان شماره کارت سانسور شود)
    spans.sort(key=lambda item: (item[0], -item[1]))
    matched_rules: list[str] = []
    parts: list[str] = []
    cursor = 0
    for start, end, rule_name in spans:
        if start < cursor:  # داخل بازه‌ای که قبلا سانسور شده - رد می‌شود
            continue
        parts.append(text[cursor:start])
        parts.append(REDACTION_PLACEHOLDER)
        cursor = end
        if rule_name not in matched_rules:
            matched_rules.append(rule_name)
    parts.append(text[cursor:])

    return "".join(parts), tuple(matched_rules)


def check_input(text: str) -> InputGuardResult:
    """
    بررسی و پاک‌سازی ورودی کاربر قبل از ورود به بقیه Pipeline.

    قوانین:
      ۱) طول متن باید بین MIN_QUERY_LENGTH و MAX_QUERY_LENGTH باشد
         (طبق ChatRequest.query در قرارداد). این تنها حالتی است که
         is_safe=False برمی‌گردد؛ با reason="unsafe_input" و
         detail="length_out_of_range" تا لایه API بتواند به جای
         safety_block پاسخ 400 بدهد.
      ۲) اگر محتوای محرمانه پیدا شود، از متن حذف می‌شود و همان متن پاک‌شده
         در sanitized_text برمی‌گردد. جریان قطع نمی‌شود: is_safe=True و
         pii_detected=True. مقدار reason ("pii_detected") فقط برچسب دلیل
         برای لاگ و متریک است - و اگر روزی سیاست سخت‌گیرانه لازم شد، همان
         مقدار مستقیما در EventSafetyBlock.reason قابل استفاده است.

    یعنی تصمیمِ «قطع جریان» فقط با is_safe گرفته می‌شود، نه با reason.

    این تابع هیچ I/O ای ندارد و باید در حد چند ده میکروثانیه اجرا شود.
    """
    # ۱) اعتبارسنجی طول
    stripped = text.strip()
    if len(stripped) < MIN_QUERY_LENGTH:
        return InputGuardResult(
            is_safe=False,
            reason="unsafe_input",
            sanitized_text=stripped,
            detail="length_out_of_range",
            message=_MSG_TOO_SHORT,
        )
    if len(text) > MAX_QUERY_LENGTH:
        return InputGuardResult(
            is_safe=False,
            reason="unsafe_input",
            sanitized_text=text[:MAX_QUERY_LENGTH],
            detail="length_out_of_range",
            message=_MSG_TOO_LONG,
        )

    # ۲) تشخیص و سانسور محتوای محرمانه - ورودی رد نمی‌شود، فقط پاک می‌شود
    sanitized_text, matched_rules = redact_pii(text)
    if matched_rules:
        return InputGuardResult(
            is_safe=True,
            reason="pii_detected",
            sanitized_text=sanitized_text,
            pii_detected=True,
            matched_pii=matched_rules,
            detail="pii_redacted",
            message=_MSG_PII,
        )

    # ورودی سالم است
    return InputGuardResult(
        is_safe=True,
        reason=None,
        sanitized_text=text,
    )


# ---------------------------------------------------------------------------
# self-check: تایید قید «زیر ۲۰ میلی‌ثانیه»
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import time

    samples = [
        "سلام، چند روزه سرم درد می‌کنه و سرگیجه دارم. چیکار کنم؟",
        "کد ملی من ۰۰۱۲۳۴۵۶۷۸ است و شماره تماسم ۰۹۱۲۳۴۵۶۷۸۹",
        "لطفا با 09121234567 تماس بگیرید یا کد ملی 1234567890 را ثبت کنید",
        "کارت من 6037-9975-1234-5678 و شبا IR062960000000100324200001 است",
        "جواب آزمایش را به ali.test@example.com بفرستید",
        "درد قفسه سینه دارم " * 100,          # حدود ۲۰۰۰ کاراکتر
        "الف" * (MAX_QUERY_LENGTH + 50),       # ورودی خیلی بلند
    ]

    worst_ms = 0.0
    for sample in samples:
        # هر نمونه چند بار اجرا می‌شود تا بدترین حالت واقعی‌تر دیده شود
        for _ in range(200):
            started = time.perf_counter()
            result = check_input(sample)
            elapsed_ms = (time.perf_counter() - started) * 1000
            worst_ms = max(worst_ms, elapsed_ms)
        print(
            f"len={len(sample):>5} | is_safe={result.is_safe!s:<5} "
            f"| pii={result.matched_pii}"
        )
        if result.pii_detected:
            print(f"        -> {result.sanitized_text}")

    print(f"\nبدترین زمان اجرا: {worst_ms:.3f} ms (حد مجاز: 20 ms)")
    assert worst_ms < 20.0, "check_input از بودجه ۲۰ میلی‌ثانیه عبور کرد"
    print("OK - قید کارایی رعایت شد.")
