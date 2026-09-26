# app/pipeline/guard_output.py
"""
مرحله ۸ از Pipeline: گارد خروجی.

دو چیز را در خروجی مدل می‌گیرد، با دو سیاست متفاوت:

  ۱) تجویز دارو: دوز عددی + واحد، یا عبارت‌های تکرار مصرف مثل «روزی دو بار»
     -> reason="unsafe_output"، action="redact"
     فقط همان عبارت با [REDACTED] جایگزین می‌شود و بقیه پاسخ ادامه پیدا
     می‌کند. یک «۵۰۰ میلی‌گرم» دلیل کافی برای دور ریختن کل توضیح پزشکی
     نیست؛ کاربر عملا هیچ پاسخی نمی‌گیرد و دوباره همان را می‌پرسد.
  ۲) اطلاعات محرمانه (PII): کد ملی، موبایل، کارت بانکی، شبا و ایمیل
     -> reason="pii_detected"، action="block"
     اینجا جریان قطع می‌شود تا لایه API رویداد safety_block بفرستد.

چرا PII در خروجی سانسور نمی‌شود (برخلاف مرحله ۱):
    در ورودی، متن مال خود کاربر است و ما فقط جلوی رفتنش به مدل را می‌گیریم؛
    سانسور آنجا کار درستی است چون جریان باید ادامه پیدا کند. اما اگر مدل
    کد ملی یا شماره‌ای *تولید* کند، آن داده یا توهم مدل است یا نشت از منابع
    RAG. جایگزین کردنش با [REDACTED] پاسخ را تمیز نشان می‌دهد در حالی که
    مسئله واقعی - نشت یا توهم - پنهان می‌ماند و پاسخ هم بی‌معنا می‌شود.
    پس اینجا سانسور نداریم: بلاک می‌کنیم.

مسئولیت بافر:
    check_token_stream خودش stateless است و روی هر رشته‌ای کار می‌کند - چه یک
    توکن تنها، چه یک بافر چندتوکنی. اما مدل ممکن است «۵۰۰ میلی‌گرم» را به صورت
    ["۵۰۰", " می", "لی", "‌گرم"] بفرستد؛ در این حالت هیچ توکن تکی الگو را
    فعال نمی‌کند. تابع کمکی update_buffer یک پنجره متحرک
    (SUGGESTED_BUFFER_CHARS کاراکتر آخر) نگه می‌دارد.

چرا توکن‌ها درجا yield نمی‌شوند:
    حتی با بافر، اگر هر توکن را همان لحظه بفرستیم، وقتی الگو در توکن پنجم
    کامل می‌شود «۵۰۰» از قبل روی سیم رفته و نه می‌شود سانسورش کرد نه پس
    گرفت. بنابراین guarded_token_stream یک «پنجره تاخیر» به اندازه
    DELAY_WINDOW_TOKENS توکن نگه می‌دارد: توکن ۱ فقط وقتی منتشر می‌شود که
    توکن ۴ رسیده و متن همچنان تمیز باشد. این پنجره همان چیزی است که سانسورِ
    نقطه‌ای را ممکن می‌کند - چون تا وقتی توکن‌ها در صف‌اند هنوز قابل ویرایش‌اند.
"""
import re
import sys
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, NamedTuple

# اجازه اجرای مستقیم فایل برای دموی پایین صفحه
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline.schemas import (
    PII_RULES,
    REDACTION_PLACEHOLDER,
    OutputGuardResult,
    normalize_digits,
)

# طول پیشنهادی پنجره متحرک برای فراخوان (به کاراکتر)
SUGGESTED_BUFFER_CHARS: int = 64

# چند توکن قبل از انتشار نگه داشته شوند. با ۳، توکن ۱ همزمان با رسیدن
# توکن ۴ منتشر می‌شود.
DELAY_WINDOW_TOKENS: int = 3

# ---------------------------------------------------------------------------
# اجزای مشترک الگوها
# ---------------------------------------------------------------------------
# ارقام قبل از اجرای regex با normalize_digits لاتین می‌شوند، پس \d هم ارقام
# فارسی (۰-۹) و هم عربی (٠-٩) را پوشش می‌دهد.

# عدد: چه رقمی (با اعشار یا کسری) و چه نوشته‌شده با حروف فارسی
_NUMBER = (
    r"(?:\d+(?:[.,/]\d+)?|نیم|یک|یه|دو|سه|چهار|پنج|شش|شیش|هفت|هشت|نه|ده"
    r"|دوازده|یکی|دوتا|سه‌تا|سه تا)"
)

# فاصله یا نیم‌فاصله (ZWNJ) - کلمات مرکب فارسی هر دو شکل نوشته می‌شوند
_SEP = r"[\s\u200c]*"

# واحدهای دوز: انگلیسی و فارسی
_UNIT = (
    r"(?:mg|mcg|ug|µg|ml|cc|g|گرم"
    rf"|میلی{_SEP}گرم|میلی{_SEP}لیتر|میکرو{_SEP}گرم|سی{_SEP}سی|واحد|یونیت)"
)

# شکل‌های دارویی که در عبارت‌های تکرار مصرف می‌آیند
_DOSE_FORM = r"(?:قرص|عدد|کپسول|آمپول|قطره|شربت|اسپری|پاف|شیاف|ست|واحد|سی‌سی)"

# ---------------------------------------------------------------------------
# الگوهای تجویز دارو (کامپایل‌شده در زمان ایمپورت، نه در زمان اجرا)
# ---------------------------------------------------------------------------
_DOSAGE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # ۱) دوز عددی + واحد: «۵۰۰ میلی‌گرم»، «500mg»، «۱۰ سی‌سی»
    (
        "dosage_amount",
        re.compile(rf"(?<![\w\u0600-\u06FF]){_NUMBER}{_SEP}{_UNIT}(?![\w\u0600-\u06FF])"),
    ),
    # ۲) «روزی دو بار» / «روزانه ۳ نوبت»
    (
        "frequency_per_day",
        re.compile(rf"(?:روزی|روزانه){_SEP}(?:{_NUMBER}{_SEP})?(?:بار|نوبت|وعده|{_DOSE_FORM})"),
    ),
    # ۳) «هر ۸ ساعت» / «هر ۱۲ ساعت یک بار» / «هر دو روز»
    (
        "frequency_interval",
        re.compile(rf"هر{_SEP}{_NUMBER}{_SEP}(?:ساعت|روز|هفته|شب)"),
    ),
    # ۴) «دو قرص در روز» / «۱ کپسول هر شب»
    (
        "dose_form_per_period",
        re.compile(rf"{_NUMBER}{_SEP}{_DOSE_FORM}{_SEP}(?:در|هر){_SEP}(?:روز|شب|هفته|وعده|نوبت)"),
    ),
    # ۵) «سه بار در روز» / «۲ بار در شبانه‌روز»
    (
        "times_per_period",
        re.compile(rf"{_NUMBER}{_SEP}بار{_SEP}در{_SEP}(?:روز|شب|شبانه{_SEP}روز|هفته)"),
    ),
    # ۶) اختصارات نسخه‌نویسی لاتین: BID/TID/QID/PRN/q8h
    (
        "prescription_abbreviation",
        re.compile(r"\b(?:bid|tid|qid|qod|prn|q\d+h)\b", re.IGNORECASE),
    ),
)

# ---------------------------------------------------------------------------
# پیام‌های فارسی آماده برای EventSafetyBlock.message
# ---------------------------------------------------------------------------
_MSG_UNSAFE_OUTPUT = (
    "برای ایمنی شما این پاسخ متوقف شد، چون در حال نوشتن دوز یا برنامه مصرف "
    "دارو بود. تعیین دارو و دوز فقط بر عهده پزشک معالج است."
)
_MSG_PII_OUTPUT = (
    "برای حفظ حریم خصوصی، تولید این پاسخ متوقف شد چون حاوی اطلاعات محرمانه "
    "(مانند کد ملی، شماره تماس یا شماره کارت) بود."
)

# ---------------------------------------------------------------------------
# جدول نهایی قوانین
# ---------------------------------------------------------------------------
# هر قانون سیاست خودش را دارد:
#
#   PII    -> "block"   قطع استریم. سانسور اینجا کار درستی نیست؛ اگر مدل کد
#                       ملی تولید کرده، مسئله نشت یا توهم است نه یک کلمه بد.
#   دوز    -> "redact"  فقط همان عبارت دوز با [REDACTED] جایگزین می‌شود و
#                       بقیه پاسخ - که معمولا اطلاعات مفید پزشکی است - به
#                       دست کاربر می‌رسد. پاک کردن کل پیام به خاطر یک
#                       «۵۰۰ میلی‌گرم» هزینه‌اش خیلی بیشتر از فایده‌اش است.
#
# برای عوض کردن سیاست هر خانواده، فقط همین action تغییر می‌کند.


class _Rule(NamedTuple):
    name: str
    pattern: re.Pattern[str]
    reason: str          # مقدار EventSafetyBlock.reason
    action: str          # "block" یا "redact"
    message: str         # پیام فارسی، فقط وقتی action == "block" استفاده می‌شود


# PII اول بررسی می‌شود چون نشت اطلاعات هویتی برگشت‌ناپذیرتر از یک جمله دوز است.
_RULES: tuple[_Rule, ...] = tuple(
    _Rule(name, pattern, "pii_detected", "block", _MSG_PII_OUTPUT)
    for name, pattern in PII_RULES
) + tuple(
    _Rule(name, pattern, "unsafe_output", "redact", _MSG_UNSAFE_OUTPUT)
    for name, pattern in _DOSAGE_RULES
)

# سقف تعداد سانسور پیاپی روی یک توکن. اگر از این بیشتر شد یعنی یا الگویی
# خودش را تکرار می‌کند یا مدل قفل کرده؛ در هر دو حالت جریان قطع می‌شود.
_MAX_REDACTIONS_PER_TOKEN = 8


class OutputBlocked(Exception):
    """
    استثنای کنترلی: گارد خروجی جلوی ادامه استریم را گرفته است.

    از استثنا استفاده می‌شود - نه یک مقدار برگشتی - تا مسیر «توکن سالم» و
    مسیر «بلاک» در هم قاطی نشوند و امکان نداشته باشد فراخوان به اشتباه یک
    توکنِ نگه‌داشته‌شده را بفرستد.
    """

    def __init__(self, result: OutputGuardResult) -> None:
        super().__init__(result.message or result.reason or "output blocked")
        self.result = result


def check_token_stream(token: str) -> OutputGuardResult:
    """
    بررسی یک توکن (یا بافر چندتوکنی) از خروجی مدل.

    خروجی: OutputGuardResult با کلیدهای
        needs_block: bool          -> اگر True باشد استریم باید قطع و
                                      safety_block فرستاده شود
        matched_pattern: str|None  -> متنی که با الگو مچ شده (برای لاگ)
        rule: str|None             -> نام قانون فعال‌شده (برای متریک)
        reason: str|None           -> مقدار EventSafetyBlock.reason
                                      (unsafe_output یا pii_detected)
        message: str|None          -> پیام فارسی EventSafetyBlock.message
        action: str|None           -> "block" (قطع استریم) یا "redact"
                                      (فقط همان span سانسور شود)
        span: (int, int)|None      -> بازه مچ در همین رشته ورودی

    خود این تابع چیزی را تغییر نمی‌دهد؛ فقط گزارش می‌دهد. اعمال سیاست با
    فراخوان (یا با guarded_token_stream) است.

    نکته‌ای که span را قابل اتکا می‌کند: normalize_digits تبدیل «یک کاراکتر
    به یک کاراکتر» است، پس ایندکس‌های مچ روی متن نرمال‌شده دقیقا روی همین
    رشته ورودی هم معتبرند.

    برای dict خواستن `.to_dict()` و برای ساخت مستقیم رویداد SSE
    `.to_safety_block(request_id)` صدا زده می‌شود.
    """
    if not token:
        return OutputGuardResult(needs_block=False)

    # نرمال‌سازی ارقام تا الگوها فقط با \d کار کنند (طول متن تغییر نمی‌کند)
    normalized = normalize_digits(token)

    for rule in _RULES:
        match = rule.pattern.search(normalized)
        if match:
            return OutputGuardResult(
                needs_block=True,
                matched_pattern=match.group(0),
                rule=rule.name,
                reason=rule.reason,
                message=rule.message,
                action=rule.action,
                span=match.span(),
            )

    return OutputGuardResult(needs_block=False)


def update_buffer(
    buffer: str,
    token: str,
    max_chars: int = SUGGESTED_BUFFER_CHARS,
) -> str:
    """
    تابع کمکی برای نگه داشتن پنجره متحرکِ *متن*.

    فقط پنجره کاراکتری را جلو می‌برد؛ تصمیم اینکه کدام توکن و چه زمانی
    منتشر شود با guarded_token_stream است.
    """
    return (buffer + token)[-max_chars:]


async def guarded_token_stream(
    token_stream: AsyncIterable[str],
    delay_tokens: int = DELAY_WINDOW_TOKENS,
    max_chars: int = SUGGESTED_BUFFER_CHARS,
) -> AsyncGenerator[str, None]:
    """
    استریم توکن‌های مدل را با «پنجره تاخیر» بازنشر می‌کند.

    هیچ توکنی درجا yield نمی‌شود: هر توکن اول وارد صف انتظار می‌شود و تنها
    وقتی بیرون می‌آید که delay_tokens توکن جدیدتر پشت سرش رسیده باشند و بافر
    در آن لحظه هنوز تمیز باشد. با مقدار پیش‌فرض ۳، توکن ۱ همزمان با رسیدن
    توکن ۴ منتشر می‌شود:

        ورودی:  t1 t2 t3 t4 t5 t6
        خروجی:           t1 t2 t3 t4 t5 t6   <- t1 با رسیدن t4

    وقتی الگویی فعال شود، بسته به action خودِ قانون:

      "redact" -> فقط همان بازه مچ‌شده داخل صف با [REDACTED] جایگزین و
                  استریم ادامه پیدا می‌کند. بقیه پیام دست نمی‌خورد.
      "block"  -> صف پاک می‌شود و OutputBlocked بالا می‌آید.

    محدودیتی که باید بدانید: این پنجره فقط تا delay_tokens توکنِ قبل را
    برمی‌گرداند. اگر یک الگو در بازه‌ای طولانی‌تر کامل شود و ابتدایش قبلا
    منتشر شده باشد، دیگر سانسور ممکن نیست و همان‌جا به block تبدیل می‌شود
    (حالت امن). با ۳ توکن، الگوهای دوز فارسی که معمولا در ۲ تا ۴ توکن جا
    می‌شوند پوشش داده می‌شوند.

    نمونه استفاده در لایه API:

        try:
            async for content in guarded_token_stream(stream_chat_completion(messages)):
                yield sse_event({"type": "token", "request_id": rid, "content": content})
        except OutputBlocked as blocked:
            yield sse_event(blocked.result.to_safety_block(rid))
            return
    """
    pending: deque[str] = deque()
    emitted_tail = ""      # آخرین max_chars کاراکترِ منتشرشده

    async for token in token_stream:
        if not token:
            continue

        pending.append(token)

        # یک توکن می‌تواند بیش از یک الگو را کامل کند، پس تا پاک شدن ادامه
        # می‌دهیم. جست‌وجو روی «دنباله منتشرشده + صف انتظار» انجام می‌شود تا
        # الگویی که روی مرز دو توکن نشسته هم دیده شود.
        redactions = 0
        while True:
            pending_text = "".join(pending)
            pending_start = len(emitted_tail)
            result = check_token_stream(emitted_tail + pending_text)
            if not result.needs_block:
                break

            start, end = result.span

            # سه حالتی که سانسور ممکن نیست و باید کل جریان قطع شود:
            #   ۱) سیاست خودِ قانون "block" است (PII)
            #   ۲) مچ از متنی شروع شده که قبلا منتشر شده - پس گرفتنی نیست
            #   ۳) تعداد سانسورها از حد گذشته (حلقه مشکوک)
            if (
                result.action != "redact"
                or start < pending_start
                or redactions >= _MAX_REDACTIONS_PER_TOKEN
            ):
                pending.clear()
                raise OutputBlocked(result)

            # فقط همان بازه مچ‌شده جایگزین می‌شود؛ بقیه پیام دست نمی‌خورد.
            pending = deque([
                pending_text[:start - pending_start]
                + REDACTION_PLACEHOLDER
                + pending_text[end - pending_start:]
            ])
            redactions += 1

        while len(pending) > delay_tokens:
            chunk = pending.popleft()
            emitted_tail = (emitted_tail + chunk)[-max_chars:]
            yield chunk

    # پایان استریم: صف آخرین بار بررسی و پاک شده، پس انتشارش امن است
    while pending:
        chunk = pending.popleft()
        emitted_tail = (emitted_tail + chunk)[-max_chars:]
        yield chunk


if __name__ == "__main__":  # pragma: no cover
    import asyncio

    # نمایش تفاوت «توکن تکی» و «بافر»: هیچ توکنی به تنهایی الگو را فعال
    # نمی‌کند، ولی بافر آن را می‌گیرد.
    demo_tokens = ["برای شما ", "۵۰۰", " می", "لی", "‌گرم", " استامینوفن"]
    buffer = ""
    for chunk in demo_tokens:
        single = check_token_stream(chunk)
        buffer = update_buffer(buffer, chunk)
        buffered = check_token_stream(buffer)
        print(
            f"token={chunk!r:<12} single={single.needs_block!s:<5} "
            f"buffered={buffered.needs_block!s:<5} matched={buffered.matched_pattern}"
        )

    # نمایش پنجره تاخیر: توکن‌های سازنده الگو هرگز منتشر نمی‌شوند.
    async def _demo() -> None:
        async def _fake_stream():
            for chunk in demo_tokens:
                yield chunk

        print("\n--- با پنجره تاخیر ---")
        emitted: list[str] = []
        try:
            async for content in guarded_token_stream(_fake_stream()):
                emitted.append(content)
        except OutputBlocked as blocked:
            print(f"blocked reason={blocked.result.reason} rule={blocked.result.rule}")
        print(f"منتشرشده: {emitted}")

    asyncio.run(_demo())
