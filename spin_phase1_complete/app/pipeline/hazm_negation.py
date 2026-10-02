# -*- coding: utf-8 -*-
"""
app/pipeline/hazm_negation.py   (نسخهٔ اصلاح‌شده)

تشخیص «نگیشن» برای یک کلیدواژهٔ اورژانسی که قبلاً در متن پیدا شده است.

چرا بازنویسی شد؟
-----------------
نسخهٔ قبلی می‌پرسید «آیا *هر* فعل منفی در کلاز هست؟». این سؤال غلط است:
در «دیگه طاقت ندارم نفس نمی‌تونم بکشم» فعل «ندارم» منفی است ولی به کلیدواژهٔ
اورژانسی ربطی ندارد، و نسخهٔ قدیمی اورژانس واقعی را سرکوب می‌کرد.
در triage.py هم پنجرهٔ ثابت ۱۲ کاراکتری همین خطا را داشت.

قاعدهٔ جدید (قطعی، بدون مدل آماری، بدون دانلود):
  یک کلیدواژه فقط وقتی «نفی‌شده» است که
    (الف) اولین واژهٔ معنی‌دار بعد از آن (با عبور از چند واژهٔ پرکننده مثل
          «که/اصلا/الان/دیگه») یکی از افعال منفی بسته‌ی NEG_AFTER باشد
          («درد قفسه سینه ندارم»، «تشنج که نداره»، «بیهوش شد نبود»)، یا
    (ب) درست قبل از آن «نه/بدون/هیچ» بیاید («بدون تشنج»، «نه، تشنج نیست»).
  همهٔ این‌ها فقط داخل *همان کلاز* بررسی می‌شود (کلاز با ، ؛ . ! ؟ و
  ولی/اما/بلکه جدا می‌شود).
  پیشوند «نمی» عمداً در لیست نیست: «نمی‌تونم/نمی‌کشه» خودش علامت اورژانس است.

hazm دیگر لازم نیست؛ hazm_available() برای سازگاری با triage.py همیشه True است.

حد شناخته‌شده: جمله‌های پیچیده (نگیشن دوردست، طعنه، «نفس نمی‌کشه؟ نه، نفس می‌کشه»)
را تشخیص نمی‌دهد. در این موارد عمداً «نفی‌نشده» برمی‌گردد (جهت ایمن).
"""
from __future__ import annotations

import re

# --- افعال/واژه‌های نفی (فقط شکل کامل واژه، نه پیشوند) ----------------------
NEG_AFTER = frozenset({
    "ندارم", "نداره", "ندارد", "ندارن", "نداریم", "نداشتم", "نداشت", "نداشته",
    "نبود", "نبودم", "نبوده", "نیست", "نیستم", "نیستن", "نشد", "نشده", "نشدم",
    "نکرد", "نکرده", "نه",
})
NEG_BEFORE_1 = frozenset({"نه", "نخیر"})            # فقط واژهٔ بلافاصله قبل
NEG_BEFORE_2 = frozenset({"بدون", "هیچ", "بی"})      # در دو واژهٔ قبل

# واژه‌های پرکننده‌ای که می‌توانند بین کلیدواژه و فعل منفی بیایند
FILLERS = frozenset({
    "که", "اصلا", "اصلاً", "هم", "الان", "دیگه", "هیچ", "خوشبختانه", "خدا", "رو",
    "شکر", "فعلا", "الحمدلله", "اصلن", "تا", "حالا", "هنوز",
})
_MAX_FILLERS = 3

# گذشته + بهبود (برای وضعیت «resolved_past»؛ سرکوب نمی‌کند، فقط برچسب می‌دهد)
_PAST_MARKERS = ("قبلا", "قبلاً", "دیشب", "پارسال", "پریشب", "دیروز", "سال پیش", "ماه پیش")
_RESOLVED_MARKERS = ("الان خوبم", "حالم خوبه", "حالم خوب", "خوب شدم", "بهتر شدم",
                     "بهترم", "امروز خوبم", "امروز حالم خوبه", "خوبم")

_BOUNDARY = re.compile(r"[،,؛;.!؟?\n]|(?<!\w)(?:ولی|اما|بلکه|ولیکن)(?!\w)")


def hazm_available() -> bool:
    """سازگاری با triage.py: این ماژول دیگر به hazm نیاز ندارد، پس همیشه آماده است."""
    return True


def _clause_bounds(text: str, kw_start: int, kw_end: int):
    """(clause_start, clause_end, prev_clause_text) برای کلاز حاوی کلیدواژه."""
    c_start, c_end, prev_start = 0, len(text), 0
    for m in _BOUNDARY.finditer(text):
        if m.end() <= kw_start:
            prev_start, c_start = c_start, m.end()
        elif m.start() >= kw_end:
            c_end = m.start()
            break
    return c_start, c_end, text[prev_start:c_start]


def _first_content_token(tokens: list[str]) -> str | None:
    skipped = 0
    for tok in tokens:
        tok = tok.strip("-–—\"'«»()")
        if not tok:
            continue
        if tok in FILLERS and skipped < _MAX_FILLERS:
            skipped += 1
            continue
        return tok
    return None


def negation_status(text: str, kw_start: int, kw_end: int) -> str:
    """'negated' | 'resolved_past' | 'affirmed' برای کلیدواژه در text[kw_start:kw_end]."""
    c_start, c_end, prev_clause = _clause_bounds(text, kw_start, kw_end)
    before = text[c_start:kw_start].split()
    after = text[kw_end:c_end].split()

    # (ب) نفی قبل از کلیدواژه
    if before and before[-1] in NEG_BEFORE_1:
        return "negated"
    if any(t in NEG_BEFORE_2 for t in before[-2:]):
        return "negated"
    if not before and prev_clause.strip() in NEG_BEFORE_1:      # «نه، تشنج نیست»
        return "negated"

    # (الف) نفی بعد از کلیدواژه
    if _first_content_token(after) in NEG_AFTER:
        return "negated"

    # گذشته + بهبودیافته (فقط برچسب)
    if any(p in " ".join(before) for p in _PAST_MARKERS) or any(p in text[:kw_start] for p in _PAST_MARKERS):
        if any(r in text[c_end:] for r in _RESOLVED_MARKERS):
            return "resolved_past"
    return "affirmed"


def is_negated_at(full_text: str, keyword_char_index: int, keyword_text: str = None) -> bool:
    """امضای سازگار با نسخهٔ قبلی (triage.py همین را صدا می‌زند)."""
    end = keyword_char_index + len(keyword_text or "")
    return negation_status(full_text, keyword_char_index, end) == "negated"


def is_negated_compact(compact_text: str, kw_end: int) -> bool:
    """برای مسیر «متن بدون فاصله» در triage: فقط نفی بلافاصله بعد از کلیدواژه."""
    rest = compact_text[kw_end:]
    for w in sorted(NEG_AFTER - {"نه"}, key=len, reverse=True):
        if rest.startswith(w):
            return True
    for f in FILLERS:                       # «که»، «اصلا» چسبیده
        if rest.startswith(f):
            rest2 = rest[len(f):]
            if any(rest2.startswith(w) for w in NEG_AFTER - {"نه"}):
                return True
    return False


# سازگاری با فراخوانی‌های قدیمی
def clause_has_negated_verb(clause_text: str) -> bool:      # pragma: no cover
    return _first_content_token(clause_text.split()) in NEG_AFTER


if __name__ == "__main__":
    cases = [
        ("درد قفسه سینه ندارم فقط خسته‌م", "درد قفسه سینه", True),
        ("درد نداره ولی نفس نمیتونم بکشم", "نفس نمیتونم بکشم", False),
        ("دیگه طاقت ندارم نفس نمیتونم بکشم", "نفس نمیتونم بکشم", False),
        ("نفس نمیتونم بکشم طاقت ندارم", "نفس نمیتونم بکشم", False),
        ("داره تشنج میکنه که نداره فقط لرزه", "داره تشنج میکنه", True),
        ("نه، بیهوش شد نیست", "بیهوش شد", True),
        ("بدون خونریزی شدید اومدم", "خونریزی شدید", True),
        ("دیشب خون بالا میارم ولی امروز حالم خوبه", "خون بالا میارم", False),
    ]
    for t, k, want in cases:
        i = t.find(k)
        got = is_negated_at(t, i, k)
        print("OK " if got == want else "BUG", t, "->", got, negation_status(t, i, i + len(k)))
