# -*- coding: utf-8 -*-
"""نرمال‌سازی سبک متن فارسی (بدون وابستگی به hazm).

ParsBERT روی متن نرمال‌شده آموزش دیده است؛ یکسان‌سازی حروف عربی/فارسی،
حذف اعراب و اصلاح نیم‌فاصله دقت را به‌طور محسوس بالا می‌برد.
"""

import re
import unicodedata

# ی/ک عربی → فارسی، ارقام عربی/فارسی → لاتین
_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ﻻ": "لا", "ۀ": "ه", "ة": "ه",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "\u200c": " ",   # نیم‌فاصله → فاصله (توکنایزر uncased با فاصله بهتر کنار می‌آید)
    "\u200f": "", "\u200e": "", "\ufeff": "",
})

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")          # اعراب و کشیده
_NON_PERSIAN = re.compile(r"[^\u0600-\u06FF0-9a-zA-Z\s\.\،\؟\!\-\+\%]")
_MULTISPACE = re.compile(r"\s+")


def normalize(text: str, keep_zwnj: bool = False) -> str:
    """متن خام کاربر را به شکل استاندارد برای توکنایزر تبدیل می‌کند."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    if keep_zwnj:
        # برای مدل‌های zwnj نیم‌فاصله باید حفظ شود
        text = text.replace("\u200c", "\x00")
    text = text.translate(_CHAR_MAP)
    if keep_zwnj:
        text = text.replace("\x00", "\u200c")
    text = _DIACRITICS.sub("", text)
    text = _NON_PERSIAN.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def setup_console():
    """کنسول ویندوز پیش‌فرض cp1252 است و فارسی را خراب نشان می‌دهد.

    این تابع stdout/stderr را به UTF-8 سوییچ می‌کند تا خروجی سالم چاپ شود.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# ---- انکودینگ فایل‌های CSV ------------------------------------------------
# نوشتن با utf-8-sig یعنی BOM ابتدای فایل قرار می‌گیرد؛ اکسل با دیدن BOM
# فایل را UTF-8 تشخیص می‌دهد و متن فارسی به‌جای Ø§Ø³Øª درست نمایش داده می‌شود.
# خواندن با utf-8-sig هم فایل BOM‌دار و هم بدون BOM را درست می‌خواند.
CSV_WRITE_ENCODING = "utf-8-sig"
CSV_READ_ENCODING = "utf-8-sig"
