"""
app/pipeline/2_triage.py

Two-stage emergency triage classifier:
  1. Rule-based keyword matching (fast, deterministic, catches known danger
     phrases verbatim).
  2. FastText model (app/pipeline/triage_model.bin) as a fallback for cases
     the keyword list doesn't cover -- generalizes beyond exact keywords,
     and (via character n-grams) tolerates typos/misspellings, which matters
     since a patient in distress may not type carefully.

is_emergency(query) -> bool  is the single public entry point.
"""
import os
import re
import time

# ---------------------------------------------------------------------------
# Text normalization -- shared by both stages
# ---------------------------------------------------------------------------
# Handles common real-world input quirks: Arabic vs Persian keyboard
# variants of the same letter, stray whitespace, and repeated punctuation
# used for emphasis when someone is panicking ("نفس نمیکشه!!!!").
_ARABIC_TO_PERSIAN = str.maketrans({
    "ك": "ک", "ي": "ی", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "ؤ": "و",
})


def normalize(text: str) -> str:
    """Best-effort normalization so minor typing variation doesn't cause a
    missed match. Does NOT fix genuine misspellings -- that's the FastText
    stage's job (via character n-grams); this only removes cosmetic noise."""
    text = (text or "").strip()
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = text.replace("\u200c", "")  # remove ZWNJ (نیم‌فاصله): "می‌تونم"
    # and "میتونم" must normalize to the same thing, or a keyword match can
    # silently fail depending on whether the writer happened to type a
    # half-space there.
    text = re.sub(r"\s+", " ", text)              # collapse repeated whitespace
    text = re.sub(r"([!؟?.]){2,}", r"\1", text)     # !!! -> !
    return text


# ---------------------------------------------------------------------------
# Stage 1: rule-based keyword set
# ---------------------------------------------------------------------------
# Deliberately broad and colloquial -- these are phrases/fragments a real
# Persian-speaking patient might actually type, not formal medical terms.
EMERGENCY_KEYWORDS: set[str] = {
    # stroke
    "نصف صورتم", "نصف بدنم بیحس", "زبونم نمیچرخه", "صورتم کج شده",
    "یه طرف بدنم بیحس", "دستم بیحس شد و حرفم قاطی",
    "دست راستم بیحس شده", "دست چپم بیحس شده", "پام یهو بیحس شده",
    # heart attack
    "قفسه سینم فشار", "درد قفسه سینه", "سینم داره میسوزه",
    "درد سینه با عرق سرد", "سینم فشار میاره",
    # poisoning / overdose
    "مسموم شدم", "قرص خورده", "وایتکس خورد", "سم خورده", "سم خوردم", "مواد شوینده رو خورده",
    "الکل صنعتی خوردم", "قارچ جنگلی خوردیم",
    "قرص زیادی خوردم", "قرص خواب زیادی خوردم", "زیادی قرص خوردم", "اشتباهی قرص زیاد خوردم",
    "دارو زیادی خوردم", "اوردوز کردم",
    # NOTE: «تیر میکشه» تنها حذف شد: دندان‌درد/زانو/کمر هم «تیر می‌کشد» و همه را
    # به ۱۱۵ می‌فرستاد. فقط ترکیب با ناحیهٔ قلبی-تنفسی نگه داشته شد (تأیید پزشک لازم).
    "سینم تیر میکشه", "قفسه سینم تیر میکشه", "تیر میکشه تو بازوی چپ",
    "تیر میکشه تو فکم", "تیر میکشه به بازوم", "دردش میزنه به بازو و فک",
    # bleeding
    "خونریزی شدید", "خون بند نمیاد", "خون زیاد میره", "استفراغ خونی",
    "خون بالا میارم", "خونریزی از واژن", "خون زیادی از دست داد",
    "خون توی مدفوع", "خون در مدفوع", "مدفوعم خونی", "خون زیادی توی مدفوع",
    "مدفوعم خون", "توی مدفوعم خون", "مدفوع خونی", "خون تو مدفوعم",
    # anaphylaxis
    "نفسم گرفت بعد خوردن", "گلوم داره میبنده", "صورتم داره ورم میکنه",
    "نفس نمیتونم بکشم بعد نیش",
    # seizure
    "داره تشنج میکنه", "بدنش داره میلرزه شدید", "کف کرده دهنش",
    "چشماش رفته بالا",
    # unconsciousness
    "بیهوش شد", "بیدار نمیشه", "جواب نمیده اصلا", "از حال رفت", "به هوش نمیاد",
    # burns
    "پوستش داره میاد", "سوختگی عمیق", "تاول زده بزرگ", "اسید ریخت رو دستم",
    # suicide/overdose
    "میخوام بمیرم", "قرص زیادی خورده", "رگشو زده", "اقدام به خودکشی",
    "میخواد خودشو بکشه", "نامه خداحافظی",
    # choking
    "نفس نمیکشه", "گیر کرده تو گلوش", "رنگش کبود شده", "خفه میشه", "گیر کرده تو گلوم",
    # trauma
    "تصادف کردیم", "استخونم زده بیرون", "زیر ماشین", "از بلندی افتادم",
    # diabetic emergency
    "قندم خیلی افتاده", "قند خونم زیر", "افت شدید قند",
    # acute abdomen
    "شکمم مثل چاقو", "شکمم سفت شده", "درد شکم غیرقابل تحمل",
    # heat/cold emergency
    "گرمازده شده", "تنش داغه ولی عرق نمیکنه", "سرما زده شدیم",
    # drowning
    "زیر آب رفته", "غرق شده", "نفس نمیکشه بعد آب",
    # electrocution
    "برق گرفتش", "برق گرفت",
    # severe asthma
    "نفس تنگی شدید", "لباش کبوده", "خس خس شدید",
    # infant emergency
    "گردنش سفت شده", "لکه‌های بنفش", "فونتانلش برجسته",
    # chemical eye injury
    "اسید پاشید تو چشمم", "پاشیده تو چشمم",
    # bites
    "مار نیش زد", "مار گزیده", "نیش عقرب خورده",
    # generic "can't breathe" -- extremely common phrasing, must be its own
    # entry rather than relying on the choking/asthma/bite-specific ones
    "نفس نمیتونم بکشم", "نمیتونم نفس بکشم", "نفسم بالا نمیاد",
}

# Keywords are normalized once at import time (same normalize() used on
# user input) so a half-space (ZWNJ) present in only one side -- e.g. a
# keyword written "لکه‌های بنفش" vs. user input "لکه های بنفش" -- can't
# cause a silent match failure in either direction.
_NORMALIZED_KEYWORDS = {normalize(kw) for kw in EMERGENCY_KEYWORDS}

# A second, more permissive comparison with ALL whitespace removed too.
# Persian has three common ways to write a compound like "نمی‌تونم" --
# with a ZWNJ, with a plain space, or with nothing at all -- and a patient
# in distress won't be consistent about which one they use. Comparing the
# space-stripped forms as a fallback catches all three without needing a
# separate keyword entry for every spacing variant of every phrase.
_COMPACT_KEYWORDS = {kw.replace(" ", "") for kw in _NORMALIZED_KEYWORDS}

# Words that negate a preceding symptom phrase ("درد قفسه سینه ندارم" =
# "I do NOT have chest pain" -- the opposite of an emergency). Checked in a
# short window right after a keyword match; if found, that particular match
# is discarded rather than counted as a positive.
_NEGATION_WORDS = ("ندارم", "نداره", "نداشتم", "نبود", "نیست", "نداشت")
_NEGATION_WINDOW = 12  # chars to look ahead after a keyword match


def _has_nearby_negation(haystack: str, match_end: int) -> bool:
    window = haystack[match_end:match_end + _NEGATION_WINDOW]
    return any(neg in window for neg in _NEGATION_WORDS)


# ---------------------------------------------------------------------------
# Optional third layer: hazm-based clause-aware negation (conditional --
# only called when a keyword already matched, see rule_based_check below)
# ---------------------------------------------------------------------------
_hazm_negation_module = None
_hazm_negation_load_attempted = False


def _get_hazm_negation_module():
    global _hazm_negation_module, _hazm_negation_load_attempted
    if _hazm_negation_load_attempted:
        return _hazm_negation_module
    _hazm_negation_load_attempted = True
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hazm_negation.py")
        spec = importlib.util.spec_from_file_location("hazm_negation", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if mod.hazm_available():
            _hazm_negation_module = mod
    except Exception:
        _hazm_negation_module = None
    return _hazm_negation_module


def _hazm_says_negated(normalized_text: str, keyword_char_index: int, keyword_text: str) -> bool:
    """Fails safe: if hazm isn't set up, always returns False (i.e. doesn't
    override the window-based check's decision)."""
    mod = _get_hazm_negation_module()
    if mod is None:
        return False
    try:
        return mod.is_negated_at(normalized_text, keyword_char_index, keyword_text=keyword_text)
    except Exception:
        return False


def rule_based_check(text: str) -> bool:
    """True اگر یکی از عبارت‌های خطر در متن باشد و *نفی‌نشده* باشد.

    تغییرات نسبت به نسخهٔ قبل:
      - همهٔ occurrenceها بررسی می‌شوند (قبلاً فقط اولین find؛ اگر اولین نفی‌شده
        بود و دومی واقعی، اورژانس گم می‌شد).
      - نگیشن فقط با hazm_negation.is_negated_at (اولین واژهٔ معنی‌دار بعد از
        کلیدواژه / نفی بلافاصله قبلش) تعیین می‌شود؛ پنجرهٔ ۱۲ کاراکتری فقط
        وقتی استفاده می‌شود که ماژول نگیشن بالا نیامده باشد.
      - مسیر «بدون فاصله» فقط برای کلیدواژه‌هایی است که در متن عادی اصلاً پیدا
        نشده‌اند (قبلاً یک نگیشن‌شدهٔ عادی از این مسیر دوباره پرچم می‌خورد).
    """
    normalized = normalize(text)
    neg = _get_hazm_negation_module()

    def negated(hay: str, idx: int, kw: str) -> bool:
        if neg is not None:
            return neg.is_negated_at(hay, idx, kw)
        return _has_nearby_negation(hay, idx + len(kw))

    compact = normalized.replace(" ", "")
    for kw in _NORMALIZED_KEYWORDS:
        found_plain = False
        start = 0
        while True:
            idx = normalized.find(kw, start)
            if idx == -1:
                break
            found_plain = True
            if not negated(normalized, idx, kw):
                return True
            start = idx + len(kw)
        if found_plain:
            continue
        ckw = kw.replace(" ", "")
        start = 0
        while True:
            idx = compact.find(ckw, start)
            if idx == -1:
                break
            end = idx + len(ckw)
            neg_c = neg.is_negated_compact(compact, end) if neg is not None else _has_nearby_negation(compact, end)
            if not neg_c:
                return True
            start = end
    return False


# ---------------------------------------------------------------------------
# Stage 2: FastText model (lazy-loaded)
# ---------------------------------------------------------------------------
_MODEL_PATH = os.environ.get(
    "TRIAGE_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "triage_model.bin"),
)
# آستانهٔ احتمال critical. 0.5 = argmax. برای ایمنی پایین‌تر است؛ مقدار نهایی را با
# evaluate روی val گروهی (کلیدواژه‌های دیده‌نشده) تعیین کنید، نه حدسی.
def _load_threshold() -> float:
    """اولویت: متغیر TRIAGE_FT_THRESHOLD ← فایل <model>.threshold.json (کنار مدل) ← 0.30."""
    env = os.environ.get("TRIAGE_FT_THRESHOLD")
    if env:
        return float(env)
    try:
        import json
        with open(_MODEL_PATH + ".threshold.json", encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return 0.30


FT_THRESHOLD = _load_threshold()
_model = None  # loaded on first use, not at import time


def _patch_fasttext_numpy2(ft_module) -> None:
    """fasttext.predict با numpy>=2 خطا می‌دهد (np.array(copy=False)). این پچ
    قبلاً فقط در اسکریپت‌های آموزش بود و مسیر production را نمی‌پوشاند."""
    import numpy as np
    orig = np.array

    def _compat(obj, copy=False, **kw):
        return orig(obj, copy=copy, **kw) if copy else np.asarray(obj, **kw)

    ft_module.np.array = _compat


def _get_model():
    """Load the FastText model once. None اگر فایل نباشد."""
    global _model
    if _model is None and os.path.exists(_MODEL_PATH):
        import fasttext  # lazy
        import fasttext.FastText as _ftm
        _patch_fasttext_numpy2(_ftm)
        _model = fasttext.load_model(_MODEL_PATH)
    return _model


def fasttext_check(text: str, threshold: float | None = None) -> bool:
    """True اگر P(critical) >= threshold. هر خطای مدل -> False (لایهٔ قانون قبلاً
    اجرا شده)؛ ولی خطا لاگ می‌شود تا بی‌صدا نماند."""
    model = _get_model()
    if model is None:
        return False
    thr = FT_THRESHOLD if threshold is None else threshold
    try:
        normalized = normalize(text).replace("\n", " ")
        labels, probs = model.predict(normalized, k=2)
        p = {l: float(x) for l, x in zip(labels, probs)}
        return p.get("__label__critical", 0.0) >= thr
    except Exception:
        import logging
        logging.getLogger(__name__).exception("fasttext_predict_failed")
        return False


# ---------------------------------------------------------------------------
# Public entry point: combined two-stage check
# ---------------------------------------------------------------------------
def is_emergency(query: str) -> bool:
    """
    Two-stage emergency detection:
      1. Rule-based keyword check (fast, exact-phrase matches).
      2. If stage 1 didn't flag it, fall back to the FastText model.

    Must run in well under 50ms per call (measured in tests/test_triage.py).
    """
    if rule_based_check(query):
        return True
    return fasttext_check(query)


def warmup():
    """
    Loads BOTH the fasttext model and the hazm negation module once,
    outside of any timed measurement. Call this once when the server
    starts (or at the beginning of a test run) -- otherwise the first
    real request that happens to need either model pays the one-time
    loading cost (fasttext: reads from disk; hazm: downloads from
    Hugging Face Hub the very first time, which can take a couple of
    seconds). Without this, that one slow first call can dominate a
    worst-case/average latency measurement even though it's a one-time
    cost that a real server only ever pays once at startup.
    """
    _get_model()
    mod = _get_hazm_negation_module()
    if mod is not None:
        try:
            mod.clause_has_negated_verb("این یک جمله‌ی تست برای گرم کردن مدل است")
        except Exception:
            pass


if __name__ == "__main__":
    examples = [
        "نفس نمی‌تونم بکشم",
        "چشمم درد می‌کنه",
        "چند تا جوش رو صورتم زده",
    ]
    for ex in examples:
        start = time.perf_counter()
        result = is_emergency(ex)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{ex!r} -> emergency={result} ({elapsed_ms:.2f} ms)")
