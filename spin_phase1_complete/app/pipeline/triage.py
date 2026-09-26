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
    # radiating/shooting pain phrase (explicitly given as an example in the task)
    "تیر میکشه", "تیر می‌کشه",
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
    """True if any known danger phrase is a substring of the (normalized) text,
    checked both with normal spacing and with all whitespace stripped (to
    tolerate ZWNJ/space/no-separator variation in compound words). A match
    immediately followed by a negation word ("ندارم", "نیست", ...) is
    discarded, so "درد قفسه سینه ندارم" isn't flagged as an emergency.

    If a keyword matches (the rare path -- most sentences won't match at
    all), and hazm is available, we ALSO run a clause-aware negation check
    with hazm (see hazm_negation.py) as a second opinion -- this is the
    "conditional pipeline": the heavier hazm check only runs when a
    keyword was already found, so the common case (no match) stays fast.
    """
    normalized = normalize(text)
    for kw in _NORMALIZED_KEYWORDS:
        idx = normalized.find(kw)
        if idx != -1 and not _has_nearby_negation(normalized, idx + len(kw)):
            if _hazm_says_negated(normalized, idx, kw):
                continue
            return True
    compact = normalized.replace(" ", "")
    for kw in _COMPACT_KEYWORDS:
        idx = compact.find(kw)
        if idx != -1 and not _has_nearby_negation(compact, idx + len(kw)):
            return True
    return False


# ---------------------------------------------------------------------------
# Stage 2: FastText model (lazy-loaded)
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triage_model.bin")
_model = None  # loaded on first use, not at import time


def _get_model():
    """Load the FastText model once and cache it. Returns None if the model
    file isn't present yet (e.g. before training has been run)."""
    global _model
    if _model is None and os.path.exists(_MODEL_PATH):
        import fasttext  # imported lazily so rule-only usage doesn't need it installed
        _model = fasttext.load_model(_MODEL_PATH)
    return _model


def fasttext_check(text: str) -> bool:
    """True if the FastText model predicts the 'critical' label for this text."""
    model = _get_model()
    if model is None:
        return False  # no model available -- fail safe to "not flagged by this stage"
    normalized = normalize(text).replace("\n", " ")
    labels, probs = model.predict(normalized, k=1)
    return labels[0] == "__label__critical"


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
