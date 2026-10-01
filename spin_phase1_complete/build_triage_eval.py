"""
build_triage_eval.py

ساخت سه مجموعهٔ ارزیابی که train_fasttext.py هرگز نمی‌بیند:

  eval/test.jsonl              — hold-out با قالب/غلط‌املایی/لهجه متفاوت از train
  eval/golden_emergency.jsonl  — حدود ۳۰۰ نمونه شامل موارد آسان و مرزی
  eval/hard_negatives.jsonl    — نگیشن / زمان گذشته / شرطی
                                 قانون: هر سطر باید عبارتی داشته باشد که
                                 بدون نگیشن/گذشته خودش در EMERGENCY_KEYWORDS
                                 بحرانی است.

اجرا (از ریشه spin_phase1_complete):
    python build_triage_eval.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _load_triage_module():
    spec = importlib.util.spec_from_file_location("triage", _ROOT / "app" / "pipeline" / "triage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


triage = _load_triage_module()
KEYWORDS = sorted(triage.EMERGENCY_KEYWORDS)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _uniq(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r["text"].strip(), r["urgency"])
        if key[0] and key not in seen:
            seen.add(key)
            out.append({"text": key[0], "urgency": r["urgency"]})
    return out


# قالب‌هایی که در prepare_triage_data.py نیستند تا نشت سبک به test کم شود
TEST_CRITICAL_TPL = [
    "الان دیگه {kw}، زود جواب بدید",
    "تو تاکسی‌ام {kw}",
    "همسایه‌مون {kw} چیکار کنیم",
    "{kw} از سمت چپ شروع شد",
    "آقا {kw} نگید صبر کنید",
    "بچه سه ساله‌م {kw}",
    "نصف شب {kw} بیدار شدم",
    "{kw} بعد از غذا",
    "دیگه طاقت ندارم {kw}",
    "لطفا اورژانس {kw}",
]

TEST_NONCRIT_ROUTINE = [
    "چند روزه گلودرد خفیف دارم بهتر میشه کم کم",
    "دندونم یه کم حساس شده به سرما",
    "زانوم بعد پیاده روی درد میگیره",
    "میخوام برای سرماخوردگی نسخه بگیرم",
    "تب خفیف عصرها دارم بدون لرز شدید",
    "پسرم کلمات رو کمی اشتباه تلفظ میکنه",
    "قاعدگیم دو روز عقب افتاده نگرانم",
    "بعد از ورزش کمرم گرفته",
    "برای چکاپ سالانه نوبت میخوام",
    "سرفه خشک دارم از دیروز بدون تنگی نفس",
    "جوش زدم رو صورتم چیکار کنم",
    "خستگی بعد از شیفت شب طبیعیه؟",
    "گوشم کیپ شده بعد شنا",
    "اسهال خفیف از غذای بیرون",
    "میخوام واکسن آنفولانزا بزنم",
    "عقربه‌های فشارم کمی بالاس بدون علامت",
    "خارش پوست بعد شوینده جدید",
    "درد عضلانی بعد باشگاه",
    "بینی ام گرفته سرما خوردم",
    "تهوع خفیف بعد سفر",
    "لکه بینی خیلی کم بین دو دوره",
    "لثه ام موقع مسواک کمی خون میاد",
    "صدا گرفتم بعد داد زدن تو کنسرت",
    "میخوام آزمایش خون روتین بدم",
    "پاشنه پا تیر میکشه وقتی صبح پا میشم",
    "نوبت تمدید دارو فشار خون",
    "سردرد بعد کم خوابی دیشب",
    "نفخ بعد غذای چرب",
    "گزگز انگشت بعد کار طولانی با موس",
    "گلوم کمی خراش داره از هوای خشک",
    "میخوام برای کم‌خونی آزمایش تجویز کنید",
    "خار پاشنه صبح‌ها وقتی پا میشم",
    "سرفه شبانه خفیف بعد برگشتن اسید معده",
    "دندون عقل بدون درد فقط گیر غذایی",
    "لکنت خفیف وقتی عجله دارم ارائه بدم",
    "نوبت پاپ اسمیر سالانه میخوام",
    "توده چربی نرم کوچک روی بازو درد نداره",
    "خواب‌آلودگی بعد ناهار سنگینه",
    "گوشم بعد هدفون وزوز کوتاه داره",
    "میخوام نسخه سرماخوردگی تمدید کنم",
    "درد گردن از بالش بد بدون بی‌حسی دست",
    "آفت دهان کوچیک چند روزه",
    "حساسیت فصلی عطسه و چشم خارش",
    "میخوام چکاپ قند ناشتا بدم",
]


def _typo(text: str, rng: random.Random) -> str:
    swaps = [
        ("می", "مي"),
        ("ک", "ك"),
        ("ی", "ي"),
        ("تشنج", "تشنچ"),
        ("بیهوش", "بیحوش"),
        ("نفس", "نفص"),
        ("  ", " "),
    ]
    out = text
    src, dst = rng.choice(swaps)
    if src in out:
        out = out.replace(src, dst, 1)
    # جابه‌جایی فاصله در یک نقطه
    if rng.random() < 0.4 and " " in out:
        parts = out.split(" ")
        i = rng.randrange(len(parts) - 1)
        parts[i] = parts[i] + parts[i + 1]
        del parts[i + 1]
        out = " ".join(parts)
    return out


def build_test(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    kws = list(KEYWORDS)
    rng.shuffle(kws)
    for i, kw in enumerate(kws):
        tpl = TEST_CRITICAL_TPL[i % len(TEST_CRITICAL_TPL)]
        text = tpl.format(kw=kw)
        rows.append({"text": text, "urgency": "critical"})
        if i % 2 == 0:
            rows.append({"text": _typo(text, rng), "urgency": "critical"})
    for t in TEST_NONCRIT_ROUTINE:
        rows.append({"text": t, "urgency": "low"})
        rows.append({"text": _typo(t, rng), "urgency": "low"})
    rng.shuffle(rows)
    return _uniq(rows)


def build_golden(rng: random.Random, test_rows: list[dict]) -> list[dict]:
    """~۳۰۰ نمونه: آسان + غلط‌املا + لهجه + چند hard non-critical."""
    rows: list[dict] = []
    # آسان critical با ترتیب محاوره‌ای
    colloq = [
        "داغون شدم {kw}",
        "والا {kw} بگید چی کار کنم",
        "حاجی {kw} جدیه؟",
        "{kw} دیگه وایسادم تو اورژانس",
    ]
    for i, kw in enumerate(KEYWORDS):
        rows.append({"text": colloq[i % len(colloq)].format(kw=kw), "urgency": "critical"})
    # نمونه‌های مرزی critical که کلیدواژه کامل دارند ولی جمله شلوغ است
    busy = [
        "از صبح حالت تهوع داشتم ولی الان {kw} کمک کنید",
        "فکر کردم سرماخوردگیه اما {kw}",
        "بیدار شدم دیدم {kw}",
    ]
    for kw in KEYWORDS[::2]:
        rows.append({"text": busy[hash(kw) % len(busy)].format(kw=kw), "urgency": "critical"})

    # non-critical آسان (تکرار با بازنویسی)
    easy_low = [
        "گلودرد معمولی دارم و کمی عطسه",
        "دندون عقلم یه کم فشار میاره بدون تورم صورت",
        "کمرم بعد بلند کردن جعبه درد گرفته",
        "میخوام برای سرفه خشک شربت بگیرم",
        "پسرم کمی لکنت داره وقتی استرس داره",
        "پریودم نامنظمه چند ماهه",
        "خال گوشتی کوچیک دارم روی دستم",
        "خستگی بعد امتحان طبیعی؟",
        "برای گواهی سلامت مدرسه اومدم",
        "گوش درد خفیف بعد پرواز",
        "جوش چرکی کوچیک رو صورتم",
        "درد عضله ساق بعد دویدن",
        "میخوام ویتامین D چک کنم",
        "عطسه و آبریزش فصلی",
        "یبوست گاه‌گاهی بدون خون",
        "خوابم کمه سرم سنگین شده",
        "ناخن فرو رفته تو گوشت پا",
        "حساسیت خفیف به گرد و خاک",
        "درد پشت گردن از موبایل زیاد",
        "میخوام نوبت چکاپ تیروئید بگیرم",
    ]
    for t in easy_low:
        rows.append({"text": t, "urgency": "low"})

    # hard non-critical: گذشته / نگیشن روی کلیدواژه واقعی
    for item in build_hard_negatives():
        rows.append(item)

    # کمی از test را هم مخلوط نکن — golden باید مستقل بماند
    rng.shuffle(rows)
    rows = _uniq(rows)
    # نزدیک ۳۰۰ تا نگه داریم (اول criticalها حفظ شوند)
    if len(rows) > 320:
        crit = [r for r in rows if r["urgency"] == "critical"]
        low = [r for r in rows if r["urgency"] != "critical"]
        need_low = max(0, 300 - len(crit))
        rows = crit + low[:need_low]
        rng.shuffle(rows)
    return rows


def build_hard_negatives() -> list[dict]:
    """هر متن شامل حداقل یک کلیدواژهٔ اورژانس است؛ برچسب همه non_critical.

    urgency در فایل‌های پایپلاین: critical | non_critical
    (evaluate_triage.py برای کلاس منفی «هر چیز غیر critical» را می‌پذیرد؛
    این‌جا صریحاً non_critical می‌نویسیم تا با evaluate_triage_pipeline یکی باشد.)
    """
    tpls = [
        "{kw} ندارم",
        "نه، {kw} نیست",
        "قبلا {kw} ولی الان خوبم",
        "دیشب {kw} ولی امروز حالم خوبه",
        "اگر {kw} باید چیکار کنم؟ الان که نیست",
        "مادرم {kw} نبود، فقط خوابش برده بود",
    ]
    rows: list[dict] = []
    for kw in KEYWORDS:
        for tpl in tpls:
            try:
                text = tpl.format(kw=kw)
            except Exception:
                text = tpl.replace("{kw}", kw)
            if kw not in text:
                continue
            rows.append({"text": text, "urgency": "non_critical"})
    # چند جملهٔ دستی با کلیدواژه کامل + نگیشن بند/گذشته محاوره‌ای
    handmade = [
        "درد قفسه سینه ندارم، فقط ترسم زیاده از فیلم",
        "نفس نمیتونم بکشم رو برای دیشب میگفتم، الان راحت نفس میکشم",
        "داره تشنج میکنه نیست، لرز سرماخوردگیه",
        "بیهوش شد نبود، فقط چرت زده بود روی مبل",
        "خون بالا میارم که ندارم، فقط آروغ ترشه",
        "میخوام بمیرم رو جدی نمیگم، فقط خسته‌ام از کار",
        "قرص زیادی خوردم نه، طبق نسخه خوردم",
        "نفس نمیکشه؟ نه نفس میکشه، رنگش هم خوبه",
        "استفراغ خونی نداره، استفراغ غذای فاسده",
        "گلوم داره میبنده نیست، فقط از سرما کمی گرفته",
        "قندم خیلی افتاده نبود، عددش ۹۲ بود",
        "تصادف کردیم ولی کسی بیهوش شد نبود، همه راه رفتن",
        "از بلندی افتادم رو شوخی بود، از پله یک پله اومدم پایین",
        "رگشو زده نیست، فقط خراش کوچیکه",
        "خونریزی شدید ندارم، لکه بینی معمولیه",
        "صورتم کج شده نیست، خواب روی دستم بوده",
        "زبونم نمیچرخه نه، فقط تند حرف میزنم",
        "شکمم مثل چاقو نیست، نفخ بعد حبوبات",
        "رنگش کبود شده نبود، سرما خورده بود لبش",
        "چشماش رفته بالا نه، فقط چرت میزد",
        "گیر کرده تو گلوش نیست، آب رفته بود تو گلوش و سرفه کرد",
        "اوردوز کردم رو دروغ گفتم برای جلب توجه، قرصی نخوردم",
        "سم خوردم نه، فقط داروی گیاهی تلخ خوردم",
        "غرق شده نبود، فقط آب تو بینی‌ش رفته بود",
        "برق گرفت ولی الان گزگز خفیفه و برق گرفت جدی نبود",
    ]
    for t in handmade:
        # تضمین: حداقل یک کلیدواژهٔ واقعی در متن باشد
        if any(kw in t for kw in KEYWORDS):
            rows.append({"text": t, "urgency": "non_critical"})
    return _uniq(rows)


def main() -> None:
    rng = random.Random(42)
    test_rows = build_test(rng)
    hard_rows = build_hard_negatives()
    golden_rows = build_golden(random.Random(42), test_rows)

    out = _ROOT / "eval"
    _write_jsonl(out / "test.jsonl", test_rows)
    _write_jsonl(out / "golden_emergency.jsonl", golden_rows)
    _write_jsonl(out / "hard_negatives.jsonl", hard_rows)

    def counts(rows):
        c = sum(1 for r in rows if r["urgency"] == "critical")
        return len(rows), c, len(rows) - c

    tn, tc, tl = counts(test_rows)
    gn, gc, gl = counts(golden_rows)
    hn, hc, hl = counts(hard_rows)
    print(f"test.jsonl: {tn}  (critical={tc}, other={tl})")
    print(f"golden_emergency.jsonl: {gn}  (critical={gc}, other={gl})")
    print(f"hard_negatives.jsonl: {hn}  (critical={hc}, non_critical={hl})")
    print("هر سطر hard_negatives شامل حداقل یک EMERGENCY_KEYWORDS است.")
    print(f"نوشته شد در: {out}")


if __name__ == "__main__":
    main()
