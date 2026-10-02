"""
prepare_triage_data.py

ساخت eval/train.jsonl و eval/val.jsonl - ورودی موردنیاز train_fasttext.py
(مدل تریاژ اورژانس، لایه دوم fallback بعد از قوانین کلیدواژه‌ای).

دو منبع، هر دو بدون نیاز به برچسب‌گذاری دستی جدید:

  ۱) critical: از خودِ EMERGENCY_KEYWORDS در app/pipeline/triage.py -
     همان عبارت‌هایی که لایه اول (قانون‌محور) از قبل به‌عنوان اورژانس
     می‌شناسد، هرکدام در چند قالب جمله واقعی‌تر بیمار (نه عبارت خام تنها)
     تا fastText روی جمله یاد بگیرد، نه فقط تطبیق دقیق رشته.

  ۲) non_critical: (الف) نسخه منفی‌شده همان کلیدواژه‌ها ("تشنج نمیکنه") تا
     مرز بحرانی/غیربحرانی برای مدل تیزتر شود، و (ب) سوالات معمولی پزشکی از
     ParsBert/data/*.csv (که prepare_data.py آن‌جا ساخته) به‌عنوان نمونه‌های
     واقعی غیر-اورژانسی روزمره.

نکته مهم درباره صحت: هیچ معیار پزشکی جدیدی اینجا اختراع نشده - critical
دقیقا همان چیزی است که تیم از قبل در EMERGENCY_KEYWORDS به‌عنوان اورژانس
تایید کرده؛ این اسکریپت فقط همان‌ها را برای آموزش یک مدل آماری بازنویسی/
تکثیر می‌کند، نه اینکه فهرست پزشکی جدیدی بسازد.

اجرا:
    python ParsBert/prepare_data.py      # اگر قبلا نزده‌اید (منبع non_critical)
    python prepare_triage_data.py        # train/val با قالب محاوره + غلط‌املایی
    python build_triage_eval.py          # test / golden / hard_negatives (خارج از train)
    python train_fasttext.py
"""
import csv
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

# قالب‌های آموزش: جملات محاوره‌ای با ترتیب کلمهٔ متفاوت (نه فقط «کلیدواژه در انتها»).
# غلط‌املایی/لهجه جداگانه با perturb() اعمال می‌شود تا fastText روی n-gram کار کند.
CRITICAL_TEMPLATES = [
    "{kw}",
    "کمک کنید {kw}",
    "دکتر {kw}",
    "فوریه، {kw}",
    "الان {kw}",
    "یهو {kw}",
    "بچم {kw}",
    "پدرم {kw}",
    "مادرم {kw}",
    "همین الان {kw} چیکار کنم",
    "{kw}، بگید چیکار کنم",
    "خواهش میکنم زود بگید {kw}",
    "{kw} کمک فوری میخوام",
    "همسرم {kw} چیکار کنم",
    "آقای دکتر فوریه {kw}",
    "والا {kw} داغون شدم",
    "بابا دیگه {kw} بگید چی کار کنم",
    "از دیشب تا الان {kw}",
    "{kw} تو راه اومدم اورژانس",
    "لطفا بگید {kw} خطرناکه؟ فوریه",
]

NEGATION_TEMPLATES = [
    "{kw} ندارم",
    "{kw} نداره",
    "خدا رو شکر {kw} نشد",
    "نه، {kw} نیست",
    "نگران {kw} بودم ولی نشد",
    "{kw} که نداره، فقط سرما خورده",
]

def perturb_keyword(kw: str, variant: int) -> str:
    """چند ریخت واقعی تایپ بیمار: فاصله اضافه، حروف نزدیک، لهجه محاوره."""
    if variant == 0:
        return kw
    replacements = [
        ("نمی", "نمي"),
        ("می‌", "مي"),
        ("ک", "ك"),
        ("ی", "ي"),
        ("تشنج", "تشنچ"),
        ("بیهوش", "بیحوش"),
        ("خونریزی", "خونریضی"),
        ("قفسه", "قفصه"),
        ("نفس", "نفص"),
        ("نمیکشه", "نمیكشه"),
        ("نمیتونم", "نمیتونم"),
        ("گلوش", "گلوش"),
        ("سینه", "سینه"),
    ]
    out = kw
    # هر variant یک یا دو جایگزینی اعمال می‌کند تا تنوع باشد نه درهم‌ریختگی کامل
    applied = 0
    for src, dst in replacements:
        if src in out and src != dst:
            out = out.replace(src, dst, 1)
            applied += 1
            if applied >= variant:
                break
    if applied == 0:
        out = kw.replace(" ", "") if " " in kw else (kw + " ")
    return out.strip() or kw


def gen_critical() -> list[str]:
    rows = []
    for kw in sorted(triage.EMERGENCY_KEYWORDS):
        for tpl in CRITICAL_TEMPLATES:
            rows.append(tpl.format(kw=kw))
            for v in (1, 2):
                rows.append(tpl.format(kw=perturb_keyword(kw, v)))
    return rows


def gen_negated_non_critical() -> list[str]:
    rows = []
    for kw in sorted(triage.EMERGENCY_KEYWORDS):
        for tpl in NEGATION_TEMPLATES:
            rows.append(tpl.format(kw=kw))
            rows.append(tpl.format(kw=perturb_keyword(kw, 1)))
    return rows


def gen_routine_non_critical() -> list[str]:
    rows = []
    for csv_name in ("train.csv", "validation.csv", "test.csv"):
        path = _ROOT / "ParsBert" / "data" / csv_name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                text = (r.get("text") or "").strip()
                if text:
                    rows.append(text)
    return rows


def main() -> None:
    critical = gen_critical()
    non_critical = gen_negated_non_critical() + gen_routine_non_critical()

    if not non_critical:
        raise SystemExit(
            "هیچ نمونه non_critical پیدا نشد. اول این را بزنید:\n"
            "  python ParsBert/prepare_data.py"
        )

    rng = random.Random(42)
    rng.shuffle(critical)
    rng.shuffle(non_critical)

    # fastText به عدم توازن شدید کلاس حساس است؛ non_critical را به حداکثر
    # ۳ برابر critical محدود می‌کنیم (نه کمتر، نه خیلی بیشتر).
    max_non_critical = min(len(non_critical), len(critical) * 3)
    non_critical = non_critical[:max_non_critical]

    rows = (
        [{"text": t, "urgency": "critical"} for t in critical]
        + [{"text": t, "urgency": "low"} for t in non_critical]
    )
    rng.shuffle(rows)

    split = int(len(rows) * 0.85)
    train_rows, val_rows = rows[:split], rows[split:]

    out_dir = _ROOT / "eval"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "val.jsonl").open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"critical: {len(critical)}   non_critical (استفاده‌شده): {len(non_critical)}")
    print(f"train: {len(train_rows)}   val: {len(val_rows)}")
    print(f"نوشته شد در: {out_dir}")


if __name__ == "__main__":
    main()
