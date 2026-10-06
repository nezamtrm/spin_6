# -*- coding: utf-8 -*-
# """ساخت data/train.csv, data/validation.csv, data/test.csv برای ۶ تخصص.

# این اسکریپت جایگزین جریان کاری قبلی build_dataset.py شده و دو منبع را
# با هم ترکیب می‌کند:

#   ۱) داده واقعی برچسب‌خورده در data/raw/parsbert_finetune_samples.json
#      (فرمت: لیستی از آبجکت‌های {id, specialty, chief_complaint,
#      follow_up_question, answer}) — برای تخصص‌های:
#      دندانپزشکی، زنان و زایمان، ارتوپدی، جراحی عمومی، گفتار درمانی

#   ۲) داده مصنوعی bootstrap برای «پزشک عمومی» که داده واقعی ندارد؛
#      از ترکیب الگوهای شکایت بیمار (TEMPLATES) × واژگان کلیدی GP در
#      specialties.py ساخته می‌شود (دقیقاً همان تکنیک build_dataset.py قدیم).

# اگر بعداً برای پزشک عمومی هم داده واقعی جمع کردید، همان را به فرمت
# chief_complaint/specialty در فایل JSON اضافه کنید یا مستقیماً به
# data/train.csv (ستون‌های text,label) اضافه/جایگزین کنید و دیگر نیازی به
# تولید مصنوعی GP نیست (با --no-synthetic-gp غیرفعالش کنید).

# اجرا:
#     python prepare_data.py
#     python prepare_data.py --no-followup-augment     # فقط متن اصلی شکایت، بدون تقویت با سوال پیگیری
#     python prepare_data.py --no-synthetic-gp          # اگر برای GP هم داده واقعی دارید
#     python prepare_data.py --gp-multiplier 1.5        # حجم دیتای مصنوعی GP را کم/زیاد کنید
# """

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

from specialties import BY_KEY, LABEL2ID, SPECIALTIES
from text_utils import CSV_WRITE_ENCODING, normalize, setup_console

# مسیرها نسبت به محل خود این فایل‌اند، نه نسبت به پوشه‌ای که از آن اجرا می‌کنید
# (قبلاً از ریشهٔ پروژه اجرا می‌شد و FileNotFoundError می‌داد).
_HERE = Path(__file__).resolve().parent
RAW_JSON = _HERE / "data" / "raw" / "parsbert_finetune_samples.json"
RAW_DIR = _HERE / "data" / "raw"

# نام‌های فارسی تخصص‌هایی که در فایل JSON داده واقعی دارند (باید دقیقا با
# specialties.py یکی باشند)
JSON_COVERED_SPECIALTIES = {
    "دندانپزشکی", "زنان و زایمان", "ارتوپدی", "جراحی عمومی", "گفتار درمانی",
}

# ----------------------------------------------------------------------------
# بخش ۱: خواندن و تبدیل داده واقعی JSON
# ----------------------------------------------------------------------------

def load_real_rows(path: Path, augment_with_followup: bool = True):
    """داده واقعی را به لیستی از (text, label) تبدیل می‌کند.

    برای هر رکورد:
      - chief_complaint به‌تنهایی (متنی که خودِ بیمار می‌نویسد) همیشه اضافه می‌شود.
      - در صورت augment_with_followup=True، ترکیب
        «chief_complaint + سوال پیگیری» هم به‌عنوان یک نمونه‌ی متنیِ غنی‌تر
        اضافه می‌شود تا واژگان تخصصی بیشتری در دیتاست حضور داشته باشد.
      - فیلد answer (پاسخ/توضیح) استفاده نمی‌شود، چون متنی است که پزشک
        می‌نویسد نه بیمار، و در inference واقعی وجود نخواهد داشت.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"فایل داده واقعی پیدا نشد: {path}\n"
            f"فایل parsbert_finetune_samples.json را در همین مسیر قرار دهید."
        )
    records = json.loads(path.read_text(encoding="utf-8"))

    rows, seen = [], set()

    def add(text, label):
        text = normalize(text)
        if text and (text, label) not in seen:
            seen.add((text, label))
            rows.append((text, label))

    for r in records:
        label = r["specialty"].strip()
        if label not in LABEL2ID:
            raise ValueError(
                f"برچسب {label!r} در specialties.py تعریف نشده. "
                f"نام‌ها باید دقیقاً یکسان باشند."
            )
        add(r["chief_complaint"], label)
        if augment_with_followup and r.get("follow_up_question"):
            combined = f"{r['chief_complaint']}؛ {r['follow_up_question']}"
            add(combined, label)

    return rows


# ----------------------------------------------------------------------------
# بخش ۲: تولید داده مصنوعی برای «پزشک عمومی» (یا هر تخصص بدون داده واقعی)
# ----------------------------------------------------------------------------

GP_TEMPLATES = [
    "سلام دکتر، {kw} دارم چیکار کنم؟",
    "چند روزه {kw} دارم و بهتر نمیشه.",
    "برای {kw} باید حضوری بیام یا نه؟",
    "میخوام برای {kw} نوبت بگیرم.",
    "بابت {kw} اومدم، اول باید پیش کدوم دکتر برم؟",
    "مادرم {kw} داره، خطرناکه یا نه؟",
    "{kw} دارم، به تخصص خاصی نیاز داره یا نه؟",
    "چند وقته با {kw} درگیرم و نمیدونم مشکلم چیه.",
    "برای {kw} چه آزمایشی نیاز دارم؟",
    "دو هفته‌ست {kw} دارم، باید نگران باشم؟",
    "همسرم {kw} داره، توصیه‌تون چیه؟",
    "{kw} همراه با کسالت عمومی دارم.",
]

GP_PAIR_TEMPLATES = [
    "هم {kw1} دارم هم {kw2}، نمیدونم مشکلم چیه.",
    "{kw1} به همراه {kw2} از چند روز پیش شروع شده.",
    "دکتر، {kw1} دارم و کم‌کم {kw2} هم اضافه شده.",
]


def build_synthetic_rows(spec_key: str, seed: int = 42, pairs: int = 40):
    """داده مصنوعی برای یک تخصص (پیش‌فرض: GP) از روی TEMPLATES × keywords می‌سازد."""
    spec = BY_KEY[spec_key]
    rng = random.Random(seed)
    rows = []
    for kw, tpl in itertools.product(spec.keywords, GP_TEMPLATES):
        rows.append((normalize(tpl.format(kw=kw)), spec.name))
    for _ in range(pairs):
        kw1, kw2 = rng.sample(spec.keywords, 2)
        tpl = rng.choice(GP_PAIR_TEMPLATES)
        rows.append((normalize(tpl.format(kw1=kw1, kw2=kw2)), spec.name))
    for kw in spec.keywords:
        rows.append((normalize(kw), spec.name))
    rows.append((normalize(spec.description), spec.name))
    return rows


# ----------------------------------------------------------------------------
# بخش ۳: تقسیم لایه‌ای train/validation/test
# ----------------------------------------------------------------------------

def stratified_split(rows, val_ratio=0.1, test_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for text, label in rows:
        by_label.setdefault(label, []).append((text, label))

    train, val, test = [], [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]

    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_WRITE_ENCODING, newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-json", default=str(RAW_JSON))
    ap.add_argument("--out", default=str(_HERE / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-followup-augment", action="store_true",
                     help="فقط chief_complaint خام استفاده شود (بدون ترکیب با سوال پیگیری)")
    ap.add_argument("--no-synthetic-gp", action="store_true",
                     help="داده مصنوعی برای پزشک عمومی ساخته نشود (وقتی خودتان داده واقعی GP دارید)")
    ap.add_argument("--gp-multiplier", type=float, default=1.0,
                     help="ضریب حجم داده مصنوعی GP نسبت به میانگین بقیه تخصص‌ها")
    args = ap.parse_args()

    all_rows = []

    # ۱) داده واقعی (۵ تخصص) + فایل‌های active learning در data/raw/*.json
    json_paths = [Path(args.raw_json)]
    raw_dir = Path(args.raw_json).parent if Path(args.raw_json).parent.exists() else RAW_DIR
    for extra in sorted(raw_dir.glob("*.json")):
        if extra.resolve() != Path(args.raw_json).resolve():
            json_paths.append(extra)

    real_rows = []
    for jp in json_paths:
        part = load_real_rows(jp, augment_with_followup=not args.no_followup_augment)
        real_rows += part
        print(f"  {jp.name}: {len(part)} نمونه")
    all_rows += real_rows
    print(f"داده واقعی بارگذاری شد: {len(real_rows)} نمونه "
          f"({'با' if not args.no_followup_augment else 'بدون'} تقویت سوال پیگیری)")

    # ۲) داده مصنوعی برای پزشک عمومی
    if not args.no_synthetic_gp:
        gp_rows = build_synthetic_rows("general_practice", seed=args.seed,
                                        pairs=int(40 * args.gp_multiplier))
        all_rows += gp_rows
        print(f"داده مصنوعی «پزشک عمومی» ساخته شد: {len(gp_rows)} نمونه")
    else:
        print("تولید داده مصنوعی GP غیرفعال است — مطمئن شوید data/train.csv "
              "شامل نمونه‌های «پزشک عمومی» از منبع دیگری هست.")

    # حذف تکراری‌های کامل با حفظ ترتیب
    seen, uniq = set(), []
    for r in all_rows:
        if r[0] and r not in seen:
            seen.add(r)
            uniq.append(r)

    train, val, test = stratified_split(uniq, seed=args.seed)
    out = Path(args.out)
    write_csv(out / "train.csv", train)
    write_csv(out / "validation.csv", val)
    write_csv(out / "test.csv", test)

    print(f"\nمجموع نمونه‌های یکتا: {len(uniq)}")
    print(f"train={len(train)}  validation={len(val)}  test={len(test)}")
    print(f"ذخیره شد در: {out.resolve()}")

    # گزارش توزیع کلاس‌ها در train
    from collections import Counter
    dist = Counter(l for _, l in train)
    print("\nتوزیع کلاس‌ها در train.csv:")
    for s in SPECIALTIES:
        print(f"  {s.name:<15} {dist.get(s.name, 0)}")


if __name__ == "__main__":
    main()
