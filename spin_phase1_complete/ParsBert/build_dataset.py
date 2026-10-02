# -*- coding: utf-8 -*-
"""ساخت دیتاست اولیه (bootstrap) برای فاین‌تیون ParsBERT.

اگر دیتای واقعیِ برچسب‌خورده دارید، همان را در data/train.csv با ستون‌های
(text,label) بگذارید و این اسکریپت را اجرا نکنید. این اسکریپت یک دیتاست
مصنوعی از ترکیب «الگوهای شکایت بیمار × واژگان هر تخصص» می‌سازد تا مدل
از روز اول قابل استفاده باشد و بعداً با داده واقعی جایگزین/تقویت شود.
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

from specialties import SPECIALTIES
from text_utils import CSV_WRITE_ENCODING, normalize, setup_console

# الگوهایی که بیمار واقعی با آن‌ها شکایتش را می‌نویسد
TEMPLATES = [
    "سلام دکتر، {kw} دارم چیکار کنم؟",
    "چند روزه {kw} دارم و بهتر نمیشه.",
    "{kw} من شدید شده، به کدوم دکتر مراجعه کنم؟",
    "مادرم {kw} داره، نگرانشم.",
    "از دیشب {kw} شروع شده و خوابم نمیبره.",
    "دکتر گفت مشکل از {kw} هست، آیا خطرناکه؟",
    "برای {kw} چه آزمایشی لازمه؟",
    "{kw} بعد از خوردن دارو بدتر شد.",
    "پدرم ۶۰ سالشه و {kw} داره، وقت ویزیت میخوام.",
    "حدود دو هفته است که با {kw} درگیرم.",
    "{kw} همراه با ضعف و بی‌حالی دارم.",
    "درمان خانگی برای {kw} وجود داره؟",
    "با {kw} باید سراغ کدوم تخصص برم؟",
    "بچه‌ام {kw} داره و بی‌قراری میکنه.",
    "{kw} من عودکننده است و هر ماه تکرار میشه.",
    "آیا {kw} نیاز به جراحی داره؟",
    "{kw} و همینطور خستگی مفرط دارم.",
    "همسرم به خاطر {kw} بستری شد.",
]

# ترکیب دوتایی علائم برای شبیه‌سازی متن‌های طولانی‌تر و واقعی‌تر
PAIR_TEMPLATES = [
    "هم {kw1} دارم هم {kw2}، خیلی اذیتم میکنه.",
    "{kw1} به همراه {kw2} از هفته پیش شروع شده.",
    "دکتر، {kw1} دارم و کم کم {kw2} هم اضافه شده است.",
    "علائم من شامل {kw1} و {kw2} می‌باشد؛ لطفا راهنمایی کنید.",
]


def build_rows(seed: int = 42, pairs_per_specialty: int = 60):
    rng = random.Random(seed)
    rows = []
    for spec in SPECIALTIES:
        # ۱) تک‌کلیدواژه × همه الگوها
        for kw, tpl in itertools.product(spec.keywords, TEMPLATES):
            rows.append((normalize(tpl.format(kw=kw)), spec.name))
        # ۲) جفت کلیدواژه‌های همان تخصص
        for _ in range(pairs_per_specialty):
            kw1, kw2 = rng.sample(spec.keywords, 2)
            tpl = rng.choice(PAIR_TEMPLATES)
            rows.append((normalize(tpl.format(kw1=kw1, kw2=kw2)), spec.name))
        # ۳) خود کلیدواژه و توصیف تخصص به‌عنوان نمونه کوتاه
        for kw in spec.keywords:
            rows.append((normalize(kw), spec.name))
        rows.append((normalize(spec.description), spec.name))

    rng.shuffle(rows)
    return rows


def split(rows, val_ratio=0.1, test_ratio=0.1, seed=42):
    """تقسیم لایه‌ای (stratified) تا همه ۱۴ کلاس در هر بخش حاضر باشند."""
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
    ap.add_argument("--out", default="data", help="پوشه خروجی")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build_rows(seed=args.seed)
    # حذف تکراری‌ها با حفظ ترتیب
    seen, uniq = set(), []
    for r in rows:
        if r[0] and r not in seen:
            seen.add(r)
            uniq.append(r)

    train, val, test = split(uniq, seed=args.seed)
    out = Path(args.out)
    write_csv(out / "train.csv", train)
    write_csv(out / "validation.csv", val)
    write_csv(out / "test.csv", test)
    print(f"train={len(train)}  validation={len(val)}  test={len(test)}  (total={len(uniq)})")
    print(f"saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
