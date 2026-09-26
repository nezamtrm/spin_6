# -*- coding: utf-8 -*-
"""ارزیابی سریع مدل روی مجموعه‌ای از جملات نمونهٔ دست‌نویس (۶ تخصص).

    python evaluate.py                 # نمونه‌های داخلی
    python evaluate.py --csv data/test.csv   # روی یک فایل CSV با ستون‌های text,label
"""

import argparse
import csv
from collections import Counter

from predict import SpecialtyClassifier
from text_utils import CSV_READ_ENCODING, setup_console

# جملاتی که در دیتاست نیستند تا تعمیم مدل سنجیده شود
SAMPLES = [
    ("دو ماهه پریود نشدم و حالت تهوع صبحگاهی دارم", "زنان و زایمان"),
    ("موقع بالا رفتن از پله زانوم صدا میده و درد میگیره", "ارتوپدی"),
    ("چند روزه دندون آسیام تیر میکشه و به سرما حساس شده", "دندانپزشکی"),
    ("پسر سه ساله‌ام هنوز جمله‌بندی درست نمیکنه و کلمات رو قاطی میگه", "گفتار درمانی"),
    ("چند روزه شکمم درد میکنه، بیشتر سمت راست پایینشه و تبم هم هست", "جراحی عمومی"),
    ("سرما خوردم و میخوام یه نسخه برای سرفه و آبریزش بینی بگیرم", "پزشک عمومی"),
]


def load_csv(path):
    with open(path, encoding=CSV_READ_ENCODING, newline="") as f:
        return [(r["text"], r["label"]) for r in csv.DictReader(f)]


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    data = load_csv(args.csv) if args.csv else SAMPLES
    clf = SpecialtyClassifier(model_dir=args.model_dir)
    print(f"حالت مدل: {clf.mode} | تعداد نمونه: {len(data)}\n")

    correct, top3, errors = 0, 0, Counter()
    for text, gold in data:
        res = clf.predict(text, top_k=3)
        names = [p.specialty for p in res.ranking]
        ok = names[0] == gold
        correct += ok
        top3 += gold in names
        if not ok:
            errors[f"{gold} -> {names[0]}"] += 1
        if not args.csv:
            mark = "OK " if ok else "XX "
            print(f"{mark} {res.top.percent:6.2f}%  {names[0]:<30} | انتظار: {gold}")
            print(f"     {text}")

    n = len(data)
    print(f"\ntop-1 accuracy: {correct}/{n} = {correct / n:.1%}")
    print(f"top-3 accuracy: {top3}/{n} = {top3 / n:.1%}")
    if errors:
        print("\nپرتکرارترین خطاها:")
        for k, v in errors.most_common(10):
            print(f"  {v:>3}x  {k}")

    # گزارش کامل: macro-F1 + precision/recall به تفکیک هر تخصص + ماتریس
    # درهم‌ریختگی. macro-F1 (نه top-1 accuracy) معیار اصلی مقایسه بین
    # نسخه‌های مختلف مدل باشد - چون هر ۶ تخصص را به‌اندازه هم مهم می‌شمارد،
    # نه فقط تخصص‌هایی که نمونه بیشتری دارند.
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("\n(sklearn نصب نیست - برای گزارش macro-F1/confusion matrix: pip install scikit-learn)")
        return

    gold_labels = [g for _, g in data]
    pred_labels = [clf.predict(t, top_k=1).top.specialty for t, _ in data]
    labels_sorted = sorted(set(gold_labels) | set(pred_labels))

    print("\n=== گزارش کامل (sklearn) ===")
    print(classification_report(gold_labels, pred_labels, labels=labels_sorted, zero_division=0))

    print("ماتریس درهم‌ریختگی (سطر=واقعی، ستون=پیش‌بینی):")
    cm = confusion_matrix(gold_labels, pred_labels, labels=labels_sorted)
    header = "".join(f"{l[:8]:>10}" for l in labels_sorted)
    print(f"{'':>20}{header}")
    for label, row in zip(labels_sorted, cm):
        print(f"{label:>20}" + "".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    main()
