# -*- coding: utf-8 -*-
"""ارزیابی سریع مدل روی مجموعه‌ای از جملات نمونهٔ دست‌نویس (۶ تخصص).

    python evaluate.py                 # نمونه‌های داخلی
    python evaluate.py --csv data/test.csv   # روی یک فایل CSV با ستون‌های text,label
    python evaluate.py --csv data/test.csv --log   # ثبت در eval_history.jsonl
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

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


def dump_confusion_pairs(labels_sorted, cm, out_path: Path):
    """جفت‌های پرخطا (غیرقطر ماتریس) برای حلقه active learning."""
    pairs = []
    for i, gold in enumerate(labels_sorted):
        for j, pred in enumerate(labels_sorted):
            if i == j:
                continue
            count = int(cm[i, j])
            if count:
                pairs.append({"gold": gold, "pred": pred, "count": count})
    pairs.sort(key=lambda x: -x["count"])
    payload = {"pairs": pairs, "labels": labels_sorted}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nجفت‌های پرخطا برای active learning (نوشته شد: {out_path}):")
    if not pairs:
        print("  (خطای خارج از قطر نبود)")
    for p in pairs[:15]:
        print(f"  {p['count']:>3}x  {p['gold']} -> {p['pred']}")
    return pairs


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--dump-pairs", default="data/confusion_pairs.json")
    ap.add_argument("--log", action="store_true",
                    help="ثبت macro-F1/recall/accuracy در ../eval_history.jsonl")
    args = ap.parse_args()

    data = load_csv(args.csv) if args.csv else SAMPLES
    clf = SpecialtyClassifier(model_dir=args.model_dir)
    print(f"حالت مدل: {clf.mode} | تعداد نمونه: {len(data)}\n")

    correct, top3, errors = 0, 0, Counter()
    pred_top = []
    for text, gold in data:
        res = clf.predict(text, top_k=3)
        names = [p.specialty for p in res.ranking]
        pred_top.append(names[0])
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
    top1_acc = correct / n if n else float("nan")
    top3_acc = top3 / n if n else float("nan")
    print(f"\ntop-1 accuracy: {correct}/{n} = {top1_acc:.1%}")
    print(f"top-3 accuracy: {top3}/{n} = {top3_acc:.1%}")
    if errors:
        print("\nپرتکرارترین خطاها:")
        for k, v in errors.most_common(10):
            print(f"  {v:>3}x  {k}")

    # گزارش کامل: macro-F1 + precision/recall به تفکیک هر تخصص + ماتریس
    # درهم‌ریختگی. macro-F1 (نه top-1 accuracy) معیار اصلی مقایسه بین
    # نسخه‌های مختلف مدل باشد - چون هر ۶ تخصص را به‌اندازه هم مهم می‌شمارد،
    # نه فقط تخصص‌هایی که نمونه بیشتری دارند.
    try:
        from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
    except ImportError:
        print("\n(sklearn نصب نیست - برای گزارش macro-F1/confusion matrix: pip install scikit-learn)")
        return

    gold_labels = [g for _, g in data]
    pred_labels = pred_top
    labels_sorted = sorted(set(gold_labels) | set(pred_labels))

    print("\n=== گزارش کامل (sklearn) ===")
    print(classification_report(gold_labels, pred_labels, labels=labels_sorted, zero_division=0))

    print("ماتریس درهم‌ریختگی (سطر=واقعی، ستون=پیش‌بینی):")
    cm = confusion_matrix(gold_labels, pred_labels, labels=labels_sorted)
    header = "".join(f"{l[:8]:>10}" for l in labels_sorted)
    print(f"{'':>20}{header}")
    for label, row in zip(labels_sorted, cm):
        print(f"{label:>20}" + "".join(f"{v:>10}" for v in row))

    dump_confusion_pairs(labels_sorted, cm, Path(args.dump_pairs))

    macro_f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
    rec_macro = recall_score(gold_labels, pred_labels, average="macro", zero_division=0)

    if args.log:
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        from log_eval_result import log_result
        csv_rel = None
        if args.csv:
            csv_path = Path(args.csv).resolve()
            try:
                csv_rel = str(csv_path.relative_to(root)).replace("\\", "/")
            except ValueError:
                csv_rel = str(csv_path)
        row = log_result(
            "parsbert-router",
            csv_rel,
            {
                "macro_f1": round(float(macro_f1), 4),
                "recall_macro": round(float(rec_macro), 4),
                "top1_accuracy": round(float(top1_acc), 4),
                "top3_accuracy": round(float(top3_acc), 4),
                "n": n,
                "mode": clf.mode,
            },
            notes=f"evaluate.py --csv {args.csv}",
        )
        print("ثبت نسخه‌دار:", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
