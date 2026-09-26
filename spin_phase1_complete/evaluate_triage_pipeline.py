"""
evaluate_triage_pipeline.py

تفاوت با evaluate_triage.py: آن اسکریپت فقط خودِ مدل fastText را می‌سنجد
(model.predict مستقیم). این اسکریپت دقیقا همان تابعی را می‌سنجد که در
production واقعا صدا زده می‌شود: triage.is_emergency() - یعنی لایه اول
قانون‌محور (rule_based_check + هرس نگیشن با hazm) + fallback به fastText.

این‌جا دقیقا همان‌جایی است که «ست hard negative» (نگیشن/زمان گذشته)
استفاده می‌شود - چون نگیشن را فقط لایه اول تشخیص می‌دهد، نه fastText؛
سنجیدنش با evaluate_triage.py (که مستقیم سراغ fastText می‌رود) اصلا این
لایه را تست نمی‌کند.

فرمت eval/hard_negatives.jsonl - دقیقا مثل بقیه فایل‌های eval:
    {"text": "...", "urgency": "critical" | "non_critical"}
با این تفاوت که محتوایش عمدا نگیشن/زمان گذشته/شرطی است، مثلا:
    {"text": "قبلا تشنج داشتم ولی الان خوبم", "urgency": "non_critical"}
    {"text": "اگه تبم بالای ۳۹ بشه باید چیکار کنم", "urgency": "non_critical"}
    {"text": "دیشب خون بالا آوردم ولی امروز حالم خوبه", "urgency": "non_critical"}
(این‌ها فقط الگو/نمونه‌اند - نویسنده واقعی این فایل باید یک نفر با تجربه
بالینی باشد که می‌داند کدام «الان خوبم» واقعا مطمئن‌کننده است.)

اجرا:
    python evaluate_triage_pipeline.py
    python evaluate_triage_pipeline.py --rule-only   # فقط لایه قانون‌محور (بدون fastText)، برای دیدن اثر خالص hard negatives
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _load_triage_module():
    spec = importlib.util.spec_from_file_location("triage", _ROOT / "app" / "pipeline" / "triage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


triage = _load_triage_module()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(rows: list[dict], name: str, predict_fn) -> None:
    tp = fp = tn = fn = 0
    wrong_details = []
    for r in rows:
        pred_critical = predict_fn(r["text"])
        gold_critical = r["urgency"] == "critical"
        if gold_critical and pred_critical:
            tp += 1
        elif gold_critical and not pred_critical:
            fn += 1
            wrong_details.append(("FN (بحرانی گم‌شد)", r["text"]))
        elif not gold_critical and pred_critical:
            fp += 1
            wrong_details.append(("FP (غیربحرانی اشتباه پرچم خورد - مثلا نگیشن رد شد)", r["text"]))
        else:
            tn += 1

    total = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / total if total else float("nan")

    print(f"\n=== {name}  (n={total}) ===")
    print(f"  accuracy: {acc:.1%}   recall(critical): {recall:.1%}   precision(critical): {precision:.1%}")
    print(f"  FN={fn}  FP={fp}")
    for kind, text in wrong_details:
        print(f"    [{kind}] {text}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-only", action="store_true",
                     help="فقط rule_based_check (بدون fastText) - برای دیدن اثر خالص کلیدواژه/نگیشن")
    args = ap.parse_args()

    predict_fn = triage.rule_based_check if args.rule_only else triage.is_emergency

    files = [("hard_negatives.jsonl", "hard_negatives (نگیشن/زمان گذشته)"),
             ("test.jsonl", "test.jsonl (پایپلاین کامل)"),
             ("golden_emergency.jsonl", "golden_emergency.jsonl (پایپلاین کامل)")]
    for filename, label in files:
        path = _ROOT / "eval" / filename
        if not path.exists():
            print(f"رد شد: eval/{filename} پیدا نشد")
            continue
        evaluate(load_jsonl(path), label, predict_fn)


if __name__ == "__main__":
    main()
