"""
evaluate_triage.py

ارزیابی مدل fastText تریاژ (app/pipeline/triage_model.bin) روی eval/test.jsonl
و eval/golden_emergency.jsonl - دو مجموعه‌ای که train_fasttext.py خودش در
train/validation استفاده نمی‌کند: test.jsonl برای یک گزارش نهایی بی‌طرفانه
(مدل هرگز این‌ها را در training/tuning ندیده)، و golden_emergency.jsonl
(۳۰۰ نمونه، شامل موارد سخت/مرزی - مثلا "non_critical_hard_*") برای استرس‌تست
مرز بحرانی/غیربحرانی.

مهم‌ترین عدد این گزارش: **recall کلاس critical**، نه accuracy کلی. یک
false negative روی critical (موردی واقعا بحرانی که مدل non_critical
تشخیص بدهد) از نظر ایمنی بیمار بسیار خطرناک‌تر از چند false positive
است (که فقط یعنی چیزی غیر-اورژانسی به‌اشتباه اورژانسی علامت خورده - آزاردهنده
ولی بی‌خطر). به همین دلیل هر false negative را هم تک‌تک چاپ می‌کند تا
دستی بازبینی شود.

اجرا:
    python evaluate_triage.py
"""
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _load_triage_module():
    spec = importlib.util.spec_from_file_location("triage", _ROOT / "app" / "pipeline" / "triage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


triage = _load_triage_module()
normalize = triage.normalize

import fasttext  # noqa: E402

# --- پچ سازگاری fasttext (نسخه پیش‌نصب‌شده روی Kaggle) با NumPy>=2.0 ---
# همان توضیح train_fasttext.py: fasttext/FastText.py::predict() داخل خودش
# np.array(probs, copy=False) صدا می‌زند که با NumPy>=2.0 خطا می‌دهد.
import numpy as _np


def _patch_fasttext_numpy2() -> None:
    import fasttext.FastText as _ft_module

    _orig_array = _np.array

    def _compat_array(obj, copy=False, **kwargs):
        if copy:
            return _orig_array(obj, copy=copy, **kwargs)
        return _np.asarray(obj, **kwargs)

    _ft_module.np.array = _compat_array


_patch_fasttext_numpy2()

MODEL_PATH = str(_ROOT / "app" / "pipeline" / "triage_model.bin")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(model, rows: list[dict], name: str) -> dict:
    tp = fp = tn = fn = 0
    false_negatives = []
    for r in rows:
        text = normalize(r["text"]).replace("\n", " ")
        pred_critical = model.predict(text)[0][0] == "__label__critical"
        gold_critical = r["urgency"] == "critical"
        if gold_critical and pred_critical:
            tp += 1
        elif gold_critical and not pred_critical:
            fn += 1
            false_negatives.append(r["text"])
        elif not gold_critical and pred_critical:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / total if total else float("nan")

    print(f"\n=== {name}  (n={total}) ===")
    print(f"  accuracy کلی:             {acc:.1%}")
    print(f"  recall کلاس critical:     {recall:.1%}  ({tp}/{tp + fn})   <- مهم‌ترین عدد")
    print(f"  precision کلاس critical:  {precision:.1%}")
    print(f"  false negative (خطرناک):  {fn}")
    if false_negatives:
        print("  موارد بحرانیِ گم‌شده (باید دستی بازبینی شوند):")
        for t in false_negatives:
            print(f"    - {t}")
    return {
        "accuracy": acc,
        "critical_recall": recall,
        "critical_precision": precision,
        "fn": fn,
        "fp": fp,
        "n": total,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true", help="ثبت در eval_history.jsonl")
    args = ap.parse_args()

    model = fasttext.load_model(MODEL_PATH)
    from log_eval_result import log_result
    for name, filename in [("test.jsonl", "test.jsonl"), ("golden_emergency.jsonl", "golden_emergency.jsonl")]:
        path = _ROOT / "eval" / filename
        if not path.exists():
            print(f"رد شد: {path} پیدا نشد")
            continue
        metrics = evaluate(model, load_jsonl(path), name)
        if args.log:
            log_result(
                "triage",
                f"eval/{filename}",
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if not (isinstance(v, float) and v != v)},
                notes="evaluate_triage.py (fastText only)",
            )


if __name__ == "__main__":
    main()
