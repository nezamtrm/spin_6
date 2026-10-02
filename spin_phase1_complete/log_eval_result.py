"""
log_eval_result.py

یک فایل append-only (eval_history.jsonl در ریشه ریپو) که هر بار یک ارزیابی
(ParsBERT، تریاژ، هرچیز دیگری) اجرا می‌شود، یک سطر با نسخه‌ی دقیق کد+دیتا
به آن اضافه می‌شود. بدون این، بعد از چند بار تغییر کلیدواژه/دیتا/کد
نمی‌شود گفت کدام عدد مال کدام نسخه بوده.

هر سطر شامل: زمان، git commit hash (اگر در گیت باشد)، هش دیتاست ورودی
(برای اینکه تغییر بی‌صدا در داده هم قابل‌ردیابی باشد)، نام مدل/تسک، و
هر متریکی که بدهید (کلید=مقدار دلخواه).

اجرا (دستی، بعد از هر evaluate):
    python log_eval_result.py --task triage --dataset eval/test.jsonl \
        --metric critical_recall=0.933 --metric accuracy=0.911 \
        --notes "بعد از اضافه‌شدن کلیدواژه‌های اوردوز/خون در مدفوع"

    python log_eval_result.py --task parsbert-router --dataset ParsBert/data/test.csv \
        --metric macro_f1=0.87 --metric top1_accuracy=0.91

مشاهده تاریخچه:
    python log_eval_result.py --show                 # همه
    python log_eval_result.py --show --task triage    # فقط یک تسک
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_HISTORY_FILE = _ROOT / "eval_history.jsonl"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=3, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - نبودن گیت/کامیت نباید کل لاگ را خراب کند
        return None


def _file_hash(path: Path) -> str | None:
    """هش محتوای فایل دیتاست - برای این‌که اگر کسی محتوای test.jsonl را
    بی‌صدا عوض کرد (نه فقط اسمش)، در تاریخچه قابل تشخیص باشد."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _parse_metrics(pairs: list[str]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"فرمت نامعتبر برای --metric: {pair!r} (باید key=value باشد)")
        key, value = pair.split("=", 1)
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def log_result(task: str, dataset: str | None, metrics: dict, notes: str | None) -> dict:
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "task": task,
        "dataset_path": dataset,
        "dataset_hash": _file_hash(_ROOT / dataset) if dataset else None,
        "metrics": metrics,
        "notes": notes,
    }
    with _HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def show_history(task_filter: str | None) -> None:
    if not _HISTORY_FILE.exists():
        print("هنوز چیزی ثبت نشده.")
        return
    with _HISTORY_FILE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if task_filter and row["task"] != task_filter:
                continue
            print(f"{row['timestamp']}  [{row['task']}]  commit={row['git_commit']}  "
                  f"dataset_hash={row['dataset_hash']}")
            print(f"    metrics: {row['metrics']}")
            if row.get("notes"):
                print(f"    notes: {row['notes']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="مثلا triage یا parsbert-router")
    ap.add_argument("--dataset", help="مسیر فایل دیتاست ارزیابی‌شده (برای هش)")
    ap.add_argument("--metric", action="append", default=[], help="key=value، چندبار قابل تکرار")
    ap.add_argument("--notes")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.show:
        show_history(args.task)
        return

    if not args.task:
        raise SystemExit("--task الزامی است (مگر با --show)")

    row = log_result(args.task, args.dataset, _parse_metrics(args.metric), args.notes)
    print(f"ثبت شد در {_HISTORY_FILE}:")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
