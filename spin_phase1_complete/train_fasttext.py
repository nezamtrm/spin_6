"""
train_fasttext.py

Trains a FastText model using the pre-split eval/train.jsonl and
eval/val.jsonl files (produced once by a stratified 70/15/15 split, kept
fixed so results are comparable run to run -- no more re-shuffling the
split on every run).

Run:
    pip install fasttext
    python train_fasttext.py
"""
import json
import os
import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "triage", os.path.join(_ROOT, "app", "pipeline", "triage.py")
)
_triage_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_triage_module)
normalize = _triage_module.normalize

import fasttext

# --- پچ سازگاری fasttext (نسخه پیش‌نصب‌شده روی Kaggle) با NumPy>=2.0 ---
# مشکل: fasttext/FastText.py::predict() داخل خودش np.array(probs, copy=False)
# صدا می‌زند که با NumPy>=2.0 ValueError می‌دهد (باگ داخل خودِ پکیج، نه کد ما).
# این‌جا inline است (نه یک فایل جدا) که کپی‌پیست تک‌فایلی این اسکریپت هم
# جواب بدهد.
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

TRAIN_SOURCE = "train.jsonl"
VAL_SOURCE = "val.jsonl"
TRAIN_FILE = "_train.txt"
VAL_FILE = "_val.txt"
MODEL_OUT = "app/pipeline/triage_model.bin"


def to_fasttext_line(row: dict) -> str:
    label = "critical" if row["urgency"] == "critical" else "non_critical"
    text = normalize(row["text"]).replace("\n", " ")
    return f"__label__{label} {text}"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


train_rows = load_jsonl(TRAIN_SOURCE)
val_rows = load_jsonl(VAL_SOURCE)

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(to_fasttext_line(r) + "\n")

with open(VAL_FILE, "w", encoding="utf-8") as f:
    for r in val_rows:
        f.write(to_fasttext_line(r) + "\n")

print(f"train examples: {len(train_rows)}, validation examples: {len(val_rows)}")

model = fasttext.train_supervised(
    input=TRAIN_FILE,
    epoch=50,
    lr=1.0,
    wordNgrams=2,
    dim=100,
    loss="softmax",
    minn=2,
    maxn=5,
)

model.save_model(MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")

n, precision, recall = model.test(VAL_FILE)
print(f"validation set: n={n}, precision={precision:.3f}, recall={recall:.3f}")

tp = sum(
    1 for r in val_rows
    if r["urgency"] == "critical"
    and model.predict(normalize(r["text"]).replace("\n", " "))[0][0] == "__label__critical"
)
total_critical_val = sum(1 for r in val_rows if r["urgency"] == "critical")
if total_critical_val:
    print(f"validation-only critical recall: {tp}/{total_critical_val} = {tp/total_critical_val:.3f}")
