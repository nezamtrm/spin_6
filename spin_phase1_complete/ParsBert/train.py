# -*- coding: utf-8 -*-
"""فاین‌تیون ParsBERT برای دسته‌بندی متن بیمار به ۱۴ تخصص پزشکی.

نمونه اجرا:
    python train.py --epochs 4 --batch-size 16
    python train.py --model HooshvareLab/bert-fa-base-uncased --max-len 128
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from specialties import ID2LABEL, LABEL2ID, LABELS, NUM_LABELS
from text_utils import CSV_READ_ENCODING, normalize, setup_console

DEFAULT_MODEL = "HooshvareLab/bert-fa-base-uncased"   # ParsBERT v2
OUTPUT_DIR = "models/parsbert-specialty"


class SpecialtyDataset(Dataset):
    """دیتاست ساده روی CSV با ستون‌های text,label (بدون نیاز به کتابخانه datasets)."""

    def __init__(self, csv_path, tokenizer, max_len=128, keep_zwnj=False):
        self.samples = []
        with open(csv_path, encoding=CSV_READ_ENCODING, newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"].strip()
                if label not in LABEL2ID:
                    raise ValueError(f"برچسب ناشناخته در {csv_path}: {label!r}")
                text = normalize(row["text"], keep_zwnj=keep_zwnj)
                if text:
                    self.samples.append((text, LABEL2ID[label]))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tok(text, truncation=True, max_length=self.max_len)
        enc["labels"] = label
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="چک‌پوینت پایه ParsBERT")
    ap.add_argument("--data", default="data", help="پوشه شامل train/validation/test.csv")
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keep_zwnj = "zwnj" in args.model.lower()
    data = Path(args.data)
    print(f"مدل پایه: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = SpecialtyDataset(data / "train.csv", tokenizer, args.max_len, keep_zwnj)
    val_ds = SpecialtyDataset(data / "validation.csv", tokenizer, args.max_len, keep_zwnj)
    test_path = data / "test.csv"
    test_ds = SpecialtyDataset(test_path, tokenizer, args.max_len, keep_zwnj) if test_path.exists() else None
    print(f"نمونه‌ها → آموزش: {len(train_ds)} | اعتبارسنجی: {len(val_ds)}"
          + (f" | آزمون: {len(test_ds)}" if test_ds else ""))

    use_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        warmup_steps=int(0.1 * len(train_ds) / args.batch_size * args.epochs),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        fp16=use_cuda,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ارزیابی نهایی روی مجموعه آزمون
    if test_ds is not None:
        pred = trainer.predict(test_ds)
        y_true = pred.label_ids
        y_pred = np.argmax(pred.predictions, axis=-1)
        print("\n=== گزارش کلاس‌به‌کلاس روی مجموعه آزمون ===")
        print(classification_report(y_true, y_pred, target_names=LABELS,
                                    digits=3, zero_division=0))
        print({k: round(float(v), 4) for k, v in pred.metrics.items()
               if k.startswith("test_") and isinstance(v, (int, float))})

    # ذخیره مدل نهایی (این پوشه را predict.py می‌خواند)
    final = Path(args.out) / "final"
    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    (final / "specialties.json").write_text(
        json.dumps({"labels": LABELS, "base_model": args.model,
                    "keep_zwnj": keep_zwnj, "max_len": args.max_len},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n✅ مدل ذخیره شد در: {final.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
