# تنظیم هایپرپارامتر ParsBERT (روتر تخصص)

معیار اصلی مقایسه بین اجراها **macro-F1** است (نه فقط top-1 accuracy)،
چون هر ۶ تخصص را یکسان وزن می‌دهد. هر اجرا باید با همان
`ParsBert/evaluate.py --csv data/test.csv` سنجیده شود و نتیجه نسخه‌دار
ثبت گردد.

## شبکهٔ جست‌وجو (حداقل)

| اجرا | epochs | lr | batch-size | max-len | seed |
|------|--------|----|------------|---------|------|
| A (پایه) | 4 | 3e-5 | 16 | 128 | 42 |
| B | 2 | 3e-5 | 16 | 128 | 42 |
| C | 5 | 3e-5 | 16 | 128 | 42 |
| D | 4 | 2e-5 | 16 | 128 | 42 |
| E | 4 | 5e-5 | 16 | 128 | 42 |

فقط **یک** متغیر نسبت به پایه عوض شود تا اثر epoch و lr جدا دیده شود.
اگر GPU حافظه کم آورد، `batch-size` را به ۸ پایین بیاورید و همان شبکه را تکرار کنید؛ آن را در `--notes` بنویسید.

## فرمان‌ها

از پوشه `ParsBert/`:

```bash
python prepare_data.py

python train.py --epochs 4 --lr 3e-5 --batch-size 16 --out models/hp-A
python evaluate.py --csv data/test.csv --model-dir models/hp-A/final --log

python train.py --epochs 2 --lr 3e-5 --batch-size 16 --out models/hp-B
python evaluate.py --csv data/test.csv --model-dir models/hp-B/final --log

python train.py --epochs 5 --lr 3e-5 --batch-size 16 --out models/hp-C
python evaluate.py --csv data/test.csv --model-dir models/hp-C/final --log

python train.py --epochs 4 --lr 2e-5 --batch-size 16 --out models/hp-D
python evaluate.py --csv data/test.csv --model-dir models/hp-D/final --log

python train.py --epochs 4 --lr 5e-5 --batch-size 16 --out models/hp-E
python evaluate.py --csv data/test.csv --model-dir models/hp-E/final --log
```

`--log` این فیلدها را در `eval_history.jsonl` می‌نویسد:
git commit، هش `data/test.csv`، `macro_f1`، `recall_macro`، `top1_accuracy`.

مشاهده:

```bash
python ../log_eval_result.py --show --task parsbert-router
```

## حلقه active learning بعد از هر evaluate

`evaluate.py` جفت‌های خارج از قطر ماتریس درهم‌ریختگی را در
`data/confusion_pairs.json` می‌نویسد.

```bash
python build_active_learning_data.py --from-confusion data/confusion_pairs.json
python prepare_data.py
# سپس بهترین (epochs, lr) را دوباره روی دیتای جدید ترین کنید و evaluate.py --log
```

نمونه‌های اولیهٔ مرز تخصص‌ها (قبل از دیدن CM واقعی) از قبل در
`build_active_learning_data.py` هست؛ بعد از اولین CM، همان اسکریپت
اولویت را به جفت‌های پرخطای واقعی می‌دهد.

## تریاژ (ثبت جدا)

```bash
python prepare_triage_data.py
python build_triage_eval.py
python train_fasttext.py
python evaluate_triage.py --log
python evaluate_triage_pipeline.py --log
python evaluate_triage_pipeline.py --rule-only --log
python log_eval_result.py --show --task triage
```
