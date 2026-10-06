# -*- coding: utf-8 -*-
"""
train_fasttext.py   (نسخهٔ اصلاح‌شده)

تغییرات مهم:
 1) همهٔ مسیرها نسبت به محل خود این فایل‌اند (قبلاً نسبت به cwd بودند). اگر از پوشهٔ
    دیگری اجرا می‌کردید، بی‌صدا train.jsonl کهنهٔ یک نسخهٔ دیگر را می‌خواند یا مدل را
    جای دیگری می‌نوشت، و evaluate مدل قدیمی را می‌سنجید (۲۱۰ نمونه‌ای که دیدید).
 2) مسیرها و تعداد نمونه‌ها با مسیر مطلق چاپ می‌شود تا کهنگی فوراً دیده شود.
 3) هایپرپارامترهای ملایم‌تر به‌جای lr=1.0/epoch=50 (که روی داده کوچک احتمال‌ها را
    اشباع و آستانه را بی‌معنا می‌کند)، و --grid برای انتخاب با val.
 4) معیار انتخاب/گزارش: بیشترین recall کلاس critical با FPR ≤ --max-fpr روی val.

اجرا:
    python train_fasttext.py            # پیکربندی پیش‌فرض
    python train_fasttext.py --grid     # جستجوی کوچک (چند دقیقه روی CPU)
"""
import argparse, importlib.util, itertools, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("triage", ROOT / "app" / "pipeline" / "triage.py")
triage = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(triage)
normalize = triage.normalize

import fasttext
import numpy as _np


def _patch_fasttext_numpy2() -> None:
    import fasttext.FastText as _ft
    _orig = _np.array

    def _compat(obj, copy=False, **kw):
        return _orig(obj, copy=copy, **kw) if copy else _np.asarray(obj, **kw)

    _ft.np.array = _compat


_patch_fasttext_numpy2()

TRAIN_SOURCE = ROOT / "eval" / "train.jsonl"
VAL_SOURCE = ROOT / "eval" / "val.jsonl"
TRAIN_FILE = ROOT / "eval" / "_train.txt"
MODEL_OUT = Path(os.environ.get("TRIAGE_MODEL_PATH", ROOT / "app" / "pipeline" / "triage_model.bin"))
THRESHOLDS = [round(x, 2) for x in _np.arange(0.05, 0.96, 0.05)]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def line(row):
    lab = "critical" if row["urgency"] == "critical" else "non_critical"
    return f"__label__{lab} " + normalize(row["text"]).replace("\n", " ")


def p_critical(model, text):
    labels, probs = model.predict(normalize(text).replace("\n", " "), k=2)
    return dict(zip(labels, map(float, probs))).get("__label__critical", 0.0)


def operating_point(golds, probs, max_fpr):
    """بیشترین recall با FPR<=max_fpr؛ خروجی (recall, fpr, thr)."""
    best = (0.0, 1.0, THRESHOLDS[0])
    P = sum(golds); N = len(golds) - P
    for t in THRESHOLDS:
        tp = sum(g and p >= t for g, p in zip(golds, probs))
        fp = sum((not g) and p >= t for g, p in zip(golds, probs))
        rec, fpr = tp / max(1, P), fp / max(1, N)
        if fpr <= max_fpr and (rec > best[0] or (rec == best[0] and fpr < best[1])):
            best = (rec, fpr, t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--word-ngrams", type=int, default=1)
    ap.add_argument("--minn", type=int, default=2)
    ap.add_argument("--maxn", type=int, default=4)
    ap.add_argument("--bucket", type=int, default=100000,
                    help="تعداد bucket هش n-gram. پیش‌فرض fastText=2M که با dim=50 مدل ۴۰۰MB می‌سازد")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--max-fpr", type=float, default=0.15)
    ap.add_argument("--min-train", type=int, default=500,
                    help="اگر train کمتر از این بود متوقف شو (نشانهٔ فایل کهنه)")
    a = ap.parse_args()

    train_rows, val_rows = load_jsonl(TRAIN_SOURCE), load_jsonl(VAL_SOURCE)
    nc = sum(r["urgency"] == "critical" for r in train_rows)
    print(f"ROOT        : {ROOT}")
    print(f"train       : {TRAIN_SOURCE}  ({len(train_rows)} نمونه، critical={nc}, non={len(train_rows)-nc})")
    print(f"val         : {VAL_SOURCE}  ({len(val_rows)} نمونه)")
    print(f"model خروجی : {MODEL_OUT}")
    if len(train_rows) < a.min_train:
        sys.exit(f"❌ فقط {len(train_rows)} نمونهٔ آموزشی — فایل کهنه است. اول prepare_triage_data.py "
                 f"(از همین پوشه). برای نادیده‌گرفتن: --min-train 0")

    TRAIN_FILE.write_text("\n".join(line(r) for r in train_rows) + "\n", encoding="utf-8")
    golds = [r["urgency"] == "critical" for r in val_rows]

    def run(cfg):
        m = fasttext.train_supervised(input=str(TRAIN_FILE), loss="softmax", verbose=0, **cfg)
        probs = [p_critical(m, r["text"]) for r in val_rows]
        return m, probs, operating_point(golds, probs, a.max_fpr)

    base = dict(lr=a.lr, epoch=a.epoch, dim=a.dim, wordNgrams=a.word_ngrams, minn=a.minn, maxn=a.maxn, bucket=a.bucket)
    cfgs = [base]
    if a.grid:
        cfgs = [dict(base, lr=lr, epoch=ep, wordNgrams=wn, dim=dm)
                for lr, ep, wn, dm in itertools.product((0.1, 0.25, 0.5), (15, 30), (1, 2), (30, 100))]
    best = None
    print(f"\nکانفیگ‌ها: {len(cfgs)}   معیار: بیشترین recall(critical) با FPR≤{a.max_fpr:.0%} روی val")
    for i, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        m, probs, (rec, fpr, thr) = run(cfg)
        print(f"[{i:>2}/{len(cfgs)}] lr={cfg['lr']:<4} ep={cfg['epoch']:<3} wng={cfg['wordNgrams']} dim={cfg['dim']:<3} "
              f"→ recall={rec:.1%} FPR={fpr:.1%} thr={thr}  ({time.time()-t0:.0f}s)")
        key = (rec, -fpr)
        if best is None or key > best[0]:
            best = (key, m, cfg, rec, fpr, thr)
    _, model, cfg, rec, fpr, thr = best
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"\n✔ بهترین: {cfg}")
    print(f"✔ ذخیره شد: {MODEL_OUT}  ({MODEL_OUT.stat().st_size/1e6:.1f} MB)")
    print(f"✔ val: recall={rec:.1%} در FPR={fpr:.1%} با آستانه {thr}")
    print("بعدی: python evaluate_grouped.py")
    if rec < 0.9:
        print("⚠ recall روی val زیر ۹۰٪ است: داده (به‌خصوص LLM بدون کلیدواژه) کافی نیست، نه هایپرپارامتر.")


if __name__ == "__main__":
    main()
