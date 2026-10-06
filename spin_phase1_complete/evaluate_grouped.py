# -*- coding: utf-8 -*-
"""
evaluate_grouped.py — ارزیابی صادقانهٔ تریاژ + انتخاب آستانه (threshold)

  1) روی eval/val.jsonl (گروهی) آستانه را جاروب می‌کند و بزرگ‌ترین آستانه‌ای را
     برمی‌گزیند که recall «پایپلاین» (قانون OR fastText) ≥ هدف (پیش‌فرض ۰.۹۸) بماند.
  2) همان آستانه را *یک‌بار* روی eval/test_grouped.jsonl اعمال می‌کند.
  3) مهم‌ترین عدد: recall fastText روی نمونه‌های critical که قانون نمی‌گیرد
     (جمله‌های بدون کلیدواژه) = توان واقعی تعمیم.

اجرا:   python evaluate_grouped.py [--target-recall 0.98]
قبلش:   python prepare_triage_data.py && python train_fasttext.py
"""
import argparse, importlib.util, json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("triage", ROOT / "app" / "pipeline" / "triage.py")
triage = importlib.util.module_from_spec(sp); sp.loader.exec_module(triage)
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def load(name):
    p = ROOT / "eval" / name
    if not p.exists():
        sys.exit(f"❌ {p} پیدا نشد. اول python prepare_triage_data.py")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def p_critical(model, text):
    labels, probs = model.predict(triage.normalize(text).replace("\n", " "), k=2)
    return dict(zip(labels, map(float, probs))).get("__label__critical", 0.0)


def prep(model, rows):
    out = []
    for r in rows:
        out.append(dict(text=r["text"], gold=r["urgency"] == "critical", group=r.get("group", "?"), source=r.get("source", "?"),
                        rule=bool(triage.rule_based_check(r["text"])), p=p_critical(model, r["text"])))
    return out


def score(items, thr, mode):
    f = (lambda x: x["p"] >= thr) if mode == "ft" else (lambda x: x["rule"] or x["p"] >= thr)
    tp = sum(x["gold"] and f(x) for x in items)
    fn = sum(x["gold"] and not f(x) for x in items)
    fp = sum((not x["gold"]) and f(x) for x in items)
    tn = sum((not x["gold"]) and not f(x) for x in items)
    rec = tp / max(1, tp + fn); prec = tp / max(1, tp + fp); fpr = fp / max(1, fp + tn)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=rec, precision=prec, fpr=fpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--save-threshold", action="store_true",
                    help="آستانهٔ انتخابی را کنار مدل (triage_model.bin.threshold.json) ذخیره کن")
    a = ap.parse_args()

    model = triage._get_model()
    if model is None:
        sys.exit("❌ app/pipeline/triage_model.bin پیدا نشد؛ اول train_fasttext.py")
    import time
    mp = Path(triage._MODEL_PATH)
    tr = ROOT / "eval" / "train.jsonl"
    print(f"مدل  : {mp}  ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mp.stat().st_mtime))})")
    print(f"train: {tr}  ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tr.stat().st_mtime))})")
    if mp.stat().st_mtime < tr.stat().st_mtime:
        print("⚠ مدل *قدیمی‌تر* از train.jsonl است: بعد از prepare_triage_data.py آموزش دوباره نداده‌اید "
              "(یا مدل جای دیگری ذخیره شده). نتایج زیر معتبر نیست؛ اول python train_fasttext.py.")
    val, test = prep(model, load("val.jsonl")), prep(model, load("test_grouped.jsonl"))
    nvc = sum(x["gold"] for x in val)
    print(f"val: n={len(val)} (critical={nvc})   test: n={len(test)} (critical={sum(x['gold'] for x in test)})")
    if nvc < 100:
        print("⚠ val کمتر از ۱۰۰ نمونهٔ critical دارد؛ انتخاب آستانه نویزی است.")

    # معیار انتخاب آستانه: قانون recall را روی جمله‌های دارای کلیدواژه پنهان می‌کند، پس
    # تا حد ممکن recall «fastText» روی critical‌هایی که قانون نمی‌گیرد ملاک است.
    pool = [x for x in val if x["gold"] and not x["rule"]]
    if len(pool) >= 30:
        pool_name = f"critical بدون کلیدواژه ({len(pool)} نمونه)"
    else:
        pool, pool_name = [x for x in val if x["gold"]], "همهٔ critical (⚠ بدون کلیدواژهٔ کافی در val؛ داده LLM اضافه کنید)"
    print(f"\nمعیار انتخاب آستانه: recall fastText روی {pool_name}")
    print("--- جاروب آستانه روی VAL ---")
    print(f"{'thr':>5} | {'pool recall':>11} | {'pipe recall':>11} {'pipe FPR':>9} {'pipe prec':>10}")
    table = []
    for t in THRESHOLDS:
        pr = sum(x["p"] >= t for x in pool) / max(1, len(pool))
        p = score(val, t, "pipe")
        table.append((t, pr, p))
        print(f"{t:>5.2f} | {pr:>11.1%} | {p['recall']:>11.1%} {p['fpr']:>9.1%} {p['precision']:>10.1%}")
    ok = [t for t, pr, _ in table if pr >= a.target_recall]
    if ok:
        chosen = max(ok)
        print(f"\n✔ آستانهٔ انتخابی (بزرگ‌ترین با recall≥{a.target_recall:.0%}): {chosen}")
    else:
        best = max(pr for _, pr, _ in table)
        chosen = max(t for t, pr, _ in table if pr >= best - 0.005)
        print(f"\n⚠ هیچ آستانه‌ای به recall≥{a.target_recall:.0%} نرسید (بیشینهٔ ممکن {best:.1%}). "
              f"بزرگ‌ترین آستانه‌ای که از بیشینه بیش از ۰٫۵ نقطه کم نمی‌کند انتخاب شد: {chosen} "
              f"(نه کمترین آستانه؛ آن فقط FP را بی‌دلیل بالا می‌برد).")

    print(f"\n=== TEST گروهی @thr={chosen} (فقط یک‌بار نگاه کنید؛ دوباره برای بهتر شدن تنظیم نکنید) ===")
    for mode, name in (("ft", "fastText تنها"), ("pipe", "پایپلاین (قانون OR fastText)")):
        s = score(test, chosen, mode)
        print(f"{name:<30} recall={s['recall']:.1%} ({s['tp']}/{s['tp']+s['fn']})  "
              f"precision={s['precision']:.1%}  FPR={s['fpr']:.1%}  FN={s['fn']}  FP={s['fp']}")

    nokw = [x for x in test if x["gold"] and not x["rule"]]
    if nokw:
        hit = sum(x["p"] >= chosen for x in nokw)
        print(f"\nتعمیم (critical بدون کلیدواژه، فقط fastText می‌تواند بگیرد): {hit}/{len(nokw)} = {hit/len(nokw):.1%}")
    else:
        print("\n⚠ در test هیچ نمونهٔ critical بدون کلیدواژه نیست → تعمیم سنجیده نمی‌شود (داده LLM اضافه کنید).")

    def base(g):
        return re.sub(r"#\d+$", "", g)
    groups = defaultdict(list)
    for x in test:
        groups[(x["gold"], base(x["group"]))].append(x)
    print(f"\nتفکیک test بر اساس مفهوم @thr={chosen} (پایپلاین؛ ستون ft فقط fastText):")
    print(f"  {'مفهوم':<46}{'n':>4}  {'pipe':>6}  {'ft':>6}   (critical: recall، non_critical: نرخ FP)")
    for (gold, g), xs in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
        pipe = sum(bool(x["rule"] or x["p"] >= chosen) for x in xs) / len(xs)
        ft = sum(x["p"] >= chosen for x in xs) / len(xs)
        tag = "C" if gold else "N"
        print(f"  {tag} {g[:44]:<44}{len(xs):>4}  {pipe:>6.0%}  {ft:>6.0%}")
    print("\nFP بر اساس منبع:")
    for src in sorted({x["source"] for x in test}):
        xs = [x for x in test if x["source"] == src and not x["gold"]]
        if xs:
            print(f"   {src:<14} n={len(xs):>4}  FPR پایپلاین={sum(bool(x['rule'] or x['p']>=chosen) for x in xs)/len(xs):.0%}  FPR fastText={sum(x['p']>=chosen for x in xs)/len(xs):.0%}")

    miss = [x for x in test if x["gold"] and not (x["rule"] or x["p"] >= chosen)]
    print(f"\nFN پایپلاین ({len(miss)}) — این‌ها را تک‌تک بخوانید:")
    for x in miss[:25]:
        print(f"  p={x['p']:.2f} | {x['text']}")
    fps = [x for x in test if (not x["gold"]) and (x["rule"] or x["p"] >= chosen)]
    print(f"\nنمونه‌های FP پایپلاین ({len(fps)}) — اولین ۱۰ تا:")
    for x in fps[:10]:
        print(f"  rule={x['rule']!s:<5} p={x['p']:.2f} | {x['text']}")
    print(f"\nبرای production: export TRIAGE_FT_THRESHOLD={chosen}   (یا --save-threshold تا کنار مدل ذخیره شود)")
    if a.save_threshold:
        fs, ps = score(test, chosen, "ft"), score(test, chosen, "pipe")
        info = {"threshold": chosen, "target_recall": a.target_recall, "chosen_on": "val(grouped)",
                "test_pipeline_recall": round(ps["recall"], 4), "test_pipeline_fpr": round(ps["fpr"], 4),
                "test_ft_recall": round(fs["recall"], 4), "n_test": len(test),
                "model_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mp.stat().st_mtime))}
        Path(str(mp) + ".threshold.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print("ذخیره شد:", str(mp) + ".threshold.json")


if __name__ == "__main__":
    main()
