# -*- coding: utf-8 -*-
"""
prepare_triage_data.py   (نسخهٔ اصلاح‌شده)

ساخت eval/train.jsonl ، eval/val.jsonl ، eval/test_grouped.jsonl برای
train_fasttext.py (مدل تریاژ اورژانس).

چه چیزی نسبت به نسخهٔ قبلی عوض شد و چرا
------------------------------------------
1. چک «داده روزمره» واقعی: قبلاً اگر ParsBert/data/*.csv نبود، اسکریپت بی‌صدا فقط
   با جمله‌های نگیشن (۱۲۶۰ تا) غیراورژانس می‌ساخت و مدل «همه‌چیز اورژانس است»
   یاد می‌گرفت (واژگان ۳۲۷ کلمه، precision=65٪). حالا اگر کمتر از --min-routine
   نمونه روزمره باشد، اسکریپت متوقف می‌شود.
2. split گروهی (group split): هر «مفهوم» (یک کلیدواژهٔ اورژانسی، یک شکایت پایه
   روزمره، یک concept از داده LLM) فقط در یکی از train/val/test می‌آید. با hash
   پایدار تعیین می‌شود (اضافه‌کردن داده جدید، تقسیم‌های قبلی را جابه‌جا نمی‌کند).
   قبلاً val با shuffle تصادفی ساخته می‌شد و val=1.000 فقط حفظ‌کردن بود.
3. توازن کلاس: گزارش + کاهش کلاس اکثریت در train با سقف برای هر گروه
   (--max-imbalance)، تا قالب‌های یک کلیدواژه همه‌چیز را اشغال نکنند.
4. تعارض برچسب: نمونهٔ «روزمره/غیراورژانس» که خودِ لایهٔ قانون‌محور آن را اورژانس
   می‌داند (مثلاً «بعد از زایمان خونریزی شدید دارم») از train حذف و در
   eval/quarantine_for_review.jsonl برای بازبینی پزشک ذخیره می‌شود.
5. ورودی اختیاری داده تولیدشده با LLM:
     eval/llm_critical.jsonl     {"text": "...", "concept": "..."}
     eval/llm_noncritical.jsonl  {"text": "...", "concept": "..."}
   (راهنمای تولید: TRIAGE_DATA_GENERATION.md)

اجرا (از ریشهٔ پروژه):
    python ParsBert/prepare_data.py          # اگر CSVها هنوز نیستند
    python prepare_triage_data.py
    python train_fasttext.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_triage_module():
    spec = importlib.util.spec_from_file_location("triage", ROOT / "app" / "pipeline" / "triage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


triage = _load_triage_module()

# ----------------------------------------------------------------------------
# قالب‌ها
# ----------------------------------------------------------------------------
CRITICAL_TEMPLATES = [
    "{kw}", "کمک کنید {kw}", "دکتر {kw}", "فوریه، {kw}", "الان {kw}", "یهو {kw}",
    "بچم {kw}", "پدرم {kw}", "مادرم {kw}", "همین الان {kw} چیکار کنم",
    "{kw}، بگید چیکار کنم", "خواهش میکنم زود بگید {kw}", "{kw} کمک فوری میخوام",
    "همسرم {kw} چیکار کنم", "آقای دکتر فوریه {kw}", "والا {kw} داغون شدم",
    "بابا دیگه {kw} بگید چی کار کنم", "از دیشب تا الان {kw}",
    "{kw} تو راه اومدم اورژانس", "لطفا بگید {kw} خطرناکه؟ فوریه",
]
NEGATION_TEMPLATES = [
    "{kw} ندارم", "{kw} نداره", "خدا رو شکر {kw} نشد", "نه، {kw} نیست",
    "نگران {kw} بودم ولی نشد", "{kw} که نداره، فقط سرما خورده",
]
_PERTURB = [
    ("نمی", "نمي"), ("ک", "ك"), ("ی", "ي"), ("تشنج", "تشنچ"), ("بیهوش", "بیحوش"),
    ("خونریزی", "خونریضی"), ("قفسه", "قفصه"), ("نفس", "نفص"),
]


def perturb_keyword(kw: str, rng: random.Random) -> str | None:
    """یک غلط‌املایی/ریخت تایپی؛ None اگر چیزی عوض نشد (تا نمونهٔ تکراری نسازد)."""
    cands = [(a, b) for a, b in _PERTURB if a in kw]
    if cands:
        a, b = rng.choice(cands)
        out = kw.replace(a, b, 1)
    elif " " in kw:
        out = kw.replace(" ", "", 1)
    else:
        return None
    return out if out != kw else None


# ----------------------------------------------------------------------------
# split گروهی با hash پایدار
# ----------------------------------------------------------------------------
def split_of(group: str, seed: int, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.sha1(f"{seed}:{group}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


# ----------------------------------------------------------------------------
# تولید/بارگذاری منابع. هر ردیف: text, urgency, group, source
# ----------------------------------------------------------------------------
def gen_template_rows(rng: random.Random, templates_per_kw: int):
    rows = []
    for kw in sorted(triage.EMERGENCY_KEYWORDS):
        g = f"kw:{triage.normalize(kw)}"
        tpls = rng.sample(CRITICAL_TEMPLATES, min(templates_per_kw, len(CRITICAL_TEMPLATES)))
        for tpl in tpls:
            rows.append(dict(text=tpl.format(kw=kw), urgency="critical", group=g, source="template"))
            p = perturb_keyword(kw, rng)
            if p:
                rows.append(dict(text=tpl.format(kw=p), urgency="critical", group=g, source="template_typo"))
        for tpl in NEGATION_TEMPLATES:
            rows.append(dict(text=tpl.format(kw=kw), urgency="non_critical", group=g, source="template_neg"))
    return rows


def load_routine(parsbert_data: Path):
    rows = []
    for name in ("train.csv", "validation.csv", "test.csv"):
        p = parsbert_data / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                t = (r.get("text") or "").strip()
                if t:
                    base = re.split(r"[؛;]", t)[0].strip()
                    rows.append(dict(text=t, urgency="non_critical", group=f"r:{base}", source="routine"))
    return rows


def load_llm(path: Path, urgency: str):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            t, c = (r.get("text") or "").strip(), (r.get("concept") or "").strip()
            if not t or not c:
                raise SystemExit(f"{path}:{ln}: هر خط باید text و concept داشته باشد.")
            rows.append(dict(text=t, urgency=urgency, group=f"llm:{urgency}:{c}", source="llm"))
    return rows


# ----------------------------------------------------------------------------
# کمکی‌ها
# ----------------------------------------------------------------------------
def cap_per_group(rows, target: int, rng: random.Random):
    """تعداد ردیف‌ها را با سقف یکسان برای هر گروه به <= target می‌رساند."""
    if len(rows) <= target:
        return rows
    by_g = defaultdict(list)
    for r in rows:
        by_g[r["group"]].append(r)
    for g in by_g.values():
        rng.shuffle(g)
    cap = max(len(g) for g in by_g.values())
    while cap > 1 and sum(min(len(g), cap) for g in by_g.values()) > target:
        cap -= 1
    out = [r for g in by_g.values() for r in g[:cap]]
    rng.shuffle(out)
    return out


def vocab_size(rows):
    return len({w for r in rows for w in triage.normalize(r["text"]).split()})


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("text", "urgency", "source", "group")},
                               ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15, help="سهم *گروه‌ها* برای val")
    ap.add_argument("--test-frac", type=float, default=0.15, help="سهم *گروه‌ها* برای test")
    ap.add_argument("--templates-per-kw", type=int, default=10)
    ap.add_argument("--min-routine", type=int, default=300)
    ap.add_argument("--min-groups-eval", type=int, default=8)
    ap.add_argument("--max-imbalance", type=float, default=1.5,
                    help="حداکثر نسبت کلاس اکثریت به اقلیت در train")
    ap.add_argument("--parsbert-data", default=str(ROOT / "ParsBert" / "data"))
    ap.add_argument("--llm-critical", default=str(ROOT / "eval" / "llm_critical.jsonl"))
    ap.add_argument("--llm-noncritical", default=str(ROOT / "eval" / "llm_noncritical.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval"))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    template_rows = gen_template_rows(rng, a.templates_per_kw)
    routine_rows = load_routine(Path(a.parsbert_data))
    llm_crit = load_llm(Path(a.llm_critical), "critical")
    llm_non = load_llm(Path(a.llm_noncritical), "non_critical")

    # --- چک ۱: داده روزمره واقعی حتماً باشد --------------------------------
    if len(routine_rows) < a.min_routine:
        raise SystemExit(
            f"فقط {len(routine_rows)} نمونه روزمره پیدا شد (حداقل {a.min_routine}).\n"
            f"اول اجرا کنید:  python ParsBert/prepare_data.py\n"
            f"بدون داده روزمره مدل «همه‌چیز اورژانس است» را یاد می‌گیرد."
        )

    all_rows = template_rows + routine_rows + llm_crit + llm_non

    # --- چک ۲: تعارض برچسب با لایه قانون‌محور → قرنطینه --------------------
    quarantine, kept = [], []
    for r in all_rows:
        if r["urgency"] == "non_critical" and r["source"] in ("routine", "llm") \
                and triage.rule_based_check(r["text"]):
            quarantine.append({**r, "why": "برچسب non_critical ولی قانون اورژانس می‌داند؛ پزشک بازبینی کند"})
        else:
            kept.append(r)

    # --- dedupe: متن یکسان با دو برچسب متفاوت → قرنطینه -------------------
    by_text = defaultdict(set)
    for r in kept:
        by_text[triage.normalize(r["text"])].add(r["urgency"])
    conflict = {t for t, s in by_text.items() if len(s) > 1}
    seen, rows = set(), []
    for r in kept:
        nt = triage.normalize(r["text"])
        if nt in conflict:
            quarantine.append({**r, "why": "همین متن با هر دو برچسب وجود دارد"})
            continue
        if nt in seen:
            continue
        seen.add(nt)
        rows.append(r)

    # --- split گروهی --------------------------------------------------------
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        splits[split_of(r["group"], a.seed, a.val_frac, a.test_frac)].append(r)

    # متن مشترک بین train و val/test (مثلاً دو گروه با متن یکسان) → از train حذف
    held_texts = {triage.normalize(r["text"]) for s in ("val", "test") for r in splits[s]}
    before = len(splits["train"])
    splits["train"] = [r for r in splits["train"] if triage.normalize(r["text"]) not in held_texts]
    dropped_overlap = before - len(splits["train"])

    # --- چک ۳: گروه کافی در val/test ---------------------------------------
    for s in ("val", "test"):
        for u in ("critical", "non_critical"):
            n = len({r["group"] for r in splits[s] if r["urgency"] == u})
            if n < a.min_groups_eval:
                raise SystemExit(f"{s}/{u}: فقط {n} گروه (حداقل {a.min_groups_eval}). "
                                 f"داده بیشتر اضافه کنید یا --val-frac/--test-frac را بالا ببرید.")

    # --- توازن کلاس در train ------------------------------------------------
    tr = splits["train"]
    crit = [r for r in tr if r["urgency"] == "critical"]
    non = [r for r in tr if r["urgency"] != "critical"]
    raw_counts = (len(crit), len(non))
    if len(crit) > a.max_imbalance * len(non):
        crit = cap_per_group(crit, int(a.max_imbalance * len(non)), rng)
    elif len(non) > a.max_imbalance * len(crit):
        non = cap_per_group(non, int(a.max_imbalance * len(crit)), rng)
    splits["train"] = crit + non
    for s in splits.values():
        rng.shuffle(s)

    # --- نشت: تضمین گروه‌های جدا -------------------------------------------
    g = {k: {r["group"] for r in v} for k, v in splits.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"]), "نشت گروه!"

    out = Path(a.out)
    write_jsonl(out / "train.jsonl", splits["train"])
    write_jsonl(out / "val.jsonl", splits["val"])
    write_jsonl(out / "test_grouped.jsonl", splits["test"])
    q_path = out / "quarantine_for_review.jsonl"
    q_path.parent.mkdir(parents=True, exist_ok=True)
    with q_path.open("w", encoding="utf-8") as f:
        for r in quarantine:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- گزارش ---------------------------------------------------------------
    report = {"seed": a.seed, "dropped_train_text_overlap": dropped_overlap,
              "train_before_balance": {"critical": raw_counts[0], "non_critical": raw_counts[1]},
              "quarantined": len(quarantine), "splits": {}}
    print("\n" + "=" * 74)
    print(f"{'split':<7}{'n':>7}{'critical':>10}{'non_crit':>10}{'%non':>7}{'groups':>8}"
          f"{'vocab':>7}{'crit بدون کلیدواژه':>22}")
    for name in ("train", "val", "test"):
        s = splits[name]
        c = sum(r["urgency"] == "critical" for r in s)
        n = len(s) - c
        nokw = sum(1 for r in s if r["urgency"] == "critical" and not triage.rule_based_check(r["text"]))
        pct = 100 * n / max(1, len(s))
        print(f"{name:<7}{len(s):>7}{c:>10}{n:>10}{pct:>6.0f}%{len(g[name]):>8}{vocab_size(s):>7}{nokw:>22}")
        report["splits"][name] = dict(n=len(s), critical=c, non_critical=n, groups=len(g[name]),
                                      vocab=vocab_size(s), critical_without_keyword=nokw)
    srcs = Counter((r["source"], r["urgency"]) for s in splits.values() for r in s)
    print("\nمنبع × برچسب (train+val+test):")
    for (src, u), n in sorted(srcs.items()):
        print(f"  {src:<14}{u:<14}{n}")
    print(f"\nقبل از متوازن‌سازی train: critical={raw_counts[0]}  non_critical={raw_counts[1]}")
    print(f"قرنطینه برای بازبینی پزشک: {len(quarantine)}  → {q_path}")
    if not (llm_crit or llm_non):
        print("\n⚠ داده LLM پیدا نشد. بدون جملهٔ اورژانسی «بدون کلیدواژه»، fastText فقط"
              "\n  همان ۱۰۵ عبارت را یاد می‌گیرد و به عبارت‌های جدید تعمیم نمی‌دهد.")
    nokw_train = report["splits"]["train"]["critical_without_keyword"]
    if nokw_train < 0.2 * max(1, report["splits"]["train"]["critical"]):
        print("⚠ کمتر از ۲۰٪ نمونه‌های critical در train بدون کلیدواژه‌اند؛ تعمیم ضعیف خواهد بود.")
    (out / "triage_split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print("=" * 74)
    print("بعدی: python train_fasttext.py ; سپس ارزیابی روی eval/test_grouped.jsonl")


if __name__ == "__main__":
    sys.exit(main())
