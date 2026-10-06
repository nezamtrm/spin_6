# -*- coding: utf-8 -*-
"""
import_llm_data.py — تمیزکردن و ثبت داده‌های تولیدشده با LLM برای تریاژ.

ورودی: فایل(های) متنی خام که خروجی چت را در آن‌ها paste کرده‌اید (JSONL، حتی با ``` دور آن).
خروجی (در eval/):
    llm_critical.jsonl              ← برای prepare_triage_data.py
    llm_noncritical.jsonl           ← فقط نمونه‌های «روشن و خفیف»
    llm_noncritical_borderline.jsonl← نمونه‌های non_critical که واژهٔ خطر/ابهام دارند.
                                      وارد آموزش نمی‌شوند تا پزشک برچسبشان را تأیید کند.

اجرا (ریشهٔ پروژه):
    python import_llm_data.py --critical raw_critical.txt --noncritical raw_noncritical.txt
    python import_llm_data.py --mixed eval/raw/Triage_data.jsonl            # فایل واحد با فیلد urgency
    python import_llm_data.py --mixed eval/raw/Triage_data.jsonl --critical old_c.txt --noncritical old_n.txt
چند batch هم می‌شود (همه ادغام و تکراری‌ها حذف می‌شود). --borderline keep|review (پیش‌فرض review).

محافظ جابه‌جایی فایل‌ها: اگر concept داخل فایل critical از فهرست «غیراورژانسی» باشد (یا برعکس)
متوقف می‌شود، تا برچسب‌ها وارونه وارد آموزش نشوند.
"""
import argparse, json, re, sys, importlib.util
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("triage", ROOT / "app" / "pipeline" / "triage.py")
triage = importlib.util.module_from_spec(sp); sp.loader.exec_module(triage)

KNOWN_CRITICAL = {"stroke_face_droop", "stroke_speech_arm", "chest_pressure_mi", "chest_pain_radiating",
    "severe_dyspnea", "choking_adult", "choking_child", "anaphylaxis", "seizure_active", "unresponsive_adult",
    "unresponsive_child", "poison_ingestion_child", "overdose_pills", "suicidal_statement", "heavy_bleeding_wound",
    "pregnancy_heavy_bleeding", "vomiting_blood", "blood_in_stool_large", "severe_burn", "chemical_eye",
    "head_injury_confusion", "fall_from_height", "diabetic_hypoglycemia_severe", "acute_abdomen_rigid",
    "infant_stiff_neck_rash", "heatstroke", "snake_scorpion_bite", "electric_shock", "drowning", "open_fracture"}
KNOWN_NON = {"mild_sore_throat", "common_cold", "mild_fever", "checkup", "prescription_renewal", "mild_back_pain",
    "knee_pain_walking", "toothache_mild", "gum_bleeding_brushing", "child_speech_delay", "hoarse_voice_cold",
    "period_irregular", "pregnancy_routine_question", "mild_nausea", "mild_diarrhea", "constipation_mild",
    "rash_mild", "allergy_seasonal", "fatigue", "headache_tension", "ear_blocked", "vaccination",
    "lab_test_question", "post_op_wound_ok", "muscle_pain_gym", "minor_cut", "ankle_sprain_mild", "acne",
    "insomnia", "heartburn", "back_to_normal_after_event"}

# non_critical با این واژه‌ها «مرزی» حساب می‌شود (حتی اگر LLM گفته خفیف است)
HARD = ["قفسه", "جناغ", "سینم", "سینه", "نفس", "خفگی", "تنگ", "رنگم پرید", "مرگ", "بیهوش", "فلج", "نخاع",
        "کاسه خون", "میمیر", "سیاهی", "تپش", "تشنج", "غش"]
SOFT = ["خون", "خونریزی", "تیر", "برق", "شوک", "ضعف"]
MARK = ["ترس", "فکر کردم", "فکر کنم", "وحشت", "نگران", "ولی", "اما", "قبلا", "خداروشکر", "خوشبختانه"]


# مفاهیمِ «الگو» (نه بیماری): نمونه‌های سخت عمدی. فیلتر مرزی روی آن‌ها اعمال نمی‌شود، چون
# دقیقاً همان جمله‌هایی‌اند که مدل باید یاد بگیرد (نفی علائم خطر، سابقهٔ گذشته، مزمن پایدار).
PATTERN_CONCEPTS = {"negated_critical_symptom", "past_history_no_current_symptom", "chronic_stable_complaint",
                    "negation_with_other_red_flag", "understated_critical"}


def read_raw(paths):
    rows, bad = [], 0
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = re.sub(r"^```\w*|```$", "", line.strip()).strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                t, c = r["text"].strip(), r["concept"].strip()
                assert t and c
                rows.append({"text": t, "concept": c, "urgency": r.get("urgency")})
            except Exception:
                bad += 1
                print("  ⚠ خط نامعتبر:", line[:70])
    return rows, bad


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        k = triage.normalize(r["text"])
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def is_borderline(text, concept=None):
    if concept in PATTERN_CONCEPTS:
        return False
    n = triage.normalize(text)
    return any(w in n for w in HARD) or (any(w in n for w in SOFT) and any(w in n for w in MARK))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critical", nargs="*", default=[])
    ap.add_argument("--noncritical", nargs="*", default=[])
    ap.add_argument("--mixed", nargs="*", default=[], help="JSONL با فیلدهای text, concept, urgency")
    ap.add_argument("--borderline", choices=["review", "keep"], default="review",
                    help="review: non_critical مرزی (غیرالگو) کنار گذاشته می‌شود تا پزشک ببیند")
    ap.add_argument("--out", default=str(ROOT / "eval"))
    a = ap.parse_args()
    out = Path(a.out)

    if not (a.critical or a.noncritical or a.mixed):
        sys.exit("حداقل یکی از --critical / --noncritical / --mixed لازم است.")
    crit, b1 = read_raw(a.critical) if a.critical else ([], 0)
    non, b2 = read_raw(a.noncritical) if a.noncritical else ([], 0)
    mixed, b3 = read_raw(a.mixed) if a.mixed else ([], 0)
    b2 += b3
    for r in mixed:
        u = r.get("urgency")
        if u not in ("critical", "non_critical", "low", "routine", "non-critical"):
            sys.exit(f"❌ در --mixed مقدار urgency نامعتبر: {u!r} ({r['text'][:40]})")
        (crit if u == "critical" else non).append(r)
    crit, non = dedupe(crit), dedupe(non)
    # همان متن با دو برچسب → هر دو حذف (تعارض)
    ck, nk = {triage.normalize(r["text"]) for r in crit}, {triage.normalize(r["text"]) for r in non}
    both = ck & nk
    if both:
        print(f"⚠ {len(both)} متن با هر دو برچسب (حذف شدند)")
        crit = [r for r in crit if triage.normalize(r["text"]) not in both]
        non = [r for r in non if triage.normalize(r["text"]) not in both]

    # --- محافظ جابه‌جایی فایل‌ها ---
    cc, nc = {r["concept"] for r in crit}, {r["concept"] for r in non}
    if cc & KNOWN_NON or nc & KNOWN_CRITICAL:
        sys.exit(f"❌ فایل‌ها جابه‌جا هستند!\n  در --critical مفاهیم غیراورژانسی: {sorted(cc & KNOWN_NON)}\n"
                 f"  در --noncritical مفاهیم اورژانسی: {sorted(nc & KNOWN_CRITICAL)}")
    if cc & nc:
        sys.exit(f"❌ مفهوم مشترک بین دو فایل: {sorted(cc & nc)}")

    clean, border = [], []
    for r in non:
        (border if (a.borderline == 'review' and is_borderline(r['text'], r['concept'])) else clean).append(r)

    strip = lambda rs: [{"text": r["text"], "concept": r["concept"]} for r in rs]
    crit, clean, border = strip(crit), strip(clean), strip(border)
    write(out / "llm_critical.jsonl", crit)
    write(out / "llm_noncritical.jsonl", clean)
    write(out / "llm_noncritical_borderline.jsonl", border)

    print(f"\nخطوط خراب: {b1 + b2}")
    print(f"\nCRITICAL: {len(crit)} نمونه، {len(cc)} مفهوم")
    per = Counter(r["concept"] for r in crit)
    for c, n in per.items():
        nokw = sum(1 for r in crit if r["concept"] == c and not triage.rule_based_check(r["text"]))
        print(f"   {c:<28}{n:>4}   بدون کلیدواژه (قانون نمی‌گیرد): {nokw}")
    tot_nokw = sum(1 for r in crit if not triage.rule_based_check(r["text"]))
    print(f"   → {tot_nokw}/{len(crit)} نمونه را فقط fastText می‌تواند بگیرد (این‌ها ارزشمندترین داده‌اند)")

    print(f"\nNON-CRITICAL: {len(non)} نمونه، {len(nc)} مفهوم")
    print(f"   روشن/خفیف (وارد آموزش می‌شود): {len(clean)}")
    print(f"   مرزی (کنار گذاشته شد، بازبینی پزشک): {len(border)}   [سیاست: {a.borderline}]")
    pc = sorted(c for c in (cc | nc) if c in PATTERN_CONCEPTS)
    print(f"   مفاهیم الگو (معاف از فیلتر مرزی، در prepare بین split‌ها پخش می‌شوند): {pc}")
    flagged = [r for r in clean if triage.rule_based_check(r["text"])]
    print(f"   از «روشن‌ها» که قانون اورژانس می‌داند (prepare قرنطینه‌شان می‌کند): {len(flagged)}")
    if len(cc) < 25 or len(nc) < 40:
        print(f"\n⚠ تعداد مفهوم کم است (critical={len(cc)}، non={len(nc)}؛ هدف ≥ ۲۵ و ≥ ۴۰). "
              "پرامپت را برای مفاهیم بعدی تکرار و با batchهای جدید دوباره اجرا کنید.")
    print("\nبعدی: python prepare_triage_data.py")


if __name__ == "__main__":
    main()
