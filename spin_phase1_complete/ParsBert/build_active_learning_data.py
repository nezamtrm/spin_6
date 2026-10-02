# -*- coding: utf-8 -*-
"""ساخت دادهٔ هدفمند برای جفت تخصص‌های پرخطا (حلقه active learning).

بدون مدلِ فاین‌تیون‌شده، جفت‌ها از هم‌پوشانی بالینی محتمل انتخاب شده‌اند؛
بعد از هر evaluate.py ماتریس درهم‌ریختگی جفت‌های واقعی را چاپ می‌کند
(و در data/confusion_pairs.json می‌نویسد). آن خروجی را اینجا با
--from-confusion می‌توان اولویت داد.

اجرا:
    python build_active_learning_data.py
    python prepare_data.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("data/raw/parsbert_active_learning.json")

# جفت‌های پرخطای مورد انتظار قبل از دیدن CM واقعی.
# هر نمونه chief_complaint یکتاست تا تقویت follow-up در prepare_data تکراری نشود.
TARGETED = [
    # جراحی عمومی ↔ زنان (درد شکم/لگن)
    ("al_gs_gyn_01", "جراحی عمومی", "درد شکمم سمت راست پایینه و تبم هم هست، راه رفتن سخت شده"),
    ("al_gs_gyn_02", "جراحی عمومی", "نزدیک ناف درد شدید دارم که به شانه راست میزنه بعد غذای چرب"),
    ("al_gs_gyn_03", "جراحی عمومی", "فتق کشاله رانم بیرون زده و درد میکنه وقتی سرفه میکنم"),
    ("al_gs_gyn_04", "زنان و زایمان", "درد لگن دارم همزمان با پریود و لکه بینی قهوه‌ای"),
    ("al_gs_gyn_05", "زنان و زایمان", "زیر دلم تیر میکشه وسط سیکل، قبلا کیست تخمدان داشتم"),
    ("al_gs_gyn_06", "زنان و زایمان", "باردارم و درد خفیف پایین شکم دارم بدون تب"),
    # جراحی عمومی ↔ پزشک عمومی
    ("al_gs_gp_01", "جراحی عمومی", "بعد از عمل آپاندیس زخمم قرمز و گرم شده و ترشح داره"),
    ("al_gs_gp_02", "جراحی عمومی", "توده سفت تو گردنم سمت راست بزرگ شده و درد میکنه"),
    ("al_gs_gp_03", "پزشک عمومی", "درد عمومی شکم دارم بعد نفخ، بدون تب و بدون استفراغ"),
    ("al_gs_gp_04", "پزشک عمومی", "یبوست گاه‌گاهی دارم، خون تو مدفوع ندیدم، برای رژیم میپرسم"),
    # ارتوپدی ↔ پزشک عمومی
    ("al_ortho_gp_01", "ارتوپدی", "زانوم موقع بالا رفتن از پله قفل میکنه و صدا میده"),
    ("al_ortho_gp_02", "ارتوپدی", "کمردرد سیاتیکی دارم که به پام تیر میکشه"),
    ("al_ortho_gp_03", "ارتوپدی", "بعد زمین خوردن مچ دستم ورم کرده و نمیتونم مشت کنم"),
    ("al_ortho_gp_04", "پزشک عمومی", "درد بدن و کوفتگی بعد آنفولانزا دارم، مفصل خاصی قفل نشده"),
    ("al_ortho_gp_05", "پزشک عمومی", "کمردرد خفیف از نشستن طولانی پشت میز، با حرکت بهتر میشه"),
    # دندانپزشکی ↔ گفتار درمانی
    ("al_dent_sp_01", "دندانپزشکی", "دندون آسیام ضربان داره و به گرما حساسه، لثه‌م ورم کرده"),
    ("al_dent_sp_02", "دندانپزشکی", "بعد عصب‌کشی دندونم هنوز درد شبانه دارم"),
    ("al_dent_sp_03", "گفتار درمانی", "کودکم حرف میزنه ولی حروف س و ش رو قاتی میکنه"),
    ("al_dent_sp_04", "گفتار درمانی", "بعد سکته حرف زدنم بریده بریده شده و قورت دادن سخت"),
    ("al_dent_sp_05", "دندانپزشکی", "فک موقع باز کردن دهان کلیک میکنه و درد داره نه مشکل تلفظ"),
    ("al_dent_sp_06", "گفتار درمانی", "لکنت دارم وقتی ارائه میدم، دندون‌درد ندارم"),
    # گفتار درمانی ↔ پزشک عمومی
    ("al_sp_gp_01", "گفتار درمانی", "پسر سه ساله‌ام هنوز جمله نمیسازه و فقط اشاره میکنه"),
    ("al_sp_gp_02", "گفتار درمانی", "صدا م گرفته و خش‌دار شده ماه‌هاست، نه سرماخوردگی حاد"),
    ("al_sp_gp_03", "پزشک عمومی", "گلودرد و گرفتگی صدای دو روزه بعد سرماخوردگی"),
    ("al_sp_gp_04", "پزشک عمومی", "سرفه و خشونت صدا از دیروز، تب خفیف هم دارم"),
    # زنان ↔ پزشک عمومی
    ("al_gyn_gp_01", "زنان و زایمان", "دو ماهه پریود نشدم تست بارداری مثبت بوده تهوع صبحگاهی دارم"),
    ("al_gyn_gp_02", "زنان و زایمان", "خونریزی نامنظم بین دو دوره و درد هنگام رابطه"),
    ("al_gyn_gp_03", "پزشک عمومی", "تهوع بعد غذای بیرون، پریودم منظمه و باردار نیستم"),
    ("al_gyn_gp_04", "پزشک عمومی", "دل‌پیچه و تب خفیف گوارشی، درد لگن چرخه‌ای ندارم"),
    # ارتوپدی ↔ جراحی عمومی (تروما/توده)
    ("al_ortho_gs_01", "ارتوپدی", "استخون ترقوه‌م بعد ضربه درد شدید داره و دستو بالا نمیبرم"),
    ("al_ortho_gs_02", "جراحی عمومی", "کیست چربی پشت گردن بزرگ شده و گاهی عفونت میکنه"),
    ("al_ortho_gs_03", "جراحی عمومی", "بواسیر دردناک دارم خون روشن روی کاغذ توالت"),
]


def to_records(pairs_filter: set[tuple[str, str]] | None = None) -> list[dict]:
    recs = []
    for i, spec, text in TARGETED:
        if pairs_filter:
            # اگر CM داده شد، فقط نمونه‌هایی که تخصص‌شان در جفت پرخطا آمده نگه دار
            if spec not in {a for a, _ in pairs_filter} | {b for _, b in pairs_filter}:
                continue
        recs.append({
            "id": i,
            "specialty": spec,
            "chief_complaint": text,
            "follow_up_question": "از کی شروع شده و چه چیزی بهتر یا بدترش میکنه؟",
            "answer": "نمونه هدفمند active learning برای مرز تخصص‌ها؛ پاسخ آموزشی است نه تجویز.",
        })
    return recs


def load_confusion_pairs(path: Path, min_count: int = 2) -> set[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = set()
    for item in data.get("pairs", []):
        if item.get("count", 0) >= min_count:
            pairs.add((item["gold"], item["pred"]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-confusion", default=None,
                    help="مسیر data/confusion_pairs.json خروجی evaluate.py")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    pairs = None
    if args.from_confusion:
        p = Path(args.from_confusion)
        if p.exists():
            pairs = load_confusion_pairs(p)
            print(f"جفت‌های پرخطا از CM: {len(pairs)}")
            for a, b in sorted(pairs):
                print(f"  {a} -> {b}")
        else:
            print(f"فایل CM پیدا نشد ({p})؛ همه نمونه‌های هدفمند نوشته می‌شود.")

    recs = to_records(pairs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in existing}
    for r in recs:
        by_id[r["id"]] = r
    merged = list(by_id.values())
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(recs)} targeted samples, file total {len(merged)} ({out})")


if __name__ == "__main__":
    main()
