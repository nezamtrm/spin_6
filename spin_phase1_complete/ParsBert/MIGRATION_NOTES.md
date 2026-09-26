# راهنمای مهاجرت به ۶ تخصص

این فایل خلاصه‌ی همه‌ی تغییرات لازم برای تبدیل پروژه از ۱۴ تخصص به ۶ تخصص
(زنان و زایمان، ارتوپدی، دندانپزشکی، گفتار درمانی، جراحی عمومی، پزشک عمومی) است.

## ۱. specialties.py  (بازنویسی کامل)
- لیست SPECIALTIES از ۱۴ آیتم به ۶ آیتم تغییر کرد.
- ۲ تخصص قدیمی نگه داشته شدند: زنان و زایمان (gynecology)، ارتوپدی (orthopedics).
- ۴ تخصص جدید اضافه شدند: دندانپزشکی (dentistry)، گفتار درمانی (speech_therapy)،
  جراحی عمومی (general_surgery)، پزشک عمومی (general_practice).
- خط پایانی: `assert NUM_LABELS == 6` (به‌جای ۱۴).

## ۲. data/raw/parsbert_finetune_samples.json  (فایل جدید)
- دیتای نمونه‌ای که فرستادید را دقیقاً در همین مسیر قرار دهید:
  `ParsBert/data/raw/parsbert_finetune_samples.json`

## ۳. prepare_data.py  (اسکریپت جدید، جایگزین نقش build_dataset.py)
- داده واقعی JSON (۵ تخصص) را به فرمت text,label تبدیل می‌کند.
- برای «پزشک عمومی» که داده واقعی ندارد، با الگوهای TEMPLATES × کلیدواژه‌های
  GDP در specialties.py داده مصنوعی می‌سازد.
- دو منبع را ترکیب، دیتای تکراری را حذف و به‌صورت لایه‌ای (stratified) به
  train/validation/test تقسیم می‌کند.
- build_dataset.py قدیمی همچنان در پروژه باقی مانده (برای مرجع/بازتولید
  bootstrap خالص در صورت نیاز)، ولی دیگر اسکریپت اصلیِ ساخت دیتاست نیست.

## ۴. api.py
- `top_k: int = Field(3, ge=1, le=14)` → `le=6` در دو کلاس PredictIn و BatchIn.

## ۵. evaluate.py
- لیست SAMPLES (۱۴ جمله‌ی دست‌نویس قدیمی) با ۶ جمله‌ی جدید متناسب با
  تخصص‌های جدید جایگزین شد.

## ۶. my_tests.txt
- جملات تست نمونه با ۶ جمله متناسب با تخصص‌های جدید جایگزین شدند.

## ۷. مدل فاین‌تیون‌شده قبلی را حذف/دوباره آموزش دهید
- اگر پوشه‌ی `models/parsbert-specialty/final/` از آموزش قبلی (۱۴ کلاس) دارید،
  **قابل استفاده مجدد نیست** — چون سر طبقه‌بندی (classification head) برای
  ۱۴ کلاس ساخته شده و با ۶ کلاس جدید ناسازگار است.
  آن پوشه را پاک کنید و از صفر با train.py آموزش دهید.

## ۸. README.md
- جدول «۶ تخصص» را به‌جای جدول ۱۴‌تایی قرار دهید (نمونه در انتهای همین فایل).
- بخش «استفاده از دیتای واقعی» را به‌روزرسانی کنید تا به prepare_data.py
  اشاره کند، نه build_dataset.py.

## ۹. results.csv
- فایل خروجی نمونه‌ی قدیمی (مخصوص ۱۴ تخصص) حذف شد؛ با اجرای
  `python predict.py --file my_tests.txt --out-csv results.csv` دوباره ساخته می‌شود.

## ۱۰. train.py / predict.py / text_utils.py / requirements.txt
- **بدون نیاز به تغییر.** این فایل‌ها به‌صورت پویا از specialties.py می‌خوانند
  (NUM_LABELS، LABELS، ID2LABEL و ...) و خودکار با ۶ کلاس سازگار می‌شوند.

---

## گردش کار کامل بعد از این تغییرات

```bash
# ۱) فایل JSON نمونه را در data/raw/ قرار دهید (اگر قرار ندادید)
mkdir -p data/raw
cp /path/to/parsbert_finetune_samples.json data/raw/

# ۲) ساخت train/validation/test.csv نهایی (داده واقعی + مصنوعیِ GP)
python prepare_data.py

# ۳) فاین‌تیون از صفر
python train.py --epochs 4 --batch-size 16

# ۴) تست سریع
python predict.py "دو ماهه پریود نشدم و حالت تهوع دارم"
python evaluate.py
```

## جدول جدید ۶ تخصص (برای README.md)

| # | تخصص | key |
|---|------|-----|
| ۱ | زنان و زایمان | `gynecology` |
| ۲ | ارتوپدی | `orthopedics` |
| ۳ | دندانپزشکی | `dentistry` |
| ۴ | گفتار درمانی | `speech_therapy` |
| ۵ | جراحی عمومی | `general_surgery` |
| ۶ | پزشک عمومی | `general_practice` |
