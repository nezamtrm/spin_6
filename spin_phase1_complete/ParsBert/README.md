# تشخیص تخصص پزشکی از متن فارسی با ParsBERT

متن فارسی بیمار را می‌گیرد و مرتبط‌ترین تخصص را از بین **۶ تخصص** با **درصد اطمینان** برمی‌گرداند.

## ۶ تخصص

| # | تخصص | key |
|---|------|-----|
| ۱ | زنان و زایمان | `gynecology` |
| ۲ | ارتوپدی | `orthopedics` |
| ۳ | دندانپزشکی | `dentistry` |
| ۴ | گفتار درمانی | `speech_therapy` |
| ۵ | جراحی عمومی | `general_surgery` |
| ۶ | پزشک عمومی | `general_practice` |

## ساختار پروژه

```
specialties.py     تعریف ۶ تخصص + توصیف + واژگان کلیدی (تنها منبع برچسب‌ها)
text_utils.py      نرمال‌سازی متن فارسی (ی/ک عربی، اعراب، ارقام، نیم‌فاصله)
prepare_data.py    ترکیب داده واقعی JSON (data/raw) + داده مصنوعی پزشک عمومی → train/validation/test.csv
build_dataset.py   (قدیمی/مرجع) ساخت دیتاست کاملاً مصنوعی از الگوهای شکایت بیمار × واژگان
train.py           فاین‌تیون ParsBERT (HooshvareLab/bert-fa-base-uncased)
predict.py         پیش‌بینی + درصد اطمینان (CLI و کلاس قابل import)
evaluate.py        ارزیابی روی جملات دست‌نویس یا CSV دلخواه
api.py             سرویس REST با FastAPI
```

## نصب

```bash
pip install -r requirements.txt
```

## اجرا

### ۱. بدون آموزش (zero-shot) — بلافاصله کار می‌کند

اگر مدل فاین‌تیون‌شده موجود نباشد، `predict.py` خودکار به حالت zero-shot می‌رود:
امبدینگ ParsBERT از متن با بردار نمایندهٔ هر تخصص (نام + توصیف + کلیدواژه‌ها)
مقایسه می‌شود و با تطبیق واژگانی ترکیب می‌گردد.

```bash
python predict.py "چند روزه قفسه سینه ام تیر میکشه و تپش قلب دارم"
```

```
تخصص پیشنهادی: قلب و عروق   (80.80%  اطمینان)

رتبه‌بندی (top-3):
  1. قلب و عروق                 ###################.....  80.80%
  2. ریه و مجاری تنفسی          #.......................   2.32%
  3. مغز و اعصاب                ........................   1.77%
```

### ۲. با فاین‌تیون — دقت بالاتر

```bash
python prepare_data.py           # ساخت data/train.csv و validation/test از داده واقعی + مصنوعیِ GP
python train.py --epochs 4 --batch-size 16
python predict.py "دو ماهه پریود نشدم و حالت تهوع دارم"
```

مدل در `models/parsbert-specialty/final/` ذخیره می‌شود و `predict.py` خودکار
آن را پیدا و استفاده می‌کند.

### ۳. خروجی JSON

```bash
python predict.py --json --top-k 5 "قند خونم بالا رفته و خیلی تشنه میشم"
```

```json
{
  "text": "قند خونم بالا رفته و خیلی تشنه میشم",
  "top": {
    "specialty": "غدد، متابولیسم و دیابت",
    "key": "endocrinology",
    "confidence": 0.9731,
    "percent": 97.31
  },
  "ranking": [ ... ],
  "mode": "fine-tuned",
  "certain": true
}
```

### ۴. استفاده به‌عنوان کتابخانه

```python
from predict import SpecialtyClassifier

clf = SpecialtyClassifier()              # یک بار بارگذاری
res = clf.predict("گلوم درد میکنه و لوزه هام ورم کرده", top_k=3)

print(res.top.specialty)      # گوش، حلق و بینی
print(res.top.percent)        # 96.4
print(res.certain)            # True  (اطمینان >= ۳۵٪)

for p in res.ranking:
    print(p.specialty, p.percent)
```

### ۵. سرویس REST

```bash
uvicorn api:app --port 8000
```

| مسیر | متد | توضیح |
|------|-----|-------|
| `/health` | GET | وضعیت و حالت مدل |
| `/specialties` | GET | فهرست ۱۴ تخصص |
| `/predict` | POST | `{"text": "...", "top_k": 3}` |
| `/predict/batch` | POST | `{"texts": ["...", "..."], "top_k": 3}` |

## استفاده از دیتای واقعی

`prepare_data.py` داده واقعی (`data/raw/parsbert_finetune_samples.json`) را
برای ۵ تخصص می‌خواند و فقط برای «پزشک عمومی» که داده واقعی ندارد، از الگوهای
شکایت بیمار × واژگان کلیدی، داده مصنوعی می‌سازد. اگر بعداً برای پزشک عمومی
هم داده واقعی جمع کردید، آن را به همان فرمت JSON اضافه کنید یا مستقیماً
`data/train.csv` را با فایل خودتان با همین ساختار عوض کنید:

```csv
text,label
"از دیشب قلبم تند میزنه",قلب و عروق
"سردرد شدید دارم",مغز و اعصاب
```

مقدار `label` باید دقیقاً یکی از نام‌های فارسی `specialties.py` باشد. سپس دوباره
`python train.py` را اجرا کنید.

## افزودن یا تغییر تخصص‌ها

فقط `specialties.py` را ویرایش کنید (اضافه/حذف یک `Specialty`) و `assert` انتهای
فایل را با تعداد جدید هماهنگ کنید؛ همه بخش‌های دیگر خودکار تطبیق می‌یابند.

## نکات فنی

- **آستانه اطمینان**: مقدار `LOW_CONFIDENCE = 0.35` در `predict.py`. زیر آن،
  `certain=False` برمی‌گردد تا در محصول واقعی به پزشک عمومی ارجاع داده شود.
- **مدل پایه**: پیش‌فرض `HooshvareLab/bert-fa-base-uncased`. برای نسخهٔ ZWNJ:
  `--model HooshvareLab/bert-fa-zwnj-base` (نیم‌فاصله خودکار حفظ می‌شود).
- **کالیبراسیون**: softmax اعتماد‌به‌نفس بیش‌ازحد می‌دهد. برای درصدهای واقع‌بینانه
  می‌توان روی مجموعه validation یک temperature scaling ساده اعمال کرد.
- **هشدار پزشکی**: این ابزار فقط برای *مسیریابی* بیمار به تخصص مناسب است و
  تشخیص پزشکی محسوب نمی‌شود.
