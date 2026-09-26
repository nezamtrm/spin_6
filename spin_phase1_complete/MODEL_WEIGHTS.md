# وزن‌های مدل‌های ترین‌شده - کجا نگه‌داری شوند

این ریپو دقیقا دو مدل fine-tune/train شده تولید می‌کند. **هیچ‌کدام نباید
commit شوند** (هر دو صدها مگابایت تا چند گیگابایت‌اند؛ commit کردن باینری
به این بزرگی، `git clone` را کند و بعد از چند بار train مجدد عملا
غیرقابل‌استفاده می‌کند - گیت برای دیف متن ساخته شده، نه ورژن‌بندی باینری).

| مدل | تولید می‌شود توسط | مسیر خروجی | حجم تقریبی |
|---|---|---|---|
| روتر تخصص (ParsBERT فاین‌تیون‌شده) | `ParsBert/train.py` | `ParsBert/models/parsbert-specialty/final/` | ~۴۵۰ مگابایت |
| تریاژ اورژانس (fastText) | `train_fasttext.py` | `app/pipeline/triage_model.bin` | ~۵۰ تا ۲۰۰ مگابایت (بسته به epoch/dim) |

هر دو مسیر در `.gitignore` اضافه شده‌اند.

## کجا نگه‌داری شوند (به ترتیب اولویت برای این پروژه)

چون طبق `OpenApi_Spin.yaml` این یک «سرویس داخلی شبکه» است
(`ai-service.internal`)، محتمل‌ترین گزینه یک آبجکت‌استور/رجیستری داخلی
است، نه سرویس عمومی ابری:

1. **آبجکت‌استور داخلی سازمان (S3-compatible/MinIO) - توصیه اصلی.**
   یک bucket مثلا `spin-models` بسازید با دو کلید نسخه‌دار:
   ```
   s3://spin-models/parsbert-specialty/2026-09-24_v1/  (کل پوشه final/)
   s3://spin-models/triage-fasttext/2026-09-24_v1/triage_model.bin
   ```
   در Dockerfile/entrypoint یا یک اسکریپت `scripts/fetch_models.sh`،
   دقیقا همین دو مسیر را قبل از `uvicorn app.main:app` دانلود کنید.
   مزیت: نسخه‌بندی صریح، rollback ساده (فقط تگ نسخه در `.env` عوض شود)،
   بدون وابستگی به سرویس بیرون از شبکه داخلی.

2. **Hugging Face Hub، یک private repo داخل سازمان.**
   اگر HF Hub سازمانی (Enterprise/self-hosted) دارید یا حتی HF Hub عمومی
   با repo خصوصی قابل‌قبول است: `huggingface-cli upload your-org/spin-parsbert-specialty ParsBert/models/parsbert-specialty/final`.
   مزیت: ابزار `from_pretrained(...)` مستقیم با آن کار می‌کند (کد
   `ParsBert/predict.py` تقریبا بدون تغییر، فقط `PARSBERT_MODEL_DIR` به
   جای مسیر لوکال به نام repo داده می‌شود).

3. **Git LFS - فقط اگر گزینه ۱ و ۲ ممکن نیست.**
   ریپوی گیت را کند می‌کند و هزینه ذخیره‌سازی LFS دارد؛ آخرین گزینه.

**هرگز:** در ایمیل/Slack/Google Drive شخصی رد و بدل نشود؛ نسخه‌ای که در
production است باید همیشه قابل ردیابی به یک commit مشخص از `ParsBert/`
و یک نسخه دیتاست مشخص باشد (خروجی `prepare_data.py`/`prepare_triage_data.py`
غیر-deterministic نیست چون seed=42 است، ولی خودِ `data/raw/parsbert_finetune_samples.json`
باید نسخه‌دار بماند).

## چطور به سرویس وصلشان کنیم (بدون تغییر کد، فقط env)

`app/core/config.py` از قبل برای همین طراحی شده:

```bash
# .env در production - به یک ولوم/مسیر دانلودشده در مرحله deploy اشاره می‌کند،
# نه به چیزی داخل خودِ ایمیج/ریپو
PARSBERT_MODEL_DIR=/mnt/models/parsbert-specialty/final
```

برای fastText مشابه، یک ENV اضافه کنید (`settings.TRIAGE_MODEL_PATH` از
قبل هست، فقط مقدارش را به مسیر ولوم بدهید):
```bash
TRIAGE_MODEL_PATH=/mnt/models/triage_model.bin
```

نمونه اسکریپت دانلود قبل از استارت (Dockerfile ENTRYPOINT یا k8s initContainer):
```bash
#!/bin/sh
# scripts/fetch_models.sh
set -e
mkdir -p /mnt/models/parsbert-specialty/final
aws s3 cp --recursive s3://spin-models/parsbert-specialty/${MODEL_VERSION}/ /mnt/models/parsbert-specialty/final/
aws s3 cp s3://spin-models/triage-fasttext/${MODEL_VERSION}/triage_model.bin /mnt/models/triage_model.bin
```

`MODEL_VERSION` هم یک ENV دیگر می‌شود - یعنی rollback = فقط عوض کردن یک
مقدار و ری‌استارت، نه rebuild ایمیج.

## مدل‌هایی که نیازی به این کار ندارند
- **Qwen2.5-32B-AWQ (vLLM):** فاین‌تیون نشده، مستقیم از HF Hub می‌آید؛
  `docker/compose.yml` همین الان با mount کردن `~/.cache/huggingface`
  از دانلود مجدد در هر بار بالا آمدن جلوگیری می‌کند - کار اضافه‌ای لازم نیست.
- **مدل پایه امبدینگ RAG (`app/rag/embedder.py`):** همان بک‌بون خام
  ParsBERT است (فاین‌تیون نشده)، آن هم از HF Hub cache می‌آید.
