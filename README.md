# SPIN — سرویس مشاوره پزشکی فارسی (Q&A per-body-part)

سرویس داخلی شبکه (`ai-service.internal`) که متن فارسی بیمار را می‌گیرد،
تخصص مرتبط را از بین ۶ تخصص تشخیص می‌دهد، در صورت اورژانس فوراً هشدار
می‌دهد، سوال تکمیلی برای تکمیل یک پرونده ابتدایی می‌پرسد، پاسخ را با
گارد ایمنی (بدون دوز دارو/PII) استریم می‌کند و برای سوالات تکراری از کش
معنایی استفاده می‌کند. قرارداد کامل API: [`OpenApi_Spin.yaml`](OpenApi_Spin.yaml).

> **وضعیت:** فاز ۱ — کد و پایپلاین کامل و تست‌شده (syntax/logic)؛ ترین
> واقعی مدل‌ها و محتوای RAG هنوز باقی مانده (چک‌لیست پایین همین فایل).

## ۶ تخصص فعال

| # | نام فارسی | key (در enum قرارداد) |
|---|---|---|
| ۱ | زنان و زایمان | `gynecology` |
| ۲ | ارتوپدی | `orthopedics` |
| ۳ | دندانپزشکی | `dentistry` |
| ۴ | گفتار درمانی | `speech_therapy` |
| ۵ | جراحی عمومی | `general_surgery` |
| ۶ | پزشک عمومی | `general_practice` |

تعریف رسمی + کلیدواژه‌ها: [`ParsBert/specialties.py`](ParsBert/specialties.py).

## معماری پایپلاین (ترتیب واقعی در `app/api/chat.py`)

```
درخواست کاربر
   │
   ▼
[۴] روتر تخصص (ParsBERT)  ──► رویداد "start" (specialty, confidence)
   │
   ▼
[۱] گارد ورودی (PII سانسور می‌شود، نه بلاک)
   │
   ▼
[۲] تریاژ اورژانس (rule+hazm negation، سپس fastText fallback، <100ms)
   │  اگر بحرانی بود ──► رویداد "emergency" و پایان
   ▼
[۵] کش معنایی (Qdrant، بر اساس شباهت با تخصص فعلی)
   │  اگر hit ──► رویداد "cached" + بازپخش پاسخ ذخیره‌شده و پایان
   ▼
[۵.۵] RAG (Qdrant، فقط کالکشن همان تخصص) + [۶.۵] پرونده ابتدایی بیمار
   │
   ▼
[۶] ساخت پرامپت (سوال تکمیلی در نوبت اول، ارجاع در نوبت‌های بعد)
   │
   ▼
[۷+۸] استریم از vLLM + گارد خروجی (دوز دارو/PII بلاک یا سانسور می‌شود)
   │
   ▼
رویدادهای citation ← structured (با patient_record) ← done
```

پوشه‌ها:

```
app/
  api/          endpointهای FastAPI (chat, health) + auth stub (deps.py)
  core/         تنظیمات (config.py)، لاگ، SSE helper، امنیت (خالی - فاز بعد)
  pipeline/     منطق اصلی: triage, guard_input, guard_output, router,
                cache, retriever, intake, inference, prompt, structured
  llm/          کلاینت خام HTTP به vLLM
  rag/          embedder (ParsBERT خام)، qdrant_client، reranker
  observability/ متریک‌های Prometheus
ParsBert/       آموزش/ارزیابی/پیش‌بینی مدل روتر تخصص (جدا از app، بدون FastAPI)
docker/         Dockerfile سرویس + compose (vLLM+Qdrant+Prometheus+Grafana)
eval/           همه دیتاست‌های ارزیابی (تریاژ، روتر، گارد، کش، RAG)
scripts/        ابزارهای مانیتورینگ/بار/chaos/پاک‌سازی کش
```

## شروع سریع (لوکال، بدون GPU)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # سرویس + روتر (torch/transformers)
pip install -r ParsBert/requirements.txt # اگر جدا هم می‌خواهید ترین کنید

cp .env.example .env   # و مقداردهی کنید (پایین توضیح کامل هر متغیر)
```

آموزش دو مدل (تفصیل کامل در [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md)):
```bash
cd ParsBert && python prepare_data.py && python train.py --epochs 4 --batch-size 16 && cd ..
python prepare_triage_data.py && python train_fasttext.py
```

بالا آوردن زیرساخت (vLLM نیاز به GPU واقعی دارد؛ برای تست لوکال بدون GPU
از یک مدل کوچک استفاده کنید - `Qwen/Qwen2.5-1.5B-Instruct` مثلا):
```bash
docker compose -f docker/compose.yml up -d   # vllm + qdrant + prometheus + grafana
uvicorn app.main:app --reload --port 8080
```

تست end-to-end:
```bash
python scripts/smoke_test_pipeline.py
```

## متغیرهای محیطی

همه در [`.env.example`](.env.example) با توضیح؛ خلاصه:

| متغیر | برای چی |
|---|---|
| `VLLM_API_BASE`, `VLLM_MODEL_NAME` | آدرس/نام مدل vLLM |
| `PARSBERT_MODEL_DIR` | مسیر مدل فاین‌تیون‌شده روتر تخصص (نه داخل ایمیج - نگاه کنید به MODEL_WEIGHTS.md) |
| `TRIAGE_MODEL_PATH` | مسیر `triage_model.bin` |
| `QDRANT_URL`, `QDRANT_COLLECTION_PREFIX` | زیرساخت RAG/کش |
| `CACHE_SIMILARITY_THRESHOLD` | آستانه hit کش معنایی (پیش‌فرض محافظه‌کارانه `0.93`) |
| `RAG_TOP_K`, `RERANK_VECTOR_WEIGHT`, `RERANK_LEXICAL_WEIGHT` | تنظیمات بازیابی - با `evaluate_retrieval.py` تیون شوند، حدس نزنید |
| `SERVICE_JWT_SECRET` | فعلا stub - نگاه کنید به «شکاف‌های شناخته‌شده» |

## ابزارهای ارزیابی — کدام کدام را می‌سنجد

| اسکریپت | چه چیزی را می‌سنجد | دیتای لازم |
|---|---|---|
| `ParsBert/evaluate.py --csv ParsBert/data/test.csv` | روتر تخصص: top-1/top-3 accuracy + macro-F1 + confusion matrix | `ParsBert/data/test.csv` (خودکار از `prepare_data.py`) |
| `evaluate_triage.py` | فقط مدل fastText تریاژ (بدون لایه قانون‌محور) | `eval/test.jsonl`, `eval/golden_emergency.jsonl` |
| `evaluate_triage_pipeline.py [--rule-only]` | تابع واقعی production `is_emergency()` (قانون+hazm+fastText) | همان + `eval/hard_negatives.jsonl` |
| `evaluate_guard.py` | گارد ورودی (`check_input`) و خروجی (`guarded_token_stream`) | `eval/redteam_input.jsonl`, `eval/redteam_output.jsonl` |
| `evaluate_cache_threshold.py [--thresholds ...]` | تیون `CACHE_SIMILARITY_THRESHOLD` با hit-rate/false-hit-rate واقعی | `eval/cache_pairs.jsonl` |
| `evaluate_retrieval.py [--sweep-top-k/--sweep-weights]` | recall@k و MRR بازیابی RAG؛ تیون `RAG_TOP_K`/وزن‌های reranker | `eval/gold_retrieval.jsonl` + محتوای واقعی در Qdrant |
| `scripts/smoke_test_pipeline.py` | تست end-to-end چهار سناریو (تخصص، پرونده ابتدایی، اورژانس، کش) روی سرویس زنده | سرویس + vLLM بالا باشند |
| `scripts/load_test.py --concurrency N` | تأخیر p50/p95/p99 زیر بار هم‌زمان | سرویس زنده |
| `scripts/test_sse_disconnect.py` | رفتار سرور هنگام قطع کلاینت وسط استریم | سرویس زنده |
| `scripts/chaos_test.md` | رفتار fail-safe واقعی Qdrant/vLLM خاموش (راهنمای دستی) | `docker compose` |
| `log_eval_result.py --task ... --metric k=v` | ثبت نسخه‌دار هر نتیجه ارزیابی (`eval_history.jsonl`) | — |

## مانیتورینگ

- Prometheus: `docker/prometheus/prometheus.yml` (اسکریپ `/metrics`)
- Grafana: `docker/grafana/dashboards/spin-overview.json` (۱۲ پنل - نرخ درخواست، تریاژ p99، اورژانس، توزیع تخصص، cache hit rate، ...) روی `:3000`
- بدون Grafana: `python scripts/monitor.py --once` (چک سریع `/health`+`/ready`+`/metrics` از ترمینال)
