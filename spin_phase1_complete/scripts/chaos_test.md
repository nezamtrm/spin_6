# تست خرابی عمدی (Chaos Test) — راهنمای دستی

هدف: ادعاهای «fail-safe» که در کامنت‌های کد نوشته شده‌اند (مثلا در
`router.py`, `retriever.py`, `cache.py`, `qdrant_client.py`) را عملا با
خاموش‌کردن واقعی یک سرویس امتحان کنید — نه فقط خواندن کد و باور کردنش.

پیش‌نیاز: `docker compose -f docker/compose.yml up -d` از قبل بالا باشد
و سرویس FastAPI (`uvicorn app.main:app`) هم در حال اجرا باشد.

## سناریو ۱ — Qdrant را خاموش کن (باید کم‌خطرترین باشد)
```bash
docker compose -f docker/compose.yml stop qdrant
curl -s http://localhost:8080/ready | python3 -m json.tool
```
**انتظار:** `components.qdrant: "down"`, `components.cache: "down"` (چون کش
هم روی Qdrant است)، ولی `status: "ready"` کلی (چون فقط vLLM سخت‌گیر است -
نگاه کنید به `app/api/health.py::readiness_check`).

بعد یک چت واقعی بزنید:
```bash
python3 -c "
import requests, uuid, hashlib
r = requests.post('http://localhost:8080/v1/medical/chat', json={
    'request_id': str(uuid.uuid4()), 'session_id': str(uuid.uuid4()),
    'user_hash': hashlib.sha256(b'chaos').hexdigest(),
    'query': 'دندونم درد میکنه'}, headers={'Authorization':'Bearer x'}, stream=True)
for line in r.iter_lines():
    if line: print(line.decode())
"
```
**انتظار:** جریان کامل طی می‌شود، فقط رویداد `citation` خالی (`sources: []`)
می‌آید و هیچ‌وقت `cached` نمی‌بینید (چون کش هم غیرفعال است) — نه خطای ۵۰۰،
نه timeout طولانی.

```bash
docker compose -f docker/compose.yml start qdrant   # برگردان
```

## سناریو ۲ — vLLM را خاموش کن (باید سخت‌گیر باشد، بر خلاف سناریو ۱)
```bash
docker compose -f docker/compose.yml stop vllm
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/ready
```
**انتظار:** کد `503` (نه `200`) — چون `is_ready = vllm_up` در `health.py`.

همان درخواست چت بالا را دوباره بزنید.
**انتظار:** درخواست نه تا ابد آویزان می‌ماند نه با یک traceback زشت قطع
می‌شود؛ باید طبق `aiohttp.ClientTimeout(total=30)` در `app/llm/vllm_client.py`
حداکثر ~۳۰ ثانیه بعد با یک خطای قابل‌فهم (نه hang بی‌نهایت) تمام شود. اگر
بیشتر از ۳۰ ثانیه آویزان ماند، یعنی timeout درست اعمال نمی‌شود - این را
به‌عنوان باگ گزارش کنید.

```bash
docker compose -f docker/compose.yml start vllm
```

## سناریو ۳ — هر دو با هم خاموش، حین یک درخواست در حال جریان
سناریوی واقعی‌تر: یک چت را شروع کنید، وسط استریم (بعد از چند توکن) هم
vLLM هم Qdrant را خاموش کنید. چک کنید سرور کرش نمی‌کند (پروسه uvicorn
هنوز زنده است، بقیه درخواست‌های جدید هنوز جواب می‌گیرند - حتی اگر خودشان
هم به همان خطا بخورند).

## گزارش
برای هر سناریو یک خط در `eval_history.jsonl` یا جای مشابه ثبت کنید: تاریخ،
نسخه کد (`git rev-parse --short HEAD`)، و آیا انتظار برآورده شد یا نه —
دقیقا همان الگوی `log_eval_result.py`، حتی اگر این‌جا عدد نیست بلکه
pass/fail است.
