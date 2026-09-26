# docker/api.Dockerfile
FROM python:3.11-slim

WORKDIR /srv

# فقط requirements.txt را اول کپی می‌کنیم تا لایهٔ Docker cache شود؛
# با این کار، تغییر کد اپلیکیشن باعث نصب دوبارهٔ کل وابستگی‌ها نمی‌شود.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY OpenApi_Spin.yaml ./OpenApi_Spin.yaml

# فقط کدِ ParsBert (نه دیتا، نه وزن‌ها - آن‌ها طبق MODEL_WEIGHTS.md در
# دیپلوی از بیرون mount/دانلود می‌شوند، نه داخل ایمیج). router.py دقیقا
# همین سه فایل را از ParsBert/ ایمپورت می‌کند.
COPY ParsBert/specialties.py ParsBert/predict.py ParsBert/text_utils.py ./ParsBert/

EXPOSE 8080

# --host 0.0.0.0 اجباری است چون داخل کانتینر localhost معنای دیگری دارد؛
# بدون آن، هیچ ترافیکی از بیرون کانتینر به uvicorn نمی‌رسد.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
