"""
scripts/smoke_test_pipeline.py

تست دودکش (end-to-end) فاز ۱ - سرویس (vLLM + FastAPI) باید از قبل روی
همین ماشین بالا باشد (نگاه کنید به راهنمای اجرا در گفتگو - uvicorn روی
پورت ۸۰۸۰، vLLM روی پورت ۸۰۰۰).

۴ سناریو را چک می‌کند و با assert فورا خطا می‌دهد اگر چیزی درست نبود:
  ۱) تشخیص تخصص در رویداد start
  ۲) پرونده ابتدایی (patient_record): نوبت اول in_progress و خالی،
     نوبت دوم (بعد از یک رفت‌وبرگشت سوال/جواب) completed با یک qa_pair
  ۳) تشخیص اورژانس (باید emergency بدهد، نه جریان عادی)
  ۴) کش (همان سوال دوباره -> cached=true)

اجرا (از ریشه ریپو، بعد از بالا آوردن سرویس):
    python scripts/smoke_test_pipeline.py
"""
import hashlib
import json
import uuid

import requests

BASE_URL = "http://localhost:8080"
AUTH = {"Authorization": "Bearer test"}


def _user_hash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def send_chat(query: str, history: list[dict] | None = None, session_id: str | None = None) -> list[dict]:
    payload = {
        "request_id": str(uuid.uuid4()),
        "session_id": session_id or str(uuid.uuid4()),
        "user_hash": _user_hash("smoke-test-user"),
        "query": query,
    }
    if history:
        payload["conversation_history"] = history

    resp = requests.post(f"{BASE_URL}/v1/medical/chat", json=payload, headers=AUTH, stream=True, timeout=60)
    resp.raise_for_status()

    events: list[dict] = []
    buf = ""
    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        if chunk is None:
            continue
        buf += chunk
        while "\n\n" in buf:
            raw_event, buf = buf.split("\n\n", 1)
            data_line = next((l[len("data: "):] for l in raw_event.split("\n") if l.startswith("data: ")), None)
            if data_line:
                events.append(json.loads(data_line))
    return events


def by_type(events: list[dict], t: str) -> dict | None:
    return next((e for e in events if e.get("type") == t), None)


def test_specialty_and_patient_record() -> None:
    print("\n--- تست ۱+۲: تشخیص تخصص + پرونده ابتدایی ---")
    session_id = str(uuid.uuid4())
    q1 = "چند روزه دندون آسیام تیر میکشه"

    turn1 = send_chat(q1, session_id=session_id)
    start = by_type(turn1, "start")
    assert start is not None, "رویداد start دریافت نشد"
    print(f"  تخصص تشخیص‌داده‌شده: {start['specialty']} (اطمینان={start['confidence']:.2f})")

    assistant_reply = "".join(e["content"] for e in turn1 if e.get("type") == "token")
    structured1 = by_type(turn1, "structured")
    assert structured1 is not None, "رویداد structured دریافت نشد"
    pr1 = structured1["structured"].get("patient_record")
    assert pr1 is not None, "patient_record در نوبت اول خالی است"
    assert pr1["status"] == "in_progress", f"نوبت اول باید in_progress باشد، شد: {pr1['status']}"
    assert pr1["qa_pairs"] == [], f"نوبت اول نباید qa_pairs داشته باشد: {pr1['qa_pairs']}"
    assert pr1["chief_complaint"] == q1
    print(f"  patient_record نوبت اول: OK (status=in_progress, chief_complaint درست ثبت شد)")

    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": assistant_reply},
    ]
    q2 = "از دیروز شروع شده و به سرما خیلی حساسه"
    turn2 = send_chat(q2, history=history, session_id=session_id)
    structured2 = by_type(turn2, "structured")
    assert structured2 is not None
    pr2 = structured2["structured"]["patient_record"]
    assert pr2["status"] == "completed", f"نوبت دوم باید completed باشد، شد: {pr2['status']}"
    assert len(pr2["qa_pairs"]) == 1, f"باید دقیقا یک qa_pair باشد: {pr2['qa_pairs']}"
    print(f"  patient_record نوبت دوم: OK (status=completed)")
    print(f"  سوال ثبت‌شده در پرونده: {pr2['qa_pairs'][0]['question'][:80]}...")
    print(f"  جواب ثبت‌شده در پرونده: {pr2['qa_pairs'][0]['answer']}")


def test_emergency() -> None:
    print("\n--- تست ۳: تشخیص اورژانس ---")
    events = send_chat("پدرم یهو نتونست نفس بکشه و داره کبود میشه")
    emergency = by_type(events, "emergency")
    assert emergency is not None, "رویداد emergency دریافت نشد - تریاژ کار نکرد!"
    assert emergency["action"] == "call_115"
    assert by_type(events, "structured") is None, "برای اورژانس نباید structured بیاید"
    print(f"  OK: emergency با severity={emergency['severity']} و action={emergency['action']}")


def test_cache() -> None:
    print("\n--- تست ۴: کش سوالات تکراری ---")
    q = "سرما خوردم و میخوام یه نسخه بگیرم"
    first = send_chat(q)
    assert by_type(first, "cached") is None, "درخواست اول نباید cache hit باشد"

    second = send_chat(q)
    cached = by_type(second, "cached")
    assert cached is not None, "درخواست تکراری باید cached بدهد ولی نداد!"
    assert cached["similarity"] == 1.0
    done = by_type(second, "done")
    assert done["cached"] is True
    print(f"  OK: درخواست دوم از کش آمد (similarity={cached['similarity']})")


if __name__ == "__main__":
    test_specialty_and_patient_record()
    test_emergency()
    test_cache()
    print("\n✅ همه تست‌های دودکش فاز ۱ پاس شدند.")
