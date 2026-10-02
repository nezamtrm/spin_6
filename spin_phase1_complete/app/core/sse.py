"""app/core/sse.py"""
import json


def _sse_lines(payload: dict, retry_ms: int = 3000) -> str:
    """
    فرمت کامل SSE طبق قرارداد: id: <request_id> + data: <json> + retry: <ms>.
    id گذاشتن روی request_id باعث می‌شود اگر کلاینت به‌خاطر قطعی شبکه reconnect کرد،
    مرورگر/کلاینت بتواند با Last-Event-ID تشخیص دهد کدام رویداد را آخرین بار گرفته.
    """
    request_id = payload.get("request_id", "")
    lines = []
    if request_id:
        lines.append(f"id: {request_id}")
    lines.append(f"data: {json.dumps(payload, ensure_ascii=False, default=str)}")
    lines.append(f"retry: {retry_ms}")
    return "\n".join(lines) + "\n\n"


def format_sse_event(event_type: str, data: dict) -> str:
    """برای ساخت رویداد از صفر (یک dict ساده + نوع رویداد)."""
    payload = {**data, "type": event_type}
    return _sse_lines(payload)


def sse_from_event(event: dict) -> str:
    """
    برای dict‌هایی که از قبل کامل ساخته شده‌اند و خودشان type/request_id دارند
    (مثل خروجی build_structured_event یا OutputGuardResult.to_safety_block) —
    اینجا دیگر چیزی اضافه/جایگزین نمی‌شود، فقط فرمت SSE پوشانده می‌شود.
    """
    return _sse_lines(event)

