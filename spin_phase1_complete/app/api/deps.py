# app/api/deps.py
"""
Dependency مشترک برای احراز هویت سرویس-به-سرویس (Backend -> این سرویس).
طبق OpenApi_Spin.yaml: securitySchemes.ServiceToken از نوع Bearer JWT است.

فعلا فقط "وجود و ساختار درست" توکن چک می‌شود، نه امضای واقعی JWT -
چون کلید امضا و مکانیزم صدور هنوز با تیم بک‌اند نهایی نشده. وقتی نهایی شد،
فقط تابع _decode_and_validate باید عوض شود؛ امضای verify_service_token
(و همهٔ routeهایی که با Depends صدایش می‌زنند) دست‌نخورده می‌ماند.
"""
from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def _decode_and_validate(token: str) -> None:
    """
    TODO (فاز بعد): جایگزینی با اعتبارسنجی واقعی JWT، مثلا با python-jose:

        from jose import jwt, JWTError
        try:
            jwt.decode(token, settings.SERVICE_JWT_SECRET, algorithms=["HS256"])
        except JWTError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc

    فعلا فقط بررسی می‌شود توکن خالی نباشد - این یک گلوگاه امنیتی موقت است،
    نه جایگزین امن برای production.
    """
    if not token or not token.strip():
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "empty token")


async def verify_service_token(authorization: str = Header(default="")) -> None:
    """
    استفاده: @router.post("/chat")
             async def medical_chat(..., _auth=Depends(verify_service_token)):

    اگر هدر Authorization غایب یا بدفرمت باشد یا توکن نامعتبر باشد، ۴۰۱
    برمی‌گردد و FastAPI خودش قبل از رسیدن به بدنهٔ route این خطا را می‌دهد -
    یعنی هیچ‌کدام از pipeline (guard_input/triage/...) برای یک درخواست
    بدون توکن معتبر اصلا اجرا نمی‌شود.
    """
    if not authorization.startswith("Bearer "):
        log.warning("auth_failed", reason="missing_bearer_prefix")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        _decode_and_validate(token)
    except HTTPException:
        log.warning("auth_failed", reason="invalid_token")
        raise
