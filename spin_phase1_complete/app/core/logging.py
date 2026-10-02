# app/core/logging.py
"""
لاگ ساختاریافته (JSON) به‌جای print پراکنده. هر خط لاگ یک شیء JSON با
حداقل timestamp/level/logger/event است، تا بعدا با هر ابزار جمع‌آوری لاگ
(ELK, Loki, ...) قابل جست‌وجو و فیلتر باشد - چیزی که با print معمولی ممکن نیست.
"""
import json
import logging
import sys
import time


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # فیلدهای اضافه‌ای که با extra={"reason": ...} یا log.warning("x", reason=...)
        # پاس داده شده باشند هم داخل همان یک خط JSON می‌آیند، نه جدا.
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info") or key.startswith("_"):
                continue
            if key not in payload and key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> "_KwargsLoggerAdapter":
    """
    برمی‌گرداند یک logger که اجازه می‌دهد به سبک structlog بنویسید:
        log.warning("auth_failed", reason="missing_bearer_prefix")
    به‌جای f-string دستی، که برای جست‌وجوی بعدی در لاگ‌ها بسیار بهتر است.
    """
    configure_logging()
    return _KwargsLoggerAdapter(logging.getLogger(name))


class _KwargsLoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _log(self, level, event, **kwargs):
        self._logger.log(level, event, extra=kwargs)

    def info(self, event, **kwargs):
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event, **kwargs):
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event, **kwargs):
        self._log(logging.ERROR, event, **kwargs)
