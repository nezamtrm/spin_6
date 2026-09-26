#!/usr/bin/env python3
"""
scripts/monitor.py

ابزار مانیتورینگ سبک برای پایش سریع سرویس از ترمینال - مکمل Grafana،
نه جایگزینش. برای زمانی مفید است که Grafana/Prometheus هنوز بالا نیامده
یا می‌خواهید سریع از یک سرور بدون مرورگر وضعیت را چک کنید.

هر N ثانیه:
  - GET /health   (زنده بودن process)
  - GET /ready    (آماده بودن vLLM/Qdrant/embedder/fasttext)
  - GET /metrics  (پارس چند متریک کلیدی و هشدار در صورت رد شدن از آستانه)

اجرا:
    python scripts/monitor.py --base-url http://localhost:8080 --interval 10
    python scripts/monitor.py --once     # فقط یک بار چک کن و خروج (برای cron/CI)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# آستانه‌های هشدار - قابل تنظیم با آرگومان خط فرمان
DEFAULT_TRIAGE_P99_MS = 100.0
DEFAULT_CACHE_HIT_MIN = 0.0  # فقط اطلاع‌رسانی، نه هشدار سخت‌گیرانه در فاز ۱
DEFAULT_ROUTER_FALLBACK_MAX_PER_HOUR = 5


def _fetch(url: str, timeout: float = 3.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        return None


def _fetch_json(url: str, timeout: float = 3.0):
    import json
    text = _fetch(url, timeout)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_prometheus_counter(metrics_text: str, name: str) -> float:
    """جمع مقدار یک متریک (بدون توجه به لیبل‌ها) از متن خام Prometheus."""
    total = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        if line.split("{")[0].split(" ")[0] == name or line.startswith(name + " "):
            try:
                total += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
    return total


def _parse_histogram_quantile_bucket_le(metrics_text: str, name: str, quantile_le: str) -> float | None:
    """مقدار یک bucket خاص هیستوگرام را برمی‌گرداند (تقریب ساده، نه interpolation
    واقعی Prometheus - فقط برای هشدار سریع ترمینالی کافی است)."""
    pattern = re.compile(rf'{re.escape(name)}_bucket\{{[^}}]*le="{re.escape(quantile_le)}"[^}}]*\}}\s+([\d.]+)')
    for line in metrics_text.splitlines():
        m = pattern.match(line)
        if m:
            return float(m.group(1))
    return None


@dataclass
class Alert:
    level: str  # "OK" | "WARN" | "CRIT"
    message: str


def check_once(base_url: str, triage_p99_threshold_ms: float) -> list[Alert]:
    alerts: list[Alert] = []

    health = _fetch_json(f"{base_url}/health")
    if health is None:
        alerts.append(Alert("CRIT", "سرویس جواب نمی‌دهد (/health در دسترس نیست)"))
        return alerts  # بقیه چک‌ها بی‌معنی است اگر پروسه اصلا بالا نیست
    alerts.append(Alert("OK", f"health: {health.get('status')} (v{health.get('version')})"))

    ready = _fetch_json(f"{base_url}/ready")
    if ready is None:
        alerts.append(Alert("CRIT", "/ready در دسترس نیست"))
    else:
        status = ready.get("status")
        level = "OK" if status == "ready" else "CRIT"
        alerts.append(Alert(level, f"readiness: {status}"))
        for comp, val in ready.get("components", {}).items():
            if val == "down":
                # فقط vllm سخت‌گیرانه است؛ بقیه (qdrant/embedder/fasttext) degraded
                comp_level = "CRIT" if comp == "vllm" else "WARN"
                alerts.append(Alert(comp_level, f"  {comp}: down"))

    metrics_text = _fetch(f"{base_url}/metrics")
    if metrics_text is None:
        alerts.append(Alert("WARN", "/metrics در دسترس نیست - نمی‌توانم آستانه‌ها را چک کنم"))
        return alerts

    hits = _parse_prometheus_counter(metrics_text, "spin_cache_hits_total")
    misses = _parse_prometheus_counter(metrics_text, "spin_cache_misses_total")
    if hits + misses > 0:
        rate = hits / (hits + misses)
        alerts.append(Alert("OK", f"cache hit rate: {rate:.1%} ({int(hits)}/{int(hits + misses)})"))

    fallback = _parse_prometheus_counter(metrics_text, "spin_router_fallback_total")
    if fallback > DEFAULT_ROUTER_FALLBACK_MAX_PER_HOUR:
        alerts.append(Alert("WARN", f"router_fallback_total بالاست ({int(fallback)}) - مدل ParsBERT شاید بارگذاری نشده"))

    emergency = _parse_prometheus_counter(metrics_text, "spin_emergency_detected_total")
    alerts.append(Alert("OK", f"emergency_detected_total (تجمعی): {int(emergency)}"))

    return alerts


def render(alerts: list[Alert]) -> str:
    colors = {"OK": "\033[92m", "WARN": "\033[93m", "CRIT": "\033[91m"}
    reset = "\033[0m"
    lines = [f"{colors.get(a.level, '')}[{a.level:4}]{reset} {a.message}" for a in alerts]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--interval", type=float, default=10.0, help="ثانیه بین هر چک (حالت پیوسته)")
    ap.add_argument("--once", action="store_true", help="فقط یک بار چک کن و با کد خروج مناسب برگرد (برای cron/CI)")
    ap.add_argument("--triage-p99-ms", type=float, default=DEFAULT_TRIAGE_P99_MS)
    args = ap.parse_args()

    def run_once() -> int:
        alerts = check_once(args.base_url, args.triage_p99_ms)
        print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(render(alerts))
        has_crit = any(a.level == "CRIT" for a in alerts)
        return 2 if has_crit else (1 if any(a.level == "WARN" for a in alerts) else 0)

    if args.once:
        sys.exit(run_once())

    try:
        while True:
            run_once()
            print()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
