"""
evaluate_cache_threshold.py

تیون app.core.config.settings.CACHE_SIMILARITY_THRESHOLD با داده واقعی،
نه حدس. مستقیم روی embedder کار می‌کند (نه از طریق Qdrant/qdrant_client)،
چون سوال فقط این است: «شباهت کسینوسیِ این دو جمله چقدر است»، و آن قبل از
هر ورود به Qdrant محاسبه‌پذیر است - سریع‌تر و بدون نیاز به سرویس Qdrant بالا.

فرمت eval/cache_pairs.jsonl - هر سطر یک جفت سوال:
    {"query_a": "...", "query_b": "...", "same_question": true}
    {"query_a": "...", "query_b": "...", "same_question": false}

"same_question": true یعنی این دو از نظر بالینی واقعا یک سوالند (کش باید
hit بدهد) - نه فقط از نظر لغوی شبیه.
"same_question": false یعنی شبیه به نظر می‌رسند ولی بالینی متفاوتند - این‌ها
خطرناک‌ترین ردیف‌ها هستند: اگر آستانه خیلی پایین باشد، کش پاسخ یک سوال
دیگر را جای این یکی می‌دهد.

اجرا:
    python evaluate_cache_threshold.py
    python evaluate_cache_threshold.py --thresholds 0.85,0.90,0.93,0.95,0.97
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.rag import embedder  # noqa: E402


def load_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    # embedder.embed_query از قبل بردار را نرمال می‌کند (F.normalize) پس
    # dot product همان cosine similarity است.
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="eval/cache_pairs.jsonl")
    ap.add_argument("--thresholds", default="0.85,0.88,0.90,0.93,0.95,0.97,0.99")
    args = ap.parse_args()

    path = Path(args.pairs)
    if not path.exists():
        raise SystemExit(f"{path} پیدا نشد - اول این فایل را طبق فرمت بالا بسازید.")
    pairs = load_pairs(path)

    print("محاسبه شباهت برای هر جفت (ممکن است بارگذاری مدل کمی طول بکشد)...")
    scored = []
    for p in pairs:
        va = embedder.embed_query(p["query_a"])
        vb = embedder.embed_query(p["query_b"])
        scored.append({**p, "similarity": cosine(va, vb)})

    thresholds = [float(t) for t in args.thresholds.split(",")]
    same = [s for s in scored if s["same_question"]]
    diff = [s for s in scored if not s["same_question"]]

    print(f"\n{len(same)} جفت هم‌سوال، {len(diff)} جفت شبیه‌ولی‌متفاوت\n")
    print(f"{'threshold':>10} {'hit-rate واقعی (recall)':>26} {'false-hit-rate (خطرناک)':>26}")
    for t in thresholds:
        recall = sum(1 for s in same if s["similarity"] >= t) / len(same) if same else float("nan")
        false_hit = sum(1 for s in diff if s["similarity"] >= t) / len(diff) if diff else float("nan")
        marker = "  <- false-hit صفر و recall بالا: کاندید خوب" if false_hit == 0 and recall > 0.5 else ""
        print(f"{t:>10.2f} {recall:>25.1%} {false_hit:>25.1%}{marker}")

    print("\nهر جفتی که در پایین‌ترین آستانه هم false-hit شد را دستی نگاه کنید:")
    for s in sorted(diff, key=lambda x: -x["similarity"])[:5]:
        print(f"  sim={s['similarity']:.3f}  A: {s['query_a']}  |  B: {s['query_b']}")


if __name__ == "__main__":
    main()
