# -*- coding: utf-8 -*-
"""پیش‌بینی تخصص پزشکی مرتبط با متن فارسی، به همراه درصد اطمینان.

دو حالت کاری:
  1) fine-tuned : اگر مدل آموزش‌دیده در models/parsbert-specialty/final موجود باشد
                  (دقیق‌ترین حالت — خروجی softmax سر طبقه‌بندی)
  2) zero-shot  : اگر مدل آموزش‌دیده نباشد، از امبدینگ خام ParsBERT + واژگان
                  کلیدی هر تخصص استفاده می‌شود (بدون نیاز به آموزش، دقت کمتر)

نمونه اجرا:
    python predict.py "چند روزه قفسه سینه‌ام تیر می‌کشه و تپش قلب دارم"
    python predict.py --top-k 5 --json "سرم خیلی درد می‌کنه و سرگیجه دارم"
    python predict.py            # حالت تعاملی
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from specialties import NUM_LABELS, LABELS, SPECIALTIES
from text_utils import CSV_WRITE_ENCODING, normalize, setup_console

FINETUNED_DIR = Path(__file__).parent / "models" / "parsbert-specialty" / "final"
BASE_MODEL = "HooshvareLab/bert-fa-base-uncased"
LOW_CONFIDENCE = 0.35          # زیر این آستانه، پاسخ «نامطمئن» علامت می‌خورد


@dataclass
class Prediction:
    specialty: str             # نام فارسی تخصص
    key: str                   # شناسه انگلیسی
    confidence: float          # 0..1
    percent: float             # 0..100


@dataclass
class Result:
    text: str
    top: Prediction
    ranking: List[Prediction]
    mode: str                  # "fine-tuned" یا "zero-shot"
    certain: bool              # آیا اطمینان بالای آستانه است؟

    def to_dict(self):
        return asdict(self)


class SpecialtyClassifier:
    """دسته‌بندِ متن فارسی به یکی از ۱۴ تخصص، با درصد اطمینان."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        base_model: str = BASE_MODEL,
        device: Optional[str] = None,
        max_len: int = 128,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_len = max_len
        path = Path(model_dir) if model_dir else FINETUNED_DIR

        if (path / "config.json").exists():
            self.mode = "fine-tuned"
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            cfg = self.model.config
            self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
            meta = path / "specialties.json"
            self.keep_zwnj = (
                json.loads(meta.read_text(encoding="utf-8")).get("keep_zwnj", False)
                if meta.exists() else False
            )
        else:
            self.mode = "zero-shot"
            self.keep_zwnj = "zwnj" in base_model.lower()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                self.model = AutoModel.from_pretrained(base_model)
            except OSError as e:
                # پیام واضح‌تر از exception خام transformers/huggingface_hub:
                # این حالت دقیقا زمانی رخ می‌دهد که (۱) مدل فاین‌تیون‌شده در
                # path بالا پیدا نشد (یعنی ParsBert/train.py هنوز در همین
                # session/محیط اجرا نشده - مثلا یک سشن تازه Kaggle که
                # /kaggle/working آن ریست شده) و (۲) اینترنت هم برای دانلود
                # fallback خاموش است. راه‌حل تقریبا همیشه یکی از این دو است،
                # نه یک مشکل واقعی در کد.
                raise RuntimeError(
                    f"نه مدل فاین‌تیون‌شده در {path} پیدا شد، نه اینترنت برای دانلود "
                    f"fallback ({base_model}) در دسترس بود.\n"
                    f"راه‌حل: یا (الف) اول 'python train.py' را در همین session اجرا کنید "
                    f"تا {path} ساخته شود، یا (ب) در تنظیمات نوتبوک Internet را On کنید.\n"
                    f"خطای اصلی: {e}"
                ) from e
            self.labels = LABELS

        self.model.to(self.device).eval()
        self._proto = self._build_prototypes() if self.mode == "zero-shot" else None

    # ---------------------------------------------------------------- zero-shot
    @torch.no_grad()
    def _embed(self, texts: List[str]) -> torch.Tensor:
        """امبدینگ جمله با mean-pooling روی توکن‌های غیر pad."""
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state              # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, dim=-1)

    def _build_prototypes(self) -> torch.Tensor:
        """برای هر تخصص یک بردار نماینده از نام + توصیف + واژگان کلیدی می‌سازد."""
        protos = []
        for spec in SPECIALTIES:
            phrases = [spec.name, spec.description] + spec.keywords
            phrases = [normalize(p, self.keep_zwnj) for p in phrases]
            protos.append(F.normalize(self._embed(phrases).mean(0), dim=-1))
        return torch.stack(protos)                             # (14, H)

    def _lexicon_scores(self, text: str) -> torch.Tensor:
        """امتیاز تطبیق واژگانی: چند کلیدواژه از هر تخصص در متن آمده است."""
        scores = torch.zeros(NUM_LABELS, device=self.device)
        for i, spec in enumerate(SPECIALTIES):
            hit = 0.0
            for kw in spec.keywords:
                k = normalize(kw, self.keep_zwnj)
                if k and k in text:
                    # کلیدواژه چندکلمه‌ای نشانه قوی‌تری است
                    hit += 1.0 + 0.5 * k.count(" ")
            scores[i] = hit
        return scores

    # ------------------------------------------------------------------- عمومی
    @torch.no_grad()
    def _probabilities(self, text: str) -> torch.Tensor:
        if self.mode == "fine-tuned":
            enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                                 return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits[0]
            return F.softmax(logits, dim=-1)

        # zero-shot: ترکیب شباهت معنایی و تطبیق واژگانی
        sim = self._embed([text])[0] @ self._proto.T           # (14,) در بازه [-1, 1]
        lex = self._lexicon_scores(text)
        logits = sim * 12.0 + lex * 2.0                        # وزن/دمای تجربی
        return F.softmax(logits, dim=-1)

    def predict(self, text: str, top_k: int = 3) -> Result:
        clean = normalize(text, self.keep_zwnj)
        if not clean:
            raise ValueError("متن ورودی خالی است.")

        probs = self._probabilities(clean).cpu()
        k = max(1, min(top_k, NUM_LABELS))
        order = torch.argsort(probs, descending=True)[:k]

        ranking = []
        for idx in order.tolist():
            name = self.labels[idx]
            key = SPECIALTIES[idx].key if name == LABELS[idx] else name
            p = float(probs[idx])
            ranking.append(Prediction(name, key, round(p, 4), round(p * 100, 2)))

        top = ranking[0]
        return Result(text=text, top=top, ranking=ranking, mode=self.mode,
                      certain=top.confidence >= LOW_CONFIDENCE)

    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Result]:
        return [self.predict(t, top_k) for t in texts]


# -------------------------------------------------------------------------- CLI
BAR_W = 24


def render(res: Result, top_k: int) -> str:
    lines = [
        "-" * 62,
        f"متن ورودی : {res.text}",
        f"حالت مدل  : {res.mode}",
        "-" * 62,
        f"تخصص پیشنهادی: {res.top.specialty}   ({res.top.percent:.2f}%  اطمینان)",
    ]
    if not res.certain:
        lines.append("هشدار: اطمینان پایین است؛ بهتر است ابتدا به پزشک عمومی مراجعه شود.")
    lines.append("")
    lines.append(f"رتبه‌بندی (top-{top_k}):")
    for i, p in enumerate(res.ranking, 1):
        filled = int(round(p.confidence * BAR_W))
        bar = "#" * filled + "." * (BAR_W - filled)
        lines.append(f"  {i}. {p.specialty:<26} {bar} {p.percent:6.2f}%")
    lines.append("-" * 62)
    return "\n".join(lines)


def save_csv(path: str, results: List[Result], top_k: int):
    """ذخیرهٔ نتایج در CSV با BOM تا اکسل فارسی را درست نمایش دهد."""
    import csv
    with open(path, "w", encoding=CSV_WRITE_ENCODING, newline="") as f:
        w = csv.writer(f)
        header = ["text", "specialty", "confidence_percent", "certain"]
        for i in range(2, top_k + 1):
            header += [f"alt{i}_specialty", f"alt{i}_percent"]
        w.writerow(header)
        for r in results:
            row = [r.text, r.top.specialty, f"{r.top.percent:.2f}", int(r.certain)]
            for p in r.ranking[1:top_k]:
                row += [p.specialty, f"{p.percent:.2f}"]
            w.writerow(row)
    print(f"\nنتایج ذخیره شد در: {Path(path).resolve()}")


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="تشخیص تخصص پزشکی از روی متن فارسی")
    ap.add_argument("text", nargs="*", help="متن بیمار (اگر خالی باشد، حالت تعاملی)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--model-dir", default=None, help="مسیر مدل فاین‌تیون‌شده")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--json", action="store_true", help="خروجی JSON")
    ap.add_argument("--file", default=None,
                    help="فایل متنی UTF-8، هر خط یک متن برای تست دسته‌ای")
    ap.add_argument("--out-csv", default=None,
                    help="ذخیرهٔ نتایج در CSV سازگار با اکسل")
    args = ap.parse_args()

    clf = SpecialtyClassifier(model_dir=args.model_dir, base_model=args.base_model)
    if clf.mode == "zero-shot" and not args.json:
        print("توجه: مدل فاین‌تیون‌شده یافت نشد؛ حالت zero-shot فعال است.")
        print("برای دقت بالاتر:  python build_dataset.py  &&  python train.py\n")

    def show(text: str) -> Result:
        res = clf.predict(text, args.top_k)
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2)
              if args.json else render(res, args.top_k))
        return res

    # حالت دسته‌ای: هر خط فایل یک متن
    if args.file:
        lines = [ln.strip() for ln in
                 Path(args.file).read_text(encoding="utf-8-sig").splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
        results = [show(ln) for ln in lines]
        if args.out_csv:
            save_csv(args.out_csv, results, args.top_k)
        return 0

    if args.text:
        res = show(" ".join(args.text))
        if args.out_csv:
            save_csv(args.out_csv, [res], args.top_k)
        return 0

    print("متن را وارد کنید (برای خروج: exit)")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit", "خروج"}:
            break
        if not text:
            continue
        try:
            show(text)
        except ValueError as e:
            print(f"خطا: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
