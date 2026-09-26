# app/rag/embedder.py
"""
امبدینگ متن فارسی برای RAG (مرحله بازیابی، قبل از prompt).

عمدا مدل امبدینگ جدا نصب نشده: همان بک‌بونِ ParsBERT
(HooshvareLab/bert-fa-base-uncased) که در ParsBert/predict.py برای حالت
zero-shot استفاده می‌شود، اینجا هم با mean-pooling به‌عنوان امبدینگ به کار
می‌رود. دلیل: یک مدل کمتر برای بارگذاری/نگهداری، و بردارها همان فضایی هستند
که واژگان کلیدی تخصص‌ها (ParsBert/specialties.py) هم در آن قرار دارند -
یعنی اگر روزی خواستید کوئری را هم با تخصص و هم با محتوای بازیابی‌شده
بسنجید، از یک فضای برداری استفاده می‌کنید.

اگر بعدا یک مدل امبدینگ اختصاصی (مثلا multilingual-e5) بهتر بود، فقط این
فایل عوض می‌شود؛ امضای embed_texts/embed_query برای بقیه پایپلاین ثابت
می‌ماند.
"""
from __future__ import annotations

from threading import Lock
from typing import List

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_EMBED_MODEL_NAME_DEFAULT = "HooshvareLab/bert-fa-base-uncased"
_MAX_LEN = 256

_tokenizer = None
_model = None
_device = None
_load_lock = Lock()


def _get_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device
    with _load_lock:
        if _model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = getattr(settings, "EMBEDDING_MODEL_NAME", None) or _EMBED_MODEL_NAME_DEFAULT
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModel.from_pretrained(model_name).to(_device).eval()
            log.info("embedder_loaded", model=model_name, device=str(_device))
    return _tokenizer, _model, _device


def embedding_dim() -> int:
    """بعد بردار امبدینگ - برای ساخت کالکشن Qdrant با اندازه درست لازم است."""
    _, model, _ = _get_model()
    return model.config.hidden_size


def embed_texts(texts: List[str]) -> List[List[float]]:
    """امبدینگ یک لیست متن با mean-pooling روی توکن‌های غیر pad (دقیقا همان
    روش ParsBert/predict.py::SpecialtyClassifier._embed)."""
    if not texts:
        return []
    import torch
    import torch.nn.functional as F

    tokenizer, model, device = _get_model()
    with torch.no_grad():
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=_MAX_LEN, return_tensors="pt"
        ).to(device)
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def warmup() -> None:
    """بارگذاری مدل هنگام startup سرویس - نه روی اولین ریکوئست واقعی."""
    try:
        _get_model()
    except Exception:  # noqa: BLE001
        log.exception("embedder_warmup_failed")
