"""
fasttext_compat.py

پچ سازگاری بین fasttext (نسخه‌ای که روی ایمیج Kaggle از قبل نصب است) و
NumPy>=2.0.

مشکل دقیق: fasttext/FastText.py::predict() داخل خودِ پکیج
np.array(probs, copy=False) صدا می‌زند. در NumPy<2 یعنی «اگر لازم نبود
کپی نکن»؛ NumPy>=2.0 سخت‌گیرتر شده و اگر واقعا کپی لازم باشد به‌جای
انجام آرام کار، ValueError می‌دهد. این باگ داخل خودِ پکیج fasttext است؛
راه تمیز واقعی‌اش `pip install "numpy<2"` یا آپدیت fasttext است - ولی
وقتی اینترنت نوتبوک خاموش است (دقیقا مورد شما) هیچ‌کدام ممکن نیست.

این پچ موقت، رفتار NumPy<2 را فقط برای همان یک call-site شبیه‌سازی می‌کند:
اگر copy=False بود و کپی لازم نبود از np.asarray استفاده می‌کند (بدون
خطا)، و اگر واقعا copy=True خواسته شده بود دست‌نخورده می‌گذارد.

استفاده - در ابتدای هر اسکریپتی که fasttext.load_model/predict صدا
می‌زند (train_fasttext.py و evaluate_triage.py از قبل این را صدا می‌زنند):

    import fasttext
    from fasttext_compat import patch_fasttext_numpy2
    patch_fasttext_numpy2()
"""
import numpy as np

_patched = False


def patch_fasttext_numpy2() -> None:
    global _patched
    if _patched:
        return
    import fasttext.FastText as _ft_module

    _orig_array = np.array

    def _compat_array(obj, copy=False, **kwargs):
        if copy:
            return _orig_array(obj, copy=copy, **kwargs)
        return np.asarray(obj, **kwargs)

    _ft_module.np.array = _compat_array
    _patched = True
