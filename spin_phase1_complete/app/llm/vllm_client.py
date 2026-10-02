# app/llm/vllm_client.py
import json
import aiohttp
from typing import AsyncGenerator, List, Dict
from app.core.config import settings

async def stream_chat_completion(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """
    این تابع به سرور vLLM متصل می‌شود و کلمات را به صورت استریم (Generator) برمی‌گرداند.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": settings.VLLM_MODEL_NAME,
        "messages": messages,
        "temperature": settings.TEMPERATURE,
        "max_tokens": settings.MAX_TOKENS,
        "stream": True,
    }

    # ایجاد یک نشستِ بازِ اینترنتی بدون قفل کردن سرور
    timeout = aiohttp.ClientTimeout(total=30)  # حداکثر ۳۰ ثانیه
    async with aiohttp.ClientSession(timeout=timeout) as session:
        url = f"{settings.VLLM_API_BASE}/chat/completions"

        async with session.post(url, headers=headers, json=payload) as response:
            # بررسی اینکه آیا سرور vLLM زنده است یا نه
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM Error {response.status}: {error_text}")

            # خواندن خط به خط داده‌هایی که از vLLM می‌ریزد
            async for line in response.content.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue

                json_str = line[6:]  # حذف کلمه data:
                if json_str == "[DONE]":
                    break

                try:
                    data = json.loads(json_str)
                    token = data["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue
