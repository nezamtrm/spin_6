# -*- coding: utf-8 -*-
"""سرویس REST برای تشخیص تخصص پزشکی از متن فارسی.

اجرا:
    uvicorn api:app --host 0.0.0.0 --port 8000

نمونه درخواست:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text":"سرم درد میکنه و سرگیجه دارم","top_k":3}'
"""

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import SpecialtyClassifier
from specialties import SPECIALTIES

app = FastAPI(title="ParsBERT Medical Specialty Router", version="1.0")
clf = SpecialtyClassifier()          # یک بار بارگذاری، استفاده مشترک بین درخواست‌ها


class PredictIn(BaseModel):
    text: str = Field(..., min_length=2, description="شرح حال یا شکایت بیمار")
    top_k: int = Field(3, ge=1, le=6)


class BatchIn(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)
    top_k: int = Field(3, ge=1, le=6)


@app.get("/health")
def health():
    return {"status": "ok", "mode": clf.mode, "num_labels": len(SPECIALTIES)}


@app.get("/specialties")
def list_specialties():
    return [{"key": s.key, "name": s.name, "description": s.description}
            for s in SPECIALTIES]


@app.post("/predict")
def predict(body: PredictIn):
    try:
        return clf.predict(body.text, body.top_k).to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(body: BatchIn):
    try:
        return [r.to_dict() for r in clf.predict_batch(body.texts, body.top_k)]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
