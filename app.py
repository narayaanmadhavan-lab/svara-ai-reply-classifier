from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

app = FastAPI(title="Reply Classifier")

class RequestItem(BaseModel):
    text: str

TRANSFORMER_DIR = "models/distilbert-best"
tfidf_path = "models/tfidf_vectorizer.joblib"
lgbm_path = "models/lgbm_model.joblib"
logreg_path = "models/logreg_model.joblib"

transformer = None
tokenizer = None
if os.path.isdir(TRANSFORMER_DIR):
    try:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        transformer = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
        transformer.eval()
        print("Loaded transformer model from", TRANSFORMER_DIR)
    except Exception as e:
        print("Failed to load transformer:", e)

tfidf = joblib.load(tfidf_path) if os.path.exists(tfidf_path) else None
lgbm = joblib.load(lgbm_path) if os.path.exists(lgbm_path) else None
logreg = joblib.load(logreg_path) if os.path.exists(logreg_path) else None

@app.post("/predict")
def predict(req: RequestItem):
    text = req.text
    if transformer is not None and tokenizer is not None:
        inputs = tokenizer(text, truncation=True, padding='longest', return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = transformer(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            return {"label": LABEL_MAP[pred], "confidence": float(round(float(probs[pred]), 4))}
    if tfidf is None:
        return {"error": "No model available on server. Please load a model."}
    x = tfidf.transform([text])
    if lgbm is not None:
        probs = lgbm.predict_proba(x)[0]
        pred = int(np.argmax(probs))
        return {"label": LABEL_MAP[pred], "confidence": float(round(float(probs[pred]),4))}
    if logreg is not None:
        probs = logreg.predict_proba(x)[0]
        pred = int(np.argmax(probs))
        return {"label": LABEL_MAP[pred], "confidence": float(round(float(probs[pred]),4))}
    return {"error":"No baseline model found."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
