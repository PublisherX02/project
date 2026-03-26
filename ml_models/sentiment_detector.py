"""
sentiment_detector.py
Inference wrapper for the fine-tuned CAMeL-Lab Arabic BERT model.
Loaded ONCE at module level — no repeated disk reads per request.
"""

import json
import torch
import numpy as np
from pathlib import Path

_MODEL_DIR = Path("ml_models/sentiment_model")
_model = None
_tokenizer = None
_label_map = {0: "calm", 1: "urgent"}

def _load_model():
    global _model, _tokenizer, _label_map
    if _model is not None:
        return  # Already loaded

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    try:
        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(_MODEL_DIR))

        # Load on GPU if available, else CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()

        # Load label map
        label_map_path = _MODEL_DIR / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                raw = json.load(f)
                _label_map = {int(k): v for k, v in raw.items()}

    except Exception as e:
        print(f"[sentiment_detector] WARNING: Could not load model: {e}")
        _model = None
        _tokenizer = None


# Eagerly load at import time
_load_model()


def detect_sentiment(text: str) -> dict:
    """
    Classify text as 'calm' (0) or 'urgent' (1).

    Returns:
        {"label": "urgent", "confidence": 0.94, "flag": True}
        {"label": "calm",   "confidence": 0.87, "flag": False}
    On any error, returns a safe default (flag=False) so the API never crashes.
    """
    if _model is None or _tokenizer is None:
        return {"label": "calm", "confidence": 0.0, "flag": False, "error": "model_not_loaded"}

    try:
        device = next(_model.parameters()).device
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = _model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        label = _label_map.get(pred_class, "calm")
        flag = pred_class == 1

        return {"label": label, "confidence": round(confidence, 4), "flag": flag}

    except Exception as e:
        return {"label": "calm", "confidence": 0.0, "flag": False, "error": str(e)}
