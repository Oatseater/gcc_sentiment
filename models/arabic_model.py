"""
Arabic Sentiment Model — CAMeL-BERT
Handles MSA, Gulf Arabic, and mixed Arabizi text.
"""

import re
import unicodedata
from typing import Union
import numpy as np

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()
    return _model, _tokenizer


def _remove_diacritics(text: str) -> str:
    """Strip Arabic diacritics (tashkeel)."""
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
    return diacritics.sub('', text)


def _normalize_arabic(text: str) -> str:
    """Normalize alef, ya, and ta marbuta variants."""
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\w]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_arabic(text: str) -> str:
    text = _remove_diacritics(text)
    text = _normalize_arabic(text)
    return text


def predict_arabic(text: str) -> dict:
    """
    Predict sentiment for Arabic text.
    Returns: {sentiment, confidence, label_scores}
    """
    import torch
    model, tokenizer = _load_model()

    processed = preprocess_arabic(text)
    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

    labels = ['negative', 'neutral', 'positive']
    # CAMeL-BERT label ordering: 0=negative, 1=neutral, 2=positive
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    pred_idx = int(np.argmax(probs))
    sentiment = label_map[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        'sentiment': sentiment,
        'confidence': round(confidence, 4),
        'label_scores': {
            'negative': round(float(probs[0]), 4),
            'neutral': round(float(probs[1]), 4),
            'positive': round(float(probs[2]), 4),
        },
        'model': 'CAMeL-BERT'
    }


def predict_arabic_batch(texts: list) -> list:
    """Batch predict for efficiency."""
    import torch
    model, tokenizer = _load_model()

    processed = [preprocess_arabic(t) for t in texts]
    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()

    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    results = []
    for i, p in enumerate(probs):
        pred_idx = int(np.argmax(p))
        results.append({
            'sentiment': label_map[pred_idx],
            'confidence': round(float(p[pred_idx]), 4),
            'label_scores': {
                'negative': round(float(p[0]), 4),
                'neutral': round(float(p[1]), 4),
                'positive': round(float(p[2]), 4),
            },
            'model': 'CAMeL-BERT'
        })
    return results
