"""
English Sentiment Model — RoBERTa Twitter
Optimized for social/review text (informal, slang-tolerant).
"""

import numpy as np
from typing import Union

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()
    return _model, _tokenizer


def predict_english(text: str) -> dict:
    """
    Predict sentiment for English/Arabizi text.
    Returns: {sentiment, confidence, label_scores}
    """
    import torch
    model, tokenizer = _load_model()

    # RoBERTa preprocess — replace @user and URLs
    text_clean = text
    import re
    text_clean = re.sub(r'@\w+', '@user', text_clean)
    text_clean = re.sub(r'http\S+', 'http', text_clean)

    inputs = tokenizer(
        text_clean,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

    # cardiffnlp latest: labels are negative, neutral, positive
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    pred_idx = int(np.argmax(probs))

    return {
        'sentiment': label_map[pred_idx],
        'confidence': round(float(probs[pred_idx]), 4),
        'label_scores': {
            'negative': round(float(probs[0]), 4),
            'neutral': round(float(probs[1]), 4),
            'positive': round(float(probs[2]), 4),
        },
        'model': 'RoBERTa-Twitter'
    }


def predict_english_batch(texts: list) -> list:
    import torch
    model, tokenizer = _load_model()
    import re

    processed = []
    for t in texts:
        t = re.sub(r'@\w+', '@user', t)
        t = re.sub(r'http\S+', 'http', t)
        processed.append(t)

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
    for p in probs:
        pred_idx = int(np.argmax(p))
        results.append({
            'sentiment': label_map[pred_idx],
            'confidence': round(float(p[pred_idx]), 4),
            'label_scores': {
                'negative': round(float(p[0]), 4),
                'neutral': round(float(p[1]), 4),
                'positive': round(float(p[2]), 4),
            },
            'model': 'RoBERTa-Twitter'
        })
    return results
