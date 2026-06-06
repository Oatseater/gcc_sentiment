"""
Unified NLP Pipeline
Language detection → model routing → unified output schema.
"""

import re
import pandas as pd
from datetime import datetime
from typing import Union, List


def detect_language(text: str) -> str:
    """
    Classify text as 'arabic', 'english', or 'mixed'.
    Uses character-level heuristics + langdetect fallback.
    """
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total = arabic_chars + latin_chars

    if total == 0:
        return 'english'

    arabic_ratio = arabic_chars / total
    if arabic_ratio > 0.7:
        return 'arabic'
    elif arabic_ratio < 0.15:
        # Arabizi check: Arabic words in Latin (3adi, sho, yallah, etc.)
        arabizi_markers = re.findall(
            r'\b(3adi|7abibi|yallah|habibi|sho|wain|inshallah|mashallah|wallah|khalas|tamam|zain|ahlan)\b',
            text.lower()
        )
        if arabizi_markers:
            return 'mixed'
        return 'english'
    else:
        return 'mixed'


def detect_emotion(sentiment: str, confidence: float, text: str) -> str:
    """Rule-based emotion layer on top of sentiment."""
    text_lower = text.lower()

    if sentiment == 'positive':
        if confidence > 0.9:
            if any(w in text_lower for w in ['love', 'amazing', '❤', 'best', 'excellent', 'رائع', 'ممتاز', 'أحبه']):
                return 'Delight'
            return 'Satisfaction'
        return 'Mild Approval'

    elif sentiment == 'negative':
        if confidence > 0.85:
            if any(w in text_lower for w in ['worst', 'terrible', 'disgusting', 'awful', 'horrible', 'سيء', 'فظيع', 'كارثة']):
                return 'Frustration'
            if any(w in text_lower for w in ['wait', 'slow', 'late', 'delay', 'بطيء', 'متأخر']):
                return 'Impatience'
        return 'Disappointment'

    return 'Indifference'


def analyze(text: str) -> dict:
    """
    Full pipeline: detect → route → unify output.
    """
    from analysis.aspect_extractor import extract_aspects

    if not text or not text.strip():
        return {}

    lang = detect_language(text)

    if lang == 'arabic':
        from models.arabic_model import predict_arabic
        result = predict_arabic(text)
    else:
        from models.english_model import predict_english
        result = predict_english(text)

    emotion = detect_emotion(result['sentiment'], result['confidence'], text)
    aspects = extract_aspects(text, lang)

    return {
        'text': text,
        'language': lang,
        'sentiment': result['sentiment'],
        'confidence': result['confidence'],
        'label_scores': result.get('label_scores', {}),
        'emotion': emotion,
        'aspects': aspects,
        'model': result['model'],
        'timestamp': datetime.utcnow().isoformat()
    }


def analyze_dataframe(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Batch process a DataFrame. Adds sentiment columns in place.
    Splits by language for efficient batching.
    """
    from analysis.aspect_extractor import extract_aspects

    results = []
    texts = df[text_col].fillna('').tolist()

    # Detect languages first
    languages = [detect_language(t) for t in texts]

    arabic_indices = [i for i, l in enumerate(languages) if l == 'arabic']
    english_indices = [i for i, l in enumerate(languages) if l != 'arabic']

    arabic_results = {}
    english_results = {}

    if arabic_indices:
        from models.arabic_model import predict_arabic_batch
        arabic_texts = [texts[i] for i in arabic_indices]
        preds = predict_arabic_batch(arabic_texts)
        arabic_results = dict(zip(arabic_indices, preds))

    if english_indices:
        from models.english_model import predict_english_batch
        english_texts = [texts[i] for i in english_indices]
        preds = predict_english_batch(english_texts)
        english_results = dict(zip(english_indices, preds))

    all_results = {**arabic_results, **english_results}

    output_rows = []
    for i, text in enumerate(texts):
        r = all_results.get(i, {'sentiment': 'neutral', 'confidence': 0.5, 'model': 'unknown', 'label_scores': {}})
        lang = languages[i]
        emotion = detect_emotion(r['sentiment'], r['confidence'], text)
        aspects = extract_aspects(text, lang)

        row = df.iloc[i].to_dict()
        row.update({
            'language': lang,
            'sentiment': r['sentiment'],
            'confidence': r['confidence'],
            'emotion': emotion,
            'model': r['model'],
            'aspects_summary': ', '.join([f"{a}: {s}" for a, s in aspects.items() if s != 'none']),
        })
        # Flatten label scores
        for label, score in r.get('label_scores', {}).items():
            row[f'score_{label}'] = score

        output_rows.append(row)

    return pd.DataFrame(output_rows)
