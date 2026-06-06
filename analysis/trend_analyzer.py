"""
Trend Analysis Engine
Sentiment velocity, brand scoring, peak detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


def sentiment_score(sentiment: str, confidence: float = 1.0) -> float:
    """Convert sentiment label to numeric score [-1, 1]."""
    mapping = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
    return mapping.get(sentiment, 0.0) * confidence


def compute_brand_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores per brand.
    Returns DataFrame with brand, score, review_count, trend.
    """
    if 'sentiment' not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df['score'] = df.apply(
        lambda r: sentiment_score(r['sentiment'], r.get('confidence', 0.8)),
        axis=1
    )

    grouped = df.groupby('brand').agg(
        avg_score=('score', 'mean'),
        review_count=('score', 'count'),
        pos_count=('sentiment', lambda x: (x == 'positive').sum()),
        neg_count=('sentiment', lambda x: (x == 'negative').sum()),
        neu_count=('sentiment', lambda x: (x == 'neutral').sum()),
    ).reset_index()

    grouped['pos_pct'] = (grouped['pos_count'] / grouped['review_count'] * 100).round(1)
    grouped['neg_pct'] = (grouped['neg_count'] / grouped['review_count'] * 100).round(1)
    grouped['neu_pct'] = (grouped['neu_count'] / grouped['review_count'] * 100).round(1)
    grouped['avg_score'] = grouped['avg_score'].round(3)

    return grouped.sort_values('avg_score', ascending=False).reset_index(drop=True)


def compute_sentiment_timeline(
    df: pd.DataFrame,
    brand: Optional[str] = None,
    freq: str = 'W'
) -> pd.DataFrame:
    """
    Compute sentiment over time.
    freq: 'D' = daily, 'W' = weekly, 'ME' = monthly
    """
    if 'timestamp' not in df.columns or 'sentiment' not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    if brand:
        df = df[df['brand'] == brand]

    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['score'] = df.apply(
        lambda r: sentiment_score(r['sentiment'], r.get('confidence', 0.8)),
        axis=1
    )
    df = df.set_index('timestamp').sort_index()

    timeline = df['score'].resample(freq).agg(['mean', 'count']).reset_index()
    timeline.columns = ['timestamp', 'avg_score', 'review_count']

    # Rolling average (3-period smoothing)
    timeline['smoothed'] = timeline['avg_score'].rolling(3, min_periods=1).mean()
    timeline['avg_score'] = timeline['avg_score'].round(3)
    timeline['smoothed'] = timeline['smoothed'].round(3)

    return timeline


def compute_velocity(timeline: pd.DataFrame) -> Tuple[str, float]:
    """
    Sentiment velocity: is the brand improving or declining?
    Returns (direction_label, velocity_value).
    """
    if timeline.empty or len(timeline) < 3:
        return ('stable', 0.0)

    recent = timeline['smoothed'].iloc[-3:].values
    if len(recent) < 2:
        return ('stable', 0.0)

    velocity = float(recent[-1] - recent[0])

    if velocity > 0.08:
        label = '▲ Improving'
    elif velocity < -0.08:
        label = '▼ Declining'
    else:
        label = '— Stable'

    return (label, round(velocity, 3))


def detect_peak_negativity(
    df: pd.DataFrame,
    brand: Optional[str] = None,
    top_n: int = 3
) -> List[Dict]:
    """
    Detect time windows with highest negative sentiment concentration.
    Returns list of {date, neg_count, avg_score, example_text}.
    """
    if df.empty or 'sentiment' not in df.columns:
        return []

    df = df.copy()
    if brand:
        df = df[df['brand'] == brand]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['week'] = df['timestamp'].dt.to_period('W')
    df['score'] = df.apply(
        lambda r: sentiment_score(r['sentiment'], r.get('confidence', 0.8)),
        axis=1
    )

    neg_df = df[df['sentiment'] == 'negative']
    if neg_df.empty:
        return []

    weekly = neg_df.groupby('week').agg(
        neg_count=('score', 'count'),
        avg_score=('score', 'mean'),
    ).reset_index()

    weekly = weekly.sort_values('neg_count', ascending=False).head(top_n)

    results = []
    for _, row in weekly.iterrows():
        week_neg = neg_df[neg_df['week'] == row['week']]
        example = week_neg.iloc[0]['text'][:100] if not week_neg.empty else ''
        results.append({
            'week': str(row['week']),
            'neg_count': int(row['neg_count']),
            'avg_score': round(float(row['avg_score']), 3),
            'example': example,
        })
    return results


def get_top_reviews(
    df: pd.DataFrame,
    brand: str,
    sentiment: str,
    top_n: int = 5
) -> List[str]:
    """Get top N reviews by confidence for a brand+sentiment."""
    filtered = df[(df['brand'] == brand) & (df['sentiment'] == sentiment)]
    if 'confidence' in filtered.columns:
        filtered = filtered.sort_values('confidence', ascending=False)
    texts = filtered['text'].dropna().head(top_n).tolist()
    return texts


def get_word_frequency(
    df: pd.DataFrame,
    brand: Optional[str] = None,
    lang: str = 'english',
    top_n: int = 10
) -> List[Tuple[str, int]]:
    """
    Compute word frequency for English or Arabic text.
    Returns list of (word, count) sorted descending.
    """
    import re
    from collections import Counter

    if brand:
        df = df[df['brand'] == brand]

    if lang == 'arabic':
        lang_filter = df['language'] == 'arabic' if 'language' in df.columns else pd.Series([True] * len(df))
        texts = df[lang_filter]['text'].dropna().tolist()
        stop_words = {'في', 'من', 'على', 'إلى', 'هذا', 'هذه', 'كان', 'كانت', 'مع', 'لكن', 'أو',
                      'أن', 'لأن', 'التي', 'الذي', 'جداً', 'جدا', 'أيضاً', 'أيضا', 'بشكل',
                      'يكون', 'لا', 'ما', 'كل', 'بعض', 'الـ', 'و', 'ال', 'في', 'عن', 'غير'}
        words = []
        for t in texts:
            words.extend(re.findall(r'[\u0600-\u06FF]{3,}', t))
    else:
        lang_filter = df['language'] != 'arabic' if 'language' in df.columns else pd.Series([True] * len(df))
        texts = df[lang_filter]['text'].dropna().tolist()
        stop_words = {'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of', 'and',
                      'or', 'but', 'was', 'were', 'are', 'be', 'been', 'has', 'have', 'had',
                      'my', 'me', 'i', 'this', 'that', 'they', 'their', 'with', 'from', 'very',
                      'so', 'not', 'no', 'as', 'by', 'just', 'too', 'also', 'its', 'all'}
        words = []
        for t in texts:
            words.extend(re.findall(r'[a-zA-Z]{3,}', t.lower()))

    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    return Counter(filtered_words).most_common(top_n)


def actionable_insights(df: pd.DataFrame, brand: str) -> List[Dict]:
    """
    Generate data-driven actionable insights for a brand.
    Returns list of {priority, finding, recommendation, metric}.
    """
    insights = []
    brand_df = df[df['brand'] == brand] if 'brand' in df.columns else df

    if brand_df.empty:
        return insights

    total = len(brand_df)
    neg_pct = (brand_df['sentiment'] == 'negative').sum() / total * 100 if total > 0 else 0
    pos_pct = (brand_df['sentiment'] == 'positive').sum() / total * 100 if total > 0 else 0

    # Insight 1: Overall sentiment
    if neg_pct > 40:
        insights.append({
            'priority': 'HIGH',
            'finding': f'{neg_pct:.0f}% of reviews are negative — brand reputation at risk.',
            'recommendation': 'Launch immediate customer experience audit. Identify top complaint categories.',
            'metric': f'{neg_pct:.1f}% negative sentiment'
        })
    elif pos_pct > 70:
        insights.append({
            'priority': 'LOW',
            'finding': f'{pos_pct:.0f}% positive sentiment — strong brand equity.',
            'recommendation': 'Leverage positive reviews for marketing. Collect testimonials.',
            'metric': f'{pos_pct:.1f}% positive sentiment'
        })

    # Insight 2: Aspect-level
    if 'aspects_summary' in brand_df.columns:
        all_aspects = brand_df['aspects_summary'].dropna().str.cat(sep=', ')
        delivery_neg = all_aspects.count('delivery: negative')
        service_neg = all_aspects.count('service: negative')

        if delivery_neg > 3:
            insights.append({
                'priority': 'HIGH',
                'finding': f'Delivery complaints appear {delivery_neg} times in dataset.',
                'recommendation': 'Review logistics SLA. Implement real-time delay notifications.',
                'metric': f'{delivery_neg} delivery complaints'
            })

        if service_neg > 3:
            insights.append({
                'priority': 'MEDIUM',
                'finding': f'Staff/service issues mentioned {service_neg} times.',
                'recommendation': 'Staff retraining recommended. Focus on empathy and resolution speed.',
                'metric': f'{service_neg} service complaints'
            })

    # Insight 3: Volume
    if total < 20:
        insights.append({
            'priority': 'LOW',
            'finding': 'Low review volume limits statistical confidence.',
            'recommendation': 'Deploy post-transaction review prompts to increase sample size.',
            'metric': f'{total} total reviews'
        })

    # Timeline-based
    timeline = compute_sentiment_timeline(brand_df)
    if not timeline.empty:
        vel_label, vel_val = compute_velocity(timeline)
        if '▼' in vel_label:
            insights.append({
                'priority': 'HIGH',
                'finding': f'Sentiment declining (velocity: {vel_val:+.3f}) over recent periods.',
                'recommendation': 'Monitor social channels for emerging crisis. Respond to top negative reviews.',
                'metric': f'Velocity: {vel_val:+.3f}'
            })
        elif '▲' in vel_label:
            insights.append({
                'priority': 'LOW',
                'finding': f'Sentiment improving (velocity: {vel_val:+.3f}) — positive momentum.',
                'recommendation': 'Continue current quality initiatives. Share wins with ops team.',
                'metric': f'Velocity: {vel_val:+.3f}'
            })

    return insights
