"""
Aspect-Based Sentiment Extraction
Rule-based NLP for UAE service review aspects.
"""

import re
from typing import Dict

ASPECT_KEYWORDS = {
    'food': {
        'en': ['food', 'meal', 'dish', 'taste', 'flavour', 'flavor', 'cuisine', 'menu', 'restaurant',
               'breakfast', 'lunch', 'dinner', 'eat', 'ate', 'cook', 'cooked', 'ingredient', 'fresh',
               'stale', 'delicious', 'bland', 'salty', 'spicy', 'cold', 'warm', 'hot'],
        'ar': ['أكل', 'طعام', 'وجبة', 'طبق', 'مطعم', 'مذاق', 'طازج', 'مفيد', 'فطور', 'غداء', 'عشاء',
               'لذيذ', 'طاهي', 'شيف', 'قائمة', 'مكونات', 'ملح', 'حار', 'بارد', 'دافئ'],
    },
    'delivery': {
        'en': ['delivery', 'deliver', 'arrive', 'arrived', 'shipping', 'courier', 'driver', 'pickup',
               'drop', 'package', 'parcel', 'tracking', 'late', 'fast', 'quick', 'slow', 'delay',
               'on time', 'minutes', 'hours', 'wait', 'waited'],
        'ar': ['توصيل', 'وصل', 'وصول', 'شحن', 'سائق', 'تسليم', 'متأخر', 'سريع', 'بطيء', 'تأخير',
               'انتظار', 'دقيقة', 'ساعة', 'تتبع', 'باكيج', 'طرد', 'في الوقت'],
    },
    'price': {
        'en': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'pricey', 'overpriced', 'value',
               'worth', 'fee', 'charge', 'promo', 'discount', 'deal', 'offer', 'aed', 'dirham',
               'money', 'pay', 'paid', 'refund', 'cashback'],
        'ar': ['سعر', 'أسعار', 'تكلفة', 'غالي', 'رخيص', 'معقول', 'مكلف', 'قيمة', 'خصم', 'عرض',
               'درهم', 'دفع', 'استرداد', 'فلوس', 'مال', 'تسعير', 'مبالغ', 'مناسب', 'تنافسي'],
    },
    'service': {
        'en': ['service', 'staff', 'employee', 'team', 'support', 'help', 'assist', 'rude', 'polite',
               'friendly', 'professional', 'courteous', 'manager', 'representative', 'customer service',
               'response', 'reply', 'resolve', 'experience'],
        'ar': ['خدمة', 'موظف', 'طاقم', 'مساعدة', 'دعم', 'محترف', 'مؤدب', 'وقح', 'فريق', 'مدير',
               'تواصل', 'استجابة', 'تجربة', 'عملاء', 'تعامل', 'مهني', 'موظفين'],
    },
    'ambiance': {
        'en': ['ambiance', 'atmosphere', 'decor', 'interior', 'design', 'clean', 'dirty', 'hygiene',
               'noise', 'quiet', 'comfortable', 'crowded', 'spacious', 'lighting', 'music', 'environment',
               'vibe', 'setting', 'aesthetic'],
        'ar': ['جو', 'ديكور', 'تصميم', 'نظيف', 'نظافة', 'قذر', 'ضجيج', 'هادئ', 'مريح', 'مزدحم',
               'فسيح', 'إضاءة', 'موسيقى', 'بيئة', 'أجواء', 'مكان'],
    },
    'speed': {
        'en': ['speed', 'fast', 'quick', 'slow', 'instant', 'immediate', 'rapid', 'efficient',
               'responsive', 'loading', 'app speed', 'processing', 'real-time', 'minutes', 'lag'],
        'ar': ['سرعة', 'سريع', 'بطيء', 'فوري', 'فعال', 'استجابة', 'تحميل', 'معالجة', 'وقت', 'دقائق'],
    },
    'app': {
        'en': ['app', 'application', 'website', 'platform', 'ui', 'ux', 'interface', 'crash', 'bug',
               'update', 'feature', 'notification', 'login', 'checkout', 'cart', 'payment', 'digital'],
        'ar': ['تطبيق', 'موقع', 'منصة', 'واجهة', 'تعطل', 'خلل', 'تحديث', 'ميزة', 'دفع', 'رقمي',
               'تسجيل', 'خروج', 'إشعار', 'سلة', 'برنامج'],
    },
}

POSITIVE_MODIFIERS = {
    'en': ['great', 'excellent', 'amazing', 'perfect', 'fast', 'quick', 'good', 'best', 'love', 'recommend',
           'helpful', 'professional', 'clean', 'fresh', 'impressed', 'satisfied', 'happy', 'enjoyed',
           'smooth', 'convenient', 'efficient', 'reliable', 'outstanding', 'wonderful', 'top'],
    'ar': ['ممتاز', 'رائع', 'جيد', 'مميز', 'سريع', 'نظيف', 'محترف', 'مفيد', 'راضٍ', 'أحب', 'أنصح',
           'استمتعت', 'مرتاح', 'خرافي', 'شكراً', 'يستحق', 'أفضل', 'إيجابي', 'فعال', 'موثوق'],
}

NEGATIVE_MODIFIERS = {
    'en': ['terrible', 'awful', 'bad', 'poor', 'worst', 'slow', 'rude', 'broken', 'failed', 'wrong',
           'disappointed', 'frustrating', 'unacceptable', 'useless', 'disgusting', 'cold', 'late',
           'never', 'avoid', 'never again', 'complaint', 'issue', 'problem', 'error', 'crash'],
    'ar': ['سيء', 'فظيع', 'رديء', 'بطيء', 'وقح', 'كسر', 'خاطئ', 'محبط', 'مقبول', 'غير', 'مشكلة',
           'خطأ', 'شكوى', 'بارد', 'متأخر', 'مروع', 'مخيب', 'تعطل', 'فشل', 'لن أعود'],
}


def _get_window(text: str, keyword: str, window: int = 30) -> str:
    """Get text window around a keyword match."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return ''
    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)
    return text[start:end]


def _score_aspect_sentiment(window: str, lang: str) -> str:
    """Determine if the aspect context is positive, negative, or neutral."""
    lang_key = 'ar' if lang == 'arabic' else 'en'
    alt_key = 'en' if lang_key == 'ar' else 'ar'

    window_lower = window.lower()
    pos_count = sum(1 for w in POSITIVE_MODIFIERS[lang_key] if w in window_lower)
    pos_count += sum(1 for w in POSITIVE_MODIFIERS[alt_key] if w in window_lower)
    neg_count = sum(1 for w in NEGATIVE_MODIFIERS[lang_key] if w in window_lower)
    neg_count += sum(1 for w in NEGATIVE_MODIFIERS[alt_key] if w in window_lower)

    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    return 'neutral'


def extract_aspects(text: str, lang: str = 'english') -> Dict[str, str]:
    """
    Extract aspect sentiments from review text.
    Returns {aspect: sentiment} for detected aspects.
    """
    lang_key = 'ar' if lang == 'arabic' else 'en'
    alt_key = 'en' if lang_key == 'ar' else 'ar'

    detected_aspects = {}

    for aspect, kw_dict in ASPECT_KEYWORDS.items():
        found = False
        matched_window = ''

        # Check primary language keywords
        for kw in kw_dict[lang_key]:
            if kw in text.lower():
                matched_window = _get_window(text, kw)
                found = True
                break

        # Check alternate language keywords (bilingual text)
        if not found:
            for kw in kw_dict[alt_key]:
                if kw in text:
                    matched_window = _get_window(text, kw)
                    found = True
                    break

        if found:
            sentiment = _score_aspect_sentiment(matched_window, lang)
            detected_aspects[aspect] = sentiment

    # If nothing detected, return top-level neutral for 'service'
    if not detected_aspects:
        detected_aspects['service'] = 'neutral'

    return detected_aspects


def get_aspect_summary(aspects: Dict[str, str]) -> str:
    """Human-readable one-line summary of aspect findings."""
    positives = [a for a, s in aspects.items() if s == 'positive']
    negatives = [a for a, s in aspects.items() if s == 'negative']

    parts = []
    if positives:
        parts.append(f"Strong: {', '.join(positives)}")
    if negatives:
        parts.append(f"Issues: {', '.join(negatives)}")
    return ' · '.join(parts) if parts else 'No specific aspects detected.'
