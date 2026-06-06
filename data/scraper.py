"""
Real-time data scraper.
Pulls from RSS feeds, Reddit (public JSON API), and web search snippets.
No API keys required for basic operation.
"""

import requests
import feedparser
import re
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/122.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9,ar;q=0.8',
}

# Google News RSS — no API key needed
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

BRAND_QUERIES = {
    'Talabat':      'Talabat UAE review food delivery',
    'Emirates':     'Emirates airline review experience',
    'Careem':       'Careem ride UAE experience',
    'Dubai Mall':   'Dubai Mall shopping experience review',
    'Burj Al Arab': 'Burj Al Arab hotel review luxury',
    'ADNOC':        'ADNOC UAE petrol station service',
    'Noon':         'Noon shopping UAE delivery review',
    'LuLu':         'LuLu hypermarket UAE review',
}

REDDIT_SUBS = ['dubai', 'abudhabi', 'uae', 'sharjah']


def fetch_google_news(query: str, max_items: int = 20) -> List[Dict]:
    """Pull headlines from Google News RSS."""
    url = GOOGLE_NEWS_RSS.format(query=requests.utils.quote(query))
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            published = entry.get('published', '')
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(published).isoformat()
            except Exception:
                pub_dt = datetime.utcnow().isoformat()

            items.append({
                'text': entry.get('title', '') + '. ' + re.sub(r'<[^>]+>', '', entry.get('summary', '')),
                'source': 'Google News',
                'brand': _extract_brand(entry.get('title', '') + entry.get('summary', '')),
                'timestamp': pub_dt,
                'url': entry.get('link', ''),
            })
        return items
    except Exception as e:
        return []


def fetch_reddit_posts(subreddit: str, query: str, max_items: int = 25) -> List[Dict]:
    """Pull Reddit posts via public JSON (no OAuth needed)."""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {'q': query, 'sort': 'new', 'limit': max_items, 'restrict_sr': 1}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        posts = data.get('data', {}).get('children', [])
        items = []
        for p in posts:
            d = p.get('data', {})
            text = d.get('selftext', '') or d.get('title', '')
            if len(text) < 10:
                text = d.get('title', '')
            items.append({
                'text': text[:600],
                'source': f'r/{subreddit}',
                'brand': _extract_brand(text),
                'timestamp': datetime.utcfromtimestamp(d.get('created_utc', 0)).isoformat(),
                'url': 'https://reddit.com' + d.get('permalink', ''),
                'upvotes': d.get('ups', 0),
            })
        return items
    except Exception:
        return []


def fetch_brand_live(brand: str, max_items: int = 30) -> List[Dict]:
    """Aggregate live data for a specific UAE brand."""
    results = []

    # Google News
    news = fetch_google_news(BRAND_QUERIES.get(brand, brand), max_items=15)
    results.extend(news)
    time.sleep(0.3)

    # Reddit across relevant subs
    for sub in REDDIT_SUBS[:2]:
        posts = fetch_reddit_posts(sub, brand, max_items=10)
        results.extend(posts)
        time.sleep(0.2)

    # Deduplicate by text prefix
    seen = set()
    unique = []
    for r in results:
        key = r['text'][:80].strip().lower()
        if key not in seen and len(r['text']) > 20:
            seen.add(key)
            r['brand'] = brand
            unique.append(r)

    return unique[:max_items]


def fetch_live_feed(brands: Optional[List[str]] = None, max_per_brand: int = 20) -> List[Dict]:
    """Fetch live data for all or specified brands."""
    if brands is None:
        brands = list(BRAND_QUERIES.keys())

    all_data = []
    for brand in brands:
        items = fetch_brand_live(brand, max_per_brand)
        all_data.extend(items)

    return all_data


def _extract_brand(text: str) -> str:
    """Heuristic brand tagging from text."""
    text_lower = text.lower()
    for brand in BRAND_QUERIES:
        if brand.lower() in text_lower:
            return brand
    return 'General'
