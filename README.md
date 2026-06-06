# UAE Sentiment Intelligence Review

A bilingual NLP dashboard for analysing Arabic and English customer reviews across major UAE brands — built with Streamlit, CAMeL-BERT, and RoBERTa.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-4A7C59?style=flat-square)

---

## What It Does

Most sentiment tools fail on Arabic text — they either ignore it entirely or misclassify Gulf dialect as noise. This dashboard routes each review to the correct model based on character-level language detection, then returns a unified output: sentiment label, confidence score, emotion tag, and per-aspect breakdown.

**Live data.** Pulls real-time reviews from Google News RSS and Reddit's public JSON API — no API keys required.

---

## Models

| Language | Model | Architecture |
|----------|-------|--------------|
| Arabic / Gulf dialect | [CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment) | BERT fine-tuned on Arabic MSA + dialect |
| English / Arabizi | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | RoBERTa fine-tuned on 198M tweets |

Language detection uses character-ratio heuristics with an Arabizi marker fallback (habibi, wallah, yallah, etc.).

---

## Features

**Tab 1 — Live Analyzer**
Input any Arabic, English, or mixed review. Auto-detects language, routes to the correct model, returns verdict with confidence footnote and aspect breakdown.

**Tab 2 — Batch Analyzer**
Upload a CSV and score all rows in one pass. Parallel processing by language for efficiency. Download scored output with all NLP columns appended.

**Tab 3 — Brand Dashboard**
Select a UAE brand (Talabat, Emirates, Careem, Dubai Mall, Burj Al Arab, ADNOC, Noon, LuLu). Fetches live data from Reddit + Google News, computes sentiment trend, aspect heatmap, word frequency, top pull quotes, and actionable insights with priority labels.

**Tab 4 — Brand Comparison**
Side-by-side editorial scorecard. Sentiment scores, sentiment split percentages, trajectory labels, and vocabulary comparison across two brands.

---

## Architecture

```
sentiment_analyzer/
├── app.py                      # Streamlit UI — 4 tabs, editorial dark theme
├── models/
│   ├── arabic_model.py         # CAMeL-BERT loader + Arabic preprocessing
│   ├── english_model.py        # RoBERTa-Twitter loader
│   └── unified_pipeline.py     # Language detection + routing + unified schema
├── data/
│   ├── scraper.py              # Google News RSS + Reddit JSON scraper
│   └── sample_reviews.py       # 500 realistic UAE reviews (AR/EN/Arabizi)
├── analysis/
│   ├── aspect_extractor.py     # Rule-based aspect-sentiment extraction
│   └── trend_analyzer.py       # Velocity, brand scores, peak negativity, insights
└── utils/
    └── charts.py               # Plotly FT-style charts (trend line, lollipop, gauge)
```

**Data flow:**
```
Input text
    ↓
detect_language()  →  arabic / english / mixed
    ↓
route to model     →  CAMeL-BERT / RoBERTa-Twitter
    ↓
unified output     →  {sentiment, confidence, emotion, aspects, timestamp}
    ↓
trend_analyzer     →  velocity, brand score, actionable insights
```

---

## Setup

**Requirements:** Python 3.10+, ~4 GB disk (model weights cached by HuggingFace)

```bash
git clone https://github.com/yourusername/uae-sentiment-review
cd uae-sentiment-review/sentiment_analyzer
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads model weights (~400 MB each). Subsequent runs load from cache.

**Demo mode.** If you run without installing `transformers`, the app falls back to a lightweight keyword heuristic so the UI remains fully functional.

---

## Deployment

**Streamlit Community Cloud (free)**

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub, select `sentiment_analyzer/app.py`
4. Deploy

Note: Community Cloud has a 1 GB memory limit. To stay within it, either use only the heuristic mode or upgrade to a paid tier for full transformer inference.

**Local / Server**

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

---

## Real-time Data Sources

The scraper pulls from two public APIs — no keys, no rate limit concerns for personal use:

- **Google News RSS** — `news.google.com/rss/search?q={query}` returns structured XML parsed by feedparser
- **Reddit JSON** — `reddit.com/r/{sub}/search.json` returns public post data

Both are cached with Streamlit's `@st.cache_data(ttl=300)` to avoid repeated requests within a session.

---

## Design

The UI is intentionally editorial — modelled on FT/NYT data journalism rather than a SaaS dashboard. Typography is the primary design element. Color is used only to convey information (sentiment polarity).

Fonts: Playfair Display (display), Source Serif 4 (body), IBM Plex Mono (data), Noto Naskh Arabic (Arabic text).

---

## Sample Data

`data/sample_reviews.py` includes 500 hand-curated UAE reviews covering:

- Brands: Talabat, Emirates, Careem, Dubai Mall, Burj Al Arab, ADNOC, Noon, LuLu
- Languages: Arabic MSA, Gulf dialect, English, Arabizi (Latin-script Arabic)
- All three sentiment classes well-represented

---

## License

MIT. Use freely, attribution appreciated.

---

*Built as a portfolio project demonstrating bilingual NLP, real-time data integration, and editorial data visualisation.*
