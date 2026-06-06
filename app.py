"""
UAE Sentiment Intelligence Review
Bilingual NLP Dashboard · Arabic + English · Real-time Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import sys
import os
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Sentiment Intelligence Review",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Google Fonts + CSS ───────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400;1,600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300;1,8..60,400&family=IBM+Plex+Mono:wght@300;400;500&family=Noto+Naskh+Arabic:wght@400;500&display=swap" rel="stylesheet">

<style>
  /* ── Reset & Base ── */
  html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background: #0a0a0a;
    color: #F5F0E8;
  }

  .stApp { background: #0a0a0a; }

  /* Hide Streamlit chrome */
  header[data-testid="stHeader"] { display: none; }
  .stDeployButton { display: none; }
  #MainMenu { display: none; }
  footer { display: none; }
  .block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

  /* ── Masthead ── */
  .masthead {
    border-top: 1px solid #1f1f1f;
    border-bottom: 1px solid #1f1f1f;
    padding: 2.2rem 0 1.8rem 0;
    margin-bottom: 2.5rem;
    text-align: center;
  }
  .masthead-title {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 400;
    font-size: 2.4rem;
    color: #F5F0E8;
    letter-spacing: 0.02em;
    margin: 0;
    line-height: 1.15;
  }
  .masthead-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #333330;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.55rem;
  }

  /* ── Section rules ── */
  .section-rule {
    border: none;
    border-top: 1px solid #1f1f1f;
    margin: 1.8rem 0;
  }
  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #333330;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
  }

  /* ── Verdict display ── */
  .verdict-positive {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #4A7C59;
    letter-spacing: 0.04em;
  }
  .verdict-negative {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #8B2E2E;
    letter-spacing: 0.04em;
  }
  .verdict-neutral {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2rem;
    font-style: italic;
    color: #555550;
    letter-spacing: 0.04em;
  }
  .confidence-footnote {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #555550;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
  }

  /* ── Lede ── */
  .article-lede {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.25rem;
    color: #C9A84C;
    line-height: 1.55;
    margin-bottom: 1.2rem;
  }

  /* ── Aspect list with dot leaders ── */
  .aspect-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.8rem 0;
  }
  .aspect-table tr {
    border-bottom: 1px solid #161616;
  }
  .aspect-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #888880;
    padding: 0.45rem 0;
    text-transform: lowercase;
    letter-spacing: 0.04em;
    width: 35%;
  }
  .aspect-dots {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #333330;
    letter-spacing: 0.15em;
    padding: 0 0.5rem;
  }
  .aspect-sentiment {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-align: right;
    padding: 0.45rem 0;
  }
  .asp-positive { color: #4A7C59; }
  .asp-negative { color: #8B2E2E; }
  .asp-neutral  { color: #555550; }

  /* ── Pull quotes ── */
  .pull-quote {
    border-left: 1px solid #1f1f1f;
    padding: 0.6rem 0 0.6rem 1rem;
    margin: 0.8rem 0;
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.9rem;
    color: #888880;
    line-height: 1.55;
  }
  .pull-quote-positive { border-left-color: #2d4a36; }
  .pull-quote-negative { border-left-color: #4a1a1a; }

  /* ── Word frequency table ── */
  .freq-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
  }
  .freq-table th {
    color: #333330;
    text-align: left;
    font-weight: 400;
    padding: 0.3rem 0.5rem 0.5rem 0;
    border-bottom: 1px solid #1f1f1f;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.6rem;
  }
  .freq-table td {
    color: #888880;
    padding: 0.35rem 0.5rem 0.35rem 0;
    border-bottom: 1px solid #111111;
  }
  .freq-num { color: #333330; width: 2rem; }
  .freq-word { color: #F5F0E8; }
  .freq-count { color: #555550; text-align: right; }

  /* ── Brand scorecard ── */
  .scorecard {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
  }
  .scorecard th {
    font-size: 0.58rem;
    color: #333330;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid #1f1f1f;
    text-align: right;
  }
  .scorecard th:first-child { text-align: left; }
  .scorecard td {
    font-size: 0.75rem;
    color: #888880;
    padding: 0.55rem 0.6rem;
    border-bottom: 1px solid #111111;
    text-align: right;
  }
  .scorecard td:first-child {
    text-align: left;
    color: #F5F0E8;
    font-size: 0.8rem;
  }
  .score-positive { color: #4A7C59 !important; }
  .score-negative { color: #8B2E2E !important; }
  .score-neutral  { color: #555550 !important; }

  /* ── Insight card ── */
  .insight-card {
    border-left: 2px solid #1f1f1f;
    padding: 0.7rem 0 0.7rem 1rem;
    margin: 0.6rem 0;
  }
  .insight-high   { border-left-color: #8B2E2E; }
  .insight-medium { border-left-color: #C9A84C; }
  .insight-low    { border-left-color: #4A7C59; }
  .insight-priority {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
  }
  .insight-finding {
    font-family: 'Source Serif 4', serif;
    font-size: 0.88rem;
    color: #F5F0E8;
    margin-bottom: 0.2rem;
  }
  .insight-rec {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.8rem;
    color: #888880;
  }
  .insight-metric {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #333330;
    margin-top: 0.2rem;
  }

  /* ── Arabic text ── */
  .arabic-text {
    font-family: 'Noto Naskh Arabic', serif;
    font-size: 1.05rem;
    direction: rtl;
    text-align: right;
    color: #F5F0E8;
    line-height: 1.8;
  }

  /* ── Input styling ── */
  .stTextArea textarea {
    background: #111111 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 0 !important;
    color: #F5F0E8 !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.9rem !important;
    caret-color: #FCD0A1;
  }
  .stTextArea textarea::placeholder { color: #333330 !important; }
  .stTextArea textarea:focus {
    border-color: #FCD0A1 !important;
    box-shadow: none !important;
  }

  /* ── Selectbox ── */
  .stSelectbox > div > div {
    background: #111111 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 0 !important;
    color: #F5F0E8 !important;
  }

  /* ── Button ── */
  .stButton > button {
    background: #0a0a0a !important;
    border: 1px solid #333330 !important;
    border-radius: 0 !important;
    color: #888880 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    border-color: #FCD0A1 !important;
    color: #FCD0A1 !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1f1f1f !important;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #333330 !important;
    padding: 0.6rem 1.4rem !important;
  }
  .stTabs [aria-selected="true"] {
    color: #F5F0E8 !important;
    border-bottom: 1px solid #F5F0E8 !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 2rem !important;
  }

  /* ── File uploader ── */
  .stFileUploader > div {
    background: #111111 !important;
    border: 1px dashed #1f1f1f !important;
    border-radius: 0 !important;
  }

  /* ── Metric ── */
  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
    color: #FCD0A1 !important;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6rem !important;
    color: #333330 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
  }

  /* ── Progress / spinner ── */
  .stSpinner { color: #555550 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 3px; }
  ::-webkit-scrollbar-track { background: #0a0a0a; }
  ::-webkit-scrollbar-thumb { background: #1f1f1f; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Lazy-load NLP pipeline."""
    from models.unified_pipeline import analyze, analyze_dataframe
    return analyze, analyze_dataframe


@st.cache_data(show_spinner=False, ttl=600)
def get_sample_data():
    from data.sample_reviews import get_sample_reviews
    return get_sample_reviews()


@st.cache_data(show_spinner=False, ttl=300)
def get_live_data(brand: str):
    from data.scraper import fetch_brand_live
    return fetch_brand_live(brand, max_items=30)


def detect_lang_display(lang: str) -> str:
    return {'arabic': 'العربية', 'english': 'English', 'mixed': 'Mixed'}.get(lang, lang)


def sentiment_html(sentiment: str) -> str:
    cls = f'verdict-{sentiment}'
    label_map = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
    return f'<span class="{cls}">{label_map.get(sentiment, sentiment)}</span>'


def aspect_table_html(aspects: dict) -> str:
    if not aspects:
        return ''
    rows = []
    icons = {'positive': '▲', 'negative': '▼', 'neutral': '—'}
    color_map = {'positive': 'asp-positive', 'negative': 'asp-negative', 'neutral': 'asp-neutral'}

    for aspect, sent in sorted(aspects.items(), key=lambda x: {'negative': 0, 'positive': 2, 'neutral': 1}[x[1]]):
        dots = '·' * max(4, 20 - len(aspect))
        icon = icons.get(sent, '—')
        cls = color_map.get(sent, 'asp-neutral')
        rows.append(
            f'<tr><td class="aspect-label">{aspect}</td>'
            f'<td class="aspect-dots">{dots}</td>'
            f'<td class="aspect-sentiment {cls}">{sent}&nbsp;&nbsp;{icon}</td></tr>'
        )
    return f'<table class="aspect-table">{"".join(rows)}</table>'


def word_freq_html(en_freq: list, ar_freq: list) -> str:
    max_rows = max(len(en_freq), len(ar_freq))
    rows = []
    for i in range(min(max_rows, 10)):
        num = f'{i+1:02d}'
        en_w = en_freq[i][0] if i < len(en_freq) else ''
        en_c = str(en_freq[i][1]) if i < len(en_freq) else ''
        ar_w = ar_freq[i][0] if i < len(ar_freq) else ''
        ar_c = str(ar_freq[i][1]) if i < len(ar_freq) else ''
        rows.append(
            f'<tr>'
            f'<td class="freq-num">{num}</td>'
            f'<td class="freq-word">{en_w}</td>'
            f'<td class="freq-count">{en_c}</td>'
            f'<td style="width:1.5rem"></td>'
            f'<td class="freq-num">{num}</td>'
            f'<td class="freq-word" style="font-family:\'Noto Naskh Arabic\',serif;direction:rtl;text-align:right">{ar_w}</td>'
            f'<td class="freq-count">{ar_c}</td>'
            f'</tr>'
        )
    header = (
        '<tr>'
        '<th colspan="3" style="text-align:left">English</th>'
        '<th style="width:1.5rem"></th>'
        '<th colspan="3" style="text-align:right">Arabic</th>'
        '</tr>'
    )
    return f'<table class="freq-table">{header}{"".join(rows)}</table>'


def insight_html(insights: list) -> str:
    html = ''
    priority_colors = {'HIGH': '#8B2E2E', 'MEDIUM': '#C9A84C', 'LOW': '#4A7C59'}
    priority_class = {'HIGH': 'insight-high', 'MEDIUM': 'insight-medium', 'LOW': 'insight-low'}
    for ins in insights:
        p = ins.get('priority', 'LOW')
        color = priority_colors.get(p, '#555550')
        cls = priority_class.get(p, 'insight-low')
        html += f"""
        <div class="insight-card {cls}">
          <div class="insight-priority" style="color:{color}">{p} PRIORITY</div>
          <div class="insight-finding">{ins.get('finding','')}</div>
          <div class="insight-rec">{ins.get('recommendation','')}</div>
          <div class="insight-metric">{ins.get('metric','')}</div>
        </div>"""
    return html


def scorecard_row(brand, score, pos, neg, neu, count, velocity):
    score_cls = 'score-positive' if score > 0.1 else 'score-negative' if score < -0.1 else 'score-neutral'
    return (
        f'<tr>'
        f'<td>{brand}</td>'
        f'<td class="{score_cls}">{score:+.3f}</td>'
        f'<td class="score-positive">{pos:.0f}%</td>'
        f'<td class="score-negative">{neg:.0f}%</td>'
        f'<td class="score-neutral">{neu:.0f}%</td>'
        f'<td>{count}</td>'
        f'<td style="font-size:0.65rem">{velocity}</td>'
        f'</tr>'
    )


# ── Masthead ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="masthead">
  <div class="masthead-title">Sentiment Intelligence Review</div>
  <div class="masthead-sub">VOL. 1 &nbsp;·&nbsp; UAE Market Edition &nbsp;·&nbsp; AR + EN &nbsp;·&nbsp; {datetime.now().strftime('%B %Y')}</div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Analyzer", "Batch Analyzer", "Brand Dashboard", "Brand Comparison"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE ANALYZER
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_spacer, col_result = st.columns([5, 0.3, 4])

    with col_input:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        example_texts = [
            "الطلب وصل متأخر ساعة كاملة والأكل كان بارد تماماً. مخيب للآمال جداً",
            "Talabat delivery in 20 mins as always. Love the tracking feature.",
            "Business class on EK was phenomenal. Flat bed, chef meals, impeccable crew.",
            "7addi zain Talabat, delivery was super fast w el akel kaan tazeej!",
            "The app crashed twice during checkout. Lost my promo code. Terrible UX.",
        ]
        example = st.selectbox(
            "Load example",
            ["— type your own —"] + example_texts,
            label_visibility="collapsed"
        )

        default_text = '' if example == "— type your own —" else example
        user_text = st.text_area(
            "Enter review",
            value=default_text,
            height=140,
            placeholder="Type or paste a review in Arabic, English, or mixed…",
            label_visibility="collapsed"
        )

        analyze_btn = st.button("ANALYZE →", key="analyze_btn")

    with col_result:
        if analyze_btn and user_text.strip():
            with st.spinner(""):
                try:
                    analyze_fn, _ = load_pipeline()
                    result = analyze_fn(user_text.strip())

                    # Language detected
                    lang = result.get('language', 'english')
                    st.markdown(
                        f'<div class="section-label">{detect_lang_display(lang)} · {result.get("model","")}</div>',
                        unsafe_allow_html=True
                    )

                    # Input display
                    if lang == 'arabic':
                        st.markdown(f'<div class="arabic-text">{user_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="pull-quote">{user_text[:220]}</div>', unsafe_allow_html=True)

                    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

                    # Verdict
                    sentiment = result.get('sentiment', 'neutral')
                    confidence = result.get('confidence', 0.0)
                    emotion = result.get('emotion', '')

                    st.markdown(sentiment_html(sentiment), unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="confidence-footnote">'
                        f'confidence: {confidence:.4f} · {emotion}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Label scores
                    label_scores = result.get('label_scores', {})
                    if label_scores:
                        st.markdown(
                            f'<div class="confidence-footnote" style="margin-top:0.3rem">'
                            f'pos {label_scores.get("positive",0):.3f} &nbsp;·&nbsp; '
                            f'neu {label_scores.get("neutral",0):.3f} &nbsp;·&nbsp; '
                            f'neg {label_scores.get("negative",0):.3f}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

                    # Aspects
                    aspects = result.get('aspects', {})
                    if aspects:
                        st.markdown('<div class="section-label">Aspect Breakdown</div>', unsafe_allow_html=True)
                        st.markdown(aspect_table_html(aspects), unsafe_allow_html=True)

                except ImportError as e:
                    # Fallback when transformers not installed (demo mode)
                    _demo_fallback(user_text, col_result)

        elif analyze_btn:
            st.markdown('<div class="confidence-footnote">No text provided.</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin-top:3rem">
              <div class="article-lede">Enter any review to begin.</div>
              <div class="confidence-footnote" style="line-height:1.9">
                Arabic · English · Arabizi<br>
                CAMeL-BERT for Arabic text<br>
                RoBERTa-Twitter for English<br>
                Aspect extraction · Emotion tagging
              </div>
            </div>
            """, unsafe_allow_html=True)


def _demo_fallback(text: str, col):
    """Lightweight demo when HuggingFace models not loaded."""
    import re
    # Simple heuristic for demo
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total = arabic_chars + latin_chars
    lang = 'arabic' if total > 0 and arabic_chars / total > 0.7 else 'english'

    neg_kw = ['متأخر', 'بارد', 'مخيب', 'سيء', 'terrible', 'awful', 'crashed', 'bad', 'worst', 'poor', 'cold', 'late', 'wrong', 'broken']
    pos_kw = ['ممتاز', 'رائع', 'مميز', 'excellent', 'amazing', 'love', 'great', 'best', 'fast', 'perfect', 'phenomenal', 'good']
    text_lower = text.lower()
    neg_score = sum(1 for w in neg_kw if w in text_lower)
    pos_score = sum(1 for w in pos_kw if w in text_lower)

    if pos_score > neg_score:
        sentiment, confidence = 'positive', round(0.72 + pos_score * 0.04, 4)
    elif neg_score > pos_score:
        sentiment, confidence = 'negative', round(0.70 + neg_score * 0.04, 4)
    else:
        sentiment, confidence = 'neutral', 0.6512

    confidence = min(confidence, 0.98)
    model_name = 'CAMeL-BERT (demo)' if lang == 'arabic' else 'RoBERTa (demo)'

    col.markdown(f'<div class="section-label">{detect_lang_display(lang)} · {model_name}</div>', unsafe_allow_html=True)
    if lang == 'arabic':
        col.markdown(f'<div class="arabic-text">{text}</div>', unsafe_allow_html=True)
    else:
        col.markdown(f'<div class="pull-quote">{text[:220]}</div>', unsafe_allow_html=True)
    col.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    col.markdown(sentiment_html(sentiment), unsafe_allow_html=True)
    col.markdown(
        f'<div class="confidence-footnote">confidence: {confidence:.4f} · model: {model_name}</div>',
        unsafe_allow_html=True
    )
    col.markdown(
        f'<div class="confidence-footnote" style="margin-top:0.2rem;color:#333330">'
        f'Install transformers for full inference</div>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYZER
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Upload CSV · text column required</div>', unsafe_allow_html=True)

    col_up, col_sp, col_stats = st.columns([4, 0.5, 5])

    with col_up:
        uploaded_file = st.file_uploader(
            "CSV",
            type=['csv'],
            label_visibility="collapsed"
        )

        use_sample = st.button("USE SAMPLE DATASET", key="sample_btn")

        if use_sample or uploaded_file is not None:
            if use_sample:
                df_raw = get_sample_data()
                st.markdown(
                    f'<div class="confidence-footnote">Loaded {len(df_raw)} sample UAE reviews.</div>',
                    unsafe_allow_html=True
                )
            else:
                df_raw = pd.read_csv(uploaded_file)
                st.markdown(
                    f'<div class="confidence-footnote">Loaded {len(df_raw)} rows from upload.</div>',
                    unsafe_allow_html=True
                )

            # Detect text column
            text_cols = [c for c in df_raw.columns if 'text' in c.lower() or 'review' in c.lower() or 'comment' in c.lower()]
            text_col = text_cols[0] if text_cols else df_raw.columns[0]

            st.markdown(
                f'<div class="confidence-footnote" style="margin-top:0.4rem">Text column: <span style="color:#F5F0E8">{text_col}</span></div>',
                unsafe_allow_html=True
            )

            run_batch = st.button("SCORE ALL ROWS →", key="batch_run")

            if run_batch:
                st.session_state['batch_df'] = df_raw
                st.session_state['batch_text_col'] = text_col

    with col_stats:
        if 'batch_df' in st.session_state:
            df_raw = st.session_state['batch_df']
            text_col = st.session_state.get('batch_text_col', 'text')

            with st.spinner("Scoring reviews…"):
                try:
                    _, analyze_df_fn = load_pipeline()
                    scored = analyze_df_fn(df_raw, text_col)
                except ImportError:
                    # Demo mode: heuristic scoring
                    scored = _batch_demo(df_raw, text_col)

            st.session_state['scored_df'] = scored

            total = len(scored)
            pos = (scored['sentiment'] == 'positive').sum()
            neg = (scored['sentiment'] == 'negative').sum()
            neu = (scored['sentiment'] == 'neutral').sum()

            # Editorial summary
            dominant = 'positive' if pos > neg else 'negative' if neg > pos else 'neutral'
            dom_pct = max(pos, neg, neu) / total * 100

            st.markdown(
                f'<div class="article-lede">{dom_pct:.0f}% of {total:,} reviews skew {dominant} '
                f'— {neg} critical voices require attention.</div>',
                unsafe_allow_html=True
            )

            # Stats row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", f"{total:,}")
            m2.metric("Positive", f"{pos/total*100:.0f}%")
            m3.metric("Negative", f"{neg/total*100:.0f}%")
            m4.metric("Neutral", f"{neu/total*100:.0f}%")

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

            # Preview table — styled
            preview = scored[['text', 'sentiment', 'confidence', 'language']].head(8)
            st.markdown('<div class="section-label">Sample Output</div>', unsafe_allow_html=True)

            table_rows = ''
            for _, row in preview.iterrows():
                sent_color = {'positive': '#4A7C59', 'negative': '#8B2E2E', 'neutral': '#555550'}.get(row['sentiment'], '#555550')
                excerpt = str(row['text'])[:60] + '…' if len(str(row['text'])) > 60 else str(row['text'])
                table_rows += (
                    f'<tr>'
                    f'<td style="color:#888880;font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;max-width:280px;overflow:hidden;text-overflow:ellipsis">{excerpt}</td>'
                    f'<td style="color:{sent_color};font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;text-align:center">{row["sentiment"]}</td>'
                    f'<td style="color:#555550;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;text-align:right">{float(row["confidence"]):.3f}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<table class="scorecard"><thead><tr>'
                f'<th style="text-align:left">Review</th>'
                f'<th>Sentiment</th>'
                f'<th>Conf.</th>'
                f'</tr></thead><tbody>{table_rows}</tbody></table>',
                unsafe_allow_html=True
            )

            # Download
            st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
            csv_out = scored.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "DOWNLOAD SCORED CSV",
                data=csv_out,
                file_name=f"sentiment_scored_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )


def _batch_demo(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Heuristic batch scoring for demo mode."""
    import re as re_
    neg_kw = ['متأخر', 'بارد', 'مخيب', 'سيء', 'terrible', 'awful', 'crashed', 'bad', 'worst', 'poor', 'cold', 'late', 'wrong', 'broken', 'never', 'avoid', 'rude', 'slow']
    pos_kw = ['ممتاز', 'رائع', 'excellent', 'amazing', 'love', 'great', 'best', 'fast', 'perfect', 'phenomenal', 'good', 'clean', 'helpful', 'recommend']

    rows = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ''))
        text_l = text.lower()
        arabic_chars = len(re_.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re_.findall(r'[a-zA-Z]', text))
        total = arabic_chars + latin_chars
        lang = 'arabic' if total > 0 and arabic_chars / total > 0.7 else 'english'
        neg = sum(1 for w in neg_kw if w in text_l)
        pos = sum(1 for w in pos_kw if w in text_l)
        if pos > neg:
            sent, conf = 'positive', round(0.70 + pos * 0.03, 3)
        elif neg > pos:
            sent, conf = 'negative', round(0.70 + neg * 0.03, 3)
        else:
            sent, conf = 'neutral', 0.612
        new_row = row.to_dict()
        new_row.update({'sentiment': sent, 'confidence': min(conf, 0.97), 'language': lang, 'model': f'heuristic-demo'})
        rows.append(new_row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BRAND DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    BRANDS = ['Talabat', 'Emirates', 'Careem', 'Dubai Mall', 'Burj Al Arab', 'ADNOC', 'Noon', 'LuLu']

    col_sel, col_sp3, col_dash = st.columns([2, 0.3, 7])

    with col_sel:
        st.markdown('<div class="section-label">Select Brand</div>', unsafe_allow_html=True)
        brand_sel = st.selectbox("Brand", BRANDS, label_visibility="collapsed")
        live_btn = st.button("FETCH LIVE DATA", key="live_btn")
        st.markdown(
            '<div class="confidence-footnote" style="margin-top:0.5rem">Pulls from Reddit + Google News RSS. No API key required.</div>',
            unsafe_allow_html=True
        )

    with col_dash:
        # Get data
        base_df = get_sample_data()

        if live_btn:
            with st.spinner(f"Fetching live {brand_sel} data…"):
                live_items = get_live_data(brand_sel)
                if live_items:
                    live_df = pd.DataFrame(live_items)
                    live_df['language'] = live_df['text'].apply(
                        lambda t: 'arabic' if len(re.findall(r'[\u0600-\u06FF]', str(t))) / max(len(re.findall(r'[a-zA-Z\u0600-\u06FF]', str(t))), 1) > 0.5 else 'english'
                    )
                    # Quick heuristic score live data
                    live_df = _batch_demo(live_df, 'text')
                    live_df['brand'] = brand_sel
                    live_df['timestamp'] = pd.to_datetime(live_df.get('timestamp', datetime.now().isoformat()))
                    brand_df = pd.concat([base_df[base_df['brand'] == brand_sel], live_df], ignore_index=True)
                    st.session_state[f'live_{brand_sel}'] = brand_df
                    st.markdown(
                        f'<div class="confidence-footnote">+{len(live_items)} live items ingested.</div>',
                        unsafe_allow_html=True
                    )

        # Use cached live data if available
        if f'live_{brand_sel}' in st.session_state:
            brand_df = st.session_state[f'live_{brand_sel}']
        else:
            brand_df = base_df[base_df['brand'] == brand_sel].copy()

        if brand_df.empty:
            st.markdown('<div class="confidence-footnote">No data for this brand.</div>', unsafe_allow_html=True)
        else:
            from analysis.trend_analyzer import (
                compute_sentiment_timeline, compute_velocity,
                get_top_reviews, get_word_frequency, actionable_insights
            )
            from utils.charts import sentiment_trend_line

            total = len(brand_df)
            pos = (brand_df['sentiment'] == 'positive').sum()
            neg = (brand_df['sentiment'] == 'negative').sum()
            neu = (brand_df['sentiment'] == 'neutral').sum()
            avg_conf = brand_df['confidence'].mean() if 'confidence' in brand_df.columns else 0.75

            # Lede
            dom = 'positive' if pos > neg else 'negative' if neg > pos else 'neutral'
            st.markdown(
                f'<div class="article-lede">{brand_sel} registers predominantly {dom} sentiment '
                f'across {total} reviews — {pos/total*100:.0f}% positive, {neg/total*100:.0f}% negative.</div>',
                unsafe_allow_html=True
            )

            # Stats
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Reviews", f"{total}")
            m2.metric("Positive", f"{pos/total*100:.0f}%")
            m3.metric("Negative", f"{neg/total*100:.0f}%")
            m4.metric("Neutral", f"{neu/total*100:.0f}%")
            m5.metric("Avg Conf.", f"{avg_conf:.2f}")

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

            # Trend chart
            timeline = compute_sentiment_timeline(brand_df, brand_sel, freq='W')
            if not timeline.empty:
                vel_label, vel_val = compute_velocity(timeline)
                st.markdown(
                    f'<div class="section-label">Sentiment Trend &nbsp; <span style="color:#555550">{vel_label}</span></div>',
                    unsafe_allow_html=True
                )
                fig = sentiment_trend_line(timeline, brand_sel)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Two columns: aspects + word frequency
            c_asp, c_freq = st.columns(2)

            with c_asp:
                st.markdown('<div class="section-label">Aspect Analysis</div>', unsafe_allow_html=True)
                # Aggregate aspects from sample
                from analysis.aspect_extractor import extract_aspects
                aspect_tallies = {}
                for _, row in brand_df.head(50).iterrows():
                    aspects = extract_aspects(str(row['text']), row.get('language', 'english'))
                    for asp, sent in aspects.items():
                        if asp not in aspect_tallies:
                            aspect_tallies[asp] = {'positive': 0, 'negative': 0, 'neutral': 0}
                        aspect_tallies[asp][sent] += 1

                # Compute dominant sentiment per aspect
                agg_aspects = {}
                for asp, counts in aspect_tallies.items():
                    dom_sent = max(counts, key=counts.get)
                    agg_aspects[asp] = dom_sent

                st.markdown(aspect_table_html(agg_aspects), unsafe_allow_html=True)

            with c_freq:
                st.markdown('<div class="section-label">Word Frequency</div>', unsafe_allow_html=True)
                en_freq = get_word_frequency(brand_df, brand_sel, 'english', 10)
                ar_freq = get_word_frequency(brand_df, brand_sel, 'arabic', 10)
                st.markdown(word_freq_html(en_freq, ar_freq), unsafe_allow_html=True)

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

            # Pull quotes
            c_pos, c_neg = st.columns(2)

            with c_pos:
                st.markdown('<div class="section-label">Top Positive Reviews</div>', unsafe_allow_html=True)
                for quote in get_top_reviews(brand_df, brand_sel, 'positive', 3):
                    is_ar = len(re.findall(r'[\u0600-\u06FF]', quote)) > len(quote) * 0.2
                    if is_ar:
                        st.markdown(f'<div class="pull-quote pull-quote-positive arabic-text">{quote[:180]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="pull-quote pull-quote-positive">{quote[:180]}</div>', unsafe_allow_html=True)

            with c_neg:
                st.markdown('<div class="section-label">Top Negative Reviews</div>', unsafe_allow_html=True)
                for quote in get_top_reviews(brand_df, brand_sel, 'negative', 3):
                    is_ar = len(re.findall(r'[\u0600-\u06FF]', quote)) > len(quote) * 0.2
                    if is_ar:
                        st.markdown(f'<div class="pull-quote pull-quote-negative arabic-text">{quote[:180]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="pull-quote pull-quote-negative">{quote[:180]}</div>', unsafe_allow_html=True)

            # Actionable insights
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Actionable Intelligence</div>', unsafe_allow_html=True)

            insights = actionable_insights(brand_df, brand_sel)
            if insights:
                st.markdown(insight_html(insights), unsafe_allow_html=True)
            else:
                st.markdown('<div class="confidence-footnote">Insufficient data for insight generation.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — BRAND COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Compare Two Brands</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        brand_a = st.selectbox("Brand A", BRANDS, index=0, label_visibility="collapsed", key="ba")
    with c2:
        brand_b = st.selectbox("Brand B", BRANDS, index=1, label_visibility="collapsed", key="bb")

    if brand_a == brand_b:
        st.markdown('<div class="confidence-footnote">Select two different brands.</div>', unsafe_allow_html=True)
    else:
        base_df = get_sample_data()
        from analysis.trend_analyzer import (
            compute_brand_scores, compute_sentiment_timeline,
            compute_velocity, get_word_frequency
        )
        from utils.charts import brand_comparison_chart

        # Build comparison data
        comp_df = base_df[base_df['brand'].isin([brand_a, brand_b])].copy()
        brand_scores = compute_brand_scores(comp_df)

        # Lede
        if not brand_scores.empty and len(brand_scores) >= 2:
            top_brand = brand_scores.iloc[0]['brand']
            bottom_brand = brand_scores.iloc[-1]['brand']
            gap = brand_scores.iloc[0]['avg_score'] - brand_scores.iloc[-1]['avg_score']
            st.markdown(
                f'<div class="article-lede">{top_brand} leads on sentiment score by {gap:.2f} points — '
                f'a meaningful gap that reflects measurable differences in customer experience delivery.</div>',
                unsafe_allow_html=True
            )

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # Scorecard table
        st.markdown('<div class="section-label">Editorial Scorecard</div>', unsafe_allow_html=True)

        rows_html = ''
        for _, row in brand_scores.iterrows():
            tl_a = compute_sentiment_timeline(comp_df[comp_df['brand'] == row['brand']])
            vel_label, _ = compute_velocity(tl_a) if not tl_a.empty else ('—', 0)
            rows_html += scorecard_row(
                row['brand'],
                row['avg_score'],
                row['pos_pct'],
                row['neg_pct'],
                row['neu_pct'],
                row['review_count'],
                vel_label,
            )

        st.markdown(f"""
        <table class="scorecard">
          <thead>
            <tr>
              <th>Brand</th>
              <th>Score</th>
              <th>Positive</th>
              <th>Negative</th>
              <th>Neutral</th>
              <th>Reviews</th>
              <th>Trajectory</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # Side-by-side trend charts
        tc1, tc2 = st.columns(2)
        for col_chart, brand_name in [(tc1, brand_a), (tc2, brand_b)]:
            timeline = compute_sentiment_timeline(comp_df[comp_df['brand'] == brand_name])
            with col_chart:
                st.markdown(f'<div class="section-label">{brand_name}</div>', unsafe_allow_html=True)
                if not timeline.empty:
                    from utils.charts import sentiment_trend_line
                    fig = sentiment_trend_line(timeline, brand_name)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # Word frequency comparison
        st.markdown('<div class="section-label">Vocabulary Comparison</div>', unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)

        for col_w, brand_name in [(wc1, brand_a), (wc2, brand_b)]:
            with col_w:
                st.markdown(f'<div class="section-label" style="font-size:0.55rem">{brand_name}</div>', unsafe_allow_html=True)
                en_freq = get_word_frequency(comp_df, brand_name, 'english', 8)
                ar_freq = get_word_frequency(comp_df, brand_name, 'arabic', 8)
                st.markdown(word_freq_html(en_freq, ar_freq), unsafe_allow_html=True)

        # All brands overview chart
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">All Brands Ranked</div>', unsafe_allow_html=True)
        all_scores = compute_brand_scores(base_df)
        if not all_scores.empty:
            fig_all = brand_comparison_chart(all_scores)
            st.plotly_chart(fig_all, use_container_width=True, config={'displayModeBar': False})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="section-rule" style="margin-top:3rem">
<div style="display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0">
  <div class="confidence-footnote">UAE SENTIMENT INTELLIGENCE REVIEW · AR + EN BILINGUAL NLP</div>
  <div class="confidence-footnote">CAMeL-BERT · RoBERTa-Twitter · Real-time RSS</div>
</div>
""", unsafe_allow_html=True)
