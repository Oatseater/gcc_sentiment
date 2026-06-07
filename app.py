import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(
    page_title="GCC Sentiment · UAE Market",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject fonts via link tag separately — some Streamlit versions choke on combined font+style blocks
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Noto+Naskh+Arabic:wght@400;500&display=swap" rel="stylesheet">', unsafe_allow_html=True)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; background: #0f0f0f !important; color: #e2e2e2 !important; }
.stApp { background: #0f0f0f !important; }
header[data-testid="stHeader"] { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1280px !important; }
.topbar { display: flex; align-items: center; justify-content: space-between; padding: 0.9rem 0; border-bottom: 1px solid #232323; margin-bottom: 1.4rem; }
.topbar-left { display: flex; align-items: center; gap: 1rem; }
.app-name { font-size: 0.85rem; font-weight: 600; color: #e2e2e2; letter-spacing: 0.01em; }
.app-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #555; background: #1a1a1a; border: 1px solid #2a2a2a; padding: 0.15rem 0.5rem; letter-spacing: 0.05em; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #232323 !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important; font-family: 'Inter', sans-serif !important; font-size: 0.72rem !important; font-weight: 500 !important; color: #555 !important; padding: 0.55rem 1.2rem !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; }
.stTabs [aria-selected="true"] { color: #e2e2e2 !important; border-bottom: 1px solid #e2e2e2 !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 1.8rem !important; }
.sec { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #444; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.6rem; }
.stat-row { display: flex; gap: 1px; margin: 1.2rem 0; background: #232323; }
.stat-cell { flex: 1; background: #0f0f0f; padding: 0.9rem 1rem; border-top: 2px solid #232323; }
.stat-cell.pos { border-top-color: #2d6a4f; }
.stat-cell.neg { border-top-color: #7a1f1f; }
.stat-cell.neu { border-top-color: #333; }
.stat-cell.hi  { border-top-color: #8a6a00; }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 500; color: #e2e2e2; }
.stat-lbl { font-size: 0.62rem; color: #555; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.08em; }
.verdict { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 500; letter-spacing: 0.04em; }
.v-pos { color: #4ade80; }
.v-neg { color: #f87171; }
.v-neu { color: #888; }
.conf-line { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #444; margin-top: 0.3rem; }
.dt { width: 100%; border-collapse: collapse; }
.dt th { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: #444; text-transform: uppercase; letter-spacing: 0.1em; padding: 0.4rem 0.6rem 0.5rem; border-bottom: 1px solid #232323; text-align: left; }
.dt th.r { text-align: right; }
.dt td { font-size: 0.78rem; color: #b0b0b0; padding: 0.45rem 0.6rem; border-bottom: 1px solid #1a1a1a; }
.dt td.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; }
.dt td.r { text-align: right; }
.dt td.brand { color: #e2e2e2; font-weight: 500; }
.c-pos { color: #4ade80 !important; }
.c-neg { color: #f87171 !important; }
.c-neu { color: #888 !important; }
.asp-row { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid #1a1a1a; font-size: 0.78rem; }
.asp-name { color: #888; text-transform: capitalize; }
.asp-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; padding: 0.1rem 0.45rem; letter-spacing: 0.05em; }
.ab-pos { color: #4ade80; background: #0d2318; }
.ab-neg { color: #f87171; background: #200e0e; }
.ab-neu { color: #888; background: #1a1a1a; }
.rev-card { border-left: 2px solid #232323; padding: 0.6rem 0 0.6rem 0.9rem; margin-bottom: 0.6rem; font-size: 0.8rem; color: #999; line-height: 1.55; }
.rev-card.pos { border-left-color: #2d6a4f; }
.rev-card.neg { border-left-color: #7a1f1f; }
.rev-ar { font-family: 'Noto Naskh Arabic', serif; direction: rtl; text-align: right; font-size: 0.88rem; }
.ins { border-left: 2px solid #2a2a2a; padding: 0.6rem 0 0.6rem 0.9rem; margin-bottom: 0.6rem; }
.ins.high { border-left-color: #7a1f1f; }
.ins.med  { border-left-color: #8a6a00; }
.ins.low  { border-left-color: #2d6a4f; }
.ins-p { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.2rem; }
.ip-high { color: #f87171; }
.ip-med  { color: #fbbf24; }
.ip-low  { color: #4ade80; }
.ins-f { font-size: 0.82rem; color: #e2e2e2; margin-bottom: 0.15rem; }
.ins-r { font-size: 0.76rem; color: #666; }
.ins-m { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #333; margin-top: 0.15rem; }
.div { border: none; border-top: 1px solid #1a1a1a; margin: 1.4rem 0; }
.stTextArea textarea { background: #141414 !important; border: 1px solid #2a2a2a !important; border-radius: 3px !important; color: #e2e2e2 !important; font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important; }
.stTextArea textarea:focus { border-color: #555 !important; box-shadow: none !important; }
.stSelectbox > div > div { background: #141414 !important; border: 1px solid #2a2a2a !important; border-radius: 3px !important; color: #e2e2e2 !important; }
.stButton > button { background: #1a1a1a !important; border: 1px solid #2a2a2a !important; border-radius: 3px !important; color: #888 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; padding: 0.45rem 1rem !important; }
.stButton > button:hover { border-color: #555 !important; color: #e2e2e2 !important; }
.stFileUploader > div { background: #141414 !important; border: 1px dashed #2a2a2a !important; border-radius: 3px !important; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.3rem !important; color: #e2e2e2 !important; }
[data-testid="stMetricLabel"] { font-size: 0.6rem !important; color: #444 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #0f0f0f; }
::-webkit-scrollbar-thumb { background: #2a2a2a; }
</style>
""", unsafe_allow_html=True)


# Top bar
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <span class="app-name">GCC Sentiment</span>
    <span class="app-tag">UAE · AR+EN · NLP</span>
    <span class="app-tag">CAMeL-BERT · RoBERTa</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Loaders ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    from models.unified_pipeline import analyze, analyze_dataframe
    return analyze, analyze_dataframe


@st.cache_data(show_spinner=False, ttl=600)
def get_sample_data():
    """Returns sample reviews pre-scored with heuristic so sentiment column always exists."""
    from data.sample_reviews import get_sample_reviews
    df = get_sample_reviews()
    # Score with heuristic so brand dashboard works without transformer inference
    return _score_df(df, 'text')


@st.cache_data(show_spinner=False, ttl=300)
def get_live_data(brand: str):
    from data.scraper import fetch_brand_live
    return fetch_brand_live(brand, max_items=30)


# ── Utilities ────────────────────────────────────────────────────────────────

def _heuristic(text: str):
    neg_kw = ['متأخر','بارد','مخيب','سيء','فظيع','terrible','awful','crashed','bad','worst',
              'poor','cold','late','wrong','broken','never','avoid','rude','slow','disgusting']
    pos_kw = ['ممتاز','رائع','excellent','amazing','love','great','best','fast','perfect',
              'phenomenal','good','clean','helpful','recommend','happy','smooth','impressed']
    t = text.lower()
    neg = sum(1 for w in neg_kw if w in t)
    pos = sum(1 for w in pos_kw if w in t)
    if pos > neg:   return 'positive', min(0.70 + pos * 0.04, 0.97)
    elif neg > pos: return 'negative', min(0.70 + neg * 0.04, 0.97)
    return 'neutral', 0.62


def _score_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ''))
        sent, conf = _heuristic(text)
        ar = len(re.findall(r'[\u0600-\u06FF]', text))
        en = len(re.findall(r'[a-zA-Z]', text))
        lang = 'arabic' if (ar + en) > 0 and ar / (ar + en) > 0.7 else 'english'
        nr = row.to_dict()
        nr.setdefault('sentiment', sent)   # don't overwrite if already scored
        nr.setdefault('confidence', round(conf, 3))
        nr.setdefault('language', lang)
        nr.setdefault('model', 'heuristic')
        rows.append(nr)
    return pd.DataFrame(rows)


def _is_arabic(text: str) -> bool:
    ar = len(re.findall(r'[\u0600-\u06FF]', str(text)))
    en = len(re.findall(r'[a-zA-Z]', str(text)))
    return (ar + en) > 0 and ar / (ar + en) > 0.7


def _asp_html(aspects: dict) -> str:
    cls = {'positive': 'ab-pos', 'negative': 'ab-neg', 'neutral': 'ab-neu'}
    order = {'negative': 0, 'neutral': 1, 'positive': 2}
    out = ''
    for asp, sent in sorted(aspects.items(), key=lambda x: order.get(x[1], 1)):
        out += (f'<div class="asp-row">'
                f'<span class="asp-name">{asp}</span>'
                f'<span class="asp-badge {cls.get(sent, "ab-neu")}">{sent}</span>'
                f'</div>')
    return out


def _ins_html(insights: list) -> str:
    pc = {'HIGH': 'ip-high', 'MEDIUM': 'ip-med', 'LOW': 'ip-low'}
    cc = {'HIGH': 'high', 'MEDIUM': 'med', 'LOW': 'low'}
    out = ''
    for ins in insights:
        p = ins.get('priority', 'LOW')
        out += (f'<div class="ins {cc.get(p, "low")}">'
                f'<div class="ins-p {pc.get(p, "ip-low")}">{p}</div>'
                f'<div class="ins-f">{ins.get("finding", "")}</div>'
                f'<div class="ins-r">{ins.get("recommendation", "")}</div>'
                f'<div class="ins-m">{ins.get("metric", "")}</div>'
                f'</div>')
    return out


def _freq_html(en_freq, ar_freq):
    rows = ''
    for i in range(min(max(len(en_freq), len(ar_freq)), 10)):
        n = f'{i+1:02d}'
        ew = en_freq[i][0] if i < len(en_freq) else ''
        ec = str(en_freq[i][1]) if i < len(en_freq) else ''
        aw = ar_freq[i][0] if i < len(ar_freq) else ''
        ac = str(ar_freq[i][1]) if i < len(ar_freq) else ''
        rows += (f'<tr>'
                 f'<td class="mono" style="color:#333">{n}</td>'
                 f'<td>{ew}</td>'
                 f'<td class="mono r" style="color:#333">{ec}</td>'
                 f'<td style="width:1.5rem"></td>'
                 f'<td class="mono" style="color:#333">{n}</td>'
                 f'<td style="font-family:Noto Naskh Arabic,serif;direction:rtl;text-align:right">{aw}</td>'
                 f'<td class="mono r" style="color:#333">{ac}</td>'
                 f'</tr>')
    return (f'<table class="dt"><thead><tr>'
            f'<th colspan="3">English</th><th></th>'
            f'<th colspan="3" style="text-align:right">Arabic</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')


def _scorecard_html(brand_scores, comp_df):
    from analysis.trend_analyzer import compute_sentiment_timeline, compute_velocity
    rows = ''
    for _, row in brand_scores.iterrows():
        tl = compute_sentiment_timeline(comp_df[comp_df['brand'] == row['brand']])
        vel, _ = compute_velocity(tl) if not tl.empty else ('stable', 0)
        sc = row['avg_score']
        sc_cls = 'c-pos' if sc > 0.1 else 'c-neg' if sc < -0.1 else 'c-neu'
        vel_cls = 'c-pos' if vel == 'improving' else 'c-neg' if vel == 'declining' else 'c-neu'
        rows += (f'<tr>'
                 f'<td class="brand">{row["brand"]}</td>'
                 f'<td class="mono r {sc_cls}">{sc:+.3f}</td>'
                 f'<td class="mono r c-pos">{row["pos_pct"]:.0f}%</td>'
                 f'<td class="mono r c-neg">{row["neg_pct"]:.0f}%</td>'
                 f'<td class="mono r c-neu">{row["neu_pct"]:.0f}%</td>'
                 f'<td class="mono r">{row["review_count"]}</td>'
                 f'<td class="mono r {vel_cls}">{vel}</td>'
                 f'</tr>')
    return (f'<table class="dt"><thead><tr>'
            f'<th>Brand</th><th class="r">Score</th><th class="r">Pos%</th>'
            f'<th class="r">Neg%</th><th class="r">Neu%</th><th class="r">n</th><th class="r">Trend</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')


BRANDS = ['Talabat', 'Emirates', 'Careem', 'Dubai Mall', 'Burj Al Arab', 'ADNOC', 'Noon', 'LuLu']

tab1, tab2, tab3, tab4 = st.tabs(["Live Analyzer", "Batch Analyzer", "Brand Dashboard", "Brand Comparison"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Analyzer
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    c_in, _, c_out = st.columns([5, 0.4, 4])

    with c_in:
        st.markdown('<div class="sec">Input</div>', unsafe_allow_html=True)
        examples = [
            "— select example —",
            "الطلب وصل متأخر ساعة كاملة والأكل كان بارد تماماً. مخيب للآمال جداً",
            "Talabat delivery in 20 mins as always. Love the tracking feature.",
            "Business class on EK was phenomenal. Flat bed, chef meals, impeccable crew.",
            "7addi zain Talabat, delivery was super fast w el akel kaan tazeej!",
            "The app crashed twice during checkout. Lost my promo code. Terrible UX.",
        ]
        ex = st.selectbox("Example", examples, label_visibility="collapsed")
        default = '' if ex == examples[0] else ex
        user_text = st.text_area(
            "Review text", value=default, height=120,
            placeholder="Paste any review — Arabic, English, or mixed.",
            label_visibility="collapsed"
        )
        btn = st.button("Run Analysis", key="run_live")

    with c_out:
        if btn and user_text.strip():
            with st.spinner(""):
                try:
                    analyze_fn, _ = load_pipeline()
                    result = analyze_fn(user_text.strip())
                except Exception:
                    from analysis.aspect_extractor import extract_aspects
                    lang = 'arabic' if _is_arabic(user_text) else 'english'
                    sent, conf = _heuristic(user_text.strip())
                    result = {
                        'sentiment': sent, 'confidence': conf, 'language': lang,
                        'model': 'heuristic', 'emotion': '',
                        'aspects': extract_aspects(user_text, lang),
                        'label_scores': {}
                    }

            sentiment  = result['sentiment']
            confidence = result['confidence']
            lang       = result.get('language', 'english')
            model_name = result.get('model', '—')
            aspects    = result.get('aspects', {})
            label_scores = result.get('label_scores', {})
            cls = {'positive': 'v-pos', 'negative': 'v-neg', 'neutral': 'v-neu'}.get(sentiment, 'v-neu')

            st.markdown('<div class="sec">Result</div>', unsafe_allow_html=True)

            if _is_arabic(user_text):
                st.markdown(f'<div class="rev-ar" style="color:#b0b0b0;margin-bottom:0.8rem">{user_text[:200]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:0.82rem;color:#666;font-style:italic;margin-bottom:0.8rem;line-height:1.5">"{user_text[:180]}"</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="verdict {cls}">{sentiment.upper()}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf-line">conf {confidence:.4f} · {model_name} · {lang}</div>', unsafe_allow_html=True)

            if label_scores:
                st.markdown(
                    f'<div class="conf-line" style="margin-top:0.2rem">'
                    f'pos {label_scores.get("positive", 0):.3f} · '
                    f'neu {label_scores.get("neutral", 0):.3f} · '
                    f'neg {label_scores.get("negative", 0):.3f}</div>',
                    unsafe_allow_html=True
                )

            if aspects:
                st.markdown('<hr class="div"><div class="sec">Aspects</div>', unsafe_allow_html=True)
                st.markdown(_asp_html(aspects), unsafe_allow_html=True)

        elif btn:
            st.markdown('<div class="conf-line">No input provided.</div>', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="margin-top:1rem">
              <div class="conf-line" style="line-height:2.2">
                AR / EN / Arabizi detection<br>
                CAMeL-BERT → Arabic<br>
                RoBERTa-Twitter → English<br>
                Aspect extraction · Emotion tagging
              </div>
            </div>''', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analyzer
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">Upload CSV — requires text column</div>', unsafe_allow_html=True)
    c_up, _, c_res = st.columns([4, 0.4, 5])

    with c_up:
        f = st.file_uploader("CSV file", type=['csv'], label_visibility="collapsed")
        use_sample = st.button("Use sample dataset", key="use_sample")

        if use_sample or f is not None:
            df_raw = get_sample_data() if use_sample else pd.read_csv(f)
            text_cols = [c for c in df_raw.columns if any(k in c.lower() for k in ['text', 'review', 'comment'])]
            text_col = text_cols[0] if text_cols else df_raw.columns[0]
            st.markdown(f'<div class="conf-line">{len(df_raw):,} rows · column: {text_col}</div>', unsafe_allow_html=True)

            if st.button("Score all rows →", key="run_batch"):
                st.session_state['batch_df'] = df_raw
                st.session_state['batch_col'] = text_col

    with c_res:
        if 'batch_df' in st.session_state:
            df_raw  = st.session_state['batch_df']
            text_col = st.session_state.get('batch_col', 'text')

            with st.spinner("Scoring…"):
                try:
                    _, adf = load_pipeline()
                    scored = adf(df_raw, text_col)
                except Exception:
                    scored = _score_df(df_raw, text_col)

            st.session_state['scored_df'] = scored

            total = len(scored)
            pos   = (scored['sentiment'] == 'positive').sum()
            neg   = (scored['sentiment'] == 'negative').sum()
            neu   = (scored['sentiment'] == 'neutral').sum()

            st.markdown(f'''<div class="stat-row">
              <div class="stat-cell pos"><div class="stat-val">{pos/total*100:.0f}%</div><div class="stat-lbl">Positive</div></div>
              <div class="stat-cell neg"><div class="stat-val">{neg/total*100:.0f}%</div><div class="stat-lbl">Negative</div></div>
              <div class="stat-cell neu"><div class="stat-val">{neu/total*100:.0f}%</div><div class="stat-lbl">Neutral</div></div>
              <div class="stat-cell"><div class="stat-val">{total:,}</div><div class="stat-lbl">Total</div></div>
            </div>''', unsafe_allow_html=True)

            st.markdown('<hr class="div"><div class="sec">Preview — first 8 rows</div>', unsafe_allow_html=True)
            rows_html = ''
            for _, row in scored.head(8).iterrows():
                sent = str(row.get('sentiment', '—'))
                cls  = 'c-pos' if sent == 'positive' else 'c-neg' if sent == 'negative' else 'c-neu'
                excerpt = str(row.get(text_col, ''))
                excerpt = excerpt[:65] + '…' if len(excerpt) > 65 else excerpt
                conf_val = float(row.get('confidence', 0))
                rows_html += (f'<tr><td>{excerpt}</td>'
                              f'<td class="mono r {cls}">{sent}</td>'
                              f'<td class="mono r" style="color:#444">{conf_val:.3f}</td></tr>')

            st.markdown(
                f'<table class="dt"><thead><tr>'
                f'<th>Text</th><th class="r">Sentiment</th><th class="r">Conf</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True
            )

            csv_out = scored.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "Download scored CSV", data=csv_out,
                file_name=f"scored_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Brand Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    c_sel, _, c_main = st.columns([2.2, 0.3, 7])

    with c_sel:
        st.markdown('<div class="sec">Brand</div>', unsafe_allow_html=True)
        brand_sel = st.selectbox("Brand", BRANDS, label_visibility="collapsed")
        live_btn  = st.button("Fetch live data", key="fetch_live")
        st.markdown(
            '<div class="conf-line" style="margin-top:0.4rem">Reddit · Google News RSS<br>No API key required</div>',
            unsafe_allow_html=True
        )

    with c_main:
        # get_sample_data() already pre-scored — always has sentiment column
        base_df = get_sample_data()

        if live_btn:
            with st.spinner(f"Fetching {brand_sel}…"):
                live_items = get_live_data(brand_sel)
                if live_items:
                    ldf = pd.DataFrame(live_items)
                    ldf = _score_df(ldf, 'text')
                    ldf['brand'] = brand_sel
                    ldf['timestamp'] = datetime.now().isoformat()
                    brand_df = pd.concat(
                        [base_df[base_df['brand'] == brand_sel], ldf],
                        ignore_index=True
                    )
                    st.session_state[f'live_{brand_sel}'] = brand_df
                    st.markdown(f'<div class="conf-line">+{len(live_items)} live items</div>', unsafe_allow_html=True)

        brand_df = st.session_state.get(
            f'live_{brand_sel}',
            base_df[base_df['brand'] == brand_sel].copy()
        )

        if brand_df.empty:
            st.markdown('<div class="conf-line">No data for this brand.</div>', unsafe_allow_html=True)
        else:
            from analysis.trend_analyzer import (
                compute_sentiment_timeline, compute_velocity,
                get_top_reviews, get_word_frequency, actionable_insights
            )
            from utils.charts import sentiment_trend_line

            total = len(brand_df)
            pos   = int((brand_df['sentiment'] == 'positive').sum())
            neg   = int((brand_df['sentiment'] == 'negative').sum())
            neu   = int((brand_df['sentiment'] == 'neutral').sum())
            conf  = float(brand_df['confidence'].mean()) if 'confidence' in brand_df.columns else 0.0

            st.markdown(f'''<div class="stat-row">
              <div class="stat-cell"><div class="stat-val">{total}</div><div class="stat-lbl">Reviews</div></div>
              <div class="stat-cell pos"><div class="stat-val">{pos/total*100:.0f}%</div><div class="stat-lbl">Positive</div></div>
              <div class="stat-cell neg"><div class="stat-val">{neg/total*100:.0f}%</div><div class="stat-lbl">Negative</div></div>
              <div class="stat-cell neu"><div class="stat-val">{neu/total*100:.0f}%</div><div class="stat-lbl">Neutral</div></div>
              <div class="stat-cell hi"><div class="stat-val">{conf:.2f}</div><div class="stat-lbl">Avg Conf</div></div>
            </div>''', unsafe_allow_html=True)

            # Trend chart
            timeline = compute_sentiment_timeline(brand_df, brand_sel, freq='W')
            if not timeline.empty:
                vel_label, vel_val = compute_velocity(timeline)
                vel_cls = 'c-pos' if vel_label == 'improving' else 'c-neg' if vel_label == 'declining' else 'c-neu'
                st.markdown(
                    f'<div class="sec" style="margin-top:1rem">Sentiment trend &nbsp;'
                    f'<span class="{vel_cls}" style="font-family:JetBrains Mono,monospace;font-size:0.6rem">'
                    f'{vel_label} &Delta;{vel_val:+.3f}</span></div>',
                    unsafe_allow_html=True
                )
                fig = sentiment_trend_line(timeline, brand_sel)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"trend_{brand_sel}")

            st.markdown('<hr class="div">', unsafe_allow_html=True)

            ca, cw = st.columns(2)
            with ca:
                st.markdown('<div class="sec">Aspect breakdown</div>', unsafe_allow_html=True)
                from analysis.aspect_extractor import extract_aspects
                tally = {}
                for _, row in brand_df.head(60).iterrows():
                    lang = row.get('language', 'english')
                    for asp, sent in extract_aspects(str(row['text']), lang).items():
                        tally.setdefault(asp, {'positive': 0, 'negative': 0, 'neutral': 0})[sent] += 1
                agg = {asp: max(counts, key=counts.get) for asp, counts in tally.items()}
                st.markdown(_asp_html(agg), unsafe_allow_html=True)

            with cw:
                st.markdown('<div class="sec">Top terms</div>', unsafe_allow_html=True)
                en_freq = get_word_frequency(brand_df, brand_sel, 'english', 10)
                ar_freq = get_word_frequency(brand_df, brand_sel, 'arabic', 10)
                st.markdown(_freq_html(en_freq, ar_freq), unsafe_allow_html=True)

            st.markdown('<hr class="div">', unsafe_allow_html=True)

            cp, cn = st.columns(2)
            with cp:
                st.markdown('<div class="sec">Top positive</div>', unsafe_allow_html=True)
                for q in get_top_reviews(brand_df, brand_sel, 'positive', 3):
                    ar_cls = 'rev-ar' if _is_arabic(q) else ''
                    st.markdown(f'<div class="rev-card pos {ar_cls}">{q[:200]}</div>', unsafe_allow_html=True)
            with cn:
                st.markdown('<div class="sec">Top negative</div>', unsafe_allow_html=True)
                for q in get_top_reviews(brand_df, brand_sel, 'negative', 3):
                    ar_cls = 'rev-ar' if _is_arabic(q) else ''
                    st.markdown(f'<div class="rev-card neg {ar_cls}">{q[:200]}</div>', unsafe_allow_html=True)

            st.markdown('<hr class="div">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Actionable insights</div>', unsafe_allow_html=True)
            insights = actionable_insights(brand_df, brand_sel)
            if insights:
                st.markdown(_ins_html(insights), unsafe_allow_html=True)
            else:
                st.markdown('<div class="conf-line">Insufficient data.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Brand Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        brand_a = st.selectbox("Brand A", BRANDS, index=0, label_visibility="collapsed", key="ba")
    with c2:
        brand_b = st.selectbox("Brand B", BRANDS, index=1, label_visibility="collapsed", key="bb")

    if brand_a == brand_b:
        st.markdown('<div class="conf-line">Select two different brands.</div>', unsafe_allow_html=True)
    else:
        from analysis.trend_analyzer import compute_brand_scores, compute_sentiment_timeline, compute_velocity, get_word_frequency
        from utils.charts import brand_comparison_chart, sentiment_trend_line

        base_df    = get_sample_data()
        comp_df    = base_df[base_df['brand'].isin([brand_a, brand_b])].copy()
        brand_scores = compute_brand_scores(comp_df)

        st.markdown('<hr class="div"><div class="sec">Scorecard</div>', unsafe_allow_html=True)
        st.markdown(_scorecard_html(brand_scores, comp_df), unsafe_allow_html=True)

        st.markdown('<hr class="div"><div class="sec">Sentiment trend</div>', unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        for col_c, bn in [(tc1, brand_a), (tc2, brand_b)]:
            with col_c:
                st.markdown(f'<div class="sec">{bn}</div>', unsafe_allow_html=True)
                tl = compute_sentiment_timeline(comp_df[comp_df['brand'] == bn])
                if not tl.empty:
                    st.plotly_chart(
                        sentiment_trend_line(tl, bn),
                        use_container_width=True,
                        config={'displayModeBar': False},
                        key=f"cmp_trend_{bn}"
                    )

        st.markdown('<hr class="div"><div class="sec">Top terms</div>', unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        for col_w, bn in [(wc1, brand_a), (wc2, brand_b)]:
            with col_w:
                st.markdown(f'<div class="sec" style="font-size:0.55rem">{bn}</div>', unsafe_allow_html=True)
                st.markdown(
                    _freq_html(
                        get_word_frequency(comp_df, bn, 'english', 8),
                        get_word_frequency(comp_df, bn, 'arabic', 8)
                    ),
                    unsafe_allow_html=True
                )

        st.markdown('<hr class="div"><div class="sec">All brands ranked</div>', unsafe_allow_html=True)
        all_scores = compute_brand_scores(base_df)
        if not all_scores.empty:
            st.plotly_chart(
                brand_comparison_chart(all_scores),
                use_container_width=True,
                config={'displayModeBar': False},
                key="all_brands_ranked"
            )

# Footer
st.markdown("""
<hr class="div" style="margin-top:2rem">
<div style="display:flex;justify-content:space-between;padding:0.4rem 0">
  <span class="conf-line">GCC Sentiment · UAE Market · AR+EN</span>
  <span class="conf-line">CAMeL-BERT · RoBERTa-Twitter · Reddit · Google News</span>
</div>
""", unsafe_allow_html=True)
