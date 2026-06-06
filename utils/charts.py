"""
Editorial charts — FT/NYT style.
Minimal, data-forward, no decorative elements.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Tuple, Optional

# ── Design tokens ──────────────────────────────────────────────────────────
C = {
    'bg': 'rgba(0,0,0,0)',
    'ft_salmon': '#FCD0A1',
    'gold': '#C9A84C',
    'positive': '#4A7C59',
    'negative': '#8B2E2E',
    'neutral': '#555550',
    'text_primary': '#F5F0E8',
    'text_secondary': '#888880',
    'text_muted': '#333330',
    'border': '#1f1f1f',
    'axis': '#333330',
}

FONT = {
    'body': 'Source Serif 4, Georgia, serif',
    'mono': 'IBM Plex Mono, monospace',
}

BASE_LAYOUT = dict(
    paper_bgcolor=C['bg'],
    plot_bgcolor=C['bg'],
    font=dict(family=FONT['body'], color=C['text_secondary'], size=12),
    margin=dict(l=0, r=0, t=40, b=0),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=C['axis'],
        linewidth=1,
        tickfont=dict(family=FONT['mono'], color=C['text_muted'], size=10),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(family=FONT['mono'], color=C['text_muted'], size=10),
    ),
)


def sentiment_trend_line(
    timeline: pd.DataFrame,
    brand: str = '',
    height: int = 220
) -> go.Figure:
    """
    Minimal sentiment trend line.
    FT salmon primary line, muted fill.
    """
    if timeline.empty:
        return go.Figure()

    fig = go.Figure()

    # Smoothed line
    fig.add_trace(go.Scatter(
        x=timeline['timestamp'],
        y=timeline['smoothed'],
        mode='lines',
        line=dict(color=C['ft_salmon'], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(252,208,161,0.07)',
        hovertemplate='%{x|%b %d}<br>Score: %{y:.3f}<extra></extra>',
    ))

    # Neutral zero line
    fig.add_hline(
        y=0,
        line=dict(color=C['text_muted'], width=0.5, dash='dot'),
    )

    layout = dict(**BASE_LAYOUT)
    layout['height'] = height
    layout['title'] = dict(
        text=f'{brand} Sentiment Trend' if brand else 'Sentiment Trend',
        font=dict(family=FONT['mono'], color=C['text_muted'], size=10),
        x=0,
        xanchor='left',
    )
    layout['yaxis']['range'] = [-1.1, 1.1]
    layout['yaxis']['tickvals'] = [-1, -0.5, 0, 0.5, 1]
    layout['yaxis']['ticktext'] = ['−1', '−½', '0', '+½', '+1']
    fig.update_layout(**layout)

    return fig


def brand_comparison_chart(
    brand_scores: pd.DataFrame,
    height: int = 260
) -> go.Figure:
    """Horizontal lollipop chart for brand comparison."""
    if brand_scores.empty:
        return go.Figure()

    df = brand_scores.sort_values('avg_score', ascending=True)
    colors = [C['positive'] if s > 0 else C['negative'] if s < -0.1 else C['neutral']
              for s in df['avg_score']]

    fig = go.Figure()

    # Stem lines
    for i, (brand, score) in enumerate(zip(df['brand'], df['avg_score'])):
        fig.add_trace(go.Scatter(
            x=[0, score],
            y=[brand, brand],
            mode='lines',
            line=dict(color=C['border'], width=1),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Dots
    fig.add_trace(go.Scatter(
        x=df['avg_score'],
        y=df['brand'],
        mode='markers+text',
        marker=dict(color=colors, size=8, line=dict(width=0)),
        text=[f'{s:+.2f}' for s in df['avg_score']],
        textposition='middle right',
        textfont=dict(family=FONT['mono'], color=C['text_muted'], size=9),
        hovertemplate='%{y}<br>Score: %{x:.3f}<extra></extra>',
    ))

    layout = dict(**BASE_LAYOUT)
    layout['height'] = height
    layout['xaxis']['range'] = [-1.1, 1.3]
    layout['xaxis']['zeroline'] = True
    layout['xaxis']['zerolinecolor'] = C['axis']
    layout['xaxis']['zerolinewidth'] = 0.5
    layout['yaxis']['showline'] = False
    layout['yaxis']['tickfont'] = dict(family=FONT['mono'], color=C['text_secondary'], size=11)
    layout['margin'] = dict(l=20, r=60, t=40, b=10)
    layout['title'] = dict(
        text='Brand Sentiment Scores',
        font=dict(family=FONT['mono'], color=C['text_muted'], size=10),
        x=0, xanchor='left',
    )
    fig.update_layout(**layout)

    return fig


def sentiment_distribution_bar(
    pos: float, neu: float, neg: float,
    height: int = 60
) -> go.Figure:
    """Stacked horizontal bar — sentiment split."""
    fig = go.Figure(go.Bar(
        x=[pos],
        y=[''],
        orientation='h',
        marker_color=C['positive'],
        width=0.4,
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Bar(
        x=[neu],
        y=[''],
        orientation='h',
        marker_color=C['neutral'],
        width=0.4,
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Bar(
        x=[neg],
        y=[''],
        orientation='h',
        marker_color=C['negative'],
        width=0.4,
        showlegend=False,
        hoverinfo='skip',
    ))

    layout = dict(**BASE_LAYOUT)
    layout['barmode'] = 'stack'
    layout['height'] = height
    layout['xaxis']['range'] = [0, 100]
    layout['xaxis']['showline'] = False
    layout['margin'] = dict(l=0, r=0, t=0, b=0)
    fig.update_layout(**layout)
    return fig


def confidence_gauge(confidence: float, height: int = 80) -> go.Figure:
    """Thin line gauge for confidence score."""
    color = C['positive'] if confidence > 0.75 else C['ft_salmon'] if confidence > 0.5 else C['negative']

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=confidence * 100,
        number=dict(
            suffix='%',
            font=dict(family=FONT['mono'], color=C['text_secondary'], size=14),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=0,
                tickcolor=C['bg'],
                tickfont=dict(family=FONT['mono'], color=C['text_muted'], size=8),
            ),
            bar=dict(color=color, thickness=0.2),
            bgcolor=C['bg'],
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color='rgba(139,46,46,0.08)'),
                dict(range=[50, 75], color='rgba(85,85,80,0.08)'),
                dict(range=[75, 100], color='rgba(74,124,89,0.08)'),
            ],
            threshold=dict(
                line=dict(color=C['text_muted'], width=1),
                thickness=0.6,
                value=confidence * 100,
            )
        ),
    ))
    layout = dict(**BASE_LAYOUT)
    layout['height'] = height
    layout['margin'] = dict(l=10, r=10, t=10, b=10)
    fig.update_layout(**layout)
    return fig
