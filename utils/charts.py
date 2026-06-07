import plotly.graph_objects as go
import pandas as pd

C = {
    'bg':      'rgba(0,0,0,0)',
    'line':    '#4ade80',
    'neg':     '#f87171',
    'neu':     '#555',
    'axis':    '#2a2a2a',
    'text':    '#555',
    'fill':    'rgba(74,222,128,0.05)',
}

BASE = dict(
    paper_bgcolor=C['bg'], plot_bgcolor=C['bg'],
    font=dict(family='JetBrains Mono, monospace', color=C['text'], size=10),
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=C['axis'],
               linewidth=1, tickfont=dict(size=9, color=C['text'])),
    yaxis=dict(showgrid=False, zeroline=False, showline=False,
               tickfont=dict(size=9, color=C['text'])),
)


def sentiment_trend_line(timeline: pd.DataFrame, brand: str = '', height: int = 200) -> go.Figure:
    if timeline.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline['timestamp'], y=timeline['smoothed'],
        mode='lines', line=dict(color=C['line'], width=1.2),
        fill='tozeroy', fillcolor=C['fill'],
        hovertemplate='%{x|%d %b}<br>%{y:.3f}<extra></extra>',
    ))
    fig.add_hline(y=0, line=dict(color=C['axis'], width=0.8))

    layout = dict(**BASE)
    layout['height'] = height
    layout['title'] = dict(text=brand, font=dict(size=9, color=C['text']), x=0, xanchor='left')
    layout['yaxis']['range'] = [-1.1, 1.1]
    layout['yaxis']['tickvals'] = [-1, 0, 1]
    layout['yaxis']['ticktext'] = ['-1', '0', '+1']
    fig.update_layout(**layout)
    return fig


def brand_comparison_chart(brand_scores: pd.DataFrame, height: int = 240) -> go.Figure:
    if brand_scores.empty:
        return go.Figure()

    df = brand_scores.sort_values('avg_score', ascending=True)
    colors = [('#4ade80' if s > 0.1 else '#f87171' if s < -0.1 else '#555') for s in df['avg_score']]

    fig = go.Figure()
    for i, (brand, score) in enumerate(zip(df['brand'], df['avg_score'])):
        fig.add_trace(go.Scatter(
            x=[0, score], y=[brand, brand],
            mode='lines', line=dict(color='#2a2a2a', width=1),
            showlegend=False, hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(
        x=df['avg_score'], y=df['brand'],
        mode='markers+text',
        marker=dict(color=colors, size=7),
        text=[f'{s:+.2f}' for s in df['avg_score']],
        textposition='middle right',
        textfont=dict(family='JetBrains Mono, monospace', color='#555', size=9),
        hovertemplate='%{y}: %{x:.3f}<extra></extra>',
    ))

    layout = dict(**BASE)
    layout['height'] = height
    layout['xaxis']['range'] = [-1.1, 1.4]
    layout['xaxis']['zeroline'] = True
    layout['xaxis']['zerolinecolor'] = C['axis']
    layout['xaxis']['zerolinewidth'] = 0.8
    layout['yaxis']['tickfont'] = dict(size=10, color='#888')
    layout['margin'] = dict(l=20, r=60, t=20, b=10)
    fig.update_layout(**layout)
    return fig
