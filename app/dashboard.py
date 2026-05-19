"""
Player Style Explorer
Run from the project root:
    streamlit run app/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.index import find_similar

st.set_page_config(
    page_title="Forma",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark pitch-green theme */
[data-testid="stAppViewContainer"] {
    background: #0a0f0a;
    color: #e8ede8;
}
[data-testid="stSidebar"] {
    background: #0f1a0f;
    border-right: 1px solid #1e3a1e;
}
[data-testid="stSidebar"] * {
    color: #c8d8c8 !important;
}

/* Header */
.dna-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.08em;
    color: #7fff7f;
    line-height: 1;
    margin-bottom: 0;
}
.dna-sub {
    font-size: 0.85rem;
    color: #5a8a5a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2px;
    margin-bottom: 1.5rem;
}

/* Section headers */
.section-head {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.1em;
    color: #7fff7f;
    border-bottom: 1px solid #1e3a1e;
    padding-bottom: 6px;
    margin-bottom: 1rem;
}

/* Stat card */
.stat-card {
    background: #0f1a0f;
    border: 1px solid #1e3a1e;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.stat-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a8a5a;
    margin-bottom: 3px;
}
.stat-value {
    font-size: 1.1rem;
    font-weight: 500;
    color: #e8ede8;
}

/* Player result row */
.result-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 6px;
    background: #0f1a0f;
    border: 1px solid #1e3a1e;
    transition: border-color 0.15s;
}
.result-row:hover { border-color: #3a6a3a; }
.result-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    color: #3a6a3a;
    width: 22px;
    flex-shrink: 0;
}
.result-name { font-weight: 500; font-size: 0.9rem; color: #e8ede8; flex: 1; }
.result-meta { font-size: 0.75rem; color: #5a8a5a; }
.result-dist {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.95rem;
    color: #7fff7f;
    letter-spacing: 0.05em;
}
.arch-pill {
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* Streamlit overrides */
.stSelectbox label, .stSlider label, .stTextInput label {
    color: #7fff7f !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] {
    background: #0f1a0f;
    border: 1px solid #1e3a1e;
    border-radius: 8px;
    padding: 12px 16px;
}
.stMetric label { color: #5a8a5a !important; }

hr { border-color: #1e3a1e; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/player_meta_aligned.csv', index_col='matrix_row')
    return df

df = load_data()

# Archetype colour palette (10 archetypes + uncategorised)
ARCH_COLORS = {
    'Shot-stopper':           '#FFD700',
    'Generalist':             '#A0A0A0',
    'Box-to-box engine':      '#FF6B35',
    'Ball-playing defender':  '#4ECDC4',
    'Possession system elite':'#7FFF7F',
    'Creative playmaker':     '#C77DFF',
    'Wide progressive':       '#FF9F1C',
    'Defensive anchor':       '#2196F3',
    'Aggressive stopper':     '#F44336',
    'Stopper CB':             '#78909C',
    'Attacking fullback':     '#00BCD4',
    'Aerial commander':       '#8D6E63',
    'Uncategorised':          '#444444',
}

archetypes  = sorted(df['archetype'].dropna().unique().tolist())
named_df    = df[df['name'] != '—'].copy()

with st.sidebar:
    st.markdown('<div class="dna-title">Forma</div>', unsafe_allow_html=True)
    st.markdown('<div class="dna-sub">Forma v1.0</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**FILTER STYLE MAP**")

    selected_archetypes = st.multiselect(
        "Archetypes",
        options=archetypes,
        default=archetypes,
        label_visibility="collapsed",
    )

    show_noise = st.checkbox("Show uncategorised players", value=False)

    st.markdown("---")
    st.markdown("**SIMILARITY SEARCH**")

    query = st.text_input("Player name", placeholder="e.g. Messi, Busquets…")
    top_k = st.slider("Results", min_value=5, max_value=20, value=10)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#3a6a3a;letter-spacing:0.08em">'
        'DATA: StatsBomb Open Data<br>'
        'La Liga 2012–2021<br>'
        '12-token action vocabulary<br>'
        'Word2Vec → UMAP → HDBSCAN'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="dna-title">Forma</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dna-sub">Player style embeddings from football event sequences · La Liga 2012–2021</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Players mapped",   f"{len(named_df):,}")
c2.metric("Archetypes found", str(len([a for a in archetypes if a != 'Uncategorised'])))
c3.metric("Seasons covered",  "9")
c4.metric("Action tokens",    "12")

st.markdown("---")

st.markdown('<div class="section-head">Style Map</div>', unsafe_allow_html=True)
st.caption("Every dot is a player. Position = playing style. Colour = archetype cluster.")

# Filter
mask = df['archetype'].isin(selected_archetypes)
if not show_noise:
    mask = mask & (df['label'] != -1)
plot_df = df[mask].copy()
plot_df['name_display'] = plot_df['name'].where(plot_df['name'] != '—', 'Unknown')

fig = px.scatter(
    plot_df,
    x='x', y='y',
    color='archetype',
    color_discrete_map=ARCH_COLORS,
    hover_name='name_display',
    hover_data={'position': True, 'archetype': True, 'x': False, 'y': False},
    template='plotly_dark',
    height=540,
)
fig.update_traces(
    marker=dict(size=5, opacity=0.82, line=dict(width=0)),
)
fig.update_layout(
    paper_bgcolor='#0a0f0a',
    plot_bgcolor='#0a0f0a',
    font=dict(family='DM Sans', color='#c8d8c8', size=12),
    legend=dict(
        title=dict(text='ARCHETYPE', font=dict(size=10, color='#5a8a5a')),
        bgcolor='#0f1a0f',
        bordercolor='#1e3a1e',
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, width='stretch')

st.markdown("---")
st.markdown('<div class="section-head">Archetype Breakdown</div>', unsafe_allow_html=True)

arch_counts = (
    df[df['label'] != -1]
    .groupby('archetype')
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=True)
)

bar = go.Figure(go.Bar(
    x=arch_counts['count'],
    y=arch_counts['archetype'],
    orientation='h',
    marker=dict(
        color=[ARCH_COLORS.get(a, '#444') for a in arch_counts['archetype']],
        opacity=0.85,
    ),
    text=arch_counts['count'],
    textposition='outside',
    textfont=dict(color='#5a8a5a', size=11),
))
bar.update_layout(
    paper_bgcolor='#0a0f0a',
    plot_bgcolor='#0a0f0a',
    font=dict(family='DM Sans', color='#c8d8c8'),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
    yaxis=dict(showgrid=False, zeroline=False, title=''),
    height=360,
    margin=dict(l=10, r=60, t=10, b=10),
)
st.plotly_chart(bar, width='stretch')

st.markdown("---")
st.markdown('<div class="section-head">Similarity Search</div>', unsafe_allow_html=True)

if query.strip():
    matches = named_df[named_df['name'].str.contains(query.strip(), case=False, na=False)]

    if matches.empty:
        st.warning(f'No player found matching "{query}". Try a surname only.')
    else:
        target_row = matches.iloc[0]
        target_idx = matches.index[0]

        # Query player card
        arch_color = ARCH_COLORS.get(target_row['archetype'], '#444')
        st.markdown(
            f"""
            <div class="stat-card" style="border-color:{arch_color}40;margin-bottom:1.2rem">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;
                            color:{arch_color};letter-spacing:0.06em">
                    {target_row['name']}
                </div>
                <div style="display:flex;gap:12px;margin-top:6px;flex-wrap:wrap">
                    <span class="arch-pill" style="background:{arch_color}22;color:{arch_color}">
                        {target_row['archetype']}
                    </span>
                    <span style="font-size:0.8rem;color:#5a8a5a">{target_row['position']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Run similarity
        results = find_similar(target_idx, top_k=top_k)

        # Normalise distances to a 0–100 similarity score
        raw_dists = np.array([d for _, d in results])
        max_dist  = raw_dists.max() if raw_dists.max() > 0 else 1
        scores    = 100 * (1 - raw_dists / max_dist)

        st.markdown(f"**{top_k} most similar players:**")
        for rank, ((i, dist), score) in enumerate(zip(results, scores), 1):
            p          = df.iloc[i]
            p_color    = ARCH_COLORS.get(p['archetype'], '#444')
            name_disp  = p['name'] if p['name'] != '—' else f"Player {p['player_id']}"
            st.markdown(
                f"""
                <div class="result-row">
                    <span class="result-rank">{rank}</span>
                    <div style="flex:1;min-width:0">
                        <div class="result-name">{name_disp}</div>
                        <div class="result-meta">{p['position']}</div>
                    </div>
                    <span class="arch-pill"
                          style="background:{p_color}22;color:{p_color}">
                        {p['archetype']}
                    </span>
                    <span class="result-dist">{score:.0f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Highlight query + results on the map
        st.markdown("---")
        st.markdown("**Similar players on the Style Map:**")

        result_indices = [i for i, _ in results]
        highlight_df = df.loc[[target_idx] + result_indices].copy()
        highlight_df['role'] = ['Query'] + ['Similar'] * len(result_indices)
        highlight_df['name_display'] = highlight_df['name'].where(
            highlight_df['name'] != '—', 'Unknown'
        )

        bg_df = df.copy()
        bg_df['name_display'] = bg_df['name'].where(bg_df['name'] != '—', 'Unknown')

        mini_fig = go.Figure()

        # Background players (muted)
        mini_fig.add_trace(go.Scatter(
            x=bg_df['x'], y=bg_df['y'],
            mode='markers',
            marker=dict(size=4, color='#1e3a1e', opacity=0.4),
            hoverinfo='skip',
            showlegend=False,
        ))

        # Similar players
        sim_df = highlight_df[highlight_df['role'] == 'Similar']
        mini_fig.add_trace(go.Scatter(
            x=sim_df['x'], y=sim_df['y'],
            mode='markers',
            marker=dict(size=9, color='#7fff7f', opacity=0.9,
                        line=dict(width=1, color='#0a0f0a')),
            text=sim_df['name_display'],
            hovertemplate='%{text}<extra></extra>',
            name='Similar',
        ))

        # Query player (bright)
        q_df = highlight_df[highlight_df['role'] == 'Query']
        mini_fig.add_trace(go.Scatter(
            x=q_df['x'], y=q_df['y'],
            mode='markers',
            marker=dict(size=14, color='#FFD700', symbol='star',
                        line=dict(width=1, color='#0a0f0a')),
            text=q_df['name_display'],
            hovertemplate='%{text}<extra></extra>',
            name='Query',
        ))

        mini_fig.update_layout(
            paper_bgcolor='#0a0f0a',
            plot_bgcolor='#0a0f0a',
            font=dict(family='DM Sans', color='#c8d8c8'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor='#0f1a0f', bordercolor='#1e3a1e', borderwidth=1),
        )
        st.plotly_chart(mini_fig, width='stretch')

else:
    st.info("Enter a player name in the sidebar to find similar players.")

# ── Archetype explorer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-head">Archetype Explorer</div>', unsafe_allow_html=True)
st.caption("Browse all players within a chosen archetype.")

selected_arch = st.selectbox("Select archetype", options=archetypes, label_visibility="collapsed")
arch_players  = named_df[named_df['archetype'] == selected_arch][
    ['name', 'position']
].sort_values('name')

col_a, col_b = st.columns([1, 2])
with col_a:
    arch_col = ARCH_COLORS.get(selected_arch, '#444')
    st.markdown(
        f"""
        <div class="stat-card" style="border-color:{arch_col}40">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.3rem;
                        color:{arch_col};letter-spacing:0.06em">{selected_arch}</div>
            <div style="font-size:0.8rem;color:#5a8a5a;margin-top:6px">
                {len(arch_players)} players
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    pos_counts = arch_players['position'].value_counts().head(6)
    st.markdown("**Top positions:**")
    for pos, cnt in pos_counts.items():
        pct = int(cnt / len(arch_players) * 100)
        st.markdown(
            f"""
            <div style="margin-bottom:5px">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.78rem;color:#c8d8c8;margin-bottom:2px">
                    <span>{pos}</span><span style="color:#5a8a5a">{cnt}</span>
                </div>
                <div style="height:3px;background:#1e3a1e;border-radius:2px">
                    <div style="height:100%;width:{pct}%;background:{arch_col};
                                border-radius:2px"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_b:
    st.dataframe(
        arch_players.rename(columns={'name': 'Player', 'position': 'Position'}),
        width='stretch',
        height=320,
        hide_index=True,
    )