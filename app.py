import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LinearRegression
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Global Food Crisis Observatory",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark editorial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:         #0d0f14;
    --surface:    #151820;
    --surface2:   #1c2030;
    --border:     #252b3b;
    --gold:       #e8b84b;
    --gold-light: #f5d07a;
    --crimson:    #d94f4f;
    --teal:       #3ecfb2;
    --muted:      #6b7696;
    --text:       #e8eaf2;
    --text-dim:   #a0a8c0;
}

/* ── Base reset ── */
html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headers ── */
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    color: var(--gold) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── Selectboxes & sliders ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
    margin: 2rem 0 !important;
}

/* ── Plotly charts: transparent background ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── Custom hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #151820 0%, #1a1f2e 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(232,184,75,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 8px 0;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--muted);
    margin: 0;
    max-width: 600px;
    line-height: 1.6;
}
.hero-tag {
    display: inline-block;
    background: rgba(232,184,75,0.12);
    color: var(--gold);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
    border: 1px solid rgba(232,184,75,0.25);
    margin-bottom: 16px;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 4px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--text);
    margin: 0 0 8px 0;
}
.section-desc {
    color: var(--text-dim);
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 24px;
}

/* ── Insight cards ── */
.insight-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    font-size: 0.88rem;
    color: var(--text-dim);
    line-height: 1.6;
}
.insight-card strong { color: var(--gold); }

/* ── Stat highlight ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.stat-chip {
    background: rgba(232,184,75,0.08);
    border: 1px solid rgba(232,184,75,0.2);
    border-radius: 8px;
    padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--gold-light);
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    padding: 32px 0 16px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#a0a8c0", size=12),
    title_font=dict(family="DM Serif Display, serif", color="#e8eaf2", size=18),
    legend=dict(
        bgcolor="rgba(21,24,32,0.8)", bordercolor="#252b3b", borderwidth=1,
        font=dict(color="#a0a8c0")
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    hoverlabel=dict(bgcolor="#1c2030", bordercolor="#252b3b", font=dict(color="#e8eaf2")),
)

_AXIS_STYLE = dict(
    xaxis=dict(gridcolor="#1c2030", zerolinecolor="#252b3b", tickfont=dict(color="#6b7696")),
    yaxis=dict(gridcolor="#1c2030", zerolinecolor="#252b3b", tickfont=dict(color="#6b7696")),
)

# Full layout with axes (for cartesian charts)
PLOTLY_LAYOUT = {**_BASE_LAYOUT, **_AXIS_STYLE}

# Map-safe layout without xaxis/yaxis (for choropleth/geo charts)
PLOTLY_MAP_LAYOUT = {**_BASE_LAYOUT}

GOLD = "#e8b84b"
CRIMSON = "#d94f4f"
TEAL = "#3ecfb2"
MUTED = "#6b7696"

COLOR_SEQ = [GOLD, TEAL, CRIMSON, "#9b7fe8", "#4fc3f7", "#ff8a65", "#81c784"]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file):
    df_raw = pd.read_csv(file)
    df_raw["date"] = pd.to_datetime(df_raw["TIME_PERIOD"], format="%Y-%m", errors="coerce")
    df_raw["phase"] = df_raw["COMP_BREAKDOWN_2"].str.extract(r"PHASE(\d)").astype(float)

    df = df_raw.rename(columns={
        "REF_AREA": "iso3",
        "REF_AREA_LABEL": "country",
        "UNIT_MEASURE": "unit",
        "OBS_VALUE": "value"
    })[["iso3", "country", "date", "phase", "unit", "value"]]

    df = df.dropna(subset=["date", "phase", "value"])
    df["phase"] = df["phase"].astype(int)

    west_africa = [
        "Benin","Burkina Faso","Cabo Verde","Côte d'Ivoire","Gambia",
        "Ghana","Guinea","Guinea-Bissau","Liberia","Mali","Mauritania",
        "Niger","Nigeria","Senegal","Sierra Leone","Togo","Chad"
    ]
    east_africa = [
        "Burundi","Djibouti","Eritrea","Ethiopia","Kenya","Rwanda",
        "South Sudan","Sudan","Uganda","Tanzania"
    ]

    def assign_region(c):
        if c in west_africa: return "West Africa"
        elif c in east_africa: return "East Africa"
        else: return "Other"

    df["Region"] = df["country"].apply(assign_region)

    df_people = df[df["unit"] == "PS"].copy()
    df_pct    = df[df["unit"] == "PT"].copy()

    wide_people = df_people.pivot_table(
        index=["iso3","country","Region","date"],
        columns="phase", values="value", aggfunc="sum"
    ).reset_index()

    wide_pct = df_pct.pivot_table(
        index=["iso3","country","Region","date"],
        columns="phase", values="value", aggfunc="mean"
    ).reset_index()

    wide_people.columns = [f"phase_{int(c)}_people" if isinstance(c, (int, float)) else c for c in wide_people.columns]
    wide_pct.columns    = [f"phase_{int(c)}_pct"    if isinstance(c, (int, float)) else c for c in wide_pct.columns]

    for p in [3,4,5]:
        if f"phase_{p}_people" not in wide_people.columns:
            wide_people[f"phase_{p}_people"] = 0
        if f"phase_{p}_pct" not in wide_pct.columns:
            wide_pct[f"phase_{p}_pct"] = 0

    wide_people["crisis_plus_people"] = wide_people[["phase_3_people","phase_4_people","phase_5_people"]].fillna(0).sum(axis=1)
    wide_pct["crisis_plus_pct"]       = wide_pct[["phase_3_pct","phase_4_pct","phase_5_pct"]].fillna(0).sum(axis=1)

    wide_pct["severe_share"] = np.where(
        wide_pct["crisis_plus_pct"] > 0,
        wide_pct[["phase_4_pct","phase_5_pct"]].fillna(0).sum(axis=1) / wide_pct["crisis_plus_pct"],
        np.nan
    )

    return wide_people, wide_pct

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:24px'>
      <div class='section-label'>Dashboard</div>
      <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;color:#e8eaf2;line-height:1.2'>
        Global Food Crisis<br>Observatory
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader("Upload IPC_PHASE.csv", type=["csv"])

    if not uploaded:
        st.info("Upload your IPC Phase CSV to begin. The app expects the standard IPC Global data format.")
        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.8rem;color:#6b7696;line-height:1.8'>
        <b style='color:#e8b84b'>IPC Phases</b><br>
        Phase 1 — Minimal<br>
        Phase 2 — Stressed<br>
        Phase 3 — Crisis ⚠️<br>
        Phase 4 — Emergency 🔴<br>
        Phase 5 — Catastrophe 🆘
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    wide_people, wide_pct = load_data(uploaded)

    all_countries = sorted(wide_pct["country"].unique())
    all_regions   = sorted(wide_pct["Region"].unique())

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    selected_regions = st.multiselect(
        "Regions", all_regions, default=all_regions,
        help="Filter all charts by region"
    )

    min_date = wide_pct["date"].min()
    max_date = wide_pct["date"].max()

    date_range = st.slider(
        "Date range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#6b7696'>
    Built for DSCD 611 Final Project<br>
    IPC Phase Classification Analysis
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
mask_pct = (
    wide_pct["Region"].isin(selected_regions) &
    (wide_pct["date"] >= pd.Timestamp(date_range[0])) &
    (wide_pct["date"] <= pd.Timestamp(date_range[1]))
)
mask_ppl = (
    wide_people["Region"].isin(selected_regions) &
    (wide_people["date"] >= pd.Timestamp(date_range[0])) &
    (wide_people["date"] <= pd.Timestamp(date_range[1]))
)

wp  = wide_pct[mask_pct].copy()
wpl = wide_people[mask_ppl].copy()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
  <div class='hero-tag'>IPC Phase 3+ · Acute Food Insecurity</div>
  <div class='hero-title'>Global Food Crisis<br>Observatory</div>
  <p class='hero-subtitle'>
    A comprehensive analysis of acute food insecurity across nations using
    IPC Phase Classifications — tracking severity, trends, and the concentration
    of crisis populations worldwide.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
latest_pct = wp.loc[wp.groupby("country")["date"].idxmax()]
latest_ppl = wpl.loc[wpl.groupby("country")["date"].idxmax()]

total_crisis   = latest_ppl["crisis_plus_people"].sum()
mean_pct       = latest_pct["crisis_plus_pct"].mean()
n_countries    = latest_pct["country"].nunique()
worst_country  = latest_pct.sort_values("crisis_plus_pct", ascending=False).iloc[0]

top5_share = (
    latest_ppl.sort_values("crisis_plus_people", ascending=False)
    .head(5)["crisis_plus_people"].sum() / latest_ppl["crisis_plus_people"].sum() * 100
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("People in Phase 3+", f"{total_crisis/1e6:.1f}M")
k2.metric("Global Avg Severity", f"{mean_pct:.1f}%")
k3.metric("Countries Monitored", str(n_countries))
k4.metric("Most Affected", worst_country["country"], f"{worst_country['crisis_plus_pct']:.0f}%")
k5.metric("Top 5 Countries Share", f"{top5_share:.0f}%", "of global crisis pop.")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Global Trends",
    "🏆 Country Rankings",
    "🌍 Regional Analysis",
    "📉 Deterioration & Recovery",
    "🔬 Statistical Insights",
])

# ══════════════════════════════════════════════
# TAB 1 — GLOBAL TRENDS
# ══════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-label">Question 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Global Trend in Phase 3+ Severity</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Average percentage of population classified as Phase 3 or above across all monitored countries over time.</div>', unsafe_allow_html=True)

        global_trend = wp.groupby("date")["crisis_plus_pct"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=global_trend["date"], y=global_trend["crisis_plus_pct"],
            mode="lines",
            line=dict(color=GOLD, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(232,184,75,0.08)",
            name="Phase 3+ %",
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f}%<extra></extra>"
        ))

        # Add rolling average
        global_trend["roll"] = global_trend["crisis_plus_pct"].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=global_trend["date"], y=global_trend["roll"],
            mode="lines",
            line=dict(color=TEAL, width=1.5, dash="dash"),
            name="3-period avg",
            hovertemplate="<b>%{x|%b %Y}</b><br>3-mo avg: %{y:.1f}%<extra></extra>"
        ))

        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          title="Global Average % Population in Phase 3+")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-label">Phase Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Phase Composition</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Average share of population in each IPC phase at latest snapshot.</div>', unsafe_allow_html=True)

        phase_cols = ["phase_1_pct","phase_2_pct","phase_3_pct","phase_4_pct","phase_5_pct"]
        phase_labels = ["Phase 1\nMinimal","Phase 2\nStressed","Phase 3\nCrisis",
                        "Phase 4\nEmergency","Phase 5\nCatastrophe"]
        phase_colors = ["#3a7d44","#8ab34a",GOLD,"#d97a2a",CRIMSON]

        avgs = []
        for col in phase_cols:
            if col in latest_pct.columns:
                avgs.append(latest_pct[col].mean())
            else:
                avgs.append(0)

        fig2 = go.Figure(go.Bar(
            x=avgs,
            y=phase_labels,
            orientation="h",
            marker_color=phase_colors,
            text=[f"{v:.1f}%" for v in avgs],
            textposition="outside",
            textfont=dict(color="#e8eaf2"),
            hovertemplate="<b>%{y}</b><br>Avg: %{x:.1f}%<extra></extra>"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                           title="Average % per Phase (Latest Data)",
                           xaxis_title="Percentage (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # World Map
    st.markdown("---")
    st.markdown('<div class="section-label">Geographic View</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">World Map of Phase 3+ Severity</div>', unsafe_allow_html=True)

    map_data = latest_pct[["iso3","country","crisis_plus_pct"]].dropna()

    clim = float(map_data["crisis_plus_pct"].quantile(0.97))
    fig_map = go.Figure(go.Choropleth(
        locations=map_data["iso3"],
        z=map_data["crisis_plus_pct"],
        text=map_data["country"],
        hovertemplate="<b>%{text}</b><br>Phase 3+: %{z:.1f}%<extra></extra>",
        colorscale=[
            [0,   "#1c2030"],
            [0.2, "#3a5a4a"],
            [0.4, "#8ab34a"],
            [0.6, "#e8b84b"],
            [0.8, "#d97a2a"],
            [1.0, "#d94f4f"],
        ],
        zmin=0,
        zmax=clim,
        colorbar=dict(
            title=dict(text="Phase 3+ %", font=dict(color="#a0a8c0")),
            tickfont=dict(color="#a0a8c0"),
            bgcolor="rgba(21,24,32,0.8)",
            bordercolor="#252b3b",
            borderwidth=1,
        ),
        marker_line_color="#252b3b",
        marker_line_width=0.5,
    ))
    fig_map.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=45, b=0),
        title=dict(
            text="World Map: Phase 3+ Severity (%)",
            font=dict(family="DM Serif Display, serif", color="#e8eaf2", size=18)
        ),
        font=dict(family="DM Sans, sans-serif", color="#a0a8c0"),
        hoverlabel=dict(bgcolor="#1c2030", bordercolor="#252b3b", font=dict(color="#e8eaf2")),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#252b3b",
            showland=True, landcolor="#1a1f2c",
            showocean=True, oceancolor="#0d0f14",
            showlakes=True, lakecolor="#0d0f14",
            showcountries=True, countrycolor="#252b3b",
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — COUNTRY RANKINGS
# ══════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-label">Question 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Global Burden Contributors</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Which countries account for the largest share of the world\'s food crisis population?</div>', unsafe_allow_html=True)

        lat_ppl = wpl.loc[wpl.groupby("country")["date"].idxmax()].copy()
        lat_ppl["global_share"] = lat_ppl["crisis_plus_people"] / lat_ppl["crisis_plus_people"].sum() * 100
        top10 = lat_ppl.sort_values("global_share", ascending=True).tail(10)

        fig3 = go.Figure(go.Bar(
            x=top10["global_share"], y=top10["country"],
            orientation="h",
            marker=dict(
                color=top10["global_share"],
                colorscale=[[0,"#1c3040"],[1,GOLD]],
                showscale=False,
                line=dict(width=0)
            ),
            text=[f"{v:.1f}%" for v in top10["global_share"]],
            textposition="outside",
            textfont=dict(color="#e8eaf2"),
            hovertemplate="<b>%{y}</b><br>Global share: %{x:.1f}%<extra></extra>"
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=400,
                           title="Top 10 Countries by Share of Global Crisis Pop.")
        st.plotly_chart(fig3, use_container_width=True)

        top5_val = lat_ppl.sort_values("global_share", ascending=False).head(5)["global_share"].sum()
        st.markdown(f"""
        <div class='insight-card'>
        <strong>Concentration insight:</strong> The top 5 countries account for
        <strong>{top5_val:.1f}%</strong> of the global Phase 3+ population —
        illustrating extreme geographic concentration of food insecurity burden.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-label">Question 3</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Highest % Population in Crisis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Countries where food insecurity affects the largest share of the total population.</div>', unsafe_allow_html=True)

        top10_pct = latest_pct.sort_values("crisis_plus_pct", ascending=True).tail(10)

        fig4 = go.Figure(go.Bar(
            x=top10_pct["crisis_plus_pct"], y=top10_pct["country"],
            orientation="h",
            marker=dict(
                color=top10_pct["crisis_plus_pct"],
                colorscale=[[0,"#2a1a1a"],[0.5,"#d97a2a"],[1,CRIMSON]],
                showscale=False,
            ),
            text=[f"{v:.0f}%" for v in top10_pct["crisis_plus_pct"]],
            textposition="outside",
            textfont=dict(color="#e8eaf2"),
            hovertemplate="<b>%{y}</b><br>Phase 3+: %{x:.0f}%<extra></extra>"
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=400,
                           title="Top 10 Countries by % Population in Phase 3+")
        st.plotly_chart(fig4, use_container_width=True)

    # Crisis depth
    st.markdown("---")
    st.markdown('<div class="section-label">Question 7</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Depth of Crisis: Phase 4–5 Dominance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Among countries with high Phase 3+ populations, which have the most extreme crises? This ratio shows Phase 4 & 5 as a share of all Phase 3+ people.</div>', unsafe_allow_html=True)

    depth = (
        wp.groupby("country")["severe_share"]
        .mean().reset_index()
        .dropna()
        .sort_values("severe_share", ascending=True)
        .tail(12)
    )

    fig5 = go.Figure(go.Bar(
        x=depth["severe_share"], y=depth["country"],
        orientation="h",
        marker=dict(
            color=depth["severe_share"],
            colorscale=[[0,"#1c2030"],[0.5,"#9b4dca"],[1,CRIMSON]],
            showscale=False,
        ),
        text=[f"{v:.0%}" for v in depth["severe_share"]],
        textposition="outside",
        textfont=dict(color="#e8eaf2"),
        hovertemplate="<b>%{y}</b><br>Severity ratio: %{x:.0%}<extra></extra>"
    ))
    fig5.update_layout(**{**PLOTLY_LAYOUT, "xaxis": dict(tickformat=".0%", gridcolor="#1c2030", zerolinecolor="#252b3b", tickfont=dict(color="#6b7696"))}, height=380,
                       title="Countries with Deepest Crisis (Phase 4–5 Share of Phase 3+)")
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Questions 5 & 6</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">West Africa vs East Africa</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Comparing the trajectory of acute food insecurity between the two most affected African regions over time.</div>', unsafe_allow_html=True)

    regional_trend = (
        wp[wp["Region"].isin(["West Africa","East Africa"])]
        .groupby(["Region","date"])["crisis_plus_pct"]
        .mean().reset_index()
    )

    fig6 = go.Figure()
    palette = {"West Africa": GOLD, "East Africa": TEAL}
    for region, grp in regional_trend.groupby("Region"):
        grp = grp.sort_values("date")
        fig6.add_trace(go.Scatter(
            x=grp["date"], y=grp["crisis_plus_pct"],
            mode="lines+markers",
            name=region,
            line=dict(color=palette[region], width=2.5),
            marker=dict(size=5, color=palette[region]),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(palette[region].lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.06,)}",
            hovertemplate=f"<b>{region}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>"
        ))

    fig6.update_layout(**PLOTLY_LAYOUT, height=360,
                       title="Average % Population in Phase 3+: West vs East Africa",
                       yaxis_title="Phase 3+ (%)")
    st.plotly_chart(fig6, use_container_width=True)

    # T-test result
    west = regional_trend[regional_trend["Region"]=="West Africa"]["crisis_plus_pct"]
    east = regional_trend[regional_trend["Region"]=="East Africa"]["crisis_plus_pct"]

    if len(west) > 1 and len(east) > 1:
        t_stat, p_val = ttest_ind(west, east, equal_var=False)
        sig = "statistically significant" if p_val < 0.05 else "not statistically significant"
        sig_color = TEAL if p_val < 0.05 else GOLD

        col1, col2, col3 = st.columns(3)
        col1.metric("T-Statistic", f"{t_stat:.3f}")
        col2.metric("P-Value", f"{p_val:.5f}")
        col3.metric("Significance (α=0.05)", "✓ Significant" if p_val < 0.05 else "✗ Not Significant")

        st.markdown(f"""
        <div class='insight-card'>
        <strong>Welch's t-test result:</strong> The difference in Phase 3+ rates between
        West and East Africa is <strong style='color:{sig_color}'>{sig}</strong>
        (t = {t_stat:.3f}, p = {p_val:.5f}).
        {"This confirms that the two regions face structurally different levels of food insecurity." if p_val < 0.05 else "This suggests the two regions have comparable levels of food insecurity."}
        </div>
        """, unsafe_allow_html=True)

    # All regions comparison
    st.markdown("---")
    st.markdown('<div class="section-label">All Regions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Regional Comparison Overview</div>', unsafe_allow_html=True)

    reg_summary = (
        wp.groupby(["Region","date"])["crisis_plus_pct"]
        .mean().reset_index()
    )

    fig7 = px.line(
        reg_summary, x="date", y="crisis_plus_pct",
        color="Region", color_discrete_sequence=COLOR_SEQ,
        labels={"crisis_plus_pct":"Phase 3+ (%)","date":"Date"},
    )
    fig7.update_traces(line_width=2)
    fig7.update_layout(**PLOTLY_LAYOUT, height=340, title="All Regions: Phase 3+ Trend")
    st.plotly_chart(fig7, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — DETERIORATION & RECOVERY
# ══════════════════════════════════════════════
with tab4:
    col_l2, col_r2 = st.columns(2)

    # Slopes computation
    slopes = []
    for country, grp in wp.groupby("country"):
        if len(grp) > 6:
            grp = grp.sort_values("date")
            X = np.arange(len(grp)).reshape(-1,1)
            y = grp["crisis_plus_pct"].values
            slope = LinearRegression().fit(X, y).coef_[0]
            slopes.append((country, slope, grp["Region"].iloc[0]))

    slope_df = pd.DataFrame(slopes, columns=["country","slope","region"])

    with col_l2:
        st.markdown('<div class="section-label">Question 8</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Fastest Deteriorating Countries</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Linear regression slope of Phase 3+ % over time — higher = worsening faster.</div>', unsafe_allow_html=True)

        fastest = slope_df.sort_values("slope", ascending=True).tail(10)

        fig8 = go.Figure(go.Bar(
            x=fastest["slope"], y=fastest["country"],
            orientation="h",
            marker=dict(
                color=fastest["slope"],
                colorscale=[[0,"#1c2030"],[1,CRIMSON]],
                showscale=False,
            ),
            text=[f"+{v:.2f}/period" for v in fastest["slope"]],
            textposition="outside",
            textfont=dict(color="#e8eaf2"),
            hovertemplate="<b>%{y}</b><br>Slope: %{x:.3f}<extra></extra>"
        ))
        fig8.update_layout(**PLOTLY_LAYOUT, height=380,
                           title="Fastest Worsening (Phase 3+ Slope)")
        st.plotly_chart(fig8, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-label">Question 10</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Countries Showing Recovery</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Countries with the most negative slope — suggesting genuine improvement in food security outcomes.</div>', unsafe_allow_html=True)

        recovery = slope_df.sort_values("slope").head(10)

        fig9 = go.Figure(go.Bar(
            x=recovery["slope"].abs(), y=recovery["country"],
            orientation="h",
            marker=dict(
                color=recovery["slope"].abs(),
                colorscale=[[0,"#1c2030"],[1,TEAL]],
                showscale=False,
            ),
            text=[f"-{abs(v):.2f}/period" for v in recovery["slope"]],
            textposition="outside",
            textfont=dict(color="#e8eaf2"),
            hovertemplate="<b>%{y}</b><br>Improvement slope: %{x:.3f}<extra></extra>"
        ))
        fig9.update_layout(**PLOTLY_LAYOUT, height=380,
                           title="Fastest Improving (Phase 3+ Decline)")
        st.plotly_chart(fig9, use_container_width=True)

    # Country deep-dive
    st.markdown("---")
    st.markdown('<div class="section-label">Country Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Individual Country Trend</div>', unsafe_allow_html=True)

    selected_countries = st.multiselect(
        "Select up to 5 countries to compare",
        options=sorted(wp["country"].unique()),
        default=slope_df.sort_values("slope", ascending=False).head(3)["country"].tolist()[:3],
        max_selections=5
    )

    if selected_countries:
        country_data = wp[wp["country"].isin(selected_countries)]
        fig10 = px.line(
            country_data, x="date", y="crisis_plus_pct", color="country",
            color_discrete_sequence=COLOR_SEQ,
            labels={"crisis_plus_pct":"Phase 3+ (%)","date":"Date"},
            markers=True
        )
        fig10.update_traces(line_width=2, marker_size=5)
        fig10.update_layout(**PLOTLY_LAYOUT, height=350,
                            title="Phase 3+ Trend: Selected Countries")
        st.plotly_chart(fig10, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — STATISTICAL INSIGHTS
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-label">Question 9</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Volatility vs Severity</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Does a higher average severity correlate with greater instability? This scatter explores the relationship between mean Phase 3+ % and its standard deviation.</div>', unsafe_allow_html=True)

    stats = wp.groupby("country")["crisis_plus_pct"].agg(["mean","std"]).reset_index()
    stats = stats.merge(wp[["country","Region"]].drop_duplicates(), on="country", how="left")
    stats = stats.dropna()

    if len(stats) > 2:
        corr, p_corr = pearsonr(stats["mean"], stats["std"])

        fig11 = go.Figure()
        # Plot each region as a separate trace
        for i, region in enumerate(stats["Region"].unique()):
            grp = stats[stats["Region"] == region]
            fig11.add_trace(go.Scatter(
                x=grp["mean"], y=grp["std"],
                mode="markers",
                name=region,
                text=grp["country"],
                marker=dict(
                    size=grp["mean"].clip(5, 30),
                    color=COLOR_SEQ[i % len(COLOR_SEQ)],
                    opacity=0.85,
                    line=dict(width=1, color="#252b3b"),
                ),
                hovertemplate="<b>%{text}</b><br>Mean: %{x:.1f}%<br>Std Dev: %{y:.2f}<extra></extra>",
            ))
        # Manual OLS trendline using numpy
        x_vals = stats["mean"].values
        y_vals = stats["std"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig11.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines",
            name="Trend",
            line=dict(color=GOLD, width=2, dash="dash"),
            hoverinfo="skip",
        ))
        fig11.update_layout(**PLOTLY_LAYOUT, height=420,
                            title="Volatility vs Average Severity in Phase 3+",
                            xaxis_title="Mean Phase 3+ (%)",
                            yaxis_title="Volatility (Std Dev)")
        st.plotly_chart(fig11, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson r", f"{corr:.3f}")
        c2.metric("P-value", f"{p_corr:.5f}")
        c3.metric("Relationship", "Strong +" if corr > 0.5 else ("Moderate +" if corr > 0.3 else "Weak"))

        interp = (
            "Countries with higher average food insecurity also tend to experience greater fluctuation over time, "
            "suggesting structural instability in the most crisis-prone nations."
            if corr > 0.4
            else "The relationship between average severity and volatility is moderate or weak."
        )
        st.markdown(f"""
        <div class='insight-card'>
        <strong>Interpretation:</strong> {interp}
        (r = {corr:.2f}, p = {p_corr:.5f})
        </div>
        """, unsafe_allow_html=True)

    # Phase heatmap
    st.markdown("---")
    st.markdown('<div class="section-label">Phase Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Phase 3+ Severity Heatmap by Country & Year</div>', unsafe_allow_html=True)

    wp_heat = wp.copy()
    wp_heat["year"] = wp_heat["date"].dt.year
    pivot_heat = wp_heat.groupby(["country","year"])["crisis_plus_pct"].mean().unstack()
    # top 20 countries by mean
    top20 = pivot_heat.mean(axis=1).sort_values(ascending=False).head(20).index
    pivot_heat = pivot_heat.loc[top20]

    fig12 = go.Figure(go.Heatmap(
        z=pivot_heat.values,
        x=pivot_heat.columns.astype(str),
        y=pivot_heat.index,
        colorscale=[
            [0,   "#1c2030"],
            [0.25,"#3a5a40"],
            [0.5, GOLD],
            [0.75,"#d97a2a"],
            [1.0, CRIMSON],
        ],
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Phase 3+: %{z:.1f}%<extra></extra>",
        colorbar=dict(
            title=dict(text="Phase 3+ %", font=dict(color="#a0a8c0")),
            tickfont=dict(color="#a0a8c0"),
            bgcolor="rgba(21,24,32,0.8)",
            bordercolor="#252b3b",
        )
    ))
    fig12.update_layout(**PLOTLY_LAYOUT, height=520,
                        title="Annual Phase 3+ Severity Heatmap (Top 20 Countries)",
                        xaxis_title="Year", yaxis_title="")
    st.plotly_chart(fig12, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  Global Food Crisis Observatory &nbsp;·&nbsp; DSCD 611 Final Project &nbsp;·&nbsp;
  Data: IPC Global Platform &nbsp;·&nbsp; Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)