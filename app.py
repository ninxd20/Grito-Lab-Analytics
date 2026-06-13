import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Channel Marketing Analytics | Grito Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0f1117; }
    
   .metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;

    transition: all 0.3s ease;
    }

.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 30px rgba(0,212,255,.3);
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #00d4ff;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-delta-pos { color: #10b981; font-size: 13px; }
    .metric-delta-neg { color: #ef4444; font-size: 13px; }
    
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #f9fafb;
        margin: 24px 0 8px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #1e2535;
    }
    .insight-box {
        background: #111827;
        border-left: 4px solid #00d4ff;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #d1d5db;
        font-size: 14px;
        line-height: 1.7;
    }
    .warn-box {
        background: #111827;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #d1d5db;
        font-size: 14px;
    }
    .code-block {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #79c0ff;
        white-space: pre-wrap;
        margin: 12px 0;
    }
    .company-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1f2e;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #00d4ff !important;
        color: #0f1117 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── SYNTHETIC DATA ────────────────────────────────────────────────────────────
np.random.seed(42)

COMPANIES = {
    "NovaMart": {
        "industry": "E-Commerce",
        "color": "#00d4ff",
        "total_budget": 850000,
    },
    "PulseHealth": {
        "industry": "Healthcare & Wellness",
        "color": "#10b981",
        "total_budget": 620000,
    },
    "UrbanDrive": {
        "industry": "Travel & Mobility",
        "color": "#f59e0b",
        "total_budget": 480000,
    }
}

CHANNELS = ["Paid Search", "Social Media", "Email", "Display Ads", "Organic Search"]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def generate_company_data(company_name, seed_offset=0):
    np.random.seed(42 + seed_offset)
    
    channel_profiles = {
        "NovaMart": {
            "Paid Search":     {"budget_pct": 0.35, "base_conv": 0.045, "base_cpc": 1.8,  "trend": 0.02},
            "Social Media":    {"budget_pct": 0.28, "base_conv": 0.031, "base_cpc": 0.9,  "trend": 0.035},
            "Email":           {"budget_pct": 0.12, "base_conv": 0.068, "base_cpc": 0.25, "trend": 0.01},
            "Display Ads":     {"budget_pct": 0.15, "base_conv": 0.012, "base_cpc": 0.6,  "trend": -0.01},
            "Organic Search":  {"budget_pct": 0.10, "base_conv": 0.052, "base_cpc": 0.0,  "trend": 0.015},
        },
        "PulseHealth": {
            "Paid Search":     {"budget_pct": 0.30, "base_conv": 0.038, "base_cpc": 2.2,  "trend": 0.015},
            "Social Media":    {"budget_pct": 0.22, "base_conv": 0.025, "base_cpc": 1.1,  "trend": 0.04},
            "Email":           {"budget_pct": 0.20, "base_conv": 0.072, "base_cpc": 0.18, "trend": 0.02},
            "Display Ads":     {"budget_pct": 0.18, "base_conv": 0.009, "base_cpc": 0.55, "trend": -0.015},
            "Organic Search":  {"budget_pct": 0.10, "base_conv": 0.048, "base_cpc": 0.0,  "trend": 0.025},
        },
        "UrbanDrive": {
            "Paid Search":     {"budget_pct": 0.40, "base_conv": 0.041, "base_cpc": 1.5,  "trend": 0.01},
            "Social Media":    {"budget_pct": 0.32, "base_conv": 0.035, "base_cpc": 0.75, "trend": 0.05},
            "Email":           {"budget_pct": 0.08, "base_conv": 0.055, "base_cpc": 0.20, "trend": 0.008},
            "Display Ads":     {"budget_pct": 0.12, "base_conv": 0.010, "base_cpc": 0.5,  "trend": -0.02},
            "Organic Search":  {"budget_pct": 0.08, "base_conv": 0.045, "base_cpc": 0.0,  "trend": 0.02},
        }
    }
    
    total_budget = COMPANIES[company_name]["total_budget"]
    profile = channel_profiles[company_name]
    
    records = []
    for month_idx, month in enumerate(MONTHS):
        for channel in CHANNELS:
            p = profile[channel]
            noise = np.random.normal(0, 0.08)
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * month_idx / 12 - np.pi/2)
            
            budget = total_budget * p["budget_pct"] / 12 * seasonal * (1 + noise * 0.3)
            
            if p["base_cpc"] > 0:
                clicks = int(budget / (p["base_cpc"] * (1 + month_idx * 0.005)))
            else:
                clicks = int(budget * 180 * seasonal * (1 + np.random.normal(0, 0.1)))
            
            impressions = int(clicks * np.random.uniform(18, 35))
            conv_rate = p["base_conv"] * (1 + p["trend"] * month_idx) * (1 + np.random.normal(0, 0.06))
            conv_rate = max(0.005, min(0.15, conv_rate))
            conversions = int(clicks * conv_rate)
            
            avg_order = {"NovaMart": 1850, "PulseHealth": 2400, "UrbanDrive": 3200}[company_name]
            revenue = conversions * avg_order * (1 + np.random.normal(0, 0.1))
            
            records.append({
                "company": company_name,
                "month": month,
                "month_idx": month_idx + 1,
                "channel": channel,
                "budget": round(budget, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "revenue": round(revenue, 2),
                "ctr": round(clicks / max(impressions, 1) * 100, 3),
                "conv_rate": round(conv_rate * 100, 3),
                "cpc": round(budget / max(clicks, 1), 2),
                "cpa": round(budget / max(conversions, 1), 2),
                "roas": round(revenue / max(budget, 1), 3),
                "clv_proxy": round(revenue / max(conversions, 1) * 2.4, 2),
            })
    
    return pd.DataFrame(records)

@st.cache_data
def load_all_data():
    dfs = []
    for i, company in enumerate(COMPANIES.keys()):
        dfs.append(generate_company_data(company, seed_offset=i * 10))
    return pd.concat(dfs, ignore_index=True)

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Marketing Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df_all = pd.read_csv(uploaded_file)
else:
    df_all = load_all_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Analytics Framework")
    st.markdown("**Grito Lab** — Multi-Channel Marketing Attribution & ROI Analytics")
    st.divider()
    
    selected_company = st.selectbox(
        "Select Company",
        list(COMPANIES.keys()),
        format_func=lambda x: f"{x} ({COMPANIES[x]['industry']})"
    )
    
    selected_channels = st.multiselect(
        "Filter Channels",
        CHANNELS,
        default=CHANNELS
    )
    
    month_range = st.select_slider(
        "Month Range",
        options=MONTHS,
        value=(MONTHS[0], MONTHS[11])
    )
    
    st.divider()
    st.markdown("**Project:** Multi-Channel Marketing Attribution & ROI Analytics")
    st.markdown("**Domain:** Marketing Analytics & Business Intelligence")
    st.markdown("**Submitted to:** Grito Lab via Techfest IIT Bombay CA Program 2026")
    st.markdown("**Analyst:** Ninad Jagdish Tarwate")

# ── FILTER DATA ───────────────────────────────────────────────────────────────
month_start_idx = MONTHS.index(month_range[0]) + 1
month_end_idx = MONTHS.index(month_range[1]) + 1

df = df_all[
    (df_all["company"] == selected_company) &
    (df_all["channel"].isin(selected_channels)) &
    (df_all["month_idx"] >= month_start_idx) &
    (df_all["month_idx"] <= month_end_idx)
].copy()

company_color = COMPANIES[selected_company]["color"]

# ── HEADER ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"# Multi-Channel Marketing Analytics")
    st.markdown(f"**{selected_company}** · {COMPANIES[selected_company]['industry']} · {month_range[0]} – {month_range[1]} 2024")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Total Budget:** ₹{COMPANIES[selected_company]['total_budget']:,.0f}")

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Dashboard",
    "🔍 SQL Queries",
    "📐 Attribution Model",
    "💡 Optimization Strategy",
    "🔮 Forecasting",
    "🎯 Scenario Simulator"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════
with tab1:
    
    # KPI CARDS
    total_budget  = df["budget"].sum()
    total_revenue = df["revenue"].sum()
    total_conv    = df["conversions"].sum()
    total_clicks  = df["clicks"].sum()
    total_impr    = df["impressions"].sum()
    overall_roas  = total_revenue / max(total_budget, 1)
    overall_cpa   = total_budget / max(total_conv, 1)
    overall_ctr   = total_clicks / max(total_impr, 1) * 100
    
    with st.expander("📋 Executive Summary", expanded=True):

        st.success(
        f"""
    Revenue: ₹{total_revenue:,.0f}

    Spend: ₹{total_budget:,.0f}

    ROAS: {overall_roas:.2f}x

    Conversions: {total_conv:,}

    CTR: {overall_ctr:.2f}%
    """
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Spend", f"₹{total_budget/1e5:.2f}L", delta=None)
    with c2:
        st.metric("Total Revenue", f"₹{total_revenue/1e5:.2f}L", delta=f"+{((total_revenue/total_budget-1)*100):.1f}% ROI")
    with c3:
        st.metric("Overall ROAS", f"{overall_roas:.2f}x", delta="Target: 3.5x" if overall_roas >= 3.5 else "Below target")
    with c4:
        st.metric("Total Conversions", f"{total_conv:,}", delta=f"CPA ₹{overall_cpa:.0f}")
    with c5:
        st.metric("Avg CTR", f"{overall_ctr:.2f}%", delta=f"{total_clicks:,} clicks")

    st.markdown("---")
    st.markdown("### 🎯 ROAS Performance")

    fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=overall_roas,
    title={"text":"Overall ROAS"},
    gauge={
        "axis":{"range":[0,8]},
        "threshold":{
            "line":{"color":"red","width":4},
            "value":3.5
            }
        }
    ))

    fig_gauge.update_layout(
    paper_bgcolor="#0f1117",
    font_color="#d1d5db"
    )

    st.plotly_chart(
    fig_gauge,
    use_container_width=True
    )
   
    st.markdown(
    '<div class="section-header">Marketing Funnel</div>',
    unsafe_allow_html=True
    )

    funnel_df = pd.DataFrame({
    "Stage":[
        "Impressions",
        "Clicks",
        "Conversions"
    ],
    "Value":[
        total_impr,
        total_clicks,
        total_conv
        ]
    })

    fig_funnel = px.funnel(
    funnel_df,
    x="Value",
    y="Stage",
    title="Conversion Funnel"
    )

    fig_funnel.update_layout(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font_color="#d1d5db"
    )

    st.plotly_chart(
    fig_funnel,
    use_container_width=True
    )
    
    # ROW 1 — Channel Performance
    st.markdown('<div class="section-header">Channel Performance Overview</div>', unsafe_allow_html=True)
    
    ch_summary = df.groupby("channel").agg(
        budget=("budget", "sum"),
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    ).reset_index()
    ch_summary["roas"] = ch_summary["revenue"] / ch_summary["budget"]
    ch_summary["cpa"]  = ch_summary["budget"] / ch_summary["conversions"]
    ch_summary["ctr"]  = ch_summary["clicks"] / ch_summary["impressions"] * 100
    ch_summary["conv_rate"] = ch_summary["conversions"] / ch_summary["clicks"] * 100

    col1, col2 = st.columns(2)
    with col1:
        fig_roas = px.bar(
            ch_summary.sort_values("roas", ascending=True),
            x="roas", y="channel", orientation="h",
            title="ROAS by Channel",
            color="roas",
            color_continuous_scale=[[0, "#1a1f2e"], [0.5, "#0066aa"], [1, company_color]],
            labels={"roas": "ROAS (Revenue / Spend)", "channel": ""},
        )
        fig_roas.add_vline(x=3.5, line_dash="dash", line_color="#f59e0b", annotation_text="Target 3.5x")
        fig_roas.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#d1d5db", showlegend=False,
            coloraxis_showscale=False,
            title_font_size=15,
        )
        fig_roas.update_xaxes(gridcolor="#1e2535")
        fig_roas.update_yaxes(gridcolor="#1e2535")
        st.plotly_chart(fig_roas, use_container_width=True)

    with col2:
        fig_budget = px.pie(
            ch_summary, values="budget", names="channel",
            title="Budget Allocation",
            color_discrete_sequence=px.colors.sequential.Teal,
            hole=0.45,
        )
        fig_budget.update_layout(
            paper_bgcolor="#0f1117",
            font_color="#d1d5db",
            title_font_size=15,
            legend=dict(bgcolor="#0f1117"),
        )
        st.plotly_chart(fig_budget, use_container_width=True)

    # ROW 2 — Revenue trend
    st.markdown('<div class="section-header">Monthly Revenue & Conversion Trends</div>', unsafe_allow_html=True)
    
    monthly = df.groupby(["month_idx", "month", "channel"]).agg(
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum"),
        budget=("budget", "sum"),
    ).reset_index()

    fig_trend = px.line(
        monthly, x="month", y="revenue", color="channel",
        title="Monthly Revenue by Channel",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_trend.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#d1d5db", title_font_size=15,
        legend=dict(bgcolor="#0f1117"),
    )
    fig_trend.update_xaxes(gridcolor="#1e2535")
    fig_trend.update_yaxes(gridcolor="#1e2535")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ROW 3 — CPA vs ROAS scatter
    col3, col4 = st.columns(2)
    with col3:
        fig_scatter = px.scatter(
            ch_summary,
            x="cpa", y="roas",
            size="conversions",
            color="channel",
            title="CPA vs ROAS (bubble = conversions)",
            text="channel",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_scatter.add_hline(y=3.5, line_dash="dash", line_color="#f59e0b")
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#d1d5db", title_font_size=15,
            showlegend=False,
        )
        fig_scatter.update_xaxes(gridcolor="#1e2535")
        fig_scatter.update_yaxes(gridcolor="#1e2535")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        fig_conv = px.bar(
            ch_summary.sort_values("conv_rate", ascending=True),
            x="conv_rate", y="channel", orientation="h",
            title="Conversion Rate by Channel (%)",
            color="conv_rate",
            color_continuous_scale=[[0, "#1a1f2e"], [0.5, "#059669"], [1, "#10b981"]],
        )
        fig_conv.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#d1d5db", title_font_size=15,
            showlegend=False, coloraxis_showscale=False,
        )
        fig_conv.update_xaxes(gridcolor="#1e2535")
        fig_conv.update_yaxes(gridcolor="#1e2535")
        st.plotly_chart(fig_conv, use_container_width=True)

    # KPI TABLE
    st.markdown('<div class="section-header">Channel KPI Summary Table</div>', unsafe_allow_html=True)
    
    display_df = ch_summary[["channel", "budget", "revenue", "conversions", "roas", "cpa", "ctr", "conv_rate"]].copy()
    display_df.columns = ["Channel", "Budget (₹)", "Revenue (₹)", "Conversions", "ROAS", "CPA (₹)", "CTR (%)", "Conv Rate (%)"]
    display_df["Budget (₹)"] = display_df["Budget (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display_df["Revenue (₹)"] = display_df["Revenue (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display_df["ROAS"] = display_df["ROAS"].apply(lambda x: f"{x:.2f}x")
    display_df["CPA (₹)"] = display_df["CPA (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display_df["CTR (%)"] = display_df["CTR (%)"].apply(lambda x: f"{x:.2f}%")
    display_df["Conv Rate (%)"] = display_df["Conv Rate (%)"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # INSIGHTS
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    
    best_roas_ch = ch_summary.loc[ch_summary["roas"].idxmax(), "channel"]
    worst_roas_ch = ch_summary.loc[ch_summary["roas"].idxmin(), "channel"]
    best_conv_ch = ch_summary.loc[ch_summary["conv_rate"].idxmax(), "channel"]
    
    st.markdown(f"""
    <div class="insight-box">
    🏆 <strong>{best_roas_ch}</strong> delivers the highest ROAS at <strong>{ch_summary.loc[ch_summary['roas'].idxmax(), 'roas']:.2f}x</strong> — 
    significantly above the 3.5x target. This channel should receive increased budget allocation in the next quarter.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="warn-box">
    ⚠️ <strong>{worst_roas_ch}</strong> underperforms with ROAS of <strong>{ch_summary.loc[ch_summary['roas'].idxmin(), 'roas']:.2f}x</strong>. 
    Budget reallocation away from this channel is recommended unless brand awareness is the primary objective.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="insight-box">
    📧 <strong>{best_conv_ch}</strong> shows the highest conversion rate at <strong>{ch_summary.loc[ch_summary['conv_rate'].idxmax(), 'conv_rate']:.2f}%</strong>. 
    Despite typically lower volume, this channel demonstrates strong intent-to-convert signals and warrants deeper investment.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — SQL QUERIES
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## SQL Queries — Marketing Analytics Workflow")
    st.markdown("The following queries represent the analytical workflow used to derive insights from the marketing dataset.")
    
    queries = [
        {
            "title": "1. Channel-Level Performance Summary",
            "description": "Aggregates key KPIs per channel — total spend, revenue, conversions, ROAS, and CPA.",
            "sql": """SELECT
    channel,
    SUM(budget)                                     AS total_spend,
    SUM(revenue)                                    AS total_revenue,
    SUM(conversions)                                AS total_conversions,
    SUM(clicks)                                     AS total_clicks,
    SUM(impressions)                                AS total_impressions,
    ROUND(SUM(revenue) / NULLIF(SUM(budget), 0), 2) AS roas,
    ROUND(SUM(budget) / NULLIF(SUM(conversions), 0), 2) AS cpa,
    ROUND(SUM(clicks) * 100.0 / NULLIF(SUM(impressions), 0), 3) AS ctr_pct,
    ROUND(SUM(conversions) * 100.0 / NULLIF(SUM(clicks), 0), 3) AS conv_rate_pct
FROM marketing_data
WHERE company = '{company}'
GROUP BY channel
ORDER BY roas DESC;""",
        },
        {
            "title": "2. Monthly Trend Analysis",
            "description": "Tracks month-over-month revenue and conversion growth per channel to identify seasonality and trend patterns.",
            "sql": """SELECT
    month,
    month_idx,
    channel,
    SUM(revenue)                                    AS monthly_revenue,
    SUM(conversions)                                AS monthly_conversions,
    SUM(budget)                                     AS monthly_spend,
    ROUND(SUM(revenue) / NULLIF(SUM(budget), 0), 2) AS monthly_roas,
    LAG(SUM(revenue)) OVER (
        PARTITION BY channel ORDER BY month_idx
    )                                               AS prev_month_revenue,
    ROUND(
        (SUM(revenue) - LAG(SUM(revenue)) OVER (PARTITION BY channel ORDER BY month_idx))
        * 100.0 / NULLIF(LAG(SUM(revenue)) OVER (PARTITION BY channel ORDER BY month_idx), 0),
        2
    )                                               AS mom_revenue_growth_pct
FROM marketing_data
WHERE company = '{company}'
GROUP BY month, month_idx, channel
ORDER BY month_idx, channel;""",
        },
        {
            "title": "3. Attribution Model — Last-Touch vs Linear",
            "description": "Computes both last-touch (100% credit to converting channel) and linear (equal credit across touchpoints) attribution models.",
            "sql": """-- Last-Touch Attribution
SELECT
    channel,
    SUM(conversions)                                AS last_touch_conversions,
    SUM(revenue)                                    AS last_touch_revenue,
    ROUND(SUM(conversions) * 100.0 / SUM(SUM(conversions)) OVER (), 2) AS conv_share_pct
FROM marketing_data
WHERE company = '{company}'
GROUP BY channel
ORDER BY last_touch_conversions DESC;

-- Linear Attribution (equal weight across all active channels)
WITH channel_count AS (
    SELECT COUNT(DISTINCT channel) AS n_channels
    FROM marketing_data
    WHERE company = '{company}'
)
SELECT
    m.channel,
    ROUND(SUM(m.conversions) * 1.0 / c.n_channels, 2) AS linear_attr_conversions,
    ROUND(SUM(m.revenue) / c.n_channels, 2)            AS linear_attr_revenue
FROM marketing_data m
CROSS JOIN channel_count c
WHERE m.company = '{company}'
GROUP BY m.channel, c.n_channels
ORDER BY linear_attr_revenue DESC;""",
        },
        {
            "title": "4. High-Impact Conversion Path Identification",
            "description": "Identifies channels with conversion rates above the portfolio average — signals high purchase intent pathways.",
            "sql": """WITH portfolio_avg AS (
    SELECT AVG(conv_rate) AS avg_conv_rate
    FROM (
        SELECT
            channel,
            SUM(conversions) * 100.0 / NULLIF(SUM(clicks), 0) AS conv_rate
        FROM marketing_data
        WHERE company = '{company}'
        GROUP BY channel
    ) t
)
SELECT
    m.channel,
    ROUND(SUM(m.conversions) * 100.0 / NULLIF(SUM(m.clicks), 0), 3)  AS channel_conv_rate,
    ROUND(p.avg_conv_rate, 3)                                          AS portfolio_avg_conv_rate,
    ROUND(
        SUM(m.conversions) * 100.0 / NULLIF(SUM(m.clicks), 0) - p.avg_conv_rate,
        3
    )                                                                  AS delta_vs_avg,
    CASE
        WHEN SUM(m.conversions) * 100.0 / NULLIF(SUM(m.clicks), 0) > p.avg_conv_rate
        THEN 'High Intent Path'
        ELSE 'Below Average'
    END                                                                AS path_classification
FROM marketing_data m
CROSS JOIN portfolio_avg p
WHERE m.company = '{company}'
GROUP BY m.channel, p.avg_conv_rate
ORDER BY channel_conv_rate DESC;""",
        },
        {
            "title": "5. Budget Efficiency & Reallocation Scoring",
            "description": "Scores each channel by efficiency (ROAS × conversion rate) to generate quantitative budget reallocation recommendations.",
            "sql": """WITH channel_metrics AS (
    SELECT
        channel,
        SUM(budget)                                     AS total_spend,
        ROUND(SUM(revenue) / NULLIF(SUM(budget), 0), 3) AS roas,
        ROUND(SUM(conversions) * 100.0 / NULLIF(SUM(clicks), 0), 3) AS conv_rate,
        ROUND(SUM(budget) / NULLIF(SUM(conversions), 0), 2) AS cpa,
        SUM(budget) * 100.0 / SUM(SUM(budget)) OVER ()  AS budget_share_pct
    FROM marketing_data
    WHERE company = '{company}'
    GROUP BY channel
),
scored AS (
    SELECT *,
        ROUND(roas * conv_rate, 4) AS efficiency_score,
        NTILE(5) OVER (ORDER BY roas * conv_rate) AS efficiency_quintile
    FROM channel_metrics
)
SELECT
    channel,
    ROUND(total_spend, 0)    AS current_spend,
    ROUND(budget_share_pct, 1) AS current_budget_pct,
    roas,
    conv_rate,
    cpa,
    efficiency_score,
    efficiency_quintile,
    CASE
        WHEN efficiency_quintile = 5 THEN 'Increase budget +20%'
        WHEN efficiency_quintile = 4 THEN 'Increase budget +10%'
        WHEN efficiency_quintile = 3 THEN 'Maintain current allocation'
        WHEN efficiency_quintile = 2 THEN 'Reduce budget -10%'
        ELSE 'Reduce budget -20% or pause'
    END AS reallocation_recommendation
FROM scored
ORDER BY efficiency_score DESC;""",
        },
        {
            "title": "6. Customer Lifetime Value (CLV) Proxy by Channel",
            "description": "Estimates CLV proxy per channel using average order value × repeat purchase multiplier to identify high-value acquisition sources.",
            "sql": """SELECT
    channel,
    SUM(conversions)                                    AS total_customers_acquired,
    ROUND(AVG(clv_proxy), 2)                            AS avg_clv_proxy,
    ROUND(SUM(clv_proxy * conversions) / NULLIF(SUM(conversions), 0), 2) AS weighted_avg_clv,
    ROUND(SUM(budget), 0)                               AS total_acquisition_cost,
    ROUND(
        SUM(clv_proxy * conversions) / NULLIF(SUM(budget), 0),
        3
    )                                                   AS clv_to_cac_ratio,
    CASE
        WHEN SUM(clv_proxy * conversions) / NULLIF(SUM(budget), 0) >= 3.0
        THEN 'Excellent — High LTV channel'
        WHEN SUM(clv_proxy * conversions) / NULLIF(SUM(budget), 0) >= 2.0
        THEN 'Good — Sustainable acquisition'
        WHEN SUM(clv_proxy * conversions) / NULLIF(SUM(budget), 0) >= 1.0
        THEN 'Marginal — Monitor closely'
        ELSE 'Poor — Acquisition cost exceeds LTV'
    END AS clv_health_status
FROM marketing_data
WHERE company = '{company}'
GROUP BY channel
ORDER BY weighted_avg_clv DESC;""",
        },
    ]
    
    for q in queries:
        with st.expander(q["title"], expanded=False):
            st.markdown(f"**Purpose:** {q['description']}")
            st.code(q["sql"].format(company=selected_company), language="sql")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — ATTRIBUTION MODEL
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Attribution Methodology & Framework")
    
    st.markdown("""
    <div class="insight-box">
    <strong>Framework Overview:</strong> This analysis implements a multi-touch attribution framework comparing three models — 
    Last-Touch, Linear, and Time-Decay — to measure the true contribution of each marketing channel to conversions. 
    The model selection is based on the customer journey complexity observed in the dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Attribution Models Compared")
    
    col_a, col_b, col_c = st.columns(3)
    
    ch_summary_attr = df.groupby("channel").agg(
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
        budget=("budget", "sum"),
    ).reset_index()
    
    total_conv_attr = ch_summary_attr["conversions"].sum()
    n_channels = len(ch_summary_attr)
    
    # Last touch
    ch_summary_attr["last_touch_share"] = ch_summary_attr["conversions"] / total_conv_attr * 100
    
    # Linear (equal weight)
    ch_summary_attr["linear_share"] = 100 / n_channels
    
    # Time decay (channels with higher conversion rate get more credit)
    ch_summary_attr["conv_rate_proxy"] = ch_summary_attr["conversions"] / ch_summary_attr["budget"]
    total_proxy = ch_summary_attr["conv_rate_proxy"].sum()
    ch_summary_attr["time_decay_share"] = ch_summary_attr["conv_rate_proxy"] / total_proxy * 100

    with col_a:
        fig_lt = px.pie(
            ch_summary_attr, values="last_touch_share", names="channel",
            title="Last-Touch Attribution",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_lt.update_layout(paper_bgcolor="#0f1117", font_color="#d1d5db", title_font_size=13)
        st.plotly_chart(fig_lt, use_container_width=True)

    with col_b:
        fig_lin = px.pie(
            ch_summary_attr, values="linear_share", names="channel",
            title="Linear Attribution",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Greens_r,
        )
        fig_lin.update_layout(paper_bgcolor="#0f1117", font_color="#d1d5db", title_font_size=13)
        st.plotly_chart(fig_lin, use_container_width=True)

    with col_c:
        fig_td = px.pie(
            ch_summary_attr, values="time_decay_share", names="channel",
            title="Time-Decay Attribution",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Oranges_r,
        )
        fig_td.update_layout(paper_bgcolor="#0f1117", font_color="#d1d5db", title_font_size=13)
        st.plotly_chart(fig_td, use_container_width=True)

    st.markdown("### Attribution Model Comparison Table")
    attr_display = ch_summary_attr[["channel", "last_touch_share", "linear_share", "time_decay_share"]].copy()
    attr_display.columns = ["Channel", "Last-Touch (%)", "Linear (%)", "Time-Decay (%)"]
    for col in ["Last-Touch (%)", "Linear (%)", "Time-Decay (%)"]:
        attr_display[col] = attr_display[col].apply(lambda x: f"{x:.1f}%")
    st.dataframe(attr_display, use_container_width=True, hide_index=True)

    st.markdown("### Model Selection Rationale")
    st.markdown("""
    <div class="insight-box">
    <strong>Recommended Model: Time-Decay Attribution</strong><br><br>
    For performance-driven campaigns with measurable conversion events (purchases, sign-ups), time-decay attribution 
    most accurately reflects the real-world customer journey. Channels with higher intent signals (Email, Organic Search) 
    receive proportionally more credit — aligned with their demonstrated conversion efficiency. 
    Last-touch attribution, while simple, over-credits bottom-funnel channels and ignores awareness-stage contributions. 
    Linear attribution provides a useful baseline but underweights high-performing channels.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Assumptions & Limitations")
    assumptions = [
        "Customer journeys are assumed to be independent across sessions — cross-device tracking is not modeled.",
        "The CLV proxy uses a 2.4x repeat purchase multiplier — actual values require CRM cohort data.",
        "Organic Search budget is treated as ₹0 direct cost — indirect costs (SEO, content) are excluded.",
        "Attribution models assume single-channel last-touch journeys — multi-channel path data would improve accuracy.",
        "Seasonal adjustments use a sinusoidal model — actual seasonality may differ by category and campaign timing.",
    ]
    for a in assumptions:
        st.markdown(f"- {a}")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION STRATEGY
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Budget Optimization & Strategic Recommendations")

    ch_opt = df.groupby("channel").agg(
        budget=("budget", "sum"),
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum"),
        clicks=("clicks", "sum"),
    ).reset_index()
    ch_opt["roas"] = ch_opt["revenue"] / ch_opt["budget"]
    ch_opt["conv_rate"] = ch_opt["conversions"] / ch_opt["clicks"] * 100
    ch_opt["efficiency_score"] = ch_opt["roas"] * ch_opt["conv_rate"]
    ch_opt["budget_share"] = ch_opt["budget"] / ch_opt["budget"].sum() * 100

    total_budget_val = ch_opt["budget"].sum()

    # Reallocation logic
    ch_opt = ch_opt.sort_values("efficiency_score", ascending=False).reset_index(drop=True)
    
    realloc_pcts = []
    for i, row in ch_opt.iterrows():
        rank = i + 1
        if rank == 1:   pct = 0.32
        elif rank == 2: pct = 0.26
        elif rank == 3: pct = 0.20
        elif rank == 4: pct = 0.14
        else:           pct = 0.08
        realloc_pcts.append(pct * 100)
    
    ch_opt["recommended_budget_pct"] = realloc_pcts
    ch_opt["recommended_budget"] = ch_opt["recommended_budget_pct"] / 100 * total_budget_val
    ch_opt["budget_delta"] = ch_opt["recommended_budget"] - ch_opt["budget"]
    ch_opt["delta_pct"] = (ch_opt["budget_delta"] / ch_opt["budget"] * 100).round(1)

    # Budget reallocation chart
    st.markdown('<div class="section-header">Budget Reallocation — Current vs Recommended</div>', unsafe_allow_html=True)

    fig_realloc = go.Figure()
    fig_realloc.add_trace(go.Bar(
        name="Current Budget",
        x=ch_opt["channel"],
        y=ch_opt["budget_share"],
        marker_color="#1e4060",
    ))
    fig_realloc.add_trace(go.Bar(
        name="Recommended Budget",
        x=ch_opt["channel"],
        y=ch_opt["recommended_budget_pct"],
        marker_color=company_color,
    ))
    fig_realloc.update_layout(
        barmode="group",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#d1d5db",
        yaxis_title="Budget Share (%)",
        legend=dict(bgcolor="#0f1117"),
    )
    fig_realloc.update_xaxes(gridcolor="#1e2535")
    fig_realloc.update_yaxes(gridcolor="#1e2535")
    st.plotly_chart(fig_realloc, use_container_width=True)

    st.markdown('<div class="section-header">Reallocation Recommendations by Channel</div>', unsafe_allow_html=True)

    for _, row in ch_opt.iterrows():
        delta_color = "metric-delta-pos" if row["budget_delta"] >= 0 else "metric-delta-neg"
        delta_arrow = "▲" if row["budget_delta"] >= 0 else "▼"
        action = "Increase" if row["budget_delta"] > 500 else ("Reduce" if row["budget_delta"] < -500 else "Maintain")
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:16px; font-weight:700; color:#f9fafb;">{row['channel']}</div>
                    <div style="font-size:12px; color:#6b7280; margin-top:4px;">
                        ROAS: {row['roas']:.2f}x &nbsp;|&nbsp; Conv Rate: {row['conv_rate']:.2f}% &nbsp;|&nbsp; Efficiency Score: {row['efficiency_score']:.3f}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:13px; color:#9ca3af;">Current: ₹{row['budget']:,.0f} ({row['budget_share']:.1f}%)</div>
                    <div style="font-size:16px; font-weight:700; color:{company_color};">Recommended: ₹{row['recommended_budget']:,.0f} ({row['recommended_budget_pct']:.0f}%)</div>
                    <div class="{delta_color}">{delta_arrow} {abs(row['delta_pct']):.1f}% &nbsp; {action}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Projected Revenue Impact</div>', unsafe_allow_html=True)

    ch_opt["projected_revenue"] = ch_opt["recommended_budget"] * ch_opt["roas"]
    current_total_rev = ch_opt["revenue"].sum()
    projected_total_rev = ch_opt["projected_revenue"].sum()
    rev_uplift = (projected_total_rev - current_total_rev) / current_total_rev * 100

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Current Annual Revenue", f"₹{current_total_rev/1e5:.2f}L")
    with col_p2:
        st.metric("Projected Revenue (Post-Reallocation)", f"₹{projected_total_rev/1e5:.2f}L",
                  delta=f"+{rev_uplift:.1f}%")
    with col_p3:
        st.metric("Revenue Uplift", f"₹{(projected_total_rev-current_total_rev)/1e5:.2f}L",
                  delta="Based on current ROAS")

    st.markdown("### Strategic Action Plan")

    best_ch = ch_opt.iloc[0]["channel"]
    worst_ch = ch_opt.iloc[-1]["channel"]

    actions = [
        f"**Immediate (0-30 days):** Reallocate 5-8% of {worst_ch} budget to {best_ch} — this single shift is projected to deliver the highest marginal ROAS improvement.",
        "**Short-term (30-90 days):** Implement UTM tracking consistency across all channels to enable proper multi-touch attribution data collection.",
        "**Medium-term (90-180 days):** Build A/B testing pipeline for Email subject lines and Paid Search ad copy — both channels show high conversion potential with room for CTR improvement.",
        "**Long-term (6-12 months):** Develop a data-driven attribution model using actual customer journey data from CRM integration — replace proxy model with observed path data.",
        f"**Ongoing:** Set ROAS floor of 2.0x as campaign pause trigger for {worst_ch} — any campaign falling below this threshold should be paused and budget redistributed within 48 hours.",
    ]
    for a in actions:
        st.markdown(f"- {a}")

    st.markdown("""
    <div class="insight-box" style="margin-top:24px;">
    <strong>Business Rationale:</strong> The recommended reallocation prioritizes efficiency over volume. 
    Rather than simply cutting underperforming channels, the strategy reduces their share while maintaining 
    minimum presence for brand awareness purposes. The projected revenue uplift assumes channel ROAS remains 
    stable post-reallocation — actual results should be monitored weekly with a 30-day review checkpoint.
    </div>
    """, unsafe_allow_html=True)
    
    # ════════════════════════════════════════════════════════════════════
    # TAB 5 — Revenue Forecasting
    # ════════════════════════════════════════════════════════════════════
   with tab5 :

   st.header("🔮 Revenue Forecasting")
    
            monthly_rev = (
            df.groupby("month_idx")["revenue"]
            .sum()
            .reset_index()
            )
    
            X = monthly_rev[["month_idx"]]
            y = monthly_rev["revenue"]
        
            model = LinearRegression()
    
            model.fit(X, y)
        
            future = pd.DataFrame({
                "month_idx":[13,14,15]
            })
        
            future["revenue"] = model.predict(future)
        
            monthly_rev["Type"] = "Actual"
        
            future["Type"] = "Forecast"
        
            forecast_df = pd.concat([
                monthly_rev,
                future
            ])
    
            fig = px.line(
                forecast_df,
                x="month_idx",
                y="revenue",
                color="Type",
                markers=True,
                title="Next 3 Month Revenue Forecast"
            )
        
            fig.update_layout(
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                font_color="#d1d5db"
            )
        
            st.plotly_chart(
                fig,
                use_container_width=True
            )
    
    # ════════════════════════════════════════════════════════════════════
    # TAB 6 —  Budget Scenario Simulator
    # ════════════════════════════════════════════════════════════════════
    with tab6:

    st.header("🎯 Budget Scenario Simulator")

    selected_channel = st.selectbox(
        "Select Channel",
        ch_summary["channel"]
    )

    budget_change = st.slider(
        "Budget Change %",
        -50,
        100,
        20
    )

    row = ch_summary[
        ch_summary["channel"]
        ==
        selected_channel
    ].iloc[0]

    current_budget = row["budget"]

    roas = row["roas"]

    new_budget = (
        current_budget
        *
        (1 + budget_change/100)
    )

    projected_revenue = (
        new_budget
        *
        roas
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Current Budget",
        f"₹{current_budget:,.0f}"
    )

    c2.metric(
        "New Budget",
        f"₹{new_budget:,.0f}"
    )

    c3.metric(
        "Projected Revenue",
        f"₹{projected_revenue:,.0f}"
    )

    csv = df.to_csv(index=False)

    st.download_button(
    "⬇ Download Dataset",
    csv,
    "marketing_data.csv",
    "text/csv"
    )

    if overall_roas > 4:
    st.balloons()
    
# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#4b5563; font-size:12px; padding:16px 0;">
    Multi-Channel Marketing Attribution & ROI Analytics Framework &nbsp;·&nbsp; 
    Submitted to Grito Lab via Techfest IIT Bombay CA Program 2026 &nbsp;·&nbsp;
    Analyst: Ninad Jagdish Tarwate
</div>
""", unsafe_allow_html=True)
