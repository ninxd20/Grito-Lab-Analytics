# Multi-Channel Marketing Attribution & ROI Analytics

**Submitted to:** Grito Lab via Techfest IIT Bombay CA Program 2026  
**Analyst:** Ninad Jagdish Tarwate  
**Domain:** Marketing Analytics & Business Intelligence

---

## Project Overview

An interactive marketing analytics framework built to analyze multi-channel campaign performance, evaluate attribution models, and generate data-driven budget optimization recommendations.

The analysis covers three companies across different industries:
- **NovaMart** — E-Commerce
- **PulseHealth** — Healthcare & Wellness
- **UrbanDrive** — Travel & Mobility

---

## What's Inside

### Tab 1 — Dashboard
- KPI cards: Total Spend, Revenue, ROAS, Conversions, CTR
- ROAS by Channel (bar chart with 3.5x target line)
- Budget Allocation (pie chart)
- Monthly Revenue & Conversion Trends
- CPA vs ROAS Scatter (bubble = conversions)
- Conversion Rate by Channel
- Full KPI Summary Table
- Automated Key Insights

### Tab 2 — SQL Queries
Six production-ready SQL queries:
1. Channel-Level Performance Summary
2. Monthly Trend Analysis with MoM Growth
3. Attribution Model (Last-Touch vs Linear)
4. High-Impact Conversion Path Identification
5. Budget Efficiency & Reallocation Scoring
6. Customer Lifetime Value (CLV) Proxy by Channel

### Tab 3 — Attribution Model
- Last-Touch, Linear, and Time-Decay attribution comparison
- Visual attribution share by model
- Model selection rationale
- Key assumptions and limitations

### Tab 4 — Optimization Strategy
- Current vs Recommended budget reallocation chart
- Per-channel reallocation recommendations with delta %
- Projected revenue impact
- Strategic action plan (0-30 days → 6-12 months)

---

## KPIs Covered
- ROAS (Return on Ad Spend)
- CPA (Cost Per Acquisition)
- CTR (Click-Through Rate)
- Conversion Rate
- CPC (Cost Per Click)
- CLV Proxy (Customer Lifetime Value)
- Budget Efficiency Score

---

## Tech Stack
- Python 3.11
- Streamlit
- Pandas
- NumPy
- Plotly

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Fork/upload this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Deploy — live URL generated automatically
