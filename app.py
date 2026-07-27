"""
Customer Segmentation & Churn Pattern Analytics — European Banking
Streamlit Dashboard  |  Unified Mentor Scholarship Submission
Author: Sanjana Allanki
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ──────────────────────────────────────────────────────────────
NAVY  = "#1B2A4A"
BLUE  = "#2E5C8A"
TEAL  = "#3D8B8B"
GOLD  = "#C9A646"
RED   = "#B0413E"
GREY  = "#8A93A3"
PAL   = [BLUE, RED, TEAL, GOLD, NAVY, GREY]

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background:#1B2A4A; }
    [data-testid="stSidebar"] * { color:#F0F4FF !important; }
    [data-testid="stMetricValue"] { font-size:1.9rem !important; font-weight:700; color:#1B2A4A; }
    [data-testid="stMetricLabel"] { font-size:.85rem; color:#5C6B7A; }
    .kpi-card {
        background:white; border-radius:10px; padding:16px 20px;
        box-shadow:0 2px 8px rgba(0,0,0,.08);
        border-left:4px solid #2E5C8A;
    }
    h1,h2,h3 { color:#1B2A4A !important; }
    .stTabs [data-baseweb="tab"] { font-weight:600; color:#1B2A4A; }
    .stTabs [aria-selected="true"] { border-bottom-color:#2E5C8A !important; }
</style>
""", unsafe_allow_html=True)

# ── Data & model loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("/home/claude/project/data/processed_bank_data.csv")
    cat_cols = ["AgeGroup", "CreditBand", "TenureGroup", "BalanceSegment", "SegmentLabel",
                "ChurnLabel", "ActiveLabel", "CrCardLabel"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

@st.cache_resource
def load_models():
    rf  = joblib.load("/home/claude/project/models/random_forest_model.joblib")
    xgb = joblib.load("/home/claude/project/models/xgboost_model.joblib")
    feat = joblib.load("/home/claude/project/models/feature_columns.joblib")
    return rf, xgb, feat

def apply_filters(df, geo, gender, age_group, bal_seg):
    mask = pd.Series([True] * len(df))
    if geo     != "All": mask &= df["Geography"]     == geo
    if gender  != "All": mask &= df["Gender"]         == gender
    if age_group != "All": mask &= df["AgeGroup"]    == age_group
    if bal_seg != "All": mask &= df["BalanceSegment"] == bal_seg
    return df[mask].copy()

def kpi_card(col, label, value, delta=None, suffix=""):
    with col:
        st.markdown(f"""<div class="kpi-card">
            <p style="margin:0;font-size:.8rem;color:#5C6B7A;font-weight:600">{label}</p>
            <p style="margin:0;font-size:1.9rem;font-weight:700;color:#1B2A4A">{value}{suffix}</p>
            {f'<p style="margin:0;font-size:.78rem;color:#B0413E">{delta}</p>' if delta else ""}
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
df_raw = load_data()

with st.sidebar:
    st.markdown("### 🏦 European Bank")
    st.markdown("**Churn Analytics Dashboard**")
    st.markdown("---")
    st.markdown("#### 🎛️ Segment Filters")
    geo       = st.selectbox("Geography",      ["All"] + sorted(df_raw["Geography"].unique()))
    gender    = st.selectbox("Gender",         ["All"] + sorted(df_raw["Gender"].unique()))
    age_group = st.selectbox("Age Group",      ["All", "<30", "30-45", "46-60", "60+"])
    bal_seg   = st.selectbox("Balance Tier",   ["All", "Zero-balance", "Low-balance", "High-balance"])
    st.markdown("---")
    st.markdown("#### ℹ️ Dataset")
    st.markdown(f"**Customers:** {len(df_raw):,}")
    st.markdown("**Geographies:** France · Spain · Germany")
    st.markdown("**Target:** Customer Churn (Exited)")

df = apply_filters(df_raw, geo, gender, age_group, bal_seg)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(135deg,{NAVY},{BLUE});
    border-radius:12px;padding:20px 28px;margin-bottom:18px;">
  <h1 style="color:white!important;margin:0;font-size:1.7rem">
    🏦 Customer Segmentation & Churn Pattern Analytics</h1>
  <p style="color:#A8C4E0;margin:4px 0 0">
    European Banking Dataset · {len(df):,} customers in current filter</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 5 KPIs (per project brief)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Key Performance Indicators")

overall_churn   = df["Exited"].mean()
segment_churns  = df.groupby("SegmentLabel")["Exited"].mean()
worst_segment   = segment_churns.idxmax()
worst_rate      = segment_churns.max()
geo_risk_idx    = df.groupby("Geography")["Exited"].mean() / df_raw["Exited"].mean()
highest_risk_geo = geo_risk_idx.idxmax()
inactive_churn  = df[df["IsActiveMember"] == 0]["Exited"].mean() if len(df) else 0
active_churn    = df[df["IsActiveMember"] == 1]["Exited"].mean() if len(df) else 0
engagement_drop = inactive_churn / active_churn if active_churn > 0 else 0
hv_churn        = df[df["IsHighValue"] == 1]["Exited"].mean() if len(df) else 0
hv_ratio        = hv_churn / overall_churn if overall_churn > 0 else 1
churned_balance = df[df["Exited"] == 1]["Balance"].sum()
total_balance   = df["Balance"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
kpi_card(c1, "1 · Overall Churn Rate",        f"{overall_churn:.1%}", "% customers who exited")
kpi_card(c2, "2 · Highest Segment Churn",     f"{worst_rate:.1%}",    worst_segment)
kpi_card(c3, "3 · High-Value Churn Ratio",    f"{hv_ratio:.2f}×",     "premium vs average")
kpi_card(c4, "4 · Geographic Risk Index",     f"{geo_risk_idx.max():.2f}×", f"highest: {highest_risk_geo}")
kpi_card(c5, "5 · Engagement Drop Indicator", f"{engagement_drop:.2f}×", "inactive vs active churn")
st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS (Core Modules per project brief)
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 Overall Churn Summary",
    "🌍 Geography-Wise Churn",
    "👥 Age & Tenure Analysis",
    "💎 High-Value Customers",
    "🤖 ML Churn Predictor",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERALL CHURN SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### Churn Distribution Overview")
    c1, c2 = st.columns([1, 2])

    with c1:
        pie_df = df["ChurnLabel"].value_counts().reset_index()
        pie_df.columns = ["Status", "Count"]
        fig = px.pie(pie_df, names="Status", values="Count",
                     color="Status", hole=0.5,
                     color_discrete_map={"Churned": RED, "Retained": BLUE})
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                          height=300, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        drivers = {
            "Age Group": df.groupby("AgeGroup", observed=True)["Exited"].mean().reindex(["<30","30-45","46-60","60+"]),
            "Products":  df.groupby("NumOfProducts")["Exited"].mean(),
        }
        sub1, sub2 = st.columns(2)
        for col_c, (label, s) in zip([sub1, sub2], drivers.items()):
            fig2 = px.bar(x=s.index.astype(str), y=(s * 100).round(1),
                          color_discrete_sequence=[BLUE],
                          labels={"x": label, "y": "Churn Rate %"}, text_auto=True)
            fig2.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10),
                               paper_bgcolor="white", plot_bgcolor="white",
                               title=f"Churn by {label}")
            fig2.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            col_c.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔍 Drill-Down: Balance & Activity")
    d1, d2 = st.columns(2)
    with d1:
        bseg = df.groupby("BalanceSegment", observed=True)["Exited"].mean().reindex(
            ["Zero-balance", "Low-balance", "High-balance"]) * 100
        fig = px.bar(x=bseg.index, y=bseg.values, color=bseg.index,
                     color_discrete_sequence=PAL, text_auto=True,
                     labels={"x": "Balance Segment", "y": "Churn %"})
        fig.update_layout(title="Churn by Balance Segment", showlegend=False,
                          height=300, paper_bgcolor="white", plot_bgcolor="white")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        act = df.groupby("ActiveLabel")["Exited"].mean() * 100
        fig = px.bar(x=act.index, y=act.values, color=act.index,
                     color_discrete_map={"Active": BLUE, "Inactive": RED},
                     text_auto=True, labels={"x": "", "y": "Churn %"})
        fig.update_layout(title="Active vs Inactive Member Churn", showlegend=False,
                          height=300, paper_bgcolor="white", plot_bgcolor="white")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Churn Summary Table"):
        dims = ["Geography", "Gender", "AgeGroup", "CreditBand", "BalanceSegment"]
        summary = df.groupby([d for d in dims if d in df.columns],
                             observed=True)["Exited"].agg(
            Count="count", Churned="sum", ChurnRate="mean"
        ).reset_index()
        summary["ChurnRate"] = (summary["ChurnRate"] * 100).round(2).astype(str) + "%"
        st.dataframe(summary, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — GEOGRAPHY-WISE CHURN
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Geographic Churn Risk Analysis")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        geo_age = df.groupby(["Geography", "AgeGroup"], observed=True)["Exited"].mean() * 100
        geo_age = geo_age.reset_index().rename(columns={"Exited": "ChurnRate"})
        order = ["<30", "30-45", "46-60", "60+"]
        geo_age["AgeGroup"] = pd.Categorical(geo_age["AgeGroup"], categories=order, ordered=True)
        geo_age = geo_age.sort_values("AgeGroup")
        fig = px.bar(geo_age, x="AgeGroup", y="ChurnRate", color="Geography",
                     barmode="group", color_discrete_sequence=PAL,
                     text_auto=True,
                     labels={"AgeGroup": "Age Group", "ChurnRate": "Churn Rate (%)"})
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(title="Churn Rate by Geography × Age Group",
                          height=380, paper_bgcolor="white", plot_bgcolor="white",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        geo_gender = df.groupby(["Geography", "Gender"])["Exited"].mean() * 100
        geo_gender = geo_gender.reset_index().rename(columns={"Exited": "ChurnRate"})
        fig2 = px.bar(geo_gender, x="Geography", y="ChurnRate", color="Gender",
                      barmode="group", color_discrete_map={"Female": GOLD, "Male": BLUE},
                      text_auto=True)
        fig2.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig2.update_layout(title="Churn by Geography × Gender",
                           height=380, paper_bgcolor="white", plot_bgcolor="white",
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 📍 Regional Risk Index (vs overall average)")
    geo_metrics = df.groupby("Geography")["Exited"].agg(["count", "mean"]).reset_index()
    geo_metrics.columns = ["Geography", "Customers", "ChurnRate"]
    geo_metrics["RiskIndex"] = (geo_metrics["ChurnRate"] / overall_churn).round(2)
    geo_metrics["ChurnRate"] = (geo_metrics["ChurnRate"] * 100).round(2).astype(str) + "%"
    geo_metrics["BalanceAtRisk(EUR)"] = df.groupby("Geography").apply(
        lambda x: x[x["Exited"] == 1]["Balance"].sum()).values.round(0)
    st.dataframe(geo_metrics.style.background_gradient(subset=["RiskIndex"], cmap="RdYlGn_r"),
                 use_container_width=True)

    with st.expander("🔍 Drill-Down: Geography by Product Count & Activity"):
        d1, d2 = st.columns(2)
        with d1:
            gp = df.groupby(["Geography", "NumOfProducts"])["Exited"].mean() * 100
            gp = gp.reset_index()
            fig = px.bar(gp, x="NumOfProducts", y="Exited", color="Geography",
                         barmode="group", color_discrete_sequence=PAL,
                         labels={"Exited": "Churn %", "NumOfProducts": "# Products"})
            fig.update_layout(title="Churn by Products × Geography", height=320,
                              paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with d2:
            ga = df.groupby(["Geography", "ActiveLabel"])["Exited"].mean() * 100
            ga = ga.reset_index()
            fig = px.bar(ga, x="Geography", y="Exited", color="ActiveLabel",
                         barmode="group",
                         color_discrete_map={"Active": BLUE, "Inactive": RED},
                         labels={"Exited": "Churn %"})
            fig.update_layout(title="Churn by Activity × Geography", height=320,
                              paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — AGE & TENURE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Age & Tenure Churn Comparison")
    c1, c2 = st.columns(2)
    with c1:
        order = ["<30", "30-45", "46-60", "60+"]
        ag = df.groupby("AgeGroup", observed=True)["Exited"].agg(["count", "mean"]).reindex(order).reset_index()
        ag["ChurnRate"] = ag["mean"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ag["AgeGroup"], y=ag["ChurnRate"],
                             marker_color=[RED if v > overall_churn*100 else BLUE for v in ag["ChurnRate"]],
                             text=[f"{v:.1f}%" for v in ag["ChurnRate"]],
                             textposition="outside", name="Churn %"))
        fig.add_hline(y=overall_churn * 100, line_dash="dash", line_color=GREY,
                      annotation_text=f"Avg {overall_churn*100:.1f}%", annotation_position="right")
        fig.update_layout(title="Churn Rate by Age Group", height=350,
                          paper_bgcolor="white", plot_bgcolor="white",
                          yaxis_title="Churn Rate (%)", xaxis_title="Age Group")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tg_order = ["New", "Mid-term", "Long-term"]
        tg = df.groupby("TenureGroup", observed=True)["Exited"].agg(["count", "mean"]).reindex(tg_order).reset_index()
        tg["ChurnRate"] = tg["mean"] * 100
        fig = px.bar(tg, x="TenureGroup", y="ChurnRate", color="TenureGroup",
                     color_discrete_sequence=[TEAL, BLUE, NAVY], text_auto=True,
                     labels={"TenureGroup": "Tenure Group", "ChurnRate": "Churn Rate (%)"})
        fig.add_hline(y=overall_churn * 100, line_dash="dash", line_color=GREY)
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(title="Churn Rate by Tenure Group", height=350, showlegend=False,
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔥 Age Group × Gender Churn Heatmap")
    pivot = df.groupby(["Gender", "AgeGroup"], observed=True)["Exited"].mean() * 100
    pivot = pivot.unstack().reindex(columns=order)
    fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn_r",
                    labels={"color": "Churn %"},
                    aspect="auto")
    fig.update_layout(title="Churn Rate (%) by Gender × Age Group", height=250,
                      paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Drill-Down: Age Group by Products & Credit Band"):
        d1, d2 = st.columns(2)
        with d1:
            ap = df.groupby(["AgeGroup", "NumOfProducts"], observed=True)["Exited"].mean() * 100
            ap = ap.reset_index()
            fig = px.line(ap, x="AgeGroup", y="Exited", color="NumOfProducts",
                          markers=True, color_discrete_sequence=PAL,
                          labels={"Exited": "Churn %", "NumOfProducts": "# Products"})
            fig.update_layout(title="Churn by Age × Products", height=320,
                              paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with d2:
            ac = df.groupby(["AgeGroup", "CreditBand"], observed=True)["Exited"].mean() * 100
            ac = ac.reset_index()
            fig = px.bar(ac, x="AgeGroup", y="Exited", color="CreditBand",
                         barmode="group", color_discrete_sequence=PAL,
                         labels={"Exited": "Churn %"})
            fig.update_layout(title="Churn by Age × Credit Band", height=320,
                              paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — HIGH-VALUE CUSTOMER CHURN EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("#### 💎 High-Value Customer Churn Explorer")

    c1, c2, c3 = st.columns(3)
    bal_p75 = df_raw["Balance"].quantile(0.75)
    hv_df = df[df["Balance"] >= bal_p75]
    retained_hv = hv_df[hv_df["Exited"] == 0]
    churned_hv  = hv_df[hv_df["Exited"] == 1]
    c1.metric("High-Value Customers", f"{len(hv_df):,}", f"{len(hv_df)/len(df)*100:.1f}% of segment")
    c2.metric("HV Churn Rate", f"{hv_df['Exited'].mean():.1%}", f"vs {overall_churn:.1%} avg")
    c3.metric("Balance at Risk (HV)", f"€{churned_hv['Balance'].sum()/1e6:.1f}M",
              f"{churned_hv['Balance'].sum()/hv_df['Balance'].sum()*100:.1f}% of HV balance")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(hv_df, x="Balance", color="ChurnLabel",
                           nbins=40, barmode="overlay", opacity=0.7,
                           color_discrete_map={"Churned": RED, "Retained": BLUE},
                           labels={"Balance": "Account Balance (EUR)", "count": "# Customers"})
        fig.update_layout(title="Balance Distribution: HV Churned vs Retained",
                          height=340, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hv_geo = hv_df.groupby("Geography")["Exited"].mean() * 100
        fig = px.bar(x=hv_geo.index, y=hv_geo.values, color=hv_geo.index,
                     color_discrete_sequence=PAL, text_auto=True,
                     labels={"x": "Geography", "y": "Churn Rate (%)"})
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(title="HV Customer Churn by Geography", showlegend=False,
                          height=340, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🎯 Data-Driven Customer Segments (K-Means)")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        scatter_df = df.sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(scatter_df, x="Age", y="Balance", color="SegmentLabel",
                         symbol="ChurnLabel",
                         color_discrete_sequence=PAL,
                         opacity=0.65, size_max=7,
                         labels={"Balance": "Account Balance (EUR)"},
                         hover_data=["Geography", "Gender", "NumOfProducts"])
        fig.update_layout(title="K-Means Segments: Age vs Balance",
                          height=420, paper_bgcolor="white", plot_bgcolor="white",
                          legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_stats = df.groupby("SegmentLabel")["Exited"].agg(
            Count="count", ChurnRate="mean"
        ).reset_index().sort_values("ChurnRate", ascending=False)
        seg_stats["ChurnRate%"] = (seg_stats["ChurnRate"] * 100).round(1)
        fig = px.bar(seg_stats, y="SegmentLabel", x="ChurnRate%",
                     orientation="h",
                     color="ChurnRate%", color_continuous_scale="RdYlGn_r",
                     text="ChurnRate%",
                     labels={"ChurnRate%": "Churn Rate (%)", "SegmentLabel": ""})
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig.update_layout(title="Churn Rate by ML Segment",
                          height=420, paper_bgcolor="white", plot_bgcolor="white",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Salary vs Balance Churn Patterns"):
        fig = px.scatter(df.sample(min(4000, len(df)), random_state=1),
                         x="EstimatedSalary", y="Balance",
                         color="ChurnLabel", opacity=0.5,
                         color_discrete_map={"Churned": RED, "Retained": BLUE},
                         labels={"EstimatedSalary": "Estimated Salary (EUR)",
                                 "Balance": "Account Balance (EUR)"})
        fig.update_layout(title="Salary vs Balance: Churned vs Retained",
                          height=380, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — ML CHURN PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("#### 🤖 Dual-Model Churn Predictor")

    sub1, sub2 = st.tabs(["🔮 Live Customer Prediction", "📉 Model Evaluation"])

    with sub1:
        st.markdown("Enter a customer profile to get an instant churn probability from both models.")
        rf_model, xgb_model, feat_cols = load_models()

        c1, c2, c3 = st.columns(3)
        with c1:
            c_score   = st.slider("Credit Score",      350, 850, 650)
            geography = st.selectbox("Geography",     ["France", "Germany", "Spain"])
            gender    = st.selectbox("Gender",        ["Male", "Female"])
            age       = st.slider("Age",               18, 90, 40)
        with c2:
            tenure    = st.slider("Tenure (years)",    0, 10, 5)
            balance   = st.number_input("Account Balance (€)", 0.0, 300000.0, 60000.0, step=1000.0)
            n_prods   = st.selectbox("Number of Products", [1, 2, 3, 4])
        with c3:
            has_card  = st.selectbox("Has Credit Card",    ["Yes", "No"])
            is_active = st.selectbox("Is Active Member",   ["Yes", "No"])
            salary    = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 100000.0, step=1000.0)

        if st.button("🔮 Predict Churn Probability", type="primary", use_container_width=True):
            row = {
                "CreditScore":     c_score,
                "Age":             age,
                "Tenure":          tenure,
                "Balance":         balance,
                "NumOfProducts":   n_prods,
                "HasCrCard":       1 if has_card == "Yes" else 0,
                "IsActiveMember":  1 if is_active == "Yes" else 0,
                "EstimatedSalary": salary,
                "Geography_Germany": int(geography == "Germany"),
                "Geography_Spain":   int(geography == "Spain"),
                "Gender_Male":       int(gender == "Male"),
            }
            X_in = pd.DataFrame([row])[feat_cols]

            rf_prob  = rf_model.predict_proba(X_in)[0][1]
            xgb_prob = xgb_model.predict_proba(X_in)[0][1]
            ensemble = (rf_prob + xgb_prob) / 2

            c1, c2, c3 = st.columns(3)
            c1.metric("🌲 Random Forest",   f"{rf_prob:.1%}")
            c2.metric("⚡ XGBoost",          f"{xgb_prob:.1%}")
            c3.metric("🎯 Ensemble Avg",     f"{ensemble:.1%}")

            risk = "🔴 HIGH RISK" if ensemble > 0.5 else ("🟡 MEDIUM RISK" if ensemble > 0.25 else "🟢 LOW RISK")
            color = RED if ensemble > 0.5 else (GOLD if ensemble > 0.25 else TEAL)
            st.markdown(f"""
            <div style="background:{color};color:white;padding:14px 22px;
                border-radius:10px;text-align:center;font-size:1.2rem;font-weight:700;margin:10px 0">
                {risk} &nbsp;|&nbsp; Ensemble Churn Probability: {ensemble:.1%}
            </div>""", unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=ensemble * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": color},
                       "steps": [{"range": [0, 25], "color": "#D5F5E3"},
                                  {"range": [25, 50], "color": "#FEF9E7"},
                                  {"range": [50, 100], "color": "#FADBD8"}],
                       "threshold": {"line": {"color": "black", "width": 3},
                                     "thickness": 0.8, "value": 50}},
                title={"text": "Churn Probability Gauge"}
            ))
            fig.update_layout(height=280, paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    with sub2:
        st.markdown("#### Model Performance on 20% Hold-Out Test Set")
        import json
        with open("/home/claude/project/models/ml_results.json") as f:
            ml_res = json.load(f)

        models_data = ml_res["churn_models"]
        perf = pd.DataFrame(models_data).T * 100
        perf = perf.round(2).reset_index().rename(columns={"index": "Model"})

        fig = go.Figure()
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        m_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        for i, row_m in perf.iterrows():
            fig.add_trace(go.Bar(
                name=row_m["Model"].replace("_", " ").title(),
                x=m_labels,
                y=[row_m[m] for m in metrics],
                text=[f"{row_m[m]:.1f}%" for m in metrics],
                textposition="outside",
                marker_color=PAL[i]
            ))
        fig.update_layout(barmode="group", title="Model Performance Metrics (%)",
                          height=400, paper_bgcolor="white", plot_bgcolor="white",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🌲 Random Forest — Feature Importance")
            rf_fi = pd.Series(ml_res["rf_feature_importance"]).sort_values(ascending=True)
            fig = px.bar(x=rf_fi.values * 100, y=rf_fi.index, orientation="h",
                         color=rf_fi.values, color_continuous_scale="Blues",
                         labels={"x": "Importance (%)", "y": ""})
            fig.update_layout(height=340, paper_bgcolor="white", plot_bgcolor="white",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("##### ⚡ XGBoost — Feature Importance")
            xgb_fi = pd.Series(ml_res["xgb_feature_importance"]).sort_values(ascending=True)
            fig = px.bar(x=xgb_fi.values * 100, y=xgb_fi.index, orientation="h",
                         color=xgb_fi.values, color_continuous_scale="Reds",
                         labels={"x": "Importance (%)", "y": ""})
            fig.update_layout(height=340, paper_bgcolor="white", plot_bgcolor="white",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""<div style="text-align:center;color:{GREY};font-size:.82rem;padding:6px 0">
    Customer Segmentation & Churn Pattern Analytics · European Banking ·
    Unified Mentor Scholarship Submission · Sanjana Allanki
    </div>""", unsafe_allow_html=True
)
