"""
pages/1_Batch_Underwriting.py
------------------------------
Dynamic Batch Underwriting & Portfolio Intelligence Engine.
Upload CSV or Excel files, run asynchronous batch predictions across all rows,
and dynamically visualize portfolio risk metrics, biometric correlations,
and actuarial exposure with 1-click export.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ehr_utils import (
    generate_portfolio_executive_summary,
    get_active_portfolio,
    render_sidebar_branding,
    render_sidebar_footer,
    score_insurance_dataframe,
    set_active_portfolio,
)


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Batch Underwriting | HealthSure",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

render_sidebar_branding()

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }
    
    .batch-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 58, 138, 0.85));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    
    .banner-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 4px;
    }
    
    .banner-sub {
        font-size: 1rem;
        color: #93c5fd;
    }

    .stat-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    .upload-box {
        background: rgba(15, 23, 42, 0.65);
        border: 2px dashed rgba(56, 189, 248, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# 1. Header Banner
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="batch-banner">
        <div class="banner-title">📁 Batch Underwriting & Dynamic Portfolio Engine</div>
        <div class="banner-sub">
            Upload custom policyholder records (CSV or Excel) to trigger high-throughput ML scoring,
            uncover actuarial risk concentrations, and generate real-time portfolio analytics.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# 2. File Upload & Ingestion Controls
# ---------------------------------------------------------------------------

upload_c1, upload_c2 = st.columns([2.5, 1.2])

with upload_c1:
    uploaded_file = st.file_uploader(
        "Upload Policyholder Dataset (CSV or Excel):",
        type=["csv", "xlsx", "xls"],
        help="Upload files containing age, weight, height, income_lpa, smoker, city, occupation."
    )

with upload_c2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    load_sample = st.button("⚡ Load 1,000 Sample Policies", use_container_width=True, help="Load benchmark sample dataset for testing")
    reset_btn = st.button("🔄 Reset to Default Portfolio", use_container_width=True)


# Process Upload or Sample Action
if reset_btn:
    from ehr_utils import reset_to_default_portfolio
    reset_to_default_portfolio()
    st.success("Reset back to benchmark Indian Insurance portfolio!")
    st.rerun()

if load_sample:
    with st.spinner("Loading and scoring 1,000 sample policies..."):
        try:
            df_sample = pd.read_csv("Indian_Insurance_Data.csv").sample(n=1000, random_state=42)
            scored_sample = score_insurance_dataframe(df_sample)
            set_active_portfolio(scored_sample, "Sample Portfolio (1,000 Random Lives)")
            st.success("Successfully ingested and scored 1,000 sample policies!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading sample dataset: {e}")

if uploaded_file is not None:
    # Check if this file was already processed
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("last_uploaded_file_key") != file_key:
        progress_bar = st.progress(0, text="Reading uploaded file...")
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            progress_bar.progress(30, text=f"Parsing {len(df_raw):,} records & normalizing features...")
            time.sleep(0.1)

            progress_bar.progress(65, text="Running machine learning batch inference across all rows...")
            t0 = time.time()
            df_scored = score_insurance_dataframe(df_raw)
            duration = round(time.time() - t0, 3)

            progress_bar.progress(100, text=f"Scoring complete! {len(df_scored):,} rows processed in {duration}s.")
            time.sleep(0.3)
            progress_bar.empty()

            set_active_portfolio(df_scored, f"Upload: {uploaded_file.name} ({len(df_scored):,} Records)")
            st.session_state["last_uploaded_file_key"] = file_key
            st.success(f"✓ Successfully processed {len(df_scored):,} records in {duration} seconds!")
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")


# ---------------------------------------------------------------------------
# 3. Active Portfolio Dynamic Dashboard
# ---------------------------------------------------------------------------

active_df, active_name = get_active_portfolio()

st.markdown("---")
st.subheader(f"📊 Live Portfolio Analytics: `{active_name}`")

total_records = len(active_df)
risk_counts = active_df["predicted_insurance_premium_category"].value_counts().to_dict()
low_count = risk_counts.get("Low", 0)
med_count = risk_counts.get("Medium", 0)
high_count = risk_counts.get("High", 0)
smoker_pct = (active_df["smoker"] == 1).mean() * 100
avg_age = active_df["age"].mean()
avg_bmi = active_df["bmi"].mean()
avg_income = active_df["income_lpa"].mean()

# Key Metric KPI Tiles
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Audited Lives", f"{total_records:,}", "Active Cohort")
k2.metric("Low Risk Tiers", f"{low_count:,}", f"{(low_count/total_records*100):.1f}%")
k3.metric("Medium Risk", f"{med_count:,}", f"{(med_count/total_records*100):.1f}%")
k4.metric("High Risk Loading", f"{high_count:,}", f"{(high_count/total_records*100):.1f}%", delta_color="inverse")
k5.metric("Smoker Rate", f"{smoker_pct:.1f}%", "Tobacco Exposure")
k6.metric("Mean BMI", f"{avg_bmi:.1f}", "kg/m²")

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Interactive Plotly Charts (100% Dynamic)
# ---------------------------------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

# Chart 1: Donut Chart of Risk Distribution
with chart_col1:
    risk_color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    df_dist = pd.DataFrame({
        "Risk Category": ["Low", "Medium", "High"],
        "Count": [low_count, med_count, high_count]
    })
    fig_donut = px.pie(
        df_dist,
        names="Risk Category",
        values="Count",
        color="Risk Category",
        color_discrete_map=risk_color_map,
        hole=0.55,
        title="<b>Actuarial Premium Category Distribution</b>"
    )
    fig_donut.update_traces(textinfo="percent+label", pull=[0.02, 0.02, 0.08])
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Chart 2: Age vs. BMI Scatter colored by Risk
with chart_col2:
    fig_scatter = px.scatter(
        active_df,
        x="age",
        y="bmi",
        color="predicted_insurance_premium_category",
        color_discrete_map=risk_color_map,
        hover_data=["city", "occupation", "income_lpa", "health_risk_index"],
        title="<b>Biometric Risk Landscape: Age vs. BMI</b>",
        labels={"age": "Age (Years)", "bmi": "BMI (kg/m²)", "predicted_insurance_premium_category": "Premium Tier"}
    )
    # Add normal BMI bounds
    fig_scatter.add_hrect(y0=18.5, y1=24.9, fillcolor="#10b981", opacity=0.1, line_width=0, annotation_text="Normal BMI Range (18.5 - 24.9)")
    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

# Chart 3: Smoker Status vs. Premium Tier Breakdown
with row2_col1:
    smoker_cross = pd.crosstab(
        active_df["smoker"].map({0: "Non-Smoker", 1: "Smoker"}),
        active_df["predicted_insurance_premium_category"],
        normalize="index"
    ) * 100
    smoker_cross_df = smoker_cross.reset_index().melt(id_vars="smoker", var_name="Premium Tier", value_name="Percentage")
    
    fig_bar = px.bar(
        smoker_cross_df,
        x="smoker",
        y="Percentage",
        color="Premium Tier",
        color_discrete_map=risk_color_map,
        barmode="group",
        title="<b>Tobacco Impact: Risk Proportion by Smoker Status</b>",
        labels={"smoker": "Applicant Tobacco Status", "Percentage": "Cohort Proportion (%)"}
    )
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Chart 4: Geographic City Tier Breakdown
with row2_col2:
    tier_counts = active_df.groupby(["city_tier", "predicted_insurance_premium_category"]).size().reset_index(name="Applicants")
    tier_counts["City Tier"] = tier_counts["city_tier"].map({1: "Tier 1 (Metro)", 2: "Tier 2 (Emerging)", 3: "Tier 3 (Regional)"})
    
    fig_geo = px.bar(
        tier_counts,
        x="City Tier",
        y="Applicants",
        color="predicted_insurance_premium_category",
        color_discrete_map=risk_color_map,
        title="<b>Geographic Exposure by Urban City Tier</b>"
    )
    fig_geo.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_geo, use_container_width=True)


# ---------------------------------------------------------------------------
# 5. Gemini AI Executive Portfolio Report
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("🤖 Gemini Actuarial Intelligence: Executive Portfolio Assessment")
st.markdown("Generate an automated executive briefing summarizing portfolio risk concentrations and underwriting policy recommendations.")

if st.button("✨ Generate AI Underwriting Portfolio Brief", use_container_width=True):
    with st.spinner("Analyzing portfolio demographics & querying Gemini Actuarial Model..."):
        ai_summary = generate_portfolio_executive_summary(active_df)
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; border-radius: 10px; padding: 20px 24px; margin: 14px 0;">
                {ai_summary}
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# 6. Filterable Scored Records Table & Data Export
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📋 Policyholder Registry & Scored Dataset Explorer")

filter_c1, filter_c2, filter_c3 = st.columns(3)
with filter_c1:
    filter_tier = st.multiselect(
        "Filter by Predicted Tier:",
        options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )
with filter_c2:
    filter_smoker = st.selectbox(
        "Filter by Smoker Status:",
        options=["All Applicants", "Smokers Only", "Non-Smokers Only"]
    )
with filter_c3:
    search_city = st.text_input("Search by City Name:", placeholder="e.g. Mumbai, Delhi, Lucknow")

df_filtered = active_df[active_df["predicted_insurance_premium_category"].isin(filter_tier)]
if filter_smoker == "Smokers Only":
    df_filtered = df_filtered[df_filtered["smoker"] == 1]
elif filter_smoker == "Non-Smokers Only":
    df_filtered = df_filtered[df_filtered["smoker"] == 0]

if search_city:
    df_filtered = df_filtered[df_filtered["city"].astype(str).str.contains(search_city, case=False, na=False)]

st.markdown(f"Displaying **{len(df_filtered):,}** of **{len(active_df):,}** policyholders:")

# Clean display columns
display_cols = [
    "age", "city", "occupation", "income_lpa", "smoker", "bmi",
    "health_risk_index", "predicted_insurance_premium_category"
]
avail_disp = [c for c in display_cols if c in df_filtered.columns]

st.dataframe(
    df_filtered[avail_disp].head(250),
    use_container_width=True,
    height=380
)

# Export Data Buttons
exp_c1, exp_c2, _ = st.columns([1.5, 1.5, 3])

with exp_c1:
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered (CSV)",
        data=csv_bytes,
        file_name="healthsure_scored_portfolio.csv",
        mime="text/csv",
        use_container_width=True
    )

with exp_c2:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_filtered.to_excel(writer, index=False, sheet_name="Scored_Policies")
    st.download_button(
        label="📥 Download Filtered (Excel)",
        data=excel_buffer.getvalue(),
        file_name="healthsure_scored_portfolio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

render_sidebar_footer()
