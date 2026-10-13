"""
pages/Comorbidity_Analysis.py
-----------------------------
Charlson Comorbidity Index (CCI) and Disease Association Analysis.
Analyzes multi-condition clustering, disease prevalence, co-occurrence lift,
demographic trends, and healthcare utilization risk tiers.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ehr_utils import load_ehr


# ---------------------------------------------------------------------------
# Page Configuration & Styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Comorbidity Analysis | HealthSure EHR",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    
    .info-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-size: 0.95rem;
        line-height: 1.55;
    }
    
    .badge-risk-low {
        color: #34d399;
        font-weight: 700;
    }
    .badge-risk-med {
        color: #fbbf24;
        font-weight: 700;
    }
    .badge-risk-high {
        color: #f87171;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Data Loading & Risk Calculation Engine
# ---------------------------------------------------------------------------

@st.cache_data
def compute_patient_comorbidities() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute patient-level Charlson Comorbidity Index scores and disease co-occurrence matrices.
    
    Calculation:
    1. Join conditions with charlson_mapping on icd10.
    2. Sum weights over DISTINCT icd10 codes per patient (no double-counting).
    3. Categorize risk: 0 = Low, 1-2 = Medium, 3+ = High.
    4. Persist patient_risk_table.csv for reuse by models and companion pages.
    """
    cleaned_dfs, _ = load_ehr()
    patients = cleaned_dfs["patients"]
    conditions = cleaned_dfs["conditions"]
    charlson = cleaned_dfs["charlson"]
    encounters = cleaned_dfs["encounters"]

    # 1. Distinct ICD-10 conditions per patient joined with weights
    dist_conds = conditions.drop_duplicates(subset=["patient_id", "icd10"]).copy()
    if not charlson.empty:
        dist_conds = dist_conds.merge(
            charlson[["icd10", "charlson_weight"]],
            on="icd10",
            how="left"
        )
        dist_conds["charlson_weight"] = dist_conds["charlson_weight"].fillna(0).astype(int)
    else:
        dist_conds["charlson_weight"] = 0

    # 2. Patient-level score aggregation
    p_agg = dist_conds.groupby("patient_id").agg(
        charlson_score=("charlson_weight", "sum"),
        condition_count=("icd10", "count")
    ).reset_index()

    # 3. Merge with all patients (patients with no conditions get 0)
    risk_table = patients[["patient_id", "name", "age", "gender", "age_group", "city"]].merge(
        p_agg,
        on="patient_id",
        how="left"
    )
    risk_table["charlson_score"] = risk_table["charlson_score"].fillna(0).astype(int)
    risk_table["condition_count"] = risk_table["condition_count"].fillna(0).astype(int)

    # 4. Assign risk group
    def _assign_risk(score: int) -> str:
        if score == 0:
            return "Low"
        elif score in (1, 2):
            return "Medium"
        else:
            return "High"

    risk_table["risk_group"] = risk_table["charlson_score"].apply(_assign_risk)

    # 5. Persist patient-level table to disk for model & page reuse
    save_cols = ["patient_id", "age", "gender", "condition_count", "charlson_score", "risk_group"]
    try:
        clean_dir = Path("clean")
        clean_dir.mkdir(parents=True, exist_ok=True)
        risk_table[save_cols].to_csv(clean_dir / "patient_risk_table.csv", index=False)
    except Exception:
        pass  # Non-fatal if filesystem is read-only in certain deployment targets

    # Store in session state for fast runtime inter-page access
    st.session_state["patient_risk_table"] = risk_table[save_cols]

    return risk_table, dist_conds, encounters, charlson


try:
    patient_risk_df, distinct_conditions_df, encounters_df, charlson_mapping_df = compute_patient_comorbidities()
except Exception as e:
    st.error(f"Failed to load comorbidity data: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar Filters for Patient Drilldown
# ---------------------------------------------------------------------------

st.sidebar.header("🔍 Cohort Filters")

all_age_groups = sorted([str(x) for x in patient_risk_df["age_group"].dropna().unique()])
all_genders = sorted([str(x) for x in patient_risk_df["gender"].dropna().unique()])
all_risk_tiers = ["Low", "Medium", "High"]

selected_age_groups = st.sidebar.multiselect(
    "Filter by Age Group:",
    options=all_age_groups,
    default=all_age_groups
)

selected_genders = st.sidebar.multiselect(
    "Filter by Gender:",
    options=all_genders,
    default=all_genders
)

selected_risks = st.sidebar.multiselect(
    "Filter by Risk Group:",
    options=all_risk_tiers,
    default=all_risk_tiers
)

# Filtered cohort view
filtered_patients = patient_risk_df[
    (patient_risk_df["age_group"].isin(selected_age_groups)) &
    (patient_risk_df["gender"].isin(selected_genders)) &
    (patient_risk_df["risk_group"].isin(selected_risks))
]


# ---------------------------------------------------------------------------
# Page Header & Clinical Note
# ---------------------------------------------------------------------------

st.markdown("## 🧬 Comorbidity & Charlson Risk Stratification")

st.markdown(
    """
    <div class="info-card">
        <b>About the Charlson Comorbidity Index (CCI):</b><br>
        The Charlson Comorbidity Index is a clinically validated prognostic scoring system predicting 
        mortality and healthcare utilization based on comorbid disease severity. In this system, 
        scoring is evaluated across <b>9 primary chronic conditions</b>:
        <ul style="margin-top: 6px; margin-bottom: 4px;">
            <li><b>Weight 2:</b> Chronic Kidney Disease (N18)</li>
            <li><b>Weight 1:</b> Type 2 Diabetes (E11), Coronary Artery Disease (I25), Asthma (J45), Pneumonia (J18)</li>
            <li><b>Weight 0:</b> Hypertension (I10), Obesity (E66), Anemia (D64), Hypothyroidism (E03)</li>
        </ul>
        <i>Patients are stratified into <b>Low (Score 0)</b>, <b>Medium (Scores 1–2)</b>, and <b>High (Scores 3+)</b> clinical risk tiers.</i>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Top KPI Cards
# ---------------------------------------------------------------------------

total_pts = len(patient_risk_df)
mean_charlson = patient_risk_df["charlson_score"].mean()
high_risk_pct = (patient_risk_df["risk_group"] == "High").mean() * 100

# Most common condition
if not distinct_conditions_df.empty:
    top_cond_name = distinct_conditions_df["condition"].value_counts().index[0]
    top_cond_count = distinct_conditions_df["condition"].value_counts().iloc[0]
    top_cond_pct = (top_cond_count / total_pts) * 100
    top_cond_label = f"{top_cond_name} ({top_cond_pct:.1f}%)"
else:
    top_cond_label = "None"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Patients", f"{total_pts:,}")
kpi2.metric("Mean Charlson Score", f"{mean_charlson:.2f}")
kpi3.metric("High-Risk Tier (3+)", f"{high_risk_pct:.1f}%")
kpi4.metric("Most Prevalent Condition", top_cond_label)

st.markdown("---")


# ---------------------------------------------------------------------------
# Chart 1: Charlson Score Distribution
# ---------------------------------------------------------------------------

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.subheader("1. Charlson Score Distribution")
    score_counts = patient_risk_df.groupby(["charlson_score", "risk_group"]).size().reset_index(name="patient_count")
    
    color_discrete_map = {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    }
    
    fig_dist = px.bar(
        score_counts,
        x="charlson_score",
        y="patient_count",
        color="risk_group",
        color_discrete_map=color_discrete_map,
        labels={"charlson_score": "Charlson Comorbidity Score", "patient_count": "Patient Count", "risk_group": "Risk Tier"},
        title="Patient Volume by Charlson Score Tier",
        template="plotly_dark",
        text="patient_count"
    )
    fig_dist.update_traces(textposition="outside")
    fig_dist.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------------------------------------------------------------
# Chart 2: Prevalence Bar Chart
# ---------------------------------------------------------------------------

with col_c2:
    st.subheader("2. Chronic Condition Prevalence")
    if not distinct_conditions_df.empty:
        prev_df = distinct_conditions_df["condition"].value_counts().reset_index()
        prev_df.columns = ["Condition", "Patients"]
        prev_df["Prevalence (%)"] = (prev_df["Patients"] / total_pts * 100).round(1)
        prev_df = prev_df.sort_values(by="Prevalence (%)", ascending=True)

        fig_prev = px.bar(
            prev_df,
            x="Prevalence (%)",
            y="Condition",
            orientation="h",
            color="Prevalence (%)",
            color_continuous_scale="Teal",
            text="Prevalence (%)",
            title="Population Disease Prevalence (% of Total Patients)",
            template="plotly_dark"
        )
        fig_prev.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_prev.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_prev, use_container_width=True)
    else:
        st.info("No condition data available for prevalence analysis.")


# ---------------------------------------------------------------------------
# Chart 3: Comorbidity Co-occurrence Heatmap & Table (Lift)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("3. Disease Co-occurrence & Associative Lift Matrix")
st.caption("Analyzes pairwise condition clustering to identify syndromic disease associations across the cohort.")

# Prepare Crosstab Matrix (Patients x Conditions)
pt_cond_matrix = pd.crosstab(
    distinct_conditions_df["patient_id"],
    distinct_conditions_df["condition"]
)
all_patient_ids = patient_risk_df["patient_id"].unique()
pt_cond_matrix = pt_cond_matrix.reindex(all_patient_ids, fill_value=0)

condition_names = sorted(pt_cond_matrix.columns.tolist())
co_matrix = pt_cond_matrix[condition_names].T.dot(pt_cond_matrix[condition_names])

# Mask diagonal for visualization contrast
matrix_masked = co_matrix.astype(float).copy()
for i in range(len(condition_names)):
    matrix_masked.iloc[i, i] = np.nan

col_hm, col_lift = st.columns([1.1, 1.0])

with col_hm:
    st.markdown("##### Co-occurrence Heatmap (Shared Patient Counts)")
    fig_hm = px.imshow(
        matrix_masked,
        labels=dict(x="Condition B", y="Condition A", color="Shared Patients"),
        x=condition_names,
        y=condition_names,
        color_continuous_scale="Purples",
        template="plotly_dark",
        text_auto=".0f"
    )
    fig_hm.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with col_lift:
    st.markdown("##### Top 10 Disease Pairs by Association (Lift)")
    
    # Calculate Lift: P(A & B) / (P(A) * P(B))
    pairs_list = []
    N = len(all_patient_ids)

    for i in range(len(condition_names)):
        for j in range(i + 1, len(condition_names)):
            cA = condition_names[i]
            cB = condition_names[j]
            shared_count = int(co_matrix.loc[cA, cB])
            p_A = pt_cond_matrix[cA].sum() / N
            p_B = pt_cond_matrix[cB].sum() / N
            p_both = shared_count / N

            lift = p_both / (p_A * p_B) if (p_A * p_B) > 0 else 0.0

            pairs_list.append({
                "Condition A": cA,
                "Condition B": cB,
                "Shared Patients": shared_count,
                "Prev A (%)": f"{p_A * 100:.1f}%",
                "Prev B (%)": f"{p_B * 100:.1f}%",
                "Lift": round(lift, 2)
            })

    lift_table_df = pd.DataFrame(pairs_list)

    st.markdown(
        """
        <div style="background: rgba(14, 165, 233, 0.1); border-left: 3px solid #0ea5e9; padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; margin-bottom: 10px;">
            <b>Clinical Interpretation:</b> A <b>Lift &gt; 1.0</b> indicates that conditions A and B co-occur more frequently than expected by random independent chance, suggesting a shared pathophysiology or underlying clinical pathway.
        </div>
        """,
        unsafe_allow_html=True
    )

    sort_criteria = st.radio(
        "Rank Pairs By:",
        options=["Highest Shared Patients", "Highest Associative Lift"],
        horizontal=True
    )

    if sort_criteria == "Highest Shared Patients":
        top_pairs = lift_table_df.sort_values(by="Shared Patients", ascending=False).head(10)
    else:
        top_pairs = lift_table_df.sort_values(by="Lift", ascending=False).head(10)

    st.dataframe(
        top_pairs,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------------------------
# Chart 4: Average Charlson Score by Age Group and Gender
# ---------------------------------------------------------------------------

st.markdown("---")
col_dem1, col_dem2 = st.columns([1.2, 1.0])

with col_dem1:
    st.subheader("4. Charlson Comorbidity by Demographics")
    
    demo_scores = patient_risk_df.groupby(["age_group", "gender"])["charlson_score"].mean().reset_index()
    demo_scores["charlson_score"] = demo_scores["charlson_score"].round(2)
    
    # Ensure correct chronological age ordering
    age_order = ["0-17", "18-39", "40-59", "60+"]
    demo_scores["age_group"] = pd.Categorical(demo_scores["age_group"], categories=age_order, ordered=True)
    demo_scores = demo_scores.sort_values(by=["age_group", "gender"])

    fig_demo = px.bar(
        demo_scores,
        x="age_group",
        y="charlson_score",
        color="gender",
        barmode="group",
        color_discrete_map={"Male": "#38bdf8", "Female": "#ec4899"},
        labels={"age_group": "Age Group", "charlson_score": "Mean Charlson Score", "gender": "Gender"},
        title="Mean Comorbidity Score across Age Brackets & Gender",
        template="plotly_dark",
        text="charlson_score"
    )
    fig_demo.update_traces(textposition="outside")
    fig_demo.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_demo, use_container_width=True)

with col_dem2:
    st.subheader("Cohort Risk Tier Breakdown")
    tier_counts = patient_risk_df["risk_group"].value_counts().reset_index()
    tier_counts.columns = ["Risk Tier", "Patient Count"]
    tier_counts["Percentage"] = (tier_counts["Patient Count"] / total_pts * 100).round(1)

    fig_pie = px.pie(
        tier_counts,
        names="Risk Tier",
        values="Patient Count",
        color="Risk Tier",
        color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
        hole=0.45,
        template="plotly_dark",
        title="Cohort Distribution by Comorbidity Risk Tier"
    )
    fig_pie.update_traces(textinfo="label+percent", pull=[0, 0, 0.08])
    fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)


# ---------------------------------------------------------------------------
# Chart 5: Healthcare Utilization & Cost by Risk Group
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("5. Clinical Utilization & Financial Impact by Risk Tier")
st.caption("Demonstrating how higher comorbidity index correlates with inpatient length of stay and billed encounter costs.")

if not encounters_df.empty:
    # Aggregate encounters per patient
    enc_patient_stats = encounters_df.groupby("patient_id").agg(
        mean_cost=("cost_inr", "mean"),
        mean_stay=("length_of_stay_days", "mean"),
        total_encounters=("encounter_date", "count")
    ).reset_index()

    util_merged = patient_risk_df.merge(enc_patient_stats, on="patient_id", how="inner")

    # Order risk tiers cleanly
    risk_order = ["Low", "Medium", "High"]
    util_merged["risk_group"] = pd.Categorical(util_merged["risk_group"], categories=risk_order, ordered=True)

    ucol1, ucol2 = st.columns(2)

    with ucol1:
        fig_cost_box = px.box(
            util_merged,
            x="risk_group",
            y="mean_cost",
            color="risk_group",
            color_discrete_map=color_discrete_map,
            points="outliers",
            labels={"risk_group": "Risk Group Tier", "mean_cost": "Mean Encounter Cost (INR)"},
            title="Encounter Billed Cost (₹ INR) Distribution by Risk Tier",
            template="plotly_dark"
        )
        fig_cost_box.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_cost_box, use_container_width=True)

    with ucol2:
        fig_stay_box = px.box(
            util_merged,
            x="risk_group",
            y="mean_stay",
            color="risk_group",
            color_discrete_map=color_discrete_map,
            points="outliers",
            labels={"risk_group": "Risk Group Tier", "mean_stay": "Average Stay (Days)"},
            title="Inpatient Length of Stay (Days) by Risk Tier",
            template="plotly_dark"
        )
        fig_stay_box.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_stay_box, use_container_width=True)
else:
    st.info("No encounter records available for utilization analysis.")


# ---------------------------------------------------------------------------
# Table: Top 20 Highest-Risk Patients
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("6. Top Highest-Risk Patients Drilldown")
st.markdown("Patients with the highest multi-system comorbidity burden. Click on any record to investigate in the **Patient 360** module.")

# Filter cohort according to sidebar selection
top_risk_cohort = filtered_patients.sort_values(
    by=["charlson_score", "condition_count", "age"],
    ascending=[False, False, False]
).head(20).copy()

# Add link indicator for Patient 360 navigation
top_risk_cohort["EHR Action"] = top_risk_cohort["patient_id"].apply(lambda pid: f"🔍 View {pid} in Patient 360")

table_display = top_risk_cohort[[
    "patient_id", "EHR Action", "name", "age", "gender", "city", "condition_count", "charlson_score", "risk_group"
]].rename(columns={
    "patient_id": "Patient ID",
    "name": "Full Name",
    "age": "Age",
    "gender": "Gender",
    "city": "City",
    "condition_count": "Conditions",
    "charlson_score": "Charlson Score",
    "risk_group": "Risk Tier"
})

def color_risk_tier(val):
    if val == "High":
        return "background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; font-weight: bold;"
    elif val == "Medium":
        return "background-color: rgba(245, 158, 11, 0.2); color: #fde68a;"
    return "background-color: rgba(16, 185, 129, 0.2); color: #a7f3d0;"

styler = table_display.style.map(color_risk_tier, subset=["Risk Tier"]) if hasattr(table_display.style, "map") else table_display.style.applymap(color_risk_tier, subset=["Risk Tier"])
st.dataframe(
    styler,
    use_container_width=True,
    hide_index=True
)

st.caption("Showing up to top 20 patients matching active sidebar filters.")

if not top_risk_cohort.empty:
    jcol1, jcol2 = st.columns([2.2, 2.8])
    with jcol1:
        jump_pid = st.selectbox(
            "Select High-Risk Patient for Longitudinal Review:",
            options=top_risk_cohort["patient_id"].tolist(),
            format_func=lambda pid: f"{pid} — {top_risk_cohort[top_risk_cohort['patient_id'] == pid]['name'].iloc[0]} (Score: {top_risk_cohort[top_risk_cohort['patient_id'] == pid]['charlson_score'].iloc[0]})",
            key="sb_jump_comorb"
        )
    with jcol2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🩺 Preselect for Patient 360", key="btn_jump_comorb"):
            st.session_state["selected_patient_id"] = jump_pid
            st.success(f"Patient {jump_pid} preselected! Click 'Patient 360' in the sidebar to view longitudinal record.")



# ---------------------------------------------------------------------------
# Download Patient Risk Table
# ---------------------------------------------------------------------------

st.divider()

dcol1, dcol2 = st.columns([1, 4])

with dcol1:
    # Prepare CSV export
    export_df = patient_risk_df[[
        "patient_id", "name", "age", "gender", "city", "age_group", "condition_count", "charlson_score", "risk_group"
    ]].copy()
    
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    
    st.download_button(
        label="📥 Download Patient Risk Table (CSV)",
        data=csv_bytes,
        file_name="patient_risk_table.csv",
        mime="text/csv",
        use_container_width=True
    )

with dcol2:
    st.caption("Exports the patient-level comorbidity and Charlson stratification table for use in insurance underwriting, machine learning, and downstream actuarial models.")
