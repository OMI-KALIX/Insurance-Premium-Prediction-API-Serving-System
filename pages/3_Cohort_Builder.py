"""
pages/Cohort_Builder.py
-----------------------
Longitudinal Cohort Builder & Statistical Comparator.
Empowers clinicians and researchers to define, analyze, and statistically compare
patient cohorts based on clinical diagnoses, medications, biomarkers, and healthcare utilization.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

from ehr_utils import load_ehr


# ---------------------------------------------------------------------------
# Page Configuration & Styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cohort Builder | HealthSure EHR",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }
    
    .cohort-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .preset-btn {
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Step 1: Patient-Level Master Table Builder (Cached)
# ---------------------------------------------------------------------------

@st.cache_data
def build_master_patient_table() -> pd.DataFrame:
    """
    Build a comprehensive longitudinal patient-level master table.
    
    Columns include:
    - Demographics: patient_id, name, age, gender, city, blood_group, height_cm, weight_kg, bmi, age_group
    - Conditions: condition_count, charlson_score, risk_group, conditions_list, icd10_list
    - Medications: medication_count, medications_list
    - Latest Labs: latest_hba1c, latest_creatinine, latest_cholesterol
    - Latest Vitals: latest_systolic_bp, latest_diastolic_bp, latest_glucose
    - Utilization: num_encounters, num_emergency, has_emergency, total_cost_inr, encounter_types
    """
    cleaned_dfs, _ = load_ehr()
    patients = cleaned_dfs["patients"]
    conditions = cleaned_dfs["conditions"]
    encounters = cleaned_dfs["encounters"]
    labs = cleaned_dfs["labs"]
    medications = cleaned_dfs["medications"]
    charlson = cleaned_dfs["charlson"]

    # 1. Conditions & Charlson Comorbidity Scoring
    dist_conds = conditions.drop_duplicates(subset=["patient_id", "icd10"]).merge(
        charlson[["icd10", "charlson_weight"]],
        on="icd10",
        how="left"
    )
    dist_conds["charlson_weight"] = dist_conds["charlson_weight"].fillna(0)

    cond_agg = dist_conds.groupby("patient_id").agg(
        conditions_list=("condition", lambda s: list(s.dropna().unique())),
        icd10_list=("icd10", lambda s: list(s.dropna().unique())),
        condition_count=("icd10", "count"),
        charlson_score=("charlson_weight", "sum")
    ).reset_index()

    # 2. Medications
    med_agg = medications.groupby("patient_id").agg(
        medications_list=("drug_name", lambda s: list(s.dropna().unique())),
        medication_count=("drug_name", lambda s: len(s.dropna().unique()))
    ).reset_index()

    # 3. Latest Labs per Patient
    labs_sorted = labs.sort_values(by=["patient_id", "lab_date"]).dropna(subset=["patient_id"])
    latest_labs = labs_sorted.groupby("patient_id").last().reset_index()
    latest_labs = latest_labs[["patient_id", "hba1c_pct", "creatinine_mg_dl", "total_cholesterol"]].rename(columns={
        "hba1c_pct": "latest_hba1c",
        "creatinine_mg_dl": "latest_creatinine",
        "total_cholesterol": "latest_cholesterol"
    })

    # 4. Latest Vitals & Encounter Aggregates
    enc_sorted = encounters.sort_values(by=["patient_id", "encounter_date"]).dropna(subset=["patient_id"])
    latest_enc = enc_sorted.groupby("patient_id").last().reset_index()
    latest_enc = latest_enc[["patient_id", "systolic_bp", "diastolic_bp", "glucose"]].rename(columns={
        "systolic_bp": "latest_systolic_bp",
        "diastolic_bp": "latest_diastolic_bp",
        "glucose": "latest_glucose"
    })

    enc_agg = encounters.groupby("patient_id").agg(
        num_encounters=("encounter_date", "count"),
        num_emergency=("encounter_type", lambda s: int((s.astype(str).str.strip().str.lower() == "emergency").sum())),
        total_cost_inr=("cost_inr", "sum"),
        encounter_types=("encounter_type", lambda s: list(s.dropna().unique())),
        min_encounter_date=("encounter_date", "min"),
        max_encounter_date=("encounter_date", "max")
    ).reset_index()

    # 5. Master Merge
    master = patients.copy()
    master = master.merge(cond_agg, on="patient_id", how="left")
    master = master.merge(med_agg, on="patient_id", how="left")
    master = master.merge(latest_labs, on="patient_id", how="left")
    master = master.merge(latest_enc, on="patient_id", how="left")
    master = master.merge(enc_agg, on="patient_id", how="left")

    # Clean list columns & defaults
    master["conditions_list"] = master["conditions_list"].apply(lambda x: x if isinstance(x, list) else [])
    master["icd10_list"] = master["icd10_list"].apply(lambda x: x if isinstance(x, list) else [])
    master["medications_list"] = master["medications_list"].apply(lambda x: x if isinstance(x, list) else [])
    master["encounter_types"] = master["encounter_types"].apply(lambda x: x if isinstance(x, list) else [])

    master["condition_count"] = master["condition_count"].fillna(0).astype(int)
    master["charlson_score"] = master["charlson_score"].fillna(0).astype(int)
    master["medication_count"] = master["medication_count"].fillna(0).astype(int)
    master["num_encounters"] = master["num_encounters"].fillna(0).astype(int)
    master["num_emergency"] = master["num_emergency"].fillna(0).astype(int)
    master["has_emergency"] = master["num_emergency"] > 0
    master["total_cost_inr"] = master["total_cost_inr"].fillna(0.0).round(2)

    def _risk(score: int) -> str:
        if score == 0:
            return "Low"
        elif score in (1, 2):
            return "Medium"
        return "High"

    master["risk_group"] = master["charlson_score"].apply(_risk)

    return master


# Load Master Table
try:
    master_df = build_master_patient_table()
    total_population = len(master_df)
except Exception as e:
    st.error(f"Error building longitudinal patient master table: {e}")
    st.stop()


# Extract Domain Value Lists for Filters
ALL_CONDITIONS = sorted(list({c for sub in master_df["conditions_list"] for c in sub}))
ALL_MEDICATIONS = sorted(list({m for sub in master_df["medications_list"] for m in sub}))
ALL_CITIES = sorted(master_df["city"].dropna().unique().tolist())
ALL_BLOOD_GROUPS = sorted(master_df["blood_group"].dropna().unique().tolist())
ALL_GENDERS = sorted(master_df["gender"].dropna().unique().tolist())
ALL_RISK_GROUPS = ["Low", "Medium", "High"]
ALL_ENC_TYPES = sorted(list({t for sub in master_df["encounter_types"] for t in sub}))


# ---------------------------------------------------------------------------
# Presets Management (Diabetics vs Non-Diabetics, Metformin vs Not, Age 60+)
# ---------------------------------------------------------------------------

st.sidebar.markdown("### ⚡ Cohort Presets")
pcol1, pcol2, pcol3 = st.sidebar.columns(3)

if pcol1.button("🩺 Diabetics vs Non", help="Compare patients with Type 2 Diabetes vs patients without"):
    st.session_state["preset_applied"] = "diabetics"

if pcol2.button("💊 Metformin vs Not", help="Compare patients prescribed Metformin vs not on Metformin"):
    st.session_state["preset_applied"] = "metformin"

if pcol3.button("👴 Age 60+ vs <60", help="Compare elderly population (60+) vs younger cohort (<60)"):
    st.session_state["preset_applied"] = "age60"


# ---------------------------------------------------------------------------
# Step 2: Reusable Sidebar Filter Panel
# ---------------------------------------------------------------------------

def render_cohort_filter_panel(
    prefix: str,
    default_name: str,
    preset: Optional[str] = None
) -> Tuple[str, pd.DataFrame]:
    """
    Render a comprehensive cohort filter panel and apply selections to master_df.
    Returns: (cohort_name, filtered_cohort_dataframe)
    """
    st.markdown(f"#### ⚙️ Define {default_name}")
    cohort_name = st.text_input(f"Cohort Name:", value=default_name, key=f"{prefix}_name")

    # Handle Presets
    default_age = (0, 100)
    default_conds: List[str] = []
    default_excl_conds: List[str] = []
    default_meds: List[str] = []

    if preset == "diabetics":
        if prefix == "A":
            default_conds = ["Type 2 Diabetes"] if "Type 2 Diabetes" in ALL_CONDITIONS else []
        elif prefix == "B":
            default_excl_conds = ["Type 2 Diabetes"] if "Type 2 Diabetes" in ALL_CONDITIONS else []
    elif preset == "metformin":
        if prefix == "A":
            default_meds = ["Metformin"] if "Metformin" in ALL_MEDICATIONS else []
        elif prefix == "B":
            default_meds = []
    elif preset == "age60":
        if prefix == "A":
            default_age = (60, 100)
        elif prefix == "B":
            default_age = (0, 59)

    # 1. Demographics
    age_range = st.slider(
        f"Age Range:",
        min_value=0,
        max_value=100,
        value=default_age,
        key=f"{prefix}_age"
    )

    sel_genders = st.multiselect(
        f"Gender:",
        options=ALL_GENDERS,
        default=ALL_GENDERS,
        key=f"{prefix}_gender"
    )

    sel_cities = st.multiselect(
        f"City:",
        options=ALL_CITIES,
        default=ALL_CITIES,
        key=f"{prefix}_city"
    )

    sel_blood_groups = st.multiselect(
        f"Blood Group:",
        options=ALL_BLOOD_GROUPS,
        default=ALL_BLOOD_GROUPS,
        key=f"{prefix}_bg"
    )

    # 2. Diagnoses & Conditions
    sel_conditions = st.multiselect(
        f"Include Chronic Conditions:",
        options=ALL_CONDITIONS,
        default=default_conds,
        key=f"{prefix}_conds"
    )

    cond_logic = st.radio(
        f"Condition Logic:",
        options=["OR (At least one)", "AND (All selected)"],
        index=0,
        horizontal=True,
        key=f"{prefix}_logic"
    )

    excl_conditions = st.multiselect(
        f"Exclude Conditions (Must NOT have):",
        options=ALL_CONDITIONS,
        default=default_excl_conds,
        key=f"{prefix}_excl_conds"
    )

    # 3. Medications
    sel_medications = st.multiselect(
        f"Prescribed Medications:",
        options=ALL_MEDICATIONS,
        default=default_meds,
        key=f"{prefix}_meds"
    )

    # 4. Clinical Biomarker Ranges
    with st.expander(f"Biomarkers & Vitals Limits", expanded=False):
        hba1c_range = st.slider(
            f"Latest HbA1c (%):",
            min_value=3.0,
            max_value=15.0,
            value=(3.0, 15.0),
            step=0.1,
            key=f"{prefix}_hba1c"
        )

        creat_range = st.slider(
            f"Latest Creatinine (mg/dL):",
            min_value=0.2,
            max_value=10.0,
            value=(0.2, 10.0),
            step=0.1,
            key=f"{prefix}_creat"
        )

        sbp_range = st.slider(
            f"Latest Systolic BP (mmHg):",
            min_value=60,
            max_value=220,
            value=(60, 220),
            key=f"{prefix}_sbp"
        )

        bmi_range = st.slider(
            f"BMI (kg/m²):",
            min_value=10.0,
            max_value=55.0,
            value=(10.0, 55.0),
            step=0.5,
            key=f"{prefix}_bmi"
        )

    # 5. Charlson Risk Group
    sel_risk = st.multiselect(
        f"Charlson Risk Group:",
        options=ALL_RISK_GROUPS,
        default=ALL_RISK_GROUPS,
        key=f"{prefix}_risk"
    )

    # 6. Utilization & Encounters
    sel_enc_types = st.multiselect(
        f"Encounter Type (Has at least one):",
        options=ALL_ENC_TYPES,
        default=ALL_ENC_TYPES,
        key=f"{prefix}_enc_types"
    )

    # --- Filtering Logic ---
    df = master_df.copy()

    # Age
    df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

    # Gender
    if sel_genders:
        df = df[df["gender"].isin(sel_genders)]
    else:
        df = df.iloc[0:0]

    # City
    if sel_cities:
        df = df[df["city"].isin(sel_cities)]
    else:
        df = df.iloc[0:0]

    # Blood group
    if sel_blood_groups:
        df = df[df["blood_group"].isin(sel_blood_groups)]
    else:
        df = df.iloc[0:0]

    # Conditions Inclusion
    if sel_conditions:
        if "AND" in cond_logic:
            df = df[df["conditions_list"].apply(lambda clist: all(c in clist for c in sel_conditions))]
        else:
            df = df[df["conditions_list"].apply(lambda clist: any(c in clist for c in sel_conditions))]

    # Conditions Exclusion
    if excl_conditions:
        df = df[df["conditions_list"].apply(lambda clist: not any(c in clist for c in excl_conditions))]

    # Medications
    if sel_medications:
        if preset == "metformin" and prefix == "B":
            df = df[df["medications_list"].apply(lambda mlist: not any(m in mlist for m in ["Metformin"]))]
        else:
            df = df[df["medications_list"].apply(lambda mlist: any(m in mlist for m in sel_medications))]

    # Biomarkers: if bounds restricted from full default, filter patients
    if hba1c_range != (3.0, 15.0):
        df = df[df["latest_hba1c"].notna() & (df["latest_hba1c"] >= hba1c_range[0]) & (df["latest_hba1c"] <= hba1c_range[1])]

    if creat_range != (0.2, 10.0):
        df = df[df["latest_creatinine"].notna() & (df["latest_creatinine"] >= creat_range[0]) & (df["latest_creatinine"] <= creat_range[1])]

    if sbp_range != (60, 220):
        df = df[df["latest_systolic_bp"].notna() & (df["latest_systolic_bp"] >= sbp_range[0]) & (df["latest_systolic_bp"] <= sbp_range[1])]

    if bmi_range != (10.0, 55.0):
        df = df[df["bmi"].notna() & (df["bmi"] >= bmi_range[0]) & (df["bmi"] <= bmi_range[1])]

    # Risk group
    if sel_risk:
        df = df[df["risk_group"].isin(sel_risk)]
    else:
        df = df.iloc[0:0]

    # Encounter types
    if sel_enc_types and len(sel_enc_types) < len(ALL_ENC_TYPES):
        df = df[df["encounter_types"].apply(lambda etypes: any(t in etypes for t in sel_enc_types))]

    # Live size metric
    cohort_size = len(df)
    cohort_pct = (cohort_size / max(1, total_population)) * 100.0
    st.info(f"**Cohort Size:** {cohort_size:,} patients ({cohort_pct:.1f}% of population)")

    return cohort_name, df


# Render Sidebar Tabs for Cohort A and Cohort B
preset_current = st.session_state.get("preset_applied", None)

sidebar_tab_a, sidebar_tab_b = st.sidebar.tabs(["🔵 Cohort A", "🟠 Cohort B"])

with sidebar_tab_a:
    cohort_a_name, cohort_a_df = render_cohort_filter_panel("A", "Cohort A", preset_current)

with sidebar_tab_b:
    cohort_b_name, cohort_b_df = render_cohort_filter_panel("B", "Cohort B", preset_current)


# ---------------------------------------------------------------------------
# Step 3: Single Cohort Profile Render Function
# ---------------------------------------------------------------------------

def render_cohort_profile(c_name: str, c_df: pd.DataFrame, theme_color: str = "#38bdf8"):
    """Render comprehensive clinical KPI summary and charts for a single patient cohort."""
    if c_df.empty:
        st.warning(f"No patients match the defined criteria for **{c_name}**. Please broaden your filter parameters.")
        return

    n_patients = len(c_df)
    pct_pop = (n_patients / total_population) * 100.0
    mean_age = float(c_df["age"].mean())
    pct_female = float((c_df["gender"].astype(str).str.lower() == "female").mean() * 100.0)
    mean_bmi = float(c_df["bmi"].dropna().mean())
    mean_sbp = float(c_df["latest_systolic_bp"].dropna().mean()) if c_df["latest_systolic_bp"].notna().any() else 0.0
    mean_hba1c = float(c_df["latest_hba1c"].dropna().mean()) if c_df["latest_hba1c"].notna().any() else 0.0
    mean_cost = float(c_df["total_cost_inr"].mean())
    er_rate = float(c_df["has_emergency"].mean() * 100.0)

    # Top KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cohort Population", f"{n_patients:,}", f"{pct_pop:.1f}% of total")
    k2.metric("Mean Age", f"{mean_age:.1f} yrs")
    k3.metric("Female Proportion", f"{pct_female:.1f}%")
    k4.metric("Mean BMI", f"{mean_bmi:.1f} kg/m²")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Mean Systolic BP", f"{mean_sbp:.0f} mmHg" if mean_sbp > 0 else "N/A")
    k6.metric("Mean HbA1c", f"{mean_hba1c:.1f}%" if mean_hba1c > 0 else "N/A")
    k7.metric("Mean Healthcare Cost", f"₹{mean_cost:,.2f}")
    k8.metric("ER Visit Rate", f"{er_rate:.1f}%")

    st.markdown("---")

    # Visualizations
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        # 1. Age Histogram
        fig_age = px.histogram(
            c_df,
            x="age",
            nbins=16,
            color_discrete_sequence=[theme_color],
            title=f"Age Distribution ({c_name})",
            template="plotly_dark",
            labels={"age": "Age (years)", "count": "Patients"}
        )
        fig_age.update_layout(height=300, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_age, use_container_width=True)

    with ch_col2:
        # 2. Gender Pie Chart
        g_counts = c_df["gender"].value_counts().reset_index()
        g_counts.columns = ["Gender", "Count"]
        fig_g = px.pie(
            g_counts,
            names="Gender",
            values="Count",
            color="Gender",
            color_discrete_map={"Male": "#38bdf8", "Female": "#ec4899"},
            hole=0.45,
            title="Gender Ratio",
            template="plotly_dark"
        )
        fig_g.update_layout(height=300, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    ch_col3, ch_col4 = st.columns(2)

    with ch_col3:
        # 3. Top 10 Conditions
        cond_flat = [c for sub in c_df["conditions_list"] for c in sub]
        if cond_flat:
            top_c = pd.Series(cond_flat).value_counts().head(10).reset_index()
            top_c.columns = ["Condition", "Patient Count"]
            fig_top_c = px.bar(
                top_c.sort_values(by="Patient Count", ascending=True),
                x="Patient Count",
                y="Condition",
                orientation="h",
                color="Patient Count",
                color_continuous_scale="Teal",
                title="Top 10 Diagnoses in Cohort",
                template="plotly_dark"
            )
            fig_top_c.update_layout(height=330, margin=dict(l=20, r=20, t=35, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_top_c, use_container_width=True)
        else:
            st.info("No recorded diagnoses in this cohort.")

    with ch_col4:
        # 4. Top 10 Medications
        med_flat = [m for sub in c_df["medications_list"] for m in sub]
        if med_flat:
            top_m = pd.Series(med_flat).value_counts().head(10).reset_index()
            top_m.columns = ["Medication", "Prescription Count"]
            fig_top_m = px.bar(
                top_m.sort_values(by="Prescription Count", ascending=True),
                x="Prescription Count",
                y="Medication",
                orientation="h",
                color="Prescription Count",
                color_continuous_scale="Purples",
                title="Top 10 Medications in Cohort",
                template="plotly_dark"
            )
            fig_top_m.update_layout(height=330, margin=dict(l=20, r=20, t=35, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_top_m, use_container_width=True)
        else:
            st.info("No active medications prescribed in this cohort.")

    ch_col5, ch_col6 = st.columns(2)

    with ch_col5:
        # 5. Average Vitals / Labs
        v_metrics = [
            ("Systolic BP (mmHg)", mean_sbp),
            ("Diastolic BP (mmHg)", float(c_df["latest_diastolic_bp"].dropna().mean()) if c_df["latest_diastolic_bp"].notna().any() else 0.0),
            ("Glucose (mg/dL)", float(c_df["latest_glucose"].dropna().mean()) if c_df["latest_glucose"].notna().any() else 0.0),
            ("HbA1c (%)", mean_hba1c),
            ("Creatinine (mg/dL)", float(c_df["latest_creatinine"].dropna().mean()) if c_df["latest_creatinine"].notna().any() else 0.0),
            ("Cholesterol (mg/dL)", float(c_df["latest_cholesterol"].dropna().mean()) if c_df["latest_cholesterol"].notna().any() else 0.0)
        ]
        v_df = pd.DataFrame(v_metrics, columns=["Biomarker", "Average Value"])
        fig_v = px.bar(
            v_df,
            x="Biomarker",
            y="Average Value",
            color="Biomarker",
            text="Average Value",
            title="Biomarker & Vital Baseline Averages",
            template="plotly_dark"
        )
        fig_v.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_v.update_layout(height=330, margin=dict(l=20, r=20, t=35, b=20), showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_v, use_container_width=True)

    with ch_col6:
        # 6. Encounter Type Distribution
        enc_types_flat = [t for sub in c_df["encounter_types"] for t in sub]
        if enc_types_flat:
            enc_t_counts = pd.Series(enc_types_flat).value_counts().reset_index()
            enc_t_counts.columns = ["Encounter Type", "Count"]
            fig_enc = px.bar(
                enc_t_counts,
                x="Encounter Type",
                y="Count",
                color="Encounter Type",
                text="Count",
                title="Encounter Types Handled in Cohort",
                template="plotly_dark"
            )
            fig_enc.update_traces(textposition="outside")
            fig_enc.update_layout(height=330, margin=dict(l=20, r=20, t=35, b=20), showlegend=False)
            st.plotly_chart(fig_enc, use_container_width=True)
        else:
            st.info("No encounter records associated with this cohort.")

    # Table of Cohort with CSV Export
    st.markdown("##### Detailed Cohort Registry")
    display_cols = [
        "patient_id", "name", "age", "gender", "city", "blood_group", "bmi",
        "condition_count", "charlson_score", "risk_group", "latest_hba1c", "latest_systolic_bp", "total_cost_inr"
    ]
    st.dataframe(c_df[display_cols], use_container_width=True, hide_index=True)

    c_dl1, c_dl2 = st.columns([1.5, 2.5])
    with c_dl1:
        csv_bytes = c_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {c_name} Patients CSV",
            data=csv_bytes,
            file_name=f"{c_name.lower().replace(' ', '_')}_patients.csv",
            mime="text/csv",
            key=f"dl_csv_{c_name}"
        )
    with c_dl2:
        if not c_df.empty:
            c_pids = c_df["patient_id"].tolist()
            sel_drill_pid = st.selectbox(
                f"Select patient from {c_name} to review in Patient 360:",
                options=c_pids,
                format_func=lambda pid: f"{pid} — {c_df[c_df['patient_id'] == pid]['name'].iloc[0]}",
                key=f"sb_drill_{c_name}"
            )
            if st.button(f"🩺 Preselect {sel_drill_pid} for Patient 360", key=f"btn_drill_{c_name}"):
                st.session_state["selected_patient_id"] = sel_drill_pid
                st.success(f"Patient {sel_drill_pid} preselected! Navigate to Patient 360 to inspect longitudinal record.")



# ---------------------------------------------------------------------------
# Step 4: Cohort Comparison Engine & Statistical Hypothesis Tests
# ---------------------------------------------------------------------------

def render_cohort_comparison(
    name_a: str, df_a: pd.DataFrame,
    name_b: str, df_b: pd.DataFrame
):
    """Side-by-side comparison with Welch's t-test and Chi-square hypothesis tests."""
    st.subheader(f"⚖️ Comparative Analysis: {name_a} vs. {name_b}")

    if df_a.empty or df_b.empty:
        st.warning("Both Cohort A and Cohort B must contain at least 1 patient to run comparisons.")
        return

    # Check for small cohort warning
    small_cohort = len(df_a) < 5 or len(df_b) < 5
    if small_cohort:
        st.warning("⚠️ One or both cohorts contain fewer than 5 patients. Hypothesis test p-values are indicative only and should be interpreted with caution due to low statistical power.")

    # 1. Summary Metrics Comparison Table
    def _metrics(df: pd.DataFrame) -> Dict[str, float]:
        return {
            "Sample Size (N)": float(len(df)),
            "Mean Age (years)": float(df["age"].mean()),
            "Female Proportion (%)": float((df["gender"].astype(str).str.lower() == "female").mean() * 100.0),
            "Mean BMI (kg/m²)": float(df["bmi"].dropna().mean()) if df["bmi"].notna().any() else 0.0,
            "Mean Systolic BP (mmHg)": float(df["latest_systolic_bp"].dropna().mean()) if df["latest_systolic_bp"].notna().any() else 0.0,
            "Mean Diastolic BP (mmHg)": float(df["latest_diastolic_bp"].dropna().mean()) if df["latest_diastolic_bp"].notna().any() else 0.0,
            "Mean HbA1c (%)": float(df["latest_hba1c"].dropna().mean()) if df["latest_hba1c"].notna().any() else 0.0,
            "Mean Creatinine (mg/dL)": float(df["latest_creatinine"].dropna().mean()) if df["latest_creatinine"].notna().any() else 0.0,
            "Mean Cost per Patient (INR)": float(df["total_cost_inr"].mean()),
            "ER Visit Rate (%)": float(df["has_emergency"].mean() * 100.0),
            "Mean Charlson Comorbidity Score": float(df["charlson_score"].mean())
        }

    m_a = _metrics(df_a)
    m_b = _metrics(df_b)

    comp_rows = []
    for k in m_a.keys():
        val_a = m_a[k]
        val_b = m_b[k]
        diff = val_b - val_a
        comp_rows.append({
            "Metric": k,
            f"{name_a}": f"{int(val_a):,}" if "Size" in k else f"{val_a:.2f}",
            f"{name_b}": f"{int(val_b):,}" if "Size" in k else f"{val_b:.2f}",
            "Delta (B - A)": f"{int(diff):+d}" if "Size" in k else f"{diff:+.2f}"
        })

    st.markdown("##### 1. Side-by-Side Baseline Summary Table")
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # 2. Grouped Bar Charts for Key Metrics
    st.markdown("##### 2. Grouped Metric Comparisons")
    chart_metrics = [
        ("Mean Age (years)", "age", "Mean Age"),
        ("Mean BMI (kg/m²)", "bmi", "Mean BMI"),
        ("Systolic BP (mmHg)", "latest_systolic_bp", "Mean SBP"),
        ("HbA1c (%)", "latest_hba1c", "Mean HbA1c"),
        ("Cost per Patient (INR)", "total_cost_inr", "Mean Cost")
    ]

    g_rows = []
    for label, col, _ in chart_metrics:
        val_a = float(df_a[col].dropna().mean()) if df_a[col].notna().any() else 0.0
        val_b = float(df_b[col].dropna().mean()) if df_b[col].notna().any() else 0.0
        g_rows.append({"Metric": label, "Cohort": name_a, "Value": round(val_a, 2)})
        g_rows.append({"Metric": label, "Cohort": name_b, "Value": round(val_b, 2)})

    g_df = pd.DataFrame(g_rows)
    fig_comp_bar = px.bar(
        g_df,
        x="Metric",
        y="Value",
        color="Cohort",
        barmode="group",
        text="Value",
        color_discrete_map={name_a: "#38bdf8", name_b: "#f59e0b"},
        template="plotly_dark",
        title=f"Comparative Clinical Parameters ({name_a} vs. {name_b})"
    )
    fig_comp_bar.update_traces(textposition="outside")
    fig_comp_bar.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_comp_bar, use_container_width=True)

    # 3. Rigorous Statistical Hypothesis Testing (scipy)
    st.markdown("##### 3. Statistical Significance Hypothesis Tests")
    st.caption("Welch's t-test (two-sample unequal variance) for continuous clinical metrics; Pearson's Chi-square contingency test for binary proportions.")

    stat_rows = []

    # Continuous tests (Welch t-test)
    cont_test_spec = [
        ("Age", "age", "years"),
        ("BMI", "bmi", "kg/m²"),
        ("Systolic BP", "latest_systolic_bp", "mmHg"),
        ("HbA1c", "latest_hba1c", "%"),
        ("Total Healthcare Cost", "total_cost_inr", "INR")
    ]

    for label, col, unit in cont_test_spec:
        s_a = df_a[col].dropna()
        s_b = df_b[col].dropna()

        if len(s_a) >= 2 and len(s_b) >= 2 and s_a.std() > 0 and s_b.std() > 0:
            t_res = stats.ttest_ind(s_a, s_b, equal_var=False)
            p_val = float(t_res.pvalue)
            sig = p_val < 0.05
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            interpretation = "Difference is statistically significant at 5% (p < 0.05)" if sig else "Difference is not statistically significant (p ≥ 0.05)"
        else:
            p_str = "N/A"
            sig = False
            interpretation = "Insufficient sample variance to calculate t-test"

        mean_a_str = f"{s_a.mean():.2f} {unit}" if not s_a.empty else "N/A"
        mean_b_str = f"{s_b.mean():.2f} {unit}" if not s_b.empty else "N/A"

        stat_rows.append({
            "Clinical Metric": label,
            f"{name_a} (Mean)": mean_a_str,
            f"{name_b} (Mean)": mean_b_str,
            "Statistical Test": "Welch's t-test",
            "p-value": p_str,
            "Significant (α=0.05)": "✅ Yes" if sig else "❌ No",
            "Plain English Interpretation": interpretation
        })

    # Categorical tests (Chi-Square)
    # Gender (% Female)
    f_a = int((df_a["gender"].astype(str).str.lower() == "female").sum())
    m_a_cnt = len(df_a) - f_a
    f_b = int((df_b["gender"].astype(str).str.lower() == "female").sum())
    m_b_cnt = len(df_b) - f_b

    if min(f_a + m_a_cnt, f_b + m_b_cnt) >= 2:
        chi_table = [[f_a, m_a_cnt], [f_b, m_b_cnt]]
        chi_res = stats.chi2_contingency(chi_table)
        p_val_g = float(chi_res.pvalue)
        sig_g = p_val_g < 0.05
        p_str_g = f"{p_val_g:.4f}" if p_val_g >= 0.0001 else "< 0.0001"
        interp_g = "Difference is statistically significant at 5% (p < 0.05)" if sig_g else "Difference is not statistically significant (p ≥ 0.05)"
    else:
        p_str_g = "N/A"
        sig_g = False
        interp_g = "Sample too small for Chi-square"

    stat_rows.append({
        "Clinical Metric": "Gender (% Female)",
        f"{name_a} (Mean)": f"{(f_a / len(df_a) * 100):.1f}%",
        f"{name_b} (Mean)": f"{(f_b / len(df_b) * 100):.1f}%",
        "Statistical Test": "Pearson's Chi-square",
        "p-value": p_str_g,
        "Significant (α=0.05)": "✅ Yes" if sig_g else "❌ No",
        "Plain English Interpretation": interp_g
    })

    # ER Visit Rate
    er_a = int(df_a["has_emergency"].sum())
    no_er_a = len(df_a) - er_a
    er_b = int(df_b["has_emergency"].sum())
    no_er_b = len(df_b) - er_b

    if min(er_a + no_er_a, er_b + no_er_b) >= 2:
        chi_er_table = [[er_a, no_er_a], [er_b, no_er_b]]
        chi_er_res = stats.chi2_contingency(chi_er_table)
        p_val_er = float(chi_er_res.pvalue)
        sig_er = p_val_er < 0.05
        p_str_er = f"{p_val_er:.4f}" if p_val_er >= 0.0001 else "< 0.0001"
        interp_er = "Difference is statistically significant at 5% (p < 0.05)" if sig_er else "Difference is not statistically significant (p ≥ 0.05)"
    else:
        p_str_er = "N/A"
        sig_er = False
        interp_er = "Sample too small for Chi-square"

    stat_rows.append({
        "Clinical Metric": "Emergency Visit Rate",
        f"{name_a} (Mean)": f"{(er_a / len(df_a) * 100):.1f}%",
        f"{name_b} (Mean)": f"{(er_b / len(df_b) * 100):.1f}%",
        "Statistical Test": "Pearson's Chi-square",
        "p-value": p_str_er,
        "Significant (α=0.05)": "✅ Yes" if sig_er else "❌ No",
        "Plain English Interpretation": interp_er
    })

    st_df = pd.DataFrame(stat_rows)
    st.dataframe(st_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main Page View Tabs
# ---------------------------------------------------------------------------

st.markdown("## 👥 Longitudinal Cohort Builder & Statistical Comparator")
st.markdown("Define complex multi-factorial clinical cohorts, analyze demographic baselines, and evaluate hypothesis tests between clinical populations.")

main_tab_a, main_tab_b, main_tab_comp = st.tabs([
    f"🔵 {cohort_a_name} Profile",
    f"🟠 {cohort_b_name} Profile",
    f"⚖️ Compare Cohorts ({cohort_a_name} vs. {cohort_b_name})"
])

with main_tab_a:
    render_cohort_profile(cohort_a_name, cohort_a_df, theme_color="#38bdf8")

with main_tab_b:
    render_cohort_profile(cohort_b_name, cohort_b_df, theme_color="#f59e0b")

with main_tab_comp:
    render_cohort_comparison(cohort_a_name, cohort_a_df, cohort_b_name, cohort_b_df)
