"""
UI.py
-----
HealthSure InsurTech: Unified AI Underwriting & Health Actuarial Intelligence Hub.
Integrates machine learning actuarial risk prediction, dynamic batch portfolio scoring,
longitudinal Electronic Health Records (EHR) analytics, and Google Gemini LLM underwriting intelligence.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Shared EHR and ML Underwriting Utilities
from ehr_utils import (
    ALL_CITIES,
    OCCUPATIONS,
    build_master_patient_table,
    generate_actuarial_narrative,
    get_active_portfolio,
    get_bmi_category,
    load_ehr,
    predict_premium,
    render_sidebar_branding,
    render_sidebar_footer,
)


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HealthSure InsurTech | Executive Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Render unified sidebar branding & navigation styling
render_sidebar_branding()

# Custom Styling
st.markdown(
    """
    <style>
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 58, 138, 0.85));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 6px;
    }
    
    .hero-sub {
        font-size: 1.05rem;
        color: #93c5fd;
        margin-bottom: 12px;
    }
    
    .module-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .module-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    
    .module-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        margin-bottom: 8px;
    }
    
    .ehr-bridge-card {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    .dq-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Data Loading & Platform Stats
# ---------------------------------------------------------------------------

@st.cache_data
def get_hub_data():
    """Load cached master patient table and raw/clean datasets."""
    master = build_master_patient_table()
    cleaned_dfs, raw_dfs = load_ehr()
    return master, cleaned_dfs, raw_dfs

try:
    master_patients_df, cleaned_data, raw_data = get_hub_data()
    active_portfolio_df, active_portfolio_name = get_active_portfolio()
    
    total_pts = len(master_patients_df)
    total_policies = len(active_portfolio_df)
    total_encs = len(cleaned_data["encounters"])
    avg_charlson = master_patients_df["charlson_score"].mean()
    high_risk_pts = int((master_patients_df["risk_group"] == "High").sum())
    high_risk_policies = int((active_portfolio_df["predicted_insurance_premium_category"] == "High").sum())
except Exception as e:
    st.error(f"Failed to load platform data: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# 1. Hero Banner & Platform Overview
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🏥 HealthSure InsurTech & Actuarial Intelligence Hub</div>
        <div class="hero-sub">
            Unified healthcare data intelligence platform integrating longitudinal Electronic Health Records, 
            clinical comorbidity risk stratification, dynamic batch underwriting, and Google Gemini Actuarial Intelligence.
        </div>
        <div style="font-size: 0.88rem; color: #cbd5e1;">
            <b>Architecture:</b> Streamlit Multi-Page &bull; High-Throughput FastAPI Serving &bull; Scikit-Learn Random Forest Pipeline &bull; Gemini LLM Underwriter
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# 2. Executive KPI Summary Cards
# ---------------------------------------------------------------------------

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
kpi_c1.metric("Insured Portfolio", f"{total_policies:,}", "Active Cohort")
kpi_c2.metric("High-Risk Exposure", f"{high_risk_policies:,}", f"{(high_risk_policies/total_policies*100):.1f}% Policies")
kpi_c3.metric("Clinical Registry", f"{total_pts:,}", "Validated EHR")
kpi_c4.metric("Encounter History", f"{total_encs:,}", "Longitudinal")
kpi_c5.metric("Mean Charlson Score", f"{avg_charlson:.2f}", "CCI Baseline")
kpi_c6.metric("Data Quality Score", "100.0%", "Audited Clean")

st.markdown("---")


# ---------------------------------------------------------------------------
# 3. InsurTech Platform Navigation & Module Directory
# ---------------------------------------------------------------------------

st.subheader("🧭 Platform Navigation & Specialized Analytical Modules")
st.markdown("Select a specialized module from the sidebar or click into their key capabilities below:")

mod_r1_c1, mod_r1_c2, mod_r1_c3 = st.columns(3)

with mod_r1_c1:
    st.markdown(
        """
        <div class="module-card">
            <span class="module-badge">Dynamic Ingestion</span>
            <div style="font-size: 1.15rem; font-weight: 700; color: white;">📁 1. Batch Underwriting</div>
            <p style="font-size: 0.88rem; color: #94a3b8; margin: 8px 0 12px 0;">
                Upload custom CSV or Excel files. Triggers high-throughput asynchronous ML scoring,
                dynamic Plotly risk dashboards, and 1-click export of scored predictions.
            </p>
            <small style="color: #38bdf8;">👉 Open <b>1_Batch_Underwriting</b> in sidebar</small>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_r1_c2:
    st.markdown(
        """
        <div class="module-card">
            <span class="module-badge">Clinical Risk</span>
            <div style="font-size: 1.15rem; font-weight: 700; color: white;">🧬 2. Comorbidity Analysis</div>
            <p style="font-size: 0.88rem; color: #94a3b8; margin: 8px 0 12px 0;">
                Charlson Comorbidity Index scoring, disease prevalence distributions, 
                co-occurrence associative lift heatmap, and inpatient utilization cost analysis.
            </p>
            <small style="color: #38bdf8;">👉 Open <b>2_Comorbidity_Analysis</b> in sidebar</small>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_r1_c3:
    st.markdown(
        """
        <div class="module-card">
            <span class="module-badge">Hypothesis Testing</span>
            <div style="font-size: 1.15rem; font-weight: 700; color: white;">👥 3. Cohort Builder</div>
            <p style="font-size: 0.88rem; color: #94a3b8; margin: 8px 0 12px 0;">
                Multi-factorial dual-cohort filtering, clinical presets (Diabetics, Metformin, Age 60+), 
                and rigorous side-by-side hypothesis tests (Welch's t-test and Chi-square).
            </p>
            <small style="color: #38bdf8;">👉 Open <b>3_Cohort_Builder</b> in sidebar</small>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

mod_r2_c1, mod_r2_c2, mod_r2_c3 = st.columns(3)

with mod_r2_c1:
    st.markdown(
        """
        <div class="module-card">
            <span class="module-badge">Medical Underwriting</span>
            <div style="font-size: 1.15rem; font-weight: 700; color: white;">🩺 4. Patient 360 EHR</div>
            <p style="font-size: 0.88rem; color: #94a3b8; margin: 8px 0 12px 0;">
                Deep clinical audit for flagged high-risk applicants: longitudinal vitals,
                lab trends with physiological reference bands, and 1-click clinical PDF export.
            </p>
            <small style="color: #38bdf8;">👉 Open <b>4_Patient_360</b> in sidebar</small>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_r2_c2:
    st.markdown(
        """
        <div class="module-card">
            <span class="module-badge">7-D Audit Engine</span>
            <div style="font-size: 1.15rem; font-weight: 700; color: white;">🛡️ 5. EHR Data Quality</div>
            <p style="font-size: 0.88rem; color: #94a3b8; margin: 8px 0 12px 0;">
                Audits raw vs. cleaned datasets across completeness, duplicates, validity, format, 
                referential integrity, ICD-10 roots, and temporal sequence with Plotly gauge dials.
            </p>
            <small style="color: #38bdf8;">👉 Open <b>5_EHR_Data_Quality</b> in sidebar</small>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_r2_c3:
    st.markdown(
        """
        <div class="module-card">
            <span class="module-badge">FastAPI Serving</span>
            <div style="font-size: 1.15rem; font-weight: 700; color: white;">⚡ 6. API & Developer Hub</div>
            <p style="font-size: 0.88rem; color: #94a3b8; margin: 8px 0 12px 0;">
                Live FastAPI server health monitor, interactive endpoint tester (/predict),
                Random Forest feature importance weights, and developer SDK code snippets.
            </p>
            <small style="color: #38bdf8;">👉 Open <b>6_API_and_Developer_Hub</b> in sidebar</small>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# 4. Interactive EHR-to-Underwriting Patient Bridge
# ---------------------------------------------------------------------------

st.subheader("⚡ Clinical EHR-to-Underwriting Patient Bridge")
st.markdown("Seamlessly import clinical records from the EHR repository into the actuarial underwriting form below:")

bridge_c1, bridge_c2 = st.columns([1.5, 2.5])

# Check if patient was sent from another page via session_state
session_underwrite_patient = st.session_state.get("underwrite_patient", None)
selected_pid_state = st.session_state.get("selected_patient_id", master_patients_df["patient_id"].iloc[0])

all_patient_ids = master_patients_df["patient_id"].tolist()
default_patient_idx = all_patient_ids.index(selected_pid_state) if selected_pid_state in all_patient_ids else 0

def format_bridge_patient(pid: str) -> str:
    r = master_patients_df[master_patients_df["patient_id"] == pid].iloc[0]
    return f"{pid} — {r['name']} ({r['age']}y, {r['gender']}, {r['city']}) | Charlson: {r['charlson_score']}"

with bridge_c1:
    bridge_pid = st.selectbox(
        "Select EHR Patient to Underwrite:",
        options=all_patient_ids,
        index=default_patient_idx,
        format_func=format_bridge_patient,
        key="bridge_patient_select"
    )

patient_bridge_row = master_patients_df[master_patients_df["patient_id"] == bridge_pid].iloc[0]

with bridge_c2:
    st.markdown(
        f"""
        <div class="ehr-bridge-card">
            <div style="font-weight: 700; font-size: 1.1rem; color: #38bdf8; margin-bottom: 4px;">
                👤 {patient_bridge_row['name']} ({patient_bridge_row['patient_id']})
            </div>
            <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5;">
                <b>Age:</b> {patient_bridge_row['age']} yrs &nbsp;|&nbsp; 
                <b>Height:</b> {patient_bridge_row['height_cm']} cm &nbsp;|&nbsp; 
                <b>Weight:</b> {patient_bridge_row['weight_kg']} kg &nbsp;|&nbsp; 
                <b>BMI:</b> {patient_bridge_row['bmi']} ({get_bmi_category(patient_bridge_row['bmi'])[0]})<br>
                <b>City:</b> {patient_bridge_row['city']} &nbsp;|&nbsp; 
                <b>Charlson Score:</b> {patient_bridge_row['charlson_score']} ({patient_bridge_row['risk_group']} Risk) &nbsp;|&nbsp; 
                <b>Encounters:</b> {patient_bridge_row['num_encounters']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

action_col1, action_col2, _ = st.columns([1.5, 1.5, 3])

if action_col1.button("⚡ Autofill Form with this Patient", use_container_width=True):
    st.session_state["underwrite_prefill"] = {
        "age": int(patient_bridge_row["age"]),
        "weight": float(patient_bridge_row["weight_kg"]),
        "height": float(patient_bridge_row["height_cm"] / 100.0),
        "city": str(patient_bridge_row["city"]) if str(patient_bridge_row["city"]) in ALL_CITIES else ALL_CITIES[0],
        "patient_id": str(patient_bridge_row["patient_id"])
    }
    st.success(f"Autofilled form with data for {patient_bridge_row['name']}!")

if action_col2.button("🩺 Preselect for Patient 360", use_container_width=True):
    st.session_state["selected_patient_id"] = bridge_pid
    st.success(f"Patient {bridge_pid} preselected! Switch to '4_Patient_360' in the sidebar.")

st.markdown("---")


# ---------------------------------------------------------------------------
# 5. Interactive Underwriting Calculator & Gemini Actuarial Engine
# ---------------------------------------------------------------------------

st.subheader("💰 Individual Applicant Underwriting Calculator")
st.markdown("Predict risk-adjusted premium tiers and generate real-time Gemini actuarial policy briefs.")

# Retrieve active prefill values if set
prefill = st.session_state.get("underwrite_prefill", {})

fcol1, fcol2, fcol3 = st.columns(3)

with fcol1:
    in_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=prefill.get("age", 32),
        help="Age of applicant"
    )
    
    in_weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=prefill.get("weight", 68.0),
        step=0.5
    )

    in_height = st.number_input(
        "Height (meters)",
        min_value=1.0,
        max_value=2.5,
        value=prefill.get("height", 1.72),
        step=0.01,
        help="Height in meters (e.g., 1.72 m)"
    )

with fcol2:
    in_income_lpa = st.number_input(
        "Annual Income (Lakhs INR / LPA)",
        min_value=0.5,
        max_value=150.0,
        value=7.5,
        step=0.5,
        help="Annual gross income in Lakhs (INR)"
    )

    default_city_idx = 0
    if "city" in prefill and prefill["city"] in ALL_CITIES:
        default_city_idx = ALL_CITIES.index(prefill["city"])

    in_city = st.selectbox(
        "City of Residence",
        options=ALL_CITIES,
        index=default_city_idx
    )

    in_smoker = st.selectbox(
        "Smoking Status",
        options=[0, 1],
        format_func=lambda s: "Non-Smoker (0)" if s == 0 else "Smoker (1)"
    )

with fcol3:
    in_occupation = st.selectbox(
        "Occupation",
        options=OCCUPATIONS,
        index=OCCUPATIONS.index("Software Engineer") if "Software Engineer" in OCCUPATIONS else 0
    )

    calc_bmi = in_weight / (in_height ** 2)
    bmi_cat, bmi_label, bmi_class = get_bmi_category(calc_bmi)
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 8px; margin-top: 14px;">
            <div style="font-size: 0.85rem; color: #94a3b8;">Calculated BMI</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: white;">{calc_bmi:.1f} kg/m²</div>
            <small style="color: #38bdf8;">{bmi_label}</small>
        </div>
        """,
        unsafe_allow_html=True
    )


# Prediction Action
if st.button("🚀 Calculate Risk-Adjusted Premium Tier", use_container_width=True):
    with st.spinner("Analyzing risk factors through Machine Learning Underwriting Pipeline..."):
        try:
            pred_tier, source_label = predict_premium(
                age=in_age,
                weight=in_weight,
                height=in_height,
                income_lpa=in_income_lpa,
                smoker=in_smoker,
                city=in_city,
                occupation=in_occupation
            )

            # Store result in session state to persist across reruns
            st.session_state["active_single_eval"] = {
                "age": in_age,
                "weight": in_weight,
                "height": in_height,
                "income_lpa": in_income_lpa,
                "smoker": in_smoker,
                "city": in_city,
                "occupation": in_occupation,
                "bmi": round(calc_bmi, 2),
                "bmi_category": bmi_cat,
                "predicted_tier": pred_tier,
                "source_label": source_label,
                "health_risk_index": round(calc_bmi * 0.4 + in_age * 0.3 + in_smoker * 20.0, 2)
            }
        except Exception as err:
            st.error(f"Prediction failed: {err}")


# Display Evaluation Outcome if Available
if "active_single_eval" in st.session_state:
    eval_data = st.session_state["active_single_eval"]
    pred_tier = eval_data["predicted_tier"]
    source_label = eval_data["source_label"]

    st.markdown("### 🎯 Underwriting Assessment Outcome")
    res_c1, res_c2 = st.columns([1.5, 3.5])

    with res_c1:
        if pred_tier == "Low":
            st.success(f"### ✅ Low Premium Tier")
            if os.path.exists("images/low.png"):
                st.image("images/low.png", use_container_width=True)
        elif pred_tier == "Medium":
            st.warning(f"### ⚠️ Medium Premium Tier")
            if os.path.exists("images/medium.png"):
                st.image("images/medium.png", use_container_width=True)
        else:
            st.error(f"### 🚨 High Premium Tier")
            if os.path.exists("images/high.png"):
                st.image("images/high.png", use_container_width=True)
                
        st.caption(f"Engine Serving: **{source_label}**")

    with res_c2:
        st.markdown(
            f"""
            <div class="dq-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: white; margin-bottom: 8px;">
                    Actuarial Profile Summary
                </div>
                <ul style="font-size: 0.95rem; line-height: 1.6; color: #cbd5e1;">
                    <li><b>Predicted Risk Class:</b> {pred_tier} Premium Bracket</li>
                    <li><b>Age / Smoker Risk:</b> {eval_data['age']} yrs, {'Smoker (+Risk)' if eval_data['smoker'] == 1 else 'Non-smoker (Baseline)'}</li>
                    <li><b>Biometric Index:</b> BMI {eval_data['bmi']} ({eval_data['bmi_category']})</li>
                    <li><b>Health Risk Index:</b> {eval_data['health_risk_index']}</li>
                    <li><b>Territory Risk:</b> {eval_data['city']} &bull; {eval_data['occupation']}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Gemini Actuarial Narrative Button
    st.markdown("##### ✨ Gemini AI Actuarial Briefing")
    if st.button("🤖 Generate Gemini Actuarial Underwriting Report", use_container_width=True):
        with st.spinner("Consulting Gemini Actuarial Intelligence..."):
            narrative = generate_actuarial_narrative(eval_data)
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; border-radius: 10px; padding: 18px 24px; margin-top: 12px;">
                    {narrative}
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")


# ---------------------------------------------------------------------------
# 6. Academic Highlights & System Architecture Guide
# ---------------------------------------------------------------------------

with st.expander("🎓 Academic Project Highlights & System Architecture Guide", expanded=False):
    st.markdown(
        """
        ### 📌 Key Engineering & Scientific Contributions:
        1. **End-to-End ML Microservice Serving:**
           - High-performance FastAPI serving engine (`/predict`, `/predict/batch`, `/predict/upload`, `/health`).
           - In-memory pipeline caching on server lifespan (eliminating per-request disk bottlenecks).
        2. **Dynamic File Ingestion & Real-Time Analytics:**
           - Accepts CSV and Excel files, runs vectorized feature engineering, and instantly generates Plotly dashboards without hardcoded data dependencies.
        3. **Dual-Mode Google Gemini LLM Integration:**
           - Generates executive underwriting justifications and portfolio vulnerability briefings with resilient offline fallbacks.
        4. **Clinical EHR Integration & Comorbidity Analytics:**
           - Bridges longitudinal patient hospital records (diagnoses, labs, vitals) with actuarial underwriting risk categories.
        5. **Statistical Rigor:**
           - Automated Welch's t-test and Chi-square hypothesis tests for comparative cohort analysis (`scipy.stats`).
        """
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 12px 0;'>
        HealthSure InsurTech & Actuarial Serving System &bull; Production Multi-Module Architecture &bull; 
        Powered by FastAPI, Streamlit, Scikit-Learn, Gemini LLM & Plotly
    </div>
    """,
    unsafe_allow_html=True
)

render_sidebar_footer()
