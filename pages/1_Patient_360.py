"""
pages/Patient_360.py
--------------------
Longitudinal Patient Record View (EHR 360).
Provides a comprehensive clinical view of patient history including:
- Demographics and BMI categorization
- Auto-generated narrative clinical summary
- Longitudinal Diagnoses, Medications, Vitals Trends, Lab Trends, Timeline, and Encounters
- PDF export capability
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import core EHR loader and clinical classification utilities
from ehr_utils import (
    classify_bp,
    classify_glucose,
    get_bmi_category,
    load_ehr,
    render_sidebar_branding,
    render_sidebar_footer,
)

# ReportLab imports for PDF export
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page Configuration & Styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Patient 360 | HealthSure EHR",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar_branding()

# Custom EHR styling
st.markdown(
    """
    <style>
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Clinical Card container */
    .ehr-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 18px;
        backdrop-filter: blur(8px);
    }
    
    .ehr-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    
    .ehr-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    
    .badge-blue { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .badge-green { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-yellow { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-red { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    
    .summary-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Helper Clinical Functions
# ---------------------------------------------------------------------------


def generate_clinical_summary(
    patient: pd.Series,
    p_conditions: pd.DataFrame,
    p_encounters: pd.DataFrame,
    p_labs: pd.DataFrame,
    p_medications: pd.DataFrame
) -> str:
    """
    Generate an automated, natural clinical narrative summary from real patient records.
    Handles patients with no conditions, encounters, labs, or medications gracefully.
    """
    # 1. Demographics
    age_str = f"{int(patient['age'])}-year-old" if pd.notna(patient.get("age")) else "Patient of unknown age"
    gender_str = str(patient.get("gender", "Individual")).strip().title()
    
    # 2. Conditions
    cond_list = []
    if not p_conditions.empty and "condition" in p_conditions.columns:
        cond_list = p_conditions["condition"].dropna().unique().tolist()
        
    if cond_list:
        if len(cond_list) == 1:
            cond_str = f"with {cond_list[0]}"
        elif len(cond_list) == 2:
            cond_str = f"with {cond_list[0]} and {cond_list[1]}"
        else:
            cond_str = f"with {', '.join(cond_list[:-1])}, and {cond_list[-1]}"
    else:
        cond_str = "with no documented chronic conditions"
        
    summary_parts = [f"{age_str} {gender_str} {cond_str}."]

    # 3. Encounters
    n_enc = len(p_encounters)
    if n_enc > 0:
        n_emerg = 0
        if "encounter_type" in p_encounters.columns:
            n_emerg = int((p_encounters["encounter_type"].astype(str).str.strip().str.lower() == "emergency").sum())
        
        enc_label = f"{n_enc} encounter" if n_enc == 1 else f"{n_enc} encounters"
        summary_parts.append(f"{enc_label} ({n_emerg} emergency).")
    else:
        summary_parts.append("No encounter records on file.")

    # 4. Last Vitals (BP)
    if not p_encounters.empty:
        bp_valid = p_encounters[p_encounters["systolic_bp"].notna() & p_encounters["diastolic_bp"].notna()]
        if not bp_valid.empty and "encounter_date" in bp_valid.columns:
            bp_sorted = bp_valid.sort_values(by="encounter_date", ascending=False)
            latest_sbp = float(bp_sorted.iloc[0]["systolic_bp"])
            latest_dbp = float(bp_sorted.iloc[0]["diastolic_bp"])
            stage = classify_bp(latest_sbp, latest_dbp) or "Unclassified"
            summary_parts.append(f"Last BP {int(latest_sbp)}/{int(latest_dbp)} ({stage}).")

    # 5. Last Labs (HbA1c)
    if not p_labs.empty and "hba1c_pct" in p_labs.columns:
        hba1c_valid = p_labs[p_labs["hba1c_pct"].notna()]
        if not hba1c_valid.empty and "lab_date" in hba1c_valid.columns:
            hba1c_sorted = hba1c_valid.sort_values(by="lab_date", ascending=False)
            latest_hba1c = float(hba1c_sorted.iloc[0]["hba1c_pct"])
            summary_parts.append(f"Last HbA1c {latest_hba1c:.1f}%.")

    # 6. Current Medications
    if not p_medications.empty and "drug_name" in p_medications.columns:
        drugs = p_medications["drug_name"].dropna().unique().tolist()
        if drugs:
            if len(drugs) == 1:
                med_str = f"Currently on {drugs[0]}."
            elif len(drugs) == 2:
                med_str = f"Currently on {drugs[0]} and {drugs[1]}."
            else:
                med_str = f"Currently on {', '.join(drugs[:-1])}, and {drugs[-1]}."
            summary_parts.append(med_str)
        else:
            summary_parts.append("No active medications recorded.")
    else:
        summary_parts.append("No active medications recorded.")

    return " ".join(summary_parts)


# ---------------------------------------------------------------------------
# PDF Generation Function
# ---------------------------------------------------------------------------

def generate_pdf_summary(
    patient: pd.Series,
    summary_text: str,
    p_conditions: pd.DataFrame,
    p_medications: pd.DataFrame,
    p_encounters: pd.DataFrame,
    p_labs: pd.DataFrame
) -> bytes:
    """
    Generate a formatted one-page clinical patient summary PDF using ReportLab.
    Falls back to structured UTF-8 text if ReportLab is unavailable.
    """
    if not REPORTLAB_AVAILABLE:
        # Fallback to text file
        buffer = io.StringIO()
        buffer.write(f"HEALTHSURE EHR - PATIENT RECORD SUMMARY\n")
        buffer.write(f"{'='*50}\n")
        buffer.write(f"Patient ID: {patient.get('patient_id')} | Name: {patient.get('name')}\n")
        buffer.write(f"Age: {patient.get('age')} | Gender: {patient.get('gender')} | City: {patient.get('city')}\n")
        buffer.write(f"Blood Group: {patient.get('blood_group')} | BMI: {patient.get('bmi')}\n\n")
        buffer.write(f"CLINICAL SUMMARY:\n{summary_text}\n\n")
        return buffer.getvalue().encode("utf-8")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a")
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    summary_style = ParagraphStyle(
        "SummaryText",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#0f172a")
    )

    story: List[Any] = []

    # Title & Hospital Header
    story.append(Paragraph("<b>HealthSure EHR</b> — Longitudinal Patient Summary", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d-%b-%Y %H:%M')}", body_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=10))

    # Demographics Table
    demo_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {patient.get('name', 'N/A')}", body_style),
            Paragraph(f"<b>Patient ID:</b> {patient.get('patient_id', 'N/A')}", body_style),
            Paragraph(f"<b>Age / Gender:</b> {patient.get('age', 'N/A')}y / {patient.get('gender', 'N/A')}", body_style)
        ],
        [
            Paragraph(f"<b>City:</b> {patient.get('city', 'N/A')}", body_style),
            Paragraph(f"<b>Blood Group:</b> {patient.get('blood_group', 'Unknown')}", body_style),
            Paragraph(f"<b>BMI:</b> {patient.get('bmi', 'N/A')} ({get_bmi_category(patient.get('bmi'))[0]})", body_style)
        ],
        [
            Paragraph(f"<b>Height:</b> {patient.get('height_cm', 'N/A')} cm", body_style),
            Paragraph(f"<b>Weight:</b> {patient.get('weight_kg', 'N/A')} kg", body_style),
            Paragraph(f"<b>Total Encounters:</b> {len(p_encounters)}", body_style)
        ]
    ]
    t_demo = Table(demo_data, colWidths=[180, 180, 180])
    t_demo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 10))

    # Narrative Summary
    story.append(Paragraph("<b>Clinical Summary</b>", section_style))
    story.append(Paragraph(summary_text, summary_style))
    story.append(Spacer(1, 8))

    # Active Diagnoses
    story.append(Paragraph("<b>Documented Diagnoses</b>", section_style))
    if not p_conditions.empty:
        cond_rows = [["ICD-10", "Condition", "Diagnosis Date"]]
        for _, row in p_conditions.sort_values(by="diagnosis_date", ascending=False).iterrows():
            d_date = pd.to_datetime(row["diagnosis_date"]).strftime("%Y-%m-%d") if pd.notna(row["diagnosis_date"]) else "N/A"
            cond_rows.append([str(row.get("icd10", "")), str(row.get("condition", "")), d_date])
        t_cond = Table(cond_rows, colWidths=[80, 310, 150])
        t_cond.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_cond)
    else:
        story.append(Paragraph("No chronic diagnoses recorded.", body_style))
    story.append(Spacer(1, 8))

    # Medications
    story.append(Paragraph("<b>Active Medications</b>", section_style))
    if not p_medications.empty:
        med_rows = [["Medication", "Dose", "Frequency", "Start Date"]]
        for _, row in p_medications.sort_values(by="start_date", ascending=False).iterrows():
            s_date = pd.to_datetime(row["start_date"]).strftime("%Y-%m-%d") if pd.notna(row["start_date"]) else "N/A"
            med_rows.append([str(row.get("drug_name", "")), str(row.get("dose", "")), str(row.get("frequency", "")), s_date])
        t_med = Table(med_rows, colWidths=[180, 110, 130, 120])
        t_med.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_med)
    else:
        story.append(Paragraph("No active medications recorded.", body_style))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Data Loading & Initialization
# ---------------------------------------------------------------------------

@st.cache_data
def get_cached_ehr_data():
    """Load clean EHR data via load_ehr()."""
    return load_ehr()

try:
    cleaned_dfs, _ = get_cached_ehr_data()
    patients_df: pd.DataFrame = cleaned_dfs["patients"]
    conditions_df: pd.DataFrame = cleaned_dfs["conditions"]
    encounters_df: pd.DataFrame = cleaned_dfs["encounters"]
    labs_df: pd.DataFrame = cleaned_dfs["labs"]
    medications_df: pd.DataFrame = cleaned_dfs["medications"]
    charlson_df: pd.DataFrame = cleaned_dfs["charlson"]
except Exception as e:
    st.error(f"Error loading EHR dataset: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Patient Search & Selection
# ---------------------------------------------------------------------------

st.markdown("## 🩺 Patient 360 | Longitudinal Record View")
st.markdown("Inspect unified longitudinal medical records, clinical trends, diagnoses, and medication regimens.")

search_col1, search_col2 = st.columns([1.5, 2.5])

with search_col1:
    search_query = st.text_input(
        "🔍 Filter Patients (by Name, ID, or City):",
        placeholder="e.g. Rahul, P00000, Kolhapur...",
        help="Type to search and filter the patient selection list."
    ).strip().lower()

# Filter patient pool
if search_query:
    filtered_patients = patients_df[
        patients_df["name"].astype(str).str.lower().str.contains(search_query) |
        patients_df["patient_id"].astype(str).str.lower().str.contains(search_query) |
        patients_df["city"].astype(str).str.lower().str.contains(search_query)
    ]
else:
    filtered_patients = patients_df

if filtered_patients.empty:
    st.warning("No patients match your search filter. Showing all patients.")
    filtered_patients = patients_df

# Format selectbox options
patient_options = filtered_patients["patient_id"].tolist()

def format_patient_label(pid: str) -> str:
    row = patients_df[patients_df["patient_id"] == pid]
    if not row.empty:
        r = row.iloc[0]
        return f"{pid} — {r.get('name', '')} ({r.get('gender', '')}, {r.get('age', '')}y, {r.get('city', '')})"
    return pid

with search_col2:
    # Connect to cross-page session state
    default_idx = 0
    curr_state_pid = st.session_state.get("selected_patient_id", None)
    if curr_state_pid and curr_state_pid in patient_options:
        default_idx = patient_options.index(curr_state_pid)

    selected_pid = st.selectbox(
        "Select Patient Record:",
        options=patient_options,
        format_func=format_patient_label,
        index=default_idx
    )
    st.session_state["selected_patient_id"] = selected_pid

if not selected_pid:
    st.info("Please select a patient to view records.")
    st.stop()

# Extract selected patient records
patient_row = patients_df[patients_df["patient_id"] == selected_pid].iloc[0]
p_conditions = conditions_df[conditions_df["patient_id"] == selected_pid].copy()
p_encounters = encounters_df[encounters_df["patient_id"] == selected_pid].copy()
p_labs = labs_df[labs_df["patient_id"] == selected_pid].copy()
p_medications = medications_df[medications_df["patient_id"] == selected_pid].copy()

# Sort dates
if not p_conditions.empty and "diagnosis_date" in p_conditions.columns:
    p_conditions["diagnosis_date"] = pd.to_datetime(p_conditions["diagnosis_date"])
    p_conditions = p_conditions.sort_values(by="diagnosis_date", ascending=False)

if not p_encounters.empty and "encounter_date" in p_encounters.columns:
    p_encounters["encounter_date"] = pd.to_datetime(p_encounters["encounter_date"])
    p_encounters = p_encounters.sort_values(by="encounter_date", ascending=False)

if not p_labs.empty and "lab_date" in p_labs.columns:
    p_labs["lab_date"] = pd.to_datetime(p_labs["lab_date"])
    p_labs = p_labs.sort_values(by="lab_date", ascending=False)

if not p_medications.empty and "start_date" in p_medications.columns:
    p_medications["start_date"] = pd.to_datetime(p_medications["start_date"])
    p_medications = p_medications.sort_values(by="start_date", ascending=False)


# ---------------------------------------------------------------------------
# 1. Header Card (Demographics & BMI Badge)
# ---------------------------------------------------------------------------

bmi_val = float(patient_row.get("bmi")) if pd.notna(patient_row.get("bmi")) else None
bmi_category, bmi_badge_label, bmi_badge_class = get_bmi_category(bmi_val)

header_html = f"""
<div class="ehr-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <div class="ehr-title">👤 {patient_row.get('name', 'Unknown Name')}</div>
            <div class="ehr-subtitle">
                <b>ID:</b> {patient_row.get('patient_id')} &nbsp;|&nbsp; 
                <b>Age:</b> {patient_row.get('age', 'N/A')} yrs &nbsp;|&nbsp; 
                <b>Gender:</b> {patient_row.get('gender', 'N/A')} &nbsp;|&nbsp; 
                <b>City:</b> {patient_row.get('city', 'N/A')} &nbsp;|&nbsp; 
                <b>Blood Group:</b> {patient_row.get('blood_group', 'Unknown')}
            </div>
        </div>
        <div style="text-align: right;">
            <span class="badge {bmi_badge_class}">{bmi_badge_label}</span>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">BMI: <b>{bmi_val if bmi_val is not None else 'N/A'}</b></div>
        </div>
    </div>
    <div style="display: flex; gap: 28px; margin-top: 10px; flex-wrap: wrap; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
        <div><small style="color: #94a3b8;">Height</small><br><b>{patient_row.get('height_cm', 'N/A')} cm</b></div>
        <div><small style="color: #94a3b8;">Weight</small><br><b>{patient_row.get('weight_kg', 'N/A')} kg</b></div>
        <div><small style="color: #94a3b8;">Age Group</small><br><b>{patient_row.get('age_group', 'N/A')}</b></div>
        <div><small style="color: #94a3b8;">Encounters</small><br><b>{len(p_encounters)}</b></div>
        <div><small style="color: #94a3b8;">Active Meds</small><br><b>{len(p_medications)}</b></div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

act_c1, act_c2 = st.columns([1.8, 3.2])
with act_c1:
    if st.button("💰 Underwrite Patient in Insurance Engine", key="btn_underwrite_p360"):
        st.session_state["underwrite_prefill"] = {
            "age": int(patient_row.get("age", 30)),
            "weight": float(patient_row.get("weight_kg", 70.0)),
            "height": float(patient_row.get("height_cm", 170.0) / 100.0),
            "city": str(patient_row.get("city", "Mumbai")),
            "patient_id": str(patient_row.get("patient_id"))
        }
        st.session_state["selected_patient_id"] = str(patient_row.get("patient_id"))
        st.success("Profile prepared for Insurance Underwriting module! Return to Home (UI.py) to run prediction.")
with act_c2:
    st.caption("Seamlessly transfers biometric indicators (Age, BMI, City, Weight, Height) into the Machine Learning Underwriting Model.")



# ---------------------------------------------------------------------------
# 2. Auto-Generated Clinical Summary
# ---------------------------------------------------------------------------

clinical_summary_text = generate_clinical_summary(
    patient=patient_row,
    p_conditions=p_conditions,
    p_encounters=p_encounters,
    p_labs=p_labs,
    p_medications=p_medications
)

st.markdown(
    f"""
    <div class="summary-box">
        <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">📋 Clinical Narrative Summary</div>
        <div>{clinical_summary_text}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# 3. Clinical Tabs
# ---------------------------------------------------------------------------

tab_diag, tab_meds, tab_vitals, tab_labs, tab_timeline, tab_enc = st.tabs([
    "🩺 Diagnoses",
    "💊 Medications",
    "📈 Vitals Trends",
    "🧪 Lab Trends",
    "🗓️ Timeline",
    "🏥 Encounters"
])


# ---------------------------------------------------------------------------
# Tab 1: Diagnoses
# ---------------------------------------------------------------------------
with tab_diag:
    st.subheader("Documented Diagnoses & Comorbidities")
    if p_conditions.empty:
        st.info("No diagnoses recorded for this patient.")
    else:
        # Merge with charlson mapping
        diag_table = p_conditions.copy()
        if not charlson_df.empty:
            diag_table = diag_table.merge(
                charlson_df[["icd10", "description", "charlson_weight"]],
                on="icd10",
                how="left"
            )
            diag_table["description"] = diag_table["description"].fillna(diag_table["condition"])
            diag_table["charlson_weight"] = diag_table["charlson_weight"].fillna(0).astype(int)
        else:
            diag_table["description"] = diag_table["condition"]
            diag_table["charlson_weight"] = 0

        # Format table
        diag_display = diag_table[[
            "condition", "icd10", "diagnosis_date", "description", "charlson_weight"
        ]].rename(columns={
            "condition": "Condition Name",
            "icd10": "ICD-10 Code",
            "diagnosis_date": "Diagnosis Date",
            "description": "Charlson Description",
            "charlson_weight": "Charlson Weight"
        })
        diag_display["Diagnosis Date"] = diag_display["Diagnosis Date"].dt.strftime("%Y-%m-%d")

        st.dataframe(
            diag_display.sort_values(by="Diagnosis Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        total_weight = int(diag_table["charlson_weight"].sum())
        st.caption(f"Total Charlson Comorbidity Weight: **{total_weight}** across **{len(diag_display)}** documented conditions.")


# ---------------------------------------------------------------------------
# Tab 2: Medications
# ---------------------------------------------------------------------------
with tab_meds:
    st.subheader("Prescribed Medications & Treatable Conditions")
    if p_medications.empty:
        st.info("No medications recorded for this patient.")
    else:
        meds_table = p_medications.copy()

        # Join on icd10 to find condition treated
        if not conditions_df.empty:
            icd_condition_map = conditions_df[["icd10", "condition"]].drop_duplicates().set_index("icd10")["condition"].to_dict()
            meds_table["Treated Condition"] = meds_table["icd10"].map(icd_condition_map)
        elif not charlson_df.empty:
            icd_condition_map = charlson_df[["icd10", "description"]].drop_duplicates().set_index("icd10")["description"].to_dict()
            meds_table["Treated Condition"] = meds_table["icd10"].map(icd_condition_map)
        else:
            meds_table["Treated Condition"] = meds_table["icd10"]

        meds_table["Treated Condition"] = meds_table["Treated Condition"].fillna("Unspecified / Indication " + meds_table["icd10"].astype(str))

        meds_display = meds_table[[
            "drug_name", "dose", "frequency", "start_date", "Treated Condition", "icd10"
        ]].rename(columns={
            "drug_name": "Drug Name",
            "dose": "Dose",
            "frequency": "Frequency",
            "start_date": "Start Date",
            "icd10": "Target ICD-10"
        })
        meds_display["Start Date"] = meds_display["Start Date"].dt.strftime("%Y-%m-%d")

        st.dataframe(
            meds_display.sort_values(by="Start Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )


# ---------------------------------------------------------------------------
# Tab 3: Vitals Trends
# ---------------------------------------------------------------------------
with tab_vitals:
    st.subheader("Longitudinal Vitals Trends")
    if p_encounters.empty:
        st.info("No encounter records available for vitals trend plotting.")
    else:
        vitals_df = p_encounters.dropna(subset=["encounter_date"]).sort_values(by="encounter_date")

        col_v1, col_v2 = st.columns(2)

        # 1. Blood Pressure Trends Chart
        with col_v1:
            st.markdown("##### Blood Pressure (SBP & DBP)")
            bp_df = vitals_df.dropna(subset=["systolic_bp", "diastolic_bp"])
            if bp_df.empty:
                st.info("No blood pressure readings available.")
            else:
                fig_bp = go.Figure()

                # Shaded Normal SBP Region (90-120)
                fig_bp.add_hrect(
                    y0=90, y1=120,
                    fillcolor="rgba(34, 197, 94, 0.12)",
                    line_width=0,
                    annotation_text="Normal SBP (90-120)",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="#10b981"
                )

                # Shaded Normal DBP Region (60-80)
                fig_bp.add_hrect(
                    y0=60, y1=80,
                    fillcolor="rgba(59, 130, 246, 0.10)",
                    line_width=0,
                    annotation_text="Normal DBP (60-80)",
                    annotation_position="bottom left",
                    annotation_font_size=10,
                    annotation_font_color="#3b82f6"
                )

                # Normal lines
                fig_bp.add_trace(go.Scatter(
                    x=bp_df["encounter_date"],
                    y=bp_df["systolic_bp"],
                    mode="lines+markers",
                    name="Systolic BP (SBP)",
                    line=dict(color="#38bdf8", width=2),
                    marker=dict(size=6)
                ))
                fig_bp.add_trace(go.Scatter(
                    x=bp_df["encounter_date"],
                    y=bp_df["diastolic_bp"],
                    mode="lines+markers",
                    name="Diastolic BP (DBP)",
                    line=dict(color="#818cf8", width=2),
                    marker=dict(size=6)
                ))

                # Highlight abnormal SBP points in Red
                sbp_abnormal = bp_df[(bp_df["systolic_bp"] < 90) | (bp_df["systolic_bp"] > 120)]
                if not sbp_abnormal.empty:
                    fig_bp.add_trace(go.Scatter(
                        x=sbp_abnormal["encounter_date"],
                        y=sbp_abnormal["systolic_bp"],
                        mode="markers",
                        name="Abnormal SBP",
                        marker=dict(color="#ef4444", size=10, symbol="x"),
                        hovertemplate="Abnormal SBP: %{y} mmHg<br>Date: %{x}<extra></extra>"
                    ))

                # Highlight abnormal DBP points in Red
                dbp_abnormal = bp_df[(bp_df["diastolic_bp"] < 60) | (bp_df["diastolic_bp"] > 80)]
                if not dbp_abnormal.empty:
                    fig_bp.add_trace(go.Scatter(
                        x=dbp_abnormal["encounter_date"],
                        y=dbp_abnormal["diastolic_bp"],
                        mode="markers",
                        name="Abnormal DBP",
                        marker=dict(color="#ef4444", size=10, symbol="circle-open", line=dict(width=2, color="#ef4444")),
                        hovertemplate="Abnormal DBP: %{y} mmHg<br>Date: %{x}<extra></extra>"
                    ))

                fig_bp.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Encounter Date",
                    yaxis_title="Pressure (mmHg)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_dark"
                )
                st.plotly_chart(fig_bp, use_container_width=True)

        # 2. Glucose Trends Chart
        with col_v2:
            st.markdown("##### Blood Glucose Trends")
            gl_df = vitals_df.dropna(subset=["glucose"])
            if gl_df.empty:
                st.info("No glucose readings available.")
            else:
                fig_gl = go.Figure()

                # Shaded Normal Glucose (70-140)
                fig_gl.add_hrect(
                    y0=70, y1=140,
                    fillcolor="rgba(34, 197, 94, 0.12)",
                    line_width=0,
                    annotation_text="Normal Glucose (70-140)",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="#10b981"
                )

                # Glucose trend line
                fig_gl.add_trace(go.Scatter(
                    x=gl_df["encounter_date"],
                    y=gl_df["glucose"],
                    mode="lines+markers",
                    name="Glucose",
                    line=dict(color="#f59e0b", width=2),
                    marker=dict(size=6)
                ))

                # Out of range glucose points in Red
                gl_abnormal = gl_df[(gl_df["glucose"] < 70) | (gl_df["glucose"] > 140)]
                if not gl_abnormal.empty:
                    fig_gl.add_trace(go.Scatter(
                        x=gl_abnormal["encounter_date"],
                        y=gl_abnormal["glucose"],
                        mode="markers",
                        name="Out of Range Glucose",
                        marker=dict(color="#ef4444", size=10, symbol="diamond"),
                        hovertemplate="Out-of-range Glucose: %{y} mg/dL<br>Date: %{x}<extra></extra>"
                    ))

                fig_gl.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Encounter Date",
                    yaxis_title="Glucose (mg/dL)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_dark"
                )
                st.plotly_chart(fig_gl, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Lab Trends
# ---------------------------------------------------------------------------
with tab_labs:
    st.subheader("Laboratory Trends & Reference Thresholds")
    if p_labs.empty:
        st.info("No laboratory records found for this patient.")
    else:
        labs_sorted = p_labs.dropna(subset=["lab_date"]).sort_values(by="lab_date")

        # 1. Abnormal Latest Values Flagging Table
        latest_lab_row = labs_sorted.iloc[-1]
        
        lab_metrics_spec = [
            ("HbA1c (%)", "hba1c_pct", "<= 6.5 %", lambda v: "High" if v > 6.5 else "Normal"),
            ("Total Cholesterol (mg/dL)", "total_cholesterol", "<= 200 mg/dL", lambda v: "High" if v > 200 else "Normal"),
            ("LDL Cholesterol (mg/dL)", "ldl", "<= 100 mg/dL", lambda v: "High" if v > 100 else "Normal"),
            ("HDL Cholesterol (mg/dL)", "hdl", ">= 40 mg/dL", lambda v: "Low" if v < 40 else "Normal"),
            ("Creatinine (mg/dL)", "creatinine_mg_dl", "<= 1.2 mg/dL", lambda v: "High" if v > 1.2 else "Normal"),
            ("Hemoglobin (g/dL)", "hemoglobin_g_dl", ">= 12.0 g/dL", lambda v: "Low" if v < 12.0 else "Normal")
        ]

        summary_lab_rows = []
        for label, col_key, ref_str, status_fn in lab_metrics_spec:
            if col_key in latest_lab_row and pd.notna(latest_lab_row[col_key]):
                val = float(latest_lab_row[col_key])
                stat = status_fn(val)
                summary_lab_rows.append({
                    "Test": label,
                    "Latest Value": f"{val:.2f}" if "creatinine" in col_key or "hba1c" in col_key else f"{val:.1f}",
                    "Reference Target": ref_str,
                    "Status": stat
                })

        if summary_lab_rows:
            st.markdown("##### Latest Lab Results Summary")
            lab_summary_df = pd.DataFrame(summary_lab_rows)

            def highlight_status(val):
                if val in ["High", "Low"]:
                    return "background-color: rgba(239, 68, 68, 0.25); color: #fca5a5; font-weight: bold;"
                return "color: #34d399;"

            lab_styler = lab_summary_df.style.map(highlight_status, subset=["Status"]) if hasattr(lab_summary_df.style, "map") else lab_summary_df.style.applymap(highlight_status, subset=["Status"])
            st.dataframe(
                lab_styler,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("##### Longitudinal Trends with Reference Limits")
        lcol1, lcol2 = st.columns(2)

        # HbA1c Chart
        with lcol1:
            fig_hba1c = px.line(
                labs_sorted.dropna(subset=["hba1c_pct"]),
                x="lab_date",
                y="hba1c_pct",
                markers=True,
                title="HbA1c (%) Trend",
                template="plotly_dark"
            )
            fig_hba1c.add_hline(
                y=6.5, line_dash="dash", line_color="#ef4444",
                annotation_text="Diagnostic Threshold (6.5%)",
                annotation_position="top right"
            )
            fig_hba1c.update_traces(line_color="#ec4899", marker=dict(size=7))
            fig_hba1c.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Lab Date", yaxis_title="HbA1c (%)")
            st.plotly_chart(fig_hba1c, use_container_width=True)

        # Cholesterol & Lipid Chart
        with lcol2:
            fig_lipids = go.Figure()
            if "total_cholesterol" in labs_sorted.columns:
                fig_lipids.add_trace(go.Scatter(
                    x=labs_sorted["lab_date"], y=labs_sorted["total_cholesterol"],
                    mode="lines+markers", name="Total Cholesterol", line=dict(color="#8b5cf6")
                ))
                fig_lipids.add_hline(
                    y=200, line_dash="dash", line_color="#8b5cf6",
                    annotation_text="Max Total Chol (200)", annotation_position="top left"
                )
            if "ldl" in labs_sorted.columns:
                fig_lipids.add_trace(go.Scatter(
                    x=labs_sorted["lab_date"], y=labs_sorted["ldl"],
                    mode="lines+markers", name="LDL", line=dict(color="#f43f5e")
                ))
                fig_lipids.add_hline(
                    y=100, line_dash="dot", line_color="#f43f5e",
                    annotation_text="Max LDL (100)", annotation_position="bottom left"
                )
            if "hdl" in labs_sorted.columns:
                fig_lipids.add_trace(go.Scatter(
                    x=labs_sorted["lab_date"], y=labs_sorted["hdl"],
                    mode="lines+markers", name="HDL", line=dict(color="#10b981")
                ))

            fig_lipids.update_layout(
                title="Lipid Profile (mg/dL)",
                height=280,
                margin=dict(l=20, r=20, t=35, b=20),
                xaxis_title="Lab Date",
                yaxis_title="mg/dL",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_lipids, use_container_width=True)

        lcol3, lcol4 = st.columns(2)

        # Creatinine Chart
        with lcol3:
            fig_creat = px.line(
                labs_sorted.dropna(subset=["creatinine_mg_dl"]),
                x="lab_date",
                y="creatinine_mg_dl",
                markers=True,
                title="Serum Creatinine (mg/dL)",
                template="plotly_dark"
            )
            fig_creat.add_hline(
                y=1.2, line_dash="dash", line_color="#ef4444",
                annotation_text="Normal Limit (1.2 mg/dL)",
                annotation_position="top right"
            )
            fig_creat.update_traces(line_color="#06b6d4", marker=dict(size=7))
            fig_creat.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Lab Date", yaxis_title="Creatinine (mg/dL)")
            st.plotly_chart(fig_creat, use_container_width=True)

        # Hemoglobin Chart
        with lcol4:
            fig_hb = px.line(
                labs_sorted.dropna(subset=["hemoglobin_g_dl"]),
                x="lab_date",
                y="hemoglobin_g_dl",
                markers=True,
                title="Hemoglobin (g/dL)",
                template="plotly_dark"
            )
            fig_hb.add_hline(
                y=12.0, line_dash="dash", line_color="#f59e0b",
                annotation_text="Anemia Cutoff (12.0 g/dL)",
                annotation_position="bottom right"
            )
            fig_hb.update_traces(line_color="#10b981", marker=dict(size=7))
            fig_hb.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Lab Date", yaxis_title="Hemoglobin (g/dL)")
            st.plotly_chart(fig_hb, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: Timeline
# ---------------------------------------------------------------------------
with tab_timeline:
    st.subheader("Longitudinal Patient Timeline")
    st.markdown("Interactive multi-track timeline capturing encounters, clinical diagnoses, and laboratory visits.")

    timeline_events = []

    # 1. Encounters
    if not p_encounters.empty:
        for _, enc in p_encounters.iterrows():
            if pd.notna(enc.get("encounter_date")):
                e_type = str(enc.get("encounter_type", "Outpatient")).strip().title()
                sbp = enc.get("systolic_bp")
                dbp = enc.get("diastolic_bp")
                gluc = enc.get("glucose")
                vitals_str = f"BP: {int(sbp)}/{int(dbp)} | Glucose: {gluc}" if pd.notna(sbp) and pd.notna(dbp) else "Vitals unrecorded"
                timeline_events.append({
                    "date": pd.to_datetime(enc["encounter_date"]),
                    "category": "Encounter",
                    "type": e_type,
                    "label": f"Encounter: {e_type}",
                    "details": f"Stay: {enc.get('length_of_stay_days', 0)} days | Cost: ₹{enc.get('cost_inr', 0):,.2f}<br>{vitals_str}",
                    "symbol": "circle",
                    "size": 12
                })

    # 2. Diagnoses (Diamond markers)
    if not p_conditions.empty:
        for _, cond in p_conditions.iterrows():
            if pd.notna(cond.get("diagnosis_date")):
                timeline_events.append({
                    "date": pd.to_datetime(cond["diagnosis_date"]),
                    "category": "Diagnosis",
                    "type": "New Diagnosis",
                    "label": f"Dx: {cond.get('condition')}",
                    "details": f"Condition: {cond.get('condition')}<br>ICD-10: {cond.get('icd10')}",
                    "symbol": "diamond",
                    "size": 14
                })

    # 3. Lab Visits (Square / circle markers)
    if not p_labs.empty:
        for _, lab in p_labs.iterrows():
            if pd.notna(lab.get("lab_date")):
                timeline_events.append({
                    "date": pd.to_datetime(lab["lab_date"]),
                    "category": "Lab",
                    "type": "Laboratory Panel",
                    "label": "Lab Panel Visit",
                    "details": f"HbA1c: {lab.get('hba1c_pct')}% | Chol: {lab.get('total_cholesterol')} | Creat: {lab.get('creatinine_mg_dl')}",
                    "symbol": "square",
                    "size": 10
                })

    if not timeline_events:
        st.info("No timeline events recorded for this patient.")
    else:
        tl_df = pd.DataFrame(timeline_events).sort_values(by="date")

        color_map = {
            "Inpatient": "#f59e0b",
            "Outpatient": "#38bdf8",
            "Emergency": "#ef4444",
            "New Diagnosis": "#c084fc",
            "Laboratory Panel": "#34d399"
        }

        fig_tl = go.Figure()

        for event_type, group in tl_df.groupby("type"):
            col = color_map.get(event_type, "#94a3b8")
            marker_sym = group["symbol"].iloc[0]
            marker_size = group["size"].iloc[0]

            fig_tl.add_trace(go.Scatter(
                x=group["date"],
                y=group["category"],
                mode="markers",
                name=event_type,
                marker=dict(
                    symbol=marker_sym,
                    size=marker_size,
                    color=col,
                    line=dict(width=1, color="white")
                ),
                text=group["label"],
                customdata=group["details"],
                hovertemplate="<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>%{customdata}<extra></extra>"
            ))

        fig_tl.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date",
            yaxis_title="Event Track",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tl, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 6: Encounters
# ---------------------------------------------------------------------------
with tab_enc:
    st.subheader("Hospital Encounters & Utilization")
    if p_encounters.empty:
        st.info("No encounter records available.")
    else:
        # Top KPI Metrics
        total_cost = float(p_encounters["cost_inr"].sum()) if "cost_inr" in p_encounters.columns else 0.0
        total_days = int(p_encounters["length_of_stay_days"].sum()) if "length_of_stay_days" in p_encounters.columns else 0
        total_encs = len(p_encounters)
        emerg_count = int((p_encounters["encounter_type"].astype(str).str.strip().str.lower() == "emergency").sum())

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Total Encounters", f"{total_encs}")
        mcol2.metric("Emergency Visits", f"{emerg_count}")
        mcol3.metric("Total Hospital Days", f"{total_days} days")
        mcol4.metric("Total Healthcare Cost", f"₹{total_cost:,.2f}")

        st.markdown("##### Detailed Encounters Log")
        enc_display = p_encounters[[
            "encounter_date", "encounter_type", "systolic_bp", "diastolic_bp", "glucose", "length_of_stay_days", "cost_inr"
        ]].copy()

        enc_display["BP (mmHg)"] = enc_display.apply(
            lambda r: f"{int(r['systolic_bp'])}/{int(r['diastolic_bp'])}" if pd.notna(r['systolic_bp']) and pd.notna(r['diastolic_bp']) else "N/A",
            axis=1
        )
        enc_display["encounter_date"] = enc_display["encounter_date"].dt.strftime("%Y-%m-%d")
        enc_display["cost_inr"] = enc_display["cost_inr"].apply(lambda v: f"₹{v:,.2f}" if pd.notna(v) else "₹0.00")

        enc_display_clean = enc_display[[
            "encounter_date", "encounter_type", "BP (mmHg)", "glucose", "length_of_stay_days", "cost_inr"
        ]].rename(columns={
            "encounter_date": "Date",
            "encounter_type": "Encounter Type",
            "glucose": "Glucose (mg/dL)",
            "length_of_stay_days": "Stay (Days)",
            "cost_inr": "Billed Cost (INR)"
        })

        st.dataframe(enc_display_clean, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 4. Download Patient Summary
# ---------------------------------------------------------------------------

st.divider()

pdf_data = generate_pdf_summary(
    patient=patient_row,
    summary_text=clinical_summary_text,
    p_conditions=p_conditions,
    p_medications=p_medications,
    p_encounters=p_encounters,
    p_labs=p_labs
)

download_filename = f"Patient_Summary_{selected_pid}_{datetime.now().strftime('%Y%m%d')}.pdf" if REPORTLAB_AVAILABLE else f"Patient_Summary_{selected_pid}.txt"
download_mime = "application/pdf" if REPORTLAB_AVAILABLE else "text/plain"

d_col1, d_col2 = st.columns([1, 4])
with d_col1:
    st.download_button(
        label="📄 Download Patient Summary",
        data=pdf_data,
        file_name=download_filename,
        mime=download_mime,
        use_container_width=True
    )
with d_col2:
    st.caption("Exports a formalized, one-page clinical patient chart summary complete with demographics, medical history, active medications, and utilization metrics.")

render_sidebar_footer()

