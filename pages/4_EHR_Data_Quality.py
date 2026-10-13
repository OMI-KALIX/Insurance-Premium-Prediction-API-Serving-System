"""
pages/EHR_Data_Quality.py
-------------------------
EHR Data Quality & Integrity Auditor.
Performs multidimensional audits across raw and cleaned EHR datasets:
1. Completeness (% non-null per column)
2. Duplicates (exact rows and duplicate patient_id)
3. Value Validity (clinical physiological ranges)
4. Format Consistency (gender codes, birth date, ISO dates)
5. Referential Integrity (orphan child records)
6. ICD-10 Validity (regex format & Charlson mapping)
7. Temporal Validity (future dates & diagnosis preceding birth)

Provides:
- Table-level and overall quality scores with Plotly gauge indicators
- Interactive drilldown tabs with issue breakdown charts and exportable CSVs
- "Cleaning impact" before vs. after comparison
- Dynamic rule-based "Recommended fixes" action items
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    page_title="EHR Data Quality | HealthSure",
    page_icon="🛡️",
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
    
    .dq-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .fix-item {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    
    .fix-item-success {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Declarative Rule Definitions (Extensible Configuration)
# ---------------------------------------------------------------------------

# Rule definition: (Table, Column, Min_Value, Max_Value, Display_Label)
# Adding a new physiological or numerical rule requires just one line here.
VALIDITY_RULES: List[Tuple[str, str, float, float, str]] = [
    ("patients", "height_cm", 100.0, 220.0, "Height (cm)"),
    ("patients", "weight_kg", 20.0, 250.0, "Weight (kg)"),
    ("encounters", "systolic_bp", 70.0, 250.0, "Systolic BP (mmHg)"),
    ("encounters", "diastolic_bp", 40.0, 150.0, "Diastolic BP (mmHg)"),
    ("encounters", "glucose", 40.0, 500.0, "Blood Glucose (mg/dL)"),
    ("labs", "hba1c_pct", 3.0, 15.0, "HbA1c (%)"),
    ("labs", "creatinine_mg_dl", 0.2, 10.0, "Creatinine (mg/dL)")
]

TABLE_REGISTRY: List[str] = ["patients", "conditions", "encounters", "labs", "medications", "charlson"]


# ---------------------------------------------------------------------------
# Modular Data Quality Check Functions
# ---------------------------------------------------------------------------

def check_completeness(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Check 1: Completeness
    Calculates % non-null values per column across all tables.
    Returns: (issues_df, overall_pass_rate, table_scores)
    """
    issues: List[Dict[str, Any]] = []
    total_cells = 0
    null_cells = 0
    table_scores: Dict[str, float] = {}

    for tname in TABLE_REGISTRY:
        if tname not in dfs or dfs[tname] is None or dfs[tname].empty:
            continue
        df = dfs[tname]
        t_cells = int(df.size)
        t_nulls = int(df.isna().sum().sum())
        total_cells += t_cells
        null_cells += t_nulls
        
        t_rate = (1.0 - (t_nulls / max(1, t_cells))) * 100.0
        table_scores[tname] = round(t_rate, 2)

        for col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                issues.append({
                    "Table": tname,
                    "Check": "Completeness",
                    "Column": col,
                    "Patient ID": "Multiple",
                    "Offending Value": f"{n_null} missing cells",
                    "Expected": "Non-null value",
                    "Description": f"Column '{col}' in {tname} has {n_null} missing entries ({n_null/len(df)*100:.1f}%)"
                })

    overall_pass = round((1.0 - (null_cells / max(1, total_cells))) * 100.0, 2)
    return pd.DataFrame(issues), overall_pass, table_scores


def check_duplicates(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Check 2: Duplicates
    Detects exact duplicate rows across tables and duplicate patient_id in patients.
    Returns: (issues_df, overall_pass_rate, table_scores)
    """
    issues: List[Dict[str, Any]] = []
    total_rows = 0
    total_duplicates = 0
    table_scores: Dict[str, float] = {}

    for tname in TABLE_REGISTRY:
        if tname not in dfs or dfs[tname] is None or dfs[tname].empty:
            continue
        df = dfs[tname]
        n_rows = len(df)
        total_rows += n_rows
        
        t_dups = int(df.duplicated().sum())

        # Check duplicate patient_ids specifically for patients table
        if tname == "patients" and "patient_id" in df.columns:
            pid_dups = int(df["patient_id"].duplicated().sum())
            t_dups += pid_dups
            if pid_dups > 0:
                dup_pids = df[df["patient_id"].duplicated()]["patient_id"].tolist()
                for pid in dup_pids:
                    issues.append({
                        "Table": tname,
                        "Check": "Duplicates",
                        "Column": "patient_id",
                        "Patient ID": str(pid),
                        "Offending Value": str(pid),
                        "Expected": "Unique patient identifier",
                        "Description": f"Duplicate primary key patient_id: '{pid}' in {tname}"
                    })

        # Exact row duplicates
        if df.duplicated().sum() > 0:
            dup_records = df[df.duplicated(keep=False)]
            for _, r in dup_records.head(50).iterrows():
                issues.append({
                    "Table": tname,
                    "Check": "Duplicates",
                    "Column": "All",
                    "Patient ID": str(r.get("patient_id", "N/A")),
                    "Offending Value": "Exact Duplicate Row",
                    "Expected": "Unique Row",
                    "Description": f"Exact duplicate record found in {tname}"
                })

        total_duplicates += t_dups
        table_scores[tname] = round(max(0.0, (1.0 - (t_dups / max(1, n_rows))) * 100.0), 2)

    overall_pass = round(max(0.0, (1.0 - (total_duplicates / max(1, total_rows))) * 100.0), 2)
    return pd.DataFrame(issues), overall_pass, table_scores


def check_validity(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Check 3: Value Validity
    Audits clinical variables against standard physiological limits using VALIDITY_RULES.
    Returns: (issues_df, overall_pass_rate, table_scores)
    """
    issues: List[Dict[str, Any]] = []
    total_checked = 0
    invalid_count = 0
    table_counts: Dict[str, List[int]] = {t: [0, 0] for t in TABLE_REGISTRY}

    for tname, col, min_v, max_v, label in VALIDITY_RULES:
        if tname in dfs and col in dfs[tname].columns:
            series = pd.to_numeric(dfs[tname][col], errors="coerce")
            valid_vals = series.dropna()
            n_vals = len(valid_vals)
            total_checked += n_vals
            table_counts[tname][0] += n_vals

            outliers = dfs[tname][(series < min_v) | (series > max_v)]
            n_outliers = len(outliers)
            invalid_count += n_outliers
            table_counts[tname][1] += n_outliers

            for _, r in outliers.iterrows():
                issues.append({
                    "Table": tname,
                    "Check": "Value Validity",
                    "Column": col,
                    "Patient ID": str(r.get("patient_id", "N/A")),
                    "Offending Value": str(r[col]),
                    "Expected": f"[{min_v}, {max_v}]",
                    "Description": f"{label} = {r[col]} is outside expected physiological range [{min_v}, {max_v}]"
                })

    table_scores: Dict[str, float] = {}
    for t, (chk, inv) in table_counts.items():
        if chk > 0:
            table_scores[t] = round((1.0 - (inv / chk)) * 100.0, 2)
        else:
            table_scores[t] = 100.0

    overall_pass = round((1.0 - (invalid_count / max(1, total_checked))) * 100.0, 2)
    return pd.DataFrame(issues), overall_pass, table_scores


def check_format_consistency(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Check 4: Format Consistency
    Checks gender codes (Male/Female), birth_date format (%d-%m-%Y), and child table dates (%Y-%m-%d).
    Returns: (issues_df, overall_pass_rate, table_scores)
    """
    issues: List[Dict[str, Any]] = []
    total_checked = 0
    format_errors = 0
    t_counts: Dict[str, List[int]] = {t: [0, 0] for t in TABLE_REGISTRY}

    # 1. Gender in patients table (expected Male or Female)
    if "patients" in dfs and "gender" in dfs["patients"].columns:
        g_series = dfs["patients"]["gender"].dropna().astype(str)
        t_counts["patients"][0] += len(g_series)
        invalid_g = dfs["patients"][~g_series.isin(["Male", "Female"])]
        t_counts["patients"][1] += len(invalid_g)
        for _, r in invalid_g.iterrows():
            issues.append({
                "Table": "patients",
                "Check": "Format Consistency",
                "Column": "gender",
                "Patient ID": str(r.get("patient_id", "N/A")),
                "Offending Value": str(r["gender"]),
                "Expected": "'Male' or 'Female'",
                "Description": f"Non-standard gender encoding '{r['gender']}'"
            })

    # 2. birth_date in patients (expected %d-%m-%Y)
    if "patients" in dfs and "birth_date" in dfs["patients"].columns:
        b_series = dfs["patients"]["birth_date"].dropna()
        t_counts["patients"][0] += len(b_series)
        
        # If already parsed as Timestamp / datetime, it is considered compliant
        if not pd.api.types.is_datetime64_any_dtype(b_series):
            dd_mm_yyyy_re = re.compile(r"^\d{2}-\d{2}-\d{4}$")
            for _, r in dfs["patients"].iterrows():
                b_str = str(r["birth_date"]).strip()
                if not dd_mm_yyyy_re.match(b_str):
                    t_counts["patients"][1] += 1
                    issues.append({
                        "Table": "patients",
                        "Check": "Format Consistency",
                        "Column": "birth_date",
                        "Patient ID": str(r.get("patient_id", "N/A")),
                        "Offending Value": b_str,
                        "Expected": "dd-mm-yyyy format",
                        "Description": f"Birth date '{b_str}' does not conform to dd-mm-yyyy standard"
                    })

    # 3. Date columns in encounters, conditions, labs, medications (expected %Y-%m-%d)
    iso_date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    other_date_targets = [
        ("encounters", "encounter_date"),
        ("conditions", "diagnosis_date"),
        ("labs", "lab_date"),
        ("medications", "start_date")
    ]
    for tname, cname in other_date_targets:
        if tname in dfs and cname in dfs[tname].columns:
            d_series = dfs[tname][cname].dropna()
            t_counts[tname][0] += len(d_series)
            
            if not pd.api.types.is_datetime64_any_dtype(d_series):
                for _, r in dfs[tname].iterrows():
                    d_str = str(r[cname]).strip()
                    if not iso_date_re.match(d_str):
                        t_counts[tname][1] += 1
                        issues.append({
                            "Table": tname,
                            "Check": "Format Consistency",
                            "Column": cname,
                            "Patient ID": str(r.get("patient_id", "N/A")),
                            "Offending Value": d_str,
                            "Expected": "yyyy-mm-dd ISO format",
                            "Description": f"Date '{d_str}' does not match standard yyyy-mm-dd format"
                        })

    table_scores: Dict[str, float] = {}
    for t, (chk, err) in t_counts.items():
        total_checked += chk
        format_errors += err
        if chk > 0:
            table_scores[t] = round((1.0 - (err / chk)) * 100.0, 2)
        else:
            table_scores[t] = 100.0

    overall_pass = round((1.0 - (format_errors / max(1, total_checked))) * 100.0, 2)
    return pd.DataFrame(issues), overall_pass, table_scores


def check_referential_integrity(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Check 5: Referential Integrity
    Identifies orphan records in child tables whose patient_id does not exist in patients.
    Returns: (issues_df, overall_pass_rate, table_scores)
    """
    issues: List[Dict[str, Any]] = []
    total_checked = 0
    orphan_count = 0
    table_scores: Dict[str, float] = {t: 100.0 for t in TABLE_REGISTRY}

    if "patients" not in dfs or "patient_id" not in dfs["patients"].columns:
        return pd.DataFrame(issues), 100.0, table_scores

    valid_pids = set(dfs["patients"]["patient_id"].dropna().unique())

    for tname in ["encounters", "conditions", "labs", "medications"]:
        if tname in dfs and "patient_id" in dfs[tname].columns:
            n_rows = len(dfs[tname])
            total_checked += n_rows
            orphans = dfs[tname][~dfs[tname]["patient_id"].isin(valid_pids)]
            n_orphans = len(orphans)
            orphan_count += n_orphans

            table_scores[tname] = round((1.0 - (n_orphans / max(1, n_rows))) * 100.0, 2)

            for _, r in orphans.iterrows():
                issues.append({
                    "Table": tname,
                    "Check": "Referential Integrity",
                    "Column": "patient_id",
                    "Patient ID": str(r.get("patient_id", "N/A")),
                    "Offending Value": str(r.get("patient_id", "N/A")),
                    "Expected": "Valid Patient ID in Patients table",
                    "Description": f"Orphan foreign key: patient_id '{r.get('patient_id')}' not found in patients"
                })

    overall_pass = round((1.0 - (orphan_count / max(1, total_checked))) * 100.0, 2)
    return pd.DataFrame(issues), overall_pass, table_scores


def check_icd10_validity(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Check 6: ICD-10 Validity
    Validates ICD-10 format against regex ^[A-Z][0-9]{2}$ and checks presence in charlson_mapping.
    Returns: (issues_df, overall_pass_rate, table_scores)
    """
    issues: List[Dict[str, Any]] = []
    total_checked = 0
    invalid_count = 0
    table_scores: Dict[str, float] = {t: 100.0 for t in TABLE_REGISTRY}

    icd_regex = re.compile(r"^[A-Z][0-9]{2}$")
    charlson_codes = set()
    if "charlson" in dfs and "icd10" in dfs["charlson"].columns:
        charlson_codes = set(dfs["charlson"]["icd10"].dropna().astype(str).str.strip().unique())

    for tname in ["conditions", "medications"]:
        if tname in dfs and "icd10" in dfs[tname].columns:
            icds = dfs[tname]["icd10"].dropna().astype(str).str.strip()
            n_codes = len(icds)
            total_checked += n_codes
            t_invalid = 0

            for _, r in dfs[tname].iterrows():
                code = str(r.get("icd10", "")).strip()
                if not icd_regex.match(code):
                    t_invalid += 1
                    issues.append({
                        "Table": tname,
                        "Check": "ICD-10 Validity",
                        "Column": "icd10",
                        "Patient ID": str(r.get("patient_id", "N/A")),
                        "Offending Value": code,
                        "Expected": "^[A-Z][0-9]{2}$ regex",
                        "Description": f"Non-conforming ICD-10 format '{code}' (expected 3-character root)"
                    })
                elif charlson_codes and code not in charlson_codes:
                    t_invalid += 1
                    issues.append({
                        "Table": tname,
                        "Check": "ICD-10 Validity",
                        "Column": "icd10",
                        "Patient ID": str(r.get("patient_id", "N/A")),
                        "Offending Value": code,
                        "Expected": "Mapped in charlson_mapping.csv",
                        "Description": f"ICD-10 '{code}' not present in Charlson Comorbidity Mapping"
                    })

            invalid_count += t_invalid
            table_scores[tname] = round((1.0 - (t_invalid / max(1, n_codes))) * 100.0, 2)

    overall_pass = round((1.0 - (invalid_count / max(1, total_checked))) * 100.0, 2)
    return pd.DataFrame(issues), overall_pass, table_scores


def check_temporal_validity(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Check 7: Temporal Validity
    Audits dates later than today and diagnosis_date earlier than patient birth_date.
    Returns: (issues_df, overall_pass_rate, table_scores)
    """
    issues: List[Dict[str, Any]] = []
    total_checked = 0
    invalid_count = 0
    today_dt = pd.Timestamp.now().normalize()
    table_scores: Dict[str, float] = {t: 100.0 for t in TABLE_REGISTRY}

    date_targets = [
        ("patients", "birth_date"),
        ("encounters", "encounter_date"),
        ("conditions", "diagnosis_date"),
        ("labs", "lab_date"),
        ("medications", "start_date")
    ]

    # 1. Dates after today
    for tname, col in date_targets:
        if tname in dfs and col in dfs[tname].columns:
            dates = pd.to_datetime(dfs[tname][col], errors="coerce")
            valid_dates = dates.dropna()
            n_d = len(valid_dates)
            total_checked += n_d

            future_mask = valid_dates > today_dt
            n_future = int(future_mask.sum())
            invalid_count += n_future

            future_rows = dfs[tname].loc[valid_dates[future_mask].index]
            for _, r in future_rows.iterrows():
                issues.append({
                    "Table": tname,
                    "Check": "Temporal Validity",
                    "Column": col,
                    "Patient ID": str(r.get("patient_id", "N/A")),
                    "Offending Value": str(r[col]),
                    "Expected": "<= Current Date (Today)",
                    "Description": f"Anachronistic future date '{r[col]}' in {tname}"
                })

            table_scores[tname] = round((1.0 - (n_future / max(1, n_d))) * 100.0, 2)

    # 2. Diagnosis date earlier than birth date
    if "conditions" in dfs and "patients" in dfs and "diagnosis_date" in dfs["conditions"].columns and "birth_date" in dfs["patients"].columns:
        p_dates = dfs["patients"][["patient_id", "birth_date"]].copy()
        p_dates["bdate"] = pd.to_datetime(p_dates["birth_date"], errors="coerce")

        c_dates = dfs["conditions"][["patient_id", "diagnosis_date", "condition"]].copy()
        c_dates["ddate"] = pd.to_datetime(c_dates["diagnosis_date"], errors="coerce")

        merged = c_dates.merge(p_dates, on="patient_id", how="inner").dropna(subset=["bdate", "ddate"])
        total_checked += len(merged)

        precedes = merged[merged["ddate"] < merged["bdate"]]
        invalid_count += len(precedes)

        for _, r in precedes.iterrows():
            issues.append({
                "Table": "conditions",
                "Check": "Temporal Validity",
                "Column": "diagnosis_date",
                "Patient ID": str(r["patient_id"]),
                "Offending Value": f"Diagnosis: {r['diagnosis_date']} vs Birth: {r['birth_date']}",
                "Expected": "diagnosis_date >= birth_date",
                "Description": f"Condition '{r.get('condition')}' recorded prior to patient birth date"
            })

    overall_pass = round((1.0 - (invalid_count / max(1, total_checked))) * 100.0, 2)
    return pd.DataFrame(issues), overall_pass, table_scores


# ---------------------------------------------------------------------------
# Comprehensive Audit Runner & Scoring Engine
# ---------------------------------------------------------------------------

CHECK_SUITE: List[Tuple[str, Callable[[Dict[str, pd.DataFrame]], Tuple[pd.DataFrame, float, Dict[str, float]]]]] = [
    ("Completeness", check_completeness),
    ("Duplicates", check_duplicates),
    ("Validity", check_validity),
    ("Format Consistency", check_format_consistency),
    ("Referential Integrity", check_referential_integrity),
    ("ICD-10 Validity", check_icd10_validity),
    ("Temporal Validity", check_temporal_validity)
]


def run_full_audit(dfs: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], pd.DataFrame, float, Dict[str, float]]:
    """
    Execute all 7 quality checks on a dictionary of DataFrames.
    Computes per-table quality scores and overall composite quality score.
    """
    all_issues_list: List[pd.DataFrame] = []
    category_results: Dict[str, Any] = {}
    table_scores_accumulator: Dict[str, List[float]] = {t: [] for t in TABLE_REGISTRY}

    for cat_name, check_fn in CHECK_SUITE:
        issues_df, pass_rate, table_breakdown = check_fn(dfs)
        category_results[cat_name] = {
            "issues_df": issues_df,
            "pass_rate": pass_rate,
            "table_breakdown": table_breakdown
        }
        if not issues_df.empty:
            all_issues_list.append(issues_df)

        for tname, score in table_breakdown.items():
            if tname in table_scores_accumulator:
                table_scores_accumulator[tname].append(score)

    combined_issues = pd.concat(all_issues_list, ignore_index=True) if all_issues_list else pd.DataFrame(
        columns=["Table", "Check", "Column", "Patient ID", "Offending Value", "Expected", "Description"]
    )

    # Per-table scores = average pass rate across applicable checks
    per_table_scores: Dict[str, float] = {}
    for tname, scores in table_scores_accumulator.items():
        if scores:
            per_table_scores[tname] = round(float(np.mean(scores)), 1)
        else:
            per_table_scores[tname] = 100.0

    # Overall composite score = mean of table scores
    overall_score = round(float(np.mean(list(per_table_scores.values()))), 1)

    return category_results, combined_issues, overall_score, per_table_scores


# ---------------------------------------------------------------------------
# Data Loading & Audit Execution
# ---------------------------------------------------------------------------

try:
    cleaned_data, raw_data = load_ehr()
    raw_results, raw_issues_all, raw_overall_score, raw_table_scores = run_full_audit(raw_data)
    clean_results, clean_issues_all, clean_overall_score, clean_table_scores = run_full_audit(cleaned_data)
except Exception as e:
    st.error(f"Error loading datasets or auditing data: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Header & Quality Score Gauges
# ---------------------------------------------------------------------------

st.markdown("## 🛡️ EHR Data Quality & Integrity Auditor")
st.markdown("Automated clinical data auditing engine comparing raw source records against cleaned production datasets across 7 validation dimensions.")

col_gauge, col_cards = st.columns([1.2, 2.0])

def build_gauge_chart(score: float, title: str) -> go.Figure:
    """Render a modern Plotly indicator gauge."""
    if score >= 90:
        bar_color = "#10b981"
    elif score >= 70:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': "white"}},
        number={'suffix': "%", 'font': {'size': 36, 'color': bar_color, 'weight': 800}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 70], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [70, 90], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [90, 100], 'color': "rgba(16, 185, 129, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "#38bdf8", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

with col_gauge:
    fig_overall = build_gauge_chart(raw_overall_score, "Raw Data Quality Score")
    st.plotly_chart(fig_overall, use_container_width=True)

with col_cards:
    st.markdown("##### Table-Level Quality Scorecards")
    
    tcols1, tcols2, tcols3 = st.columns(3)
    t_list = list(raw_table_scores.items())
    
    for i, (tname, tscore) in enumerate(t_list):
        target_col = [tcols1, tcols2, tcols3][i % 3]
        clean_score = clean_table_scores.get(tname, tscore)
        delta_str = f"{clean_score - tscore:+.1f}% clean"
        
        target_col.metric(
            label=f"📋 {tname.title()}",
            value=f"{tscore:.1f}%",
            delta=delta_str if clean_score != tscore else "Optimal"
        )

st.markdown("---")


# ---------------------------------------------------------------------------
# Interactive Audit Tabs (7 Dimensions)
# ---------------------------------------------------------------------------

st.subheader("🔍 Deep-Dive Validation Categories")

tab_comp, tab_dup, tab_val, tab_fmt, tab_ref, tab_icd, tab_temp = st.tabs([
    "📊 Completeness",
    "♊ Duplicates",
    "🎯 Value Validity",
    "🔤 Format Consistency",
    "🔗 Referential Integrity",
    "🏷️ ICD-10 Validity",
    "⏳ Temporal Validity"
])

category_tab_mapping = [
    (tab_comp, "Completeness"),
    (tab_dup, "Duplicates"),
    (tab_val, "Value Validity"),
    (tab_fmt, "Format Consistency"),
    (tab_ref, "Referential Integrity"),
    (tab_icd, "ICD-10 Validity"),
    (tab_temp, "Temporal Validity")
]


for current_tab, cat_title in category_tab_mapping:
    with current_tab:
        # Retrieve raw audit for this category
        raw_res = raw_results.get(cat_title, {})
        cat_issues = raw_res.get("issues_df", pd.DataFrame())
        cat_pass_rate = raw_res.get("pass_rate", 100.0)
        table_rates = raw_res.get("table_breakdown", {})

        m1, m2, m3 = st.columns(3)
        m1.metric("Category Pass Rate", f"{cat_pass_rate:.1f}%")
        m2.metric("Detected Issues", f"{len(cat_issues):,}")
        m3.metric("Audited Tables", f"{len(table_rates)}")

        # Visual Chart Breakdown
        if cat_title == "Completeness":
            st.markdown("##### Column-Level Non-Null Fill Rates")
            # Build completeness bar chart
            comp_rows = []
            for tname in TABLE_REGISTRY:
                if tname in raw_data and raw_data[tname] is not None:
                    df = raw_data[tname]
                    for col in df.columns:
                        fill_pct = (1.0 - (df[col].isna().sum() / len(df))) * 100.0
                        comp_rows.append({"Table": tname, "Column": f"{tname}.{col}", "Fill Rate (%)": round(fill_pct, 1)})
            
            comp_df = pd.DataFrame(comp_rows)
            fig_comp = px.bar(
                comp_df,
                x="Column",
                y="Fill Rate (%)",
                color="Table",
                template="plotly_dark",
                title="Completeness (% Non-Null) per Column across All Tables"
            )
            fig_comp.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20), xaxis_tickangle=-45)
            st.plotly_chart(fig_comp, use_container_width=True)

        else:
            # Bar chart of pass rates by table
            tb_df = pd.DataFrame(list(table_rates.items()), columns=["Table", "Pass Rate (%)"])
            fig_tb = px.bar(
                tb_df,
                x="Table",
                y="Pass Rate (%)",
                color="Pass Rate (%)",
                color_continuous_scale="Viridis",
                text="Pass Rate (%)",
                template="plotly_dark",
                title=f"{cat_title} Pass Rate by Table"
            )
            fig_tb.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_tb.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), yaxis_range=[0, 105])
            st.plotly_chart(fig_tb, use_container_width=True)

        # Offending records breakdown
        with st.expander(f"📋 Offending Records & Anomaly Details ({len(cat_issues)} issues found)", expanded=(len(cat_issues) > 0)):
            if cat_issues.empty:
                st.success(f"✅ Zero anomalies detected in {cat_title}! All records passed validation.")
            else:
                st.dataframe(cat_issues, use_container_width=True, hide_index=True)
                
                # CSV Export button
                csv_bytes = cat_issues.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download {cat_title} Issues CSV",
                    data=csv_bytes,
                    file_name=f"{cat_title.lower().replace(' ', '_')}_issues.csv",
                    mime="text/csv",
                    key=f"dl_{cat_title}"
                )


# ---------------------------------------------------------------------------
# Cleaning Impact (Before vs. After Comparison)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("🚀 Cleaning Pipeline Impact: Raw vs. Cleaned Quality")
st.markdown("Evaluating the performance of data transformations, deduplication, date normalization, and imputation implemented in `load_ehr()`.")

impact_records = []
for cat_name, _ in CHECK_SUITE:
    r_issues = len(raw_results[cat_name]["issues_df"])
    c_issues = len(clean_results[cat_name]["issues_df"])
    r_rate = raw_results[cat_name]["pass_rate"]
    c_rate = clean_results[cat_name]["pass_rate"]
    delta_rate = c_rate - r_rate

    impact_records.append({
        "Validation Check": cat_name,
        "Raw Issues Found": r_issues,
        "Cleaned Issues Remaining": c_issues,
        "Raw Pass Rate": f"{r_rate:.2f}%",
        "Cleaned Pass Rate": f"{c_rate:.2f}%",
        "Improvement (Delta)": f"{delta_rate:+.2f}%"
    })

impact_df = pd.DataFrame(impact_records)

icol1, icol2 = st.columns([1.8, 1.2])

with icol1:
    st.dataframe(impact_df, use_container_width=True, hide_index=True)

with icol2:
    overall_delta = clean_overall_score - raw_overall_score
    st.markdown(
        f"""
        <div class="dq-card" style="text-align: center; padding: 28px 20px;">
            <div style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 8px;">Composite Quality Improvement</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #34d399;">
                {overall_delta:+.1f}%
            </div>
            <div style="font-size: 0.95rem; margin-top: 8px;">
                Raw Score: <b>{raw_overall_score:.1f}%</b> &nbsp;➔&nbsp; Cleaned Score: <b>{clean_overall_score:.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Recommended Fixes Generator
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("💡 Recommended Actions & Remediation Plan")
st.caption("Actionable, rule-based clinical remediation items generated from the findings.")

recommendations: List[str] = []

# Generate rule-based recommendations from findings
if len(raw_results["Completeness"]["issues_df"]) > 0:
    recommendations.append(f"<b>Missing Values:</b> Found {len(raw_results['Completeness']['issues_df'])} columns with missing entries. Recommended fix: Apply median imputation for biometric metrics and fill categorical unknowns.")
else:
    recommendations.append("<b>Completeness:</b> All core columns are fully populated with 100% fill rate.")

if len(raw_results["Duplicates"]["issues_df"]) > 0:
    recommendations.append(f"<b>Deduplication:</b> Detected duplicate patient rows. Recommended fix: Enforce primary key constraint on patient_id (`keep='first'`).")
else:
    recommendations.append("<b>Deduplication:</b> Zero duplicate primary keys detected in patient records.")

val_issues = raw_results["Validity"]["issues_df"]
if not val_issues.empty:
    n_vitals = len(val_issues[val_issues["Column"].isin(["systolic_bp", "diastolic_bp", "glucose"])])
    recommendations.append(f"<b>Physiological Outliers:</b> Found {len(val_issues)} out-of-range clinical readings (including {n_vitals} vitals entries). Recommended fix: Clip extreme values or impute with cohort median to prevent model skew.")
else:
    recommendations.append("<b>Validity:</b> All clinical and lab values sit within permissible physiological bounds.")

fmt_issues = raw_results["Format Consistency"]["issues_df"]
if not fmt_issues.empty:
    recommendations.append(f"<b>Date & Code Formatting:</b> Identified {len(fmt_issues)} non-conforming date strings. Recommended fix: Parse with flexible datetime parser and standardize to ISO-8601 (%Y-%m-%d).")
else:
    recommendations.append("<b>Format Consistency:</b> All date strings and demographic categories strictly match target schemas.")

icd_issues = raw_results["ICD-10 Validity"]["issues_df"]
if not icd_issues.empty:
    recommendations.append(f"<b>ICD-10 Nomenclature:</b> Found {len(icd_issues)} non-standard or unmapped ICD-10 diagnostic codes. Recommended fix: Truncate decimal subdivisions to 3-character root categories matching charlson_mapping.csv.")
else:
    recommendations.append("<b>ICD-10 Validity:</b> All recorded diagnosis codes match validated 3-character Charlson roots.")

temp_issues = raw_results["Temporal Validity"]["issues_df"]
if not temp_issues.empty:
    recommendations.append(f"<b>Chronological Integrity:</b> Detected {len(temp_issues)} dates set in the future or preceding patient birth. Recommended fix: Filter out records where event_date > today or event_date < birth_date.")
else:
    recommendations.append("<b>Temporal Integrity:</b> All clinical events occur in proper chronological sequence.")

for rec in recommendations:
    css_class = "fix-item-success" if "100%" in rec or "Zero" in rec or "All" in rec else "fix-item"
    st.markdown(f'<div class="{css_class}">{rec}</div>', unsafe_allow_html=True)
