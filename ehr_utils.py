"""
ehr_utils.py
-------------
Electronic Health Records (EHR) data loading, cleaning, and clinical
classification utilities for Streamlit applications.

Includes:
- load_ehr: Streamlit-cached loader applying data cleaning and validation rules.
- calc_age: Calculates age in years as of today.
- classify_bp: Categorizes blood pressure using American Heart Association (AHA) ranges.
- classify_glucose: Categorizes blood glucose readings.
"""

from __future__ import annotations

import os
import pickle
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def calc_age(
    birth_date: Union[str, date, datetime, pd.Timestamp, pd.Series]
) -> Union[int, float, pd.Series]:
    """
    Calculate age in full years as of today from a birth date.

    Supports strings (%d-%m-%Y or %Y-%m-%d), datetime/date objects,
    pd.Timestamp, or a pandas Series of dates.

    Parameters
    ----------
    birth_date : str, date, datetime, pd.Timestamp, or pd.Series
        The birth date(s) to compute age from.

    Returns
    -------
    int, float, or pd.Series
        Age in years as of today. Returns np.nan for invalid or missing dates.
    """
    today = date.today()

    def _calc_single(b: Any) -> Union[int, float]:
        if b is None or pd.isna(b):
            return np.nan

        dt: Union[date, None] = None
        if isinstance(b, str):
            b_str = b.strip()
            # Try specified dd-mm-yyyy first, then fallback to flexible parse
            try:
                dt = datetime.strptime(b_str, "%d-%m-%Y").date()
            except ValueError:
                try:
                    dt = pd.to_datetime(b_str).date()
                except Exception:
                    return np.nan
        elif isinstance(b, (datetime, pd.Timestamp)):
            dt = b.date()
        elif isinstance(b, date):
            dt = b
        else:
            try:
                dt = pd.to_datetime(b).date()
            except Exception:
                return np.nan

        if dt is None:
            return np.nan

        # Exact age considering month and day threshold
        age = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
        return int(age) if age >= 0 else np.nan

    if isinstance(birth_date, pd.Series):
        return birth_date.apply(_calc_single)
    return _calc_single(birth_date)


def classify_bp(
    sys: Union[float, int, None],
    dia: Union[float, int, None]
) -> Union[str, None]:
    """
    Classify blood pressure based on American Heart Association (AHA) guidelines.

    Categories:
    - Crisis: Systolic > 180 and/or Diastolic > 120 mm Hg
    - Stage 2: Systolic >= 140 or Diastolic >= 90 mm Hg
    - Stage 1: Systolic 130-139 or Diastolic 80-89 mm Hg
    - Elevated: Systolic 120-129 and Diastolic < 80 mm Hg
    - Normal: Systolic < 120 and Diastolic < 80 mm Hg

    When systolic and diastolic readings fall into different categories,
    the higher category is used.

    Parameters
    ----------
    sys : float, int, or None
        Systolic blood pressure (mm Hg).
    dia : float, int, or None
        Diastolic blood pressure (mm Hg).

    Returns
    -------
    str or None
        One of 'Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis',
        or None if input is invalid/missing.
    """
    if sys is None or dia is None or pd.isna(sys) or pd.isna(dia):
        return None

    try:
        s = float(sys)
        d = float(dia)
    except (ValueError, TypeError):
        return None

    # AHA classification hierarchy (highest severity takes precedence)
    if s > 180 or d > 120:
        return "Crisis"
    if s >= 140 or d >= 90:
        return "Stage 2"
    if s >= 130 or d >= 80:
        return "Stage 1"
    if 120 <= s <= 129 and d < 80:
        return "Elevated"
    if s < 120 and d < 80:
        return "Normal"

    # Edge cases (e.g., s < 120 but d is 80-89 -> Stage 1)
    return "Stage 1"


def classify_glucose(value: Union[float, int, None]) -> Union[str, None]:
    """
    Classify blood glucose reading (mg/dL).

    Categories:
    - Low: < 70 mg/dL
    - Normal: 70 - 140 mg/dL
    - High: 141 - 180 mg/dL
    - Critical: > 180 mg/dL

    Parameters
    ----------
    value : float, int, or None
        Blood glucose reading in mg/dL.

    Returns
    -------
    str or None
        One of 'Low', 'Normal', 'High', 'Critical', or None if missing.
    """
    if value is None or pd.isna(value):
        return None

    try:
        v = float(value)
    except (ValueError, TypeError):
        return None

    if v < 70:
        return "Low"
    if 70 <= v <= 140:
        return "Normal"
    if 140 < v <= 180:
        return "High"
    return "Critical"


def get_bmi_category(bmi: Union[float, int, None]) -> Tuple[str, str, str]:
    """
    Determine BMI category, badge text, and CSS badge class.

    Returns:
        Tuple[str, str, str]: (category_name, badge_label, css_badge_class)
    """
    if bmi is None or pd.isna(bmi):
        return "Unknown", "Unknown BMI", "badge-blue"

    try:
        b = float(bmi)
    except (ValueError, TypeError):
        return "Unknown", "Unknown BMI", "badge-blue"

    if b <= 0:
        return "Unknown", "Unknown BMI", "badge-blue"
    if b < 18.5:
        return "Underweight", "Underweight", "badge-blue"
    elif 18.5 <= b < 25.0:
        return "Normal", "Normal Weight", "badge-green"
    elif 25.0 <= b < 30.0:
        return "Overweight", "Overweight", "badge-yellow"
    else:
        return "Obese", "Obese", "badge-red"


# ---------------------------------------------------------------------------
# Path Resolution Helper
# ---------------------------------------------------------------------------

def _resolve_file(data_dir: Path, possible_names: list[str]) -> Path:
    """
    Locate an EHR file by checking primary data directory, fallback directories,
    and multiple potential filename conventions.
    """
    # 1. Check directly in target directory
    for name in possible_names:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate

    # 2. Check known alternate directories relative to project root
    project_root = Path(__file__).resolve().parent
    search_dirs = [
        project_root / "data",
        project_root / "clean",
        Path("data"),
        Path("clean"),
    ]

    for d in search_dirs:
        for name in possible_names:
            candidate = d / name
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"Unable to find any of {possible_names} in '{data_dir}' or fallback directories."
    )


def _parse_date_series(series: pd.Series) -> pd.Series:
    """
    Parse a date Series, trying the expected %d-%m-%Y format first and
    falling back to ISO / standard formats if necessary.
    """
    parsed = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(series, errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


# ---------------------------------------------------------------------------
# Main Data Loader
# ---------------------------------------------------------------------------

@st.cache_data
def load_ehr(
    data_dir: str = "clean"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load EHR datasets from disk, apply cleaning and imputation rules, and cache results.

    Expected files:
    - ehr_patients.csv (or clean_patients.csv)
    - ehr_conditions.csv (or clean_conditions.csv)
    - ehr_encounters.csv (or clean_encounters.csv)
    - ehr_labs.csv (or clean_labs.csv)
    - ehr_medications.csv (or clean_medications.csv)
    - charlson_mapping.csv
    - Optional: healthsure_ehr_master_clean.csv

    Cleaning steps:
    1. Patients:
       - Drop exact duplicate rows and duplicate patient_id (keep first).
       - Standardize gender (M/m/Male -> Male; F/female/FEMALE -> Female).
       - Parse birth_date (%d-%m-%Y) and compute age as of today.
       - Replace height_cm outside [100, 220] with NaN, impute with median of same gender.
       - Impute missing weight_kg with median of same gender.
       - Compute bmi = weight_kg / (height_cm / 100) ** 2.
       - Fill missing blood_group with "Unknown".
       - Add age_group ('0-17', '18-39', '40-59', '60+').
    2. Conditions, Encounters, Labs, Medications:
       - Parse date columns.
       - Keep only rows where patient_id exists in cleaned patients table.
       - Drop rows with dates later than today.
       - In conditions, drop rows where icd10 does not match regex ^[A-Z][0-9]{2}$.
    3. Return two dictionaries: (cleaned_dfs, raw_dfs).

    Parameters
    ----------
    data_dir : str, default "clean"
        Path to the directory containing EHR CSV files.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        (cleaned_dataframes_dict, raw_dataframes_dict)
    """
    dir_path = Path(data_dir)

    # Resolve CSV file paths
    patients_file = _resolve_file(dir_path, ["ehr_patients.csv", "clean_patients.csv"])
    conditions_file = _resolve_file(dir_path, ["ehr_conditions.csv", "clean_conditions.csv"])
    encounters_file = _resolve_file(dir_path, ["ehr_encounters.csv", "clean_encounters.csv"])
    labs_file = _resolve_file(dir_path, ["ehr_labs.csv", "clean_labs.csv"])
    medications_file = _resolve_file(dir_path, ["ehr_medications.csv", "clean_medications.csv"])
    charlson_file = _resolve_file(dir_path, ["charlson_mapping.csv"])

    # Load raw DataFrames
    raw_patients = pd.read_csv(patients_file)
    raw_conditions = pd.read_csv(conditions_file)
    raw_encounters = pd.read_csv(encounters_file)
    raw_labs = pd.read_csv(labs_file)
    raw_medications = pd.read_csv(medications_file)
    raw_charlson = pd.read_csv(charlson_file)

    # Optional master dataset if present
    raw_master = None
    try:
        master_file = _resolve_file(dir_path, ["healthsure_ehr_master_clean.csv"])
        raw_master = pd.read_csv(master_file)
    except FileNotFoundError:
        raw_master = None

    # Work on copies for cleaning
    patients = raw_patients.copy()
    conditions = raw_conditions.copy()
    encounters = raw_encounters.copy()
    labs = raw_labs.copy()
    medications = raw_medications.copy()
    charlson = raw_charlson.copy()

    # -----------------------------------------------------------------------
    # 1. Clean Patients
    # -----------------------------------------------------------------------
    # Drop exact duplicate rows and duplicate patient_id (keep first)
    patients = patients.drop_duplicates()
    patients = patients.drop_duplicates(subset=["patient_id"], keep="first").copy()

    # Standardize gender (M/m/Male -> Male; F/female/FEMALE -> Female)
    def _standardize_gender(val: Any) -> Any:
        if pd.isna(val):
            return val
        s = str(val).strip().lower()
        if s in {"m", "male"}:
            return "Male"
        if s in {"f", "female"}:
            return "Female"
        return str(val).strip().title()

    patients["gender"] = patients["gender"].apply(_standardize_gender)

    # Parse birth_date (%d-%m-%Y) and compute age as of today
    patients["birth_date"] = _parse_date_series(patients["birth_date"])
    patients["age"] = calc_age(patients["birth_date"])

    # Replace height_cm outside [100, 220] with NaN, then impute with median of same gender
    patients["height_cm"] = pd.to_numeric(patients["height_cm"], errors="coerce")
    patients.loc[(patients["height_cm"] < 100) | (patients["height_cm"] > 220), "height_cm"] = np.nan
    gender_median_height = patients.groupby("gender")["height_cm"].transform("median")
    patients["height_cm"] = patients["height_cm"].fillna(gender_median_height)
    if patients["height_cm"].isna().any():
        patients["height_cm"] = patients["height_cm"].fillna(patients["height_cm"].median())
    patients["height_cm"] = patients["height_cm"].round(1)

    # Impute missing weight_kg with median by gender
    patients["weight_kg"] = pd.to_numeric(patients["weight_kg"], errors="coerce")
    gender_median_weight = patients.groupby("gender")["weight_kg"].transform("median")
    patients["weight_kg"] = patients["weight_kg"].fillna(gender_median_weight)
    if patients["weight_kg"].isna().any():
        patients["weight_kg"] = patients["weight_kg"].fillna(patients["weight_kg"].median())
    patients["weight_kg"] = patients["weight_kg"].round(1)

    # Compute BMI = weight_kg / (height_cm / 100) ** 2
    patients["bmi"] = (patients["weight_kg"] / ((patients["height_cm"] / 100.0) ** 2)).round(1)

    # Fill missing blood_group with "Unknown"
    patients["blood_group"] = patients["blood_group"].fillna("Unknown")
    patients["blood_group"] = patients["blood_group"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    # Add age_group (0-17, 18-39, 40-59, 60+)
    patients["age_group"] = pd.cut(
        patients["age"],
        bins=[-np.inf, 17, 39, 59, np.inf],
        labels=["0-17", "18-39", "40-59", "60+"],
        right=True
    ).astype(str)

    # -----------------------------------------------------------------------
    # 2. Clean Conditions, Encounters, Labs, Medications
    # -----------------------------------------------------------------------
    valid_patient_ids = set(patients["patient_id"].dropna().unique())
    today = pd.Timestamp.now().normalize()

    # -- Conditions --
    conditions["diagnosis_date"] = _parse_date_series(conditions["diagnosis_date"])
    conditions = conditions[conditions["patient_id"].isin(valid_patient_ids)].copy()
    conditions = conditions[conditions["diagnosis_date"].notna() & (conditions["diagnosis_date"] <= today)].copy()
    # Filter icd10 matching regex ^[A-Z][0-9]{2}$
    conditions = conditions[conditions["icd10"].astype(str).str.match(r"^[A-Z][0-9]{2}$")].copy()

    # -- Encounters --
    encounters["encounter_date"] = _parse_date_series(encounters["encounter_date"])
    encounters = encounters[encounters["patient_id"].isin(valid_patient_ids)].copy()
    encounters = encounters[encounters["encounter_date"].notna() & (encounters["encounter_date"] <= today)].copy()

    # -- Labs --
    labs["lab_date"] = _parse_date_series(labs["lab_date"])
    labs = labs[labs["patient_id"].isin(valid_patient_ids)].copy()
    labs = labs[labs["lab_date"].notna() & (labs["lab_date"] <= today)].copy()

    # -- Medications --
    medications["start_date"] = _parse_date_series(medications["start_date"])
    medications = medications[medications["patient_id"].isin(valid_patient_ids)].copy()
    medications = medications[medications["start_date"].notna() & (medications["start_date"] <= today)].copy()

    # -- Charlson Mapping --
    charlson = charlson.drop_duplicates().copy()

    # -----------------------------------------------------------------------
    # 3. Build Return Dictionaries
    # -----------------------------------------------------------------------
    cleaned_dfs: Dict[str, pd.DataFrame] = {
        "patients": patients,
        "conditions": conditions,
        "encounters": encounters,
        "labs": labs,
        "medications": medications,
        "charlson": charlson,
    }

    raw_dfs: Dict[str, pd.DataFrame] = {
        "patients": raw_patients,
        "conditions": raw_conditions,
        "encounters": raw_encounters,
        "labs": raw_labs,
        "medications": raw_medications,
        "charlson": raw_charlson,
    }

    if raw_master is not None:
        cleaned_dfs["master"] = raw_master.copy()
        raw_dfs["master"] = raw_master

    return cleaned_dfs, raw_dfs


# ---------------------------------------------------------------------------
# Constants for Domain Categorization & Underwriting
# ---------------------------------------------------------------------------

TIER1_CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Chennai",
    "Hyderabad", "Kolkata", "Pune", "Ahmedabad"
]

TIER2_CITIES = [
    "Jaipur", "Lucknow", "Indore", "Bhopal", "Nagpur",
    "Surat", "Vadodara", "Visakhapatnam", "Patna", "Kanpur",
    "Ludhiana", "Nashik", "Rajkot", "Faridabad", "Ghaziabad"
]

TIER3_CITIES = [
    "Agra", "Allahabad", "Srinagar", "Meerut",
    "Varanasi", "Amritsar", "Ranchi"
]

ALL_CITIES = sorted(list(set(TIER1_CITIES + TIER2_CITIES + TIER3_CITIES)))

OCCUPATIONS = [
    "Accountant", "Architect", "Banker", "Businessman", "Carpenter",
    "Chef", "Civil Servant", "Consultant", "Content Writer", "Data Analyst",
    "Doctor", "Driver", "Electrician", "Engineer", "Factory Worker",
    "Government Employee", "HR Manager", "Insurance Agent", "Lab Technician",
    "Lawyer", "Marketing Manager", "Nurse", "Pharmacist", "Plumber",
    "Real Estate Agent", "Retail Manager", "Sales Manager", "Shop Owner",
    "Software Engineer", "Teacher"
]


# ---------------------------------------------------------------------------
# Longitudinal Patient Master Table Builder (Cached)
# ---------------------------------------------------------------------------

@st.cache_data
def build_master_patient_table() -> pd.DataFrame:
    """
    Build a comprehensive longitudinal patient-level master table.
    
    Includes demographics, condition counts, Charlson scores, lists of diagnoses
    and medications, latest labs, latest vitals, and healthcare utilization.
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


# ---------------------------------------------------------------------------
# ML Model Loading & Resilient Premium Prediction
# ---------------------------------------------------------------------------

@st.cache_resource
def load_insurance_model():
    """Load local trained insurance model pipeline with cross-version scikit-learn patch."""
    import pickle
    import warnings
    import sklearn.compose._column_transformer as ct
    
    # Scikit-learn 1.7+ backward compatibility patch
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        ct._RemainderColsList = _RemainderColsList

    project_root = Path(__file__).resolve().parent
    model_paths = [
        project_root / "insurance_premium_model.pkl",
        Path("insurance_premium_model.pkl")
    ]
    for p in model_paths:
        if p.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(p, "rb") as f:
                    return pickle.load(f)
    raise FileNotFoundError("Could not find insurance_premium_model.pkl")


def predict_premium(
    age: int,
    weight: float,
    height: float,
    income_lpa: float,
    smoker: int,
    city: str,
    occupation: str,
    api_url: str = "https://insurance-premium-prediction-api-serving.onrender.com/predict"
) -> Tuple[str, str]:
    """
    Predict insurance premium category with resilient dual-mode strategy:
    1. Attempts live cloud API.
    2. Falls back seamlessly to local ML pipeline on network timeout or failure.
    
    Returns: (predicted_category, source_label)
    """
    import requests
    import warnings

    payload = {
        "age": int(age),
        "weight": float(weight),
        "height": float(height),
        "income_lpa": float(income_lpa),
        "smoker": int(smoker),
        "city": str(city),
        "occupation": str(occupation)
    }

    # 1. Attempt API (with 2.5s timeout)
    try:
        resp = requests.post(api_url, json=payload, timeout=2.5)
        if resp.status_code == 200:
            pred = resp.json().get("predicted_insurance_premium_category")
            if pred:
                return str(pred), "Live Cloud API"
    except Exception:
        pass

    # 2. Resilient fallback to local ML model
    model = load_insurance_model()

    bmi = weight / (height ** 2)
    income_per_age = income_lpa / max(1, age)
    weight_height_ratio = weight / max(0.1, height)
    income_bmi = income_lpa * bmi
    smoker_income = smoker * income_lpa
    health_risk_index = bmi * 0.4 + age * 0.3 + smoker * 20

    if city in TIER1_CITIES:
        city_tier = 1
    elif city in TIER2_CITIES:
        city_tier = 2
    else:
        city_tier = 3

    df_in = pd.DataFrame([{
        "age": age,
        "income_lpa": income_lpa,
        "occupation": occupation,
        "smoker": smoker,
        "city_tier": city_tier,
        "bmi": bmi,
        "income_per_age": income_per_age,
        "weight_height_ratio": weight_height_ratio,
        "income_bmi": income_bmi,
        "smoker_income": smoker_income,
        "health_risk_index": health_risk_index
    }])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_local = model.predict(df_in)[0]

    return str(pred_local), "Local ML Engine"


# ---------------------------------------------------------------------------
# Batch Portfolio Scoring & Dynamic Data Pipeline
# ---------------------------------------------------------------------------

def score_insurance_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize, engineer features, and predict insurance premium categories
    for any input DataFrame (uploaded CSV/Excel or default dataset).
    Returns an enriched DataFrame with all metrics and predictions.
    """
    df = df_input.copy()
    
    # 1. Standardize column names (strip, lowercase, replace spaces)
    rename_dict = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_")
        if c_clean in ["annual_income", "income", "income_inr"]:
            rename_dict[col] = "income_lpa"
        elif c_clean in ["height_in_m", "height_meters"]:
            rename_dict[col] = "height"
        elif c_clean in ["weight_in_kg", "weight_kg"]:
            rename_dict[col] = "weight"
        elif c_clean in ["smoking", "smoking_status", "is_smoker"]:
            rename_dict[col] = "smoker"
        elif c_clean != col:
            rename_dict[col] = c_clean
            
    df = df.rename(columns=rename_dict)
    
    # Defaults for missing essential columns
    if "age" not in df.columns:
        df["age"] = 35
    if "weight" not in df.columns:
        df["weight"] = 70.0
    if "height" not in df.columns:
        df["height"] = 170.0
    if "income_lpa" not in df.columns:
        df["income_lpa"] = 10.0
    if "smoker" not in df.columns:
        df["smoker"] = 0
    if "city" not in df.columns:
        df["city"] = "Mumbai"
    if "occupation" not in df.columns:
        df["occupation"] = "Software Engineer"

    # Numeric conversion with safe coercion
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(35).astype(int)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(70.0)
    df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(170.0)
    df["income_lpa"] = pd.to_numeric(df["income_lpa"], errors="coerce").fillna(10.0)

    # Standardize height: determine if height is in cm or meters
    df["height_m"] = df["height"].apply(lambda h: float(h) / 100.0 if float(h) > 3.0 else float(h))
    df["height_cm"] = df["height"].apply(lambda h: float(h) if float(h) > 3.0 else float(h) * 100.0)

    # 2. Compute Biometrics & Engineered Features
    df["bmi"] = (df["weight"] / (df["height_m"] ** 2)).round(2)
    df["bmi_category"] = df["bmi"].apply(lambda b: get_bmi_category(b)[0])
    df["income_per_age"] = (df["income_lpa"] / df["age"].clip(lower=1)).round(4)
    df["weight_height_ratio"] = (df["weight"] / df["height_cm"].clip(lower=10.0)).round(4)
    df["income_bmi"] = (df["income_lpa"] * df["bmi"]).round(2)

    def parse_smoker(val):
        v = str(val).strip().lower()
        if v in ["true", "1", "yes", "y", "smoker"]:
            return 1
        return 0

    df["smoker"] = df["smoker"].apply(parse_smoker)
    df["smoker_income"] = (df["smoker"] * df["income_lpa"]).round(2)
    df["health_risk_index"] = (df["bmi"] * 0.4 + df["age"] * 0.3 + df["smoker"] * 20.0).round(2)

    def classify_tier(c):
        c_str = str(c).strip().title()
        if c_str in TIER1_CITIES:
            return 1
        elif c_str in TIER2_CITIES:
            return 2
        return 3

    df["city_tier"] = df["city"].apply(classify_tier)

    # Clean occupation to match model training categories
    def clean_occ(o):
        o_str = str(o).strip()
        matched = [cand for cand in OCCUPATIONS if cand.lower() == o_str.lower()]
        return matched[0] if matched else "Software Engineer"

    df["occupation"] = df["occupation"].apply(clean_occ)

    # 3. Model Inference (Batch Vectorized)
    model = load_insurance_model()
    feature_cols = [
        "age", "income_lpa", "occupation", "smoker", "city_tier",
        "bmi", "income_per_age", "weight_height_ratio", "income_bmi",
        "smoker_income", "health_risk_index"
    ]
    df_features = df[feature_cols]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = model.predict(df_features)
    
    df["predicted_insurance_premium_category"] = predictions
    return df


@st.cache_data
def get_default_portfolio() -> pd.DataFrame:
    """Load and score baseline Indian Insurance dataset."""
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "Indian_Insurance_Data.csv"
    if csv_path.exists():
        df_raw = pd.read_csv(csv_path)
    else:
        # Fallback dummy data
        df_raw = pd.DataFrame([
            {"age": 35, "weight": 72, "height": 175, "income_lpa": 12.0, "smoker": 0, "city": "Mumbai", "occupation": "Software Engineer"}
        ])
    return score_insurance_dataframe(df_raw)


def get_active_portfolio() -> Tuple[pd.DataFrame, str]:
    """Retrieve active session portfolio or fallback to scored baseline dataset."""
    if "active_portfolio" in st.session_state and st.session_state["active_portfolio"] is not None:
        return st.session_state["active_portfolio"], st.session_state.get("portfolio_name", "Uploaded Portfolio")
    
    default_df = get_default_portfolio()
    st.session_state["active_portfolio"] = default_df
    st.session_state["portfolio_name"] = "Indian Insurance Benchmark (4,000 Records)"
    return default_df, st.session_state["portfolio_name"]


def set_active_portfolio(df: pd.DataFrame, source_name: str) -> None:
    """Store scored portfolio in session state."""
    st.session_state["active_portfolio"] = df
    st.session_state["portfolio_name"] = source_name


def reset_to_default_portfolio() -> None:
    """Reset active portfolio back to default benchmark."""
    default_df = get_default_portfolio()
    st.session_state["active_portfolio"] = default_df
    st.session_state["portfolio_name"] = "Indian Insurance Benchmark (4,000 Records)"


# ---------------------------------------------------------------------------
# Gemini LLM Actuarial Engine (Dual Mode: Live Gemini API + Offline Fallback)
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, api_key: Optional[str] = None) -> Optional[str]:
    """Call Google Gemini API using GEMINI_API_KEY configured securely in .env."""
    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        return None
    
    # 1. Try google.genai (new official SDK)
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        if resp and resp.text:
            return resp.text.strip()
    except Exception:
        pass

    # 2. Fallback to google.generativeai
    try:
        import google.generativeai as gai
        gai.configure(api_key=key)
        m = gai.GenerativeModel("gemini-1.5-flash")
        resp = m.generate_content(prompt)
        if resp and resp.text:
            return resp.text.strip()
    except Exception:
        pass

    return None


def generate_actuarial_narrative(applicant: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """Generate executive underwriting explanation using Gemini LLM or structured heuristics."""
    prompt = f"""
You are a senior insurance actuary and medical underwriter for HealthSure InsurTech.
Generate an executive underwriting risk report for the following applicant:

- Age: {applicant.get('age')} years
- Gender/Biometrics: Height {applicant.get('height')} m, Weight {applicant.get('weight')} kg, BMI: {applicant.get('bmi')} ({applicant.get('bmi_category')})
- Tobacco / Smoker: {'Smoker (Active)' if applicant.get('smoker') == 1 else 'Non-Smoker'}
- Annual Income: {applicant.get('income_lpa')} LPA (INR)
- City: {applicant.get('city')} (Tier {applicant.get('city_tier')})
- Occupation: {applicant.get('occupation')}
- Health Risk Index: {applicant.get('health_risk_index')}
- ML Predicted Premium Tier: {applicant.get('predicted_tier')}

Provide a structured, executive-grade evaluation with:
1. **Actuarial Risk Assessment**: Primary risk drivers (BMI, tobacco, age, occupational exposure).
2. **Policy Loading & Premium Rationale**: Why this premium tier ({applicant.get('predicted_tier')}) is justified.
3. **Recommended Underwriting Riders / Exclusions**: Suggested riders (e.g. Critical Illness, Cardiac copay, Lifestyle loading).
4. **Applicant Wellness Action Plan**: Specific clinical and lifestyle steps to reduce risk profile upon next policy renewal.

Keep tone professional, analytical, concise, and structured with markdown bullet points.
"""
    llm_result = call_gemini(prompt, api_key)
    if llm_result:
        return llm_result

    # Offline Resilient Fallback
    tier = applicant.get('predicted_tier', 'Medium')
    bmi = float(applicant.get('bmi', 24.0))
    smoker = int(applicant.get('smoker', 0))
    age = int(applicant.get('age', 35))

    bullets = []
    if smoker == 1:
        bullets.append("• **Tobacco Exposure:** Active smoking status carries a severe actuarial morbidity loading (+40-60% base rate loading).")
    else:
        bullets.append("• **Tobacco Status:** Non-smoker baseline qualifies for preferred lifestyle rating credits.")

    if bmi >= 30.0:
        bullets.append(f"• **Biometric Elevation:** BMI {bmi:.1f} indicates clinical obesity, contributing to metabolic and cardiovascular risk loading.")
    elif bmi < 18.5:
        bullets.append(f"• **Nutritional Variance:** Underweight BMI {bmi:.1f} warrants clinical exclusion review.")
    else:
        bullets.append(f"• **Biometric Stability:** BMI {bmi:.1f} sits within normal standard parameters.")

    if age >= 50:
        bullets.append(f"• **Age Cohort:** Age {age} exceeds senior actuarial threshold, triggering mandatory ECG/HbA1c diagnostic tele-mer.")

    return f"""
### 📋 Executive Underwriting Report (Actuarial Intelligence)

**Overall Actuarial Rating:** **{tier} Premium Bracket** (Health Risk Index: `{applicant.get('health_risk_index', 'N/A')}`)

#### 1. Primary Risk Drivers
{chr(10).join(bullets)}
• **Territory & Profession:** {applicant.get('city')} (Tier {applicant.get('city_tier')}) &bull; {applicant.get('occupation')}

#### 2. Underwriting Policy Recommendations
- **Recommended Policy Riders:** {'Mandatory Tobacco Cardiovascular Surcharge, Critical Illness Copay (15%)' if smoker else 'Standard Comprehensive Health Cover with Annual Preventive Checkup Rider'}
- **Deductible Optimization:** 10% co-pay recommended if applicant has pre-existing condition history.
- **Renewal Wellness Rebate:** Applicant eligible for a 15% premium reduction upon verifying smoking cessation or reaching BMI < 25 at 12-month policy review.
"""


def generate_portfolio_executive_summary(df: pd.DataFrame, api_key: Optional[str] = None) -> str:
    """Generate high-level executive portfolio risk brief using Gemini or analytics summary."""
    total = len(df)
    counts = df["predicted_insurance_premium_category"].value_counts().to_dict()
    smoker_pct = (df["smoker"] == 1).mean() * 100
    avg_age = df["age"].mean()
    avg_bmi = df["bmi"].mean()
    high_count = counts.get("High", 0)

    prompt = f"""
You are the Chief Underwriting Officer for HealthSure.
Write a concise 3-paragraph executive portfolio underwriting summary for the executive committee based on these uploaded portfolio statistics:

- Total Insured Lives / Applicants: {total:,}
- Premium Tier Breakdown: High: {high_count} ({(high_count/total*100):.1f}%), Medium: {counts.get('Medium', 0)}, Low: {counts.get('Low', 0)}
- Smoker Prevalence: {smoker_pct:.1f}%
- Mean Age: {avg_age:.1f} years
- Mean BMI: {avg_bmi:.1f} kg/m²
- Top Cities: {', '.join(df['city'].value_counts().head(3).index.tolist())}

Provide:
1. Executive Risk Diagnosis & Portfolio Exposure
2. Critical Actuarial Observations (Smoker vs. Obesity hotspots)
3. Actionable Underwriting Recommendations for Capital Reserves & Pricing
Keep it concise, professional, and impactful.
"""
    llm_result = call_gemini(prompt, api_key)
    if llm_result:
        return llm_result

    return f"""
### 📊 Executive Portfolio Underwriting Summary

**Portfolio Size:** {total:,} Lives Audited &bull; **High-Risk Concentration:** {high_count} ({(high_count/total*100):.1f}%)

1. **Portfolio Exposure Overview:**
   The portfolio exhibits an average applicant age of **{avg_age:.1f} years** and a mean BMI of **{avg_bmi:.1f} kg/m²**. Overall risk is distributed into {counts.get('Low', 0)} Low, {counts.get('Medium', 0)} Medium, and {high_count} High risk lives.
2. **Key Actuarial Observations:**
   Tobacco prevalence sits at **{smoker_pct:.1f}%**, representing the single largest contributor to the High Risk tier. Concentration of applicants in Tier 1 urban zones correlates with higher claimed lifestyle strain.
3. **Underwriting Action Items:**
   - Enforce mandatory tele-underwriting and diagnostic biomarker panels for all applicants in the High Risk bracket.
   - Introduce proactive wellness incentive riders offering premium reductions for verified smoking cessation.
"""


# ---------------------------------------------------------------------------
# Visual Styling System & Sidebar UI
# ---------------------------------------------------------------------------

def get_plotly_layout(title: str = "") -> Dict[str, Any]:
    """Returns standardized enterprise-grade dark glassmorphism Plotly layout."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=15, color="#f8fafc"),
            x=0.02,
            y=0.96
        ),
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#cbd5e1", size=12),
        margin=dict(t=50, b=30, l=30, r=30),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
        legend=dict(
            font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#94a3b8"),
            bgcolor="rgba(15, 23, 42, 0.4)",
            bordercolor="rgba(255, 255, 255, 0.06)",
            borderwidth=1
        )
    )


def render_sidebar_branding() -> None:
    """
    Renders unified enterprise design tokens, typography, glassmorphism cards,
    interactive hover micro-interactions, and status beacon across all Streamlit pages.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Clean Typography without overriding Streamlit Material Symbol Icons */
        html, body, .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Protect Material Symbols icons used by Streamlit collapse button and headers */
        [data-testid="stIconMaterial"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapsedControl"] span,
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="stHeader"] span,
        span[data-testid*="stIcon"],
        [class*="material-symbols"],
        [class*="material-icons"] {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Ambient Radial Gradient Mesh */
        .stApp {
            background: radial-gradient(circle at 12% 15%, rgba(14, 165, 233, 0.09) 0%, transparent 40%),
                        radial-gradient(circle at 88% 20%, rgba(99, 102, 241, 0.09) 0%, transparent 45%),
                        radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.06) 0%, transparent 50%),
                        #0b0f19 !important;
            color: #f1f5f9;
        }

        /* Glassmorphism Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.65) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45) !important;
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px) !important;
            border-color: rgba(56, 189, 248, 0.4) !important;
            box-shadow: 0 12px 28px -4px rgba(56, 189, 248, 0.18) !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            color: #94a3b8 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            letter-spacing: -0.025em !important;
        }

        /* Gradient Glowing Buttons */
        button[kind="primary"], .stButton > button {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 10px !important;
            padding: 0.55rem 1.35rem !important;
            box-shadow: 0 4px 16px 0 rgba(37, 99, 235, 0.35) !important;
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        button[kind="primary"]:hover, .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px 0 rgba(37, 99, 235, 0.55) !important;
            border-color: rgba(56, 189, 248, 0.6) !important;
        }
        button[kind="primary"]:active, .stButton > button:active {
            transform: translateY(0px) !important;
        }

        /* Glassmorphism Input & Selectbox Styling */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: rgba(15, 23, 42, 0.75) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 10px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
        }

        /* Modern Tabs */
        div[data-baseweb="tab-list"] {
            gap: 6px !important;
            background: rgba(15, 23, 42, 0.5) !important;
            padding: 6px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
        }
        div[data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 8px 18px !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            transition: all 0.2s ease !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(56, 189, 248, 0.16) !important;
            color: #38bdf8 !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
        }
        div[data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Modern Sidebar */
        [data-testid="stSidebarNav"] {
            padding-top: 0.5rem !important;
        }
        [data-testid="stSidebarNav"] a {
            border-radius: 10px !important;
            padding: 9px 14px !important;
            margin-bottom: 4px !important;
            font-weight: 500 !important;
            color: #cbd5e1 !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stSidebarNav"] a:hover {
            background: rgba(56, 189, 248, 0.14) !important;
            color: #38bdf8 !important;
            transform: translateX(3px) !important;
        }

        /* Live Beacon Micro-Animation */
        @keyframes pulseGlow {
            0% { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
            70% { transform: scale(1.05); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .live-beacon {
            width: 7px;
            height: 7px;
            background: #10b981;
            border-radius: 50%;
            display: inline-block;
            animation: pulseGlow 2.2s infinite;
        }

        /* Unified scrollbars */
        ::-webkit-scrollbar {
            width: 7px;
            height: 7px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.3);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar Header
    st.sidebar.markdown(
        """
        <div style="padding: 6px 2px 14px 2px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px;">
            <div style="font-size: 1.32rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; gap: 8px; letter-spacing: -0.015em;">
                🏥 HealthSure InsurTech
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                Unified AI Underwriting & EHR Intelligence
            </div>
            <div style="margin-top: 8px; display: inline-flex; align-items: center; gap: 7px; background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 9999px; padding: 2px 10px; font-size: 0.72rem; color: #34d399; font-weight: 600;">
                <span class="live-beacon"></span>
                FastAPI &bull; ML Pipeline Active
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Active Portfolio Indicator
    active_name = st.session_state.get("portfolio_name", "Benchmark Dataset (4,000)")
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 9px; padding: 9px 12px; margin-bottom: 14px;">
            <div style="font-size: 0.7rem; text-transform: uppercase; font-weight: 700; color: #38bdf8; letter-spacing: 0.04em;">📂 Active Portfolio</div>
            <div style="font-size: 0.82rem; color: #f1f5f9; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px;" title="{active_name}">{active_name}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gemini AI Status Badge (Key configured via .env)
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        st.sidebar.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 9px; padding: 7px 12px; margin-bottom: 14px; font-size: 0.74rem; color: #34d399; font-weight: 600; display: flex; align-items: center; gap: 7px;">
                <span class="live-beacon"></span>
                Gemini LLM Active (via .env)
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 9px; padding: 7px 12px; margin-bottom: 14px; font-size: 0.72rem; color: #94a3b8; font-weight: 500;">
                ✨ Gemini LLM: Standby / Rule Engine<br>
                <span style="font-size: 0.68rem; color: #64748b;">(Set GEMINI_API_KEY in .env)</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_sidebar_footer() -> None:
    """Renders clean footer branding in the sidebar."""
    st.sidebar.markdown(
        """
        <div style="padding: 16px 2px 8px 2px; border-top: 1px solid rgba(255,255,255,0.08); margin-top: 24px; font-size: 0.76rem; color: #64748b; line-height: 1.5;">
            <b style="color: #94a3b8;">HealthSure InsurTech Platform</b><br>
            Multi-Module Actuarial Intelligence<br>
            <span>FastAPI &bull; Scikit-Learn &bull; Gemini LLM &bull; Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True
    )



