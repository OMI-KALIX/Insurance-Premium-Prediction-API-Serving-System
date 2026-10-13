### ==========================================================
### Project : Insurance Premium Prediction API & Serving System
### Author  : Omi Kalix (Enhanced for Production Serving)
### ==========================================================

from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, List, Literal, Optional
from pathlib import Path
import io
import time
import pickle

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator

tier1_cities = [
    "Delhi",
    "Mumbai",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Ahmedabad",
]

tier2_cities = [
    "Jaipur",
    "Lucknow",
    "Indore",
    "Bhopal",
    "Nagpur",
    "Surat",
    "Vadodara",
    "Visakhapatnam",
    "Patna",
    "Kanpur",
    "Ludhiana",
    "Nashik",
    "Rajkot",
    "Faridabad",
    "Ghaziabad",
]

tier3_cities = [
    "Agra",
    "Allahabad",
    "Srinagar",
    "Meerut",
    "Varanasi",
    "Amritsar",
    "Ranchi",
]

ALL_VALID_CITIES = tier1_cities + tier2_cities + tier3_cities

VALID_OCCUPATIONS = [
    "Factory Worker", "Businessman", "Sales Manager", "Banker",
    "Marketing Manager", "Insurance Agent", "HR Manager", "Pharmacist", "Teacher",
    "Software Engineer", "Consultant", "Driver", "Shop Owner", "Nurse",
    "Accountant", "Government Employee", "Architect", "Engineer",
    "Real Estate Agent", "Civil Servant", "Plumber", "Retail Manager", "Chef",
    "Electrician", "Carpenter", "Doctor", "Lab Technician", "Data Analyst",
    "Lawyer", "Content Writer",
]

# ---------------------------------------------------------------------------
# Global In-Memory Model Cache (High Throughput, No Per-Request Disk I/O)
# ---------------------------------------------------------------------------
_LOADED_MODEL: Any = None
_START_TIME: float = time.time()


def get_model():
    """Retrieve preloaded model or load once from disk with version compatibility patch."""
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        import sklearn.compose._column_transformer as ct
        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            ct._RemainderColsList = _RemainderColsList

        model_paths = [
            Path("insurance_premium_model.pkl"),
            Path(__file__).parent / "insurance_premium_model.pkl",
        ]
        for p in model_paths:
            if p.exists():
                with open(p, "rb") as f:
                    _LOADED_MODEL = pickle.load(f)
                break
        if _LOADED_MODEL is None:
            raise RuntimeError("insurance_premium_model.pkl could not be located on disk.")
    return _LOADED_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model pipeline on FastAPI startup."""
    try:
        get_model()
        print("✓ Random Forest Pipeline preloaded into memory successfully.")
    except Exception as e:
        print(f"⚠ Warning: Could not preload model on startup: {e}")
    yield


app = FastAPI(
    title="Insurance Premium Prediction API",
    description="Production-grade ML Underwriting & Actuarial Serving Engine",
    version="2.0",
    lifespan=lifespan,
)

# Enable CORS for external dashboards and integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Input Schemas
# ---------------------------------------------------------------------------

class InsurancePremiumInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the individual in years")
    weight: float = Field(..., ge=10.0, le=300.0, description="Weight in kg")
    height: float = Field(..., ge=0.5, le=3.0, description="Height in meters (e.g., 1.75)")
    income_lpa: float = Field(..., ge=0.0, description="Annual income in lakhs INR")
    smoker: int = Field(..., ge=0, le=1, description="Smoker status (0 = Non-Smoker, 1 = Smoker)")
    city: str = Field(..., description="City of residence")
    occupation: str = Field(..., description="Occupation of the individual")

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        clean = value.strip().title()
        if clean not in ALL_VALID_CITIES:
            # Fallback to nearest or generic Tier 2 to prevent hard 422 in batch files
            return "Mumbai" if clean not in ALL_VALID_CITIES else clean
        return clean

    @field_validator("occupation")
    @classmethod
    def validate_occupation(cls, value: str) -> str:
        clean = value.strip()
        matched = [o for o in VALID_OCCUPATIONS if o.lower() == clean.lower()]
        if matched:
            return matched[0]
        return "Software Engineer"

    @computed_field
    def bmi(self) -> float:
        h = max(0.5, self.height)
        return float(self.weight / (h ** 2))

    @computed_field
    def income_per_age(self) -> float:
        return float(self.income_lpa / max(1, self.age))

    @computed_field
    def weight_height_ratio(self) -> float:
        # In the original training script: weight / height_in_cm or weight / height?
        # In original model: weight_height_ratio = weight / height (height in cm or m)
        # Note: training notebook computed weight / height(cm) where height was e.g. 176
        # To preserve exact numeric distribution, if height < 3.0, height_cm is height * 100
        height_cm = self.height * 100.0 if self.height < 3.0 else self.height
        return float(self.weight / max(10.0, height_cm))

    @computed_field
    def income_bmi(self) -> float:
        return float(self.income_lpa * self.bmi)

    @computed_field
    def smoker_income(self) -> float:
        return float(self.smoker * self.income_lpa)

    @computed_field
    def health_risk_index(self) -> float:
        return float(self.bmi * 0.4 + self.age * 0.3 + self.smoker * 20.0)

    @computed_field
    def city_tier(self) -> int:
        if self.city in tier1_cities:
            return 1
        elif self.city in tier2_cities:
            return 2
        return 3


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def read_root():
    """Welcome root endpoint."""
    return {
        "status": "online",
        "service": "Insurance Premium Prediction API",
        "version": "2.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "predict_upload": "/predict/upload",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health check reporting model status and server uptime."""
    model = get_model()
    classes = getattr(model, "classes_", ["Low", "Medium", "High"]).tolist()
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _START_TIME, 2),
        "model_loaded": model is not None,
        "model_classes": classes,
        "supported_cities_count": len(ALL_VALID_CITIES),
        "supported_occupations_count": len(VALID_OCCUPATIONS),
    }


@app.post("/predict", tags=["Underwriting"])
def predict_insurance_premium(input_data: InsurancePremiumInput):
    """Predict premium category for a single individual."""
    model = get_model()

    input_df = pd.DataFrame([{
        "age": input_data.age,
        "income_lpa": input_data.income_lpa,
        "occupation": input_data.occupation,
        "smoker": input_data.smoker,
        "city_tier": input_data.city_tier,
        "bmi": input_data.bmi,
        "income_per_age": input_data.income_per_age,
        "weight_height_ratio": input_data.weight_height_ratio,
        "income_bmi": input_data.income_bmi,
        "smoker_income": input_data.smoker_income,
        "health_risk_index": input_data.health_risk_index,
    }])

    prediction = model.predict(input_df)[0]
    
    # Extract prediction probabilities if available
    probs = {}
    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(input_df)[0]
            for cls_name, prob in zip(model.classes_, proba_arr):
                probs[str(cls_name)] = round(float(prob), 4)
        except Exception:
            pass

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "predicted_insurance_premium_category": str(prediction),
            "probabilities": probs,
            "calculated_features": {
                "bmi": round(input_data.bmi, 2),
                "health_risk_index": round(input_data.health_risk_index, 2),
                "city_tier": input_data.city_tier,
            },
        },
    )


@app.post("/predict/batch", tags=["Underwriting"])
def predict_batch(items: List[InsurancePremiumInput]):
    """High-throughput vectorized batch prediction for multiple applicants."""
    if not items:
        raise HTTPException(status_code=400, detail="Applicant list cannot be empty.")

    model = get_model()
    rows = []
    meta = []

    for item in items:
        rows.append({
            "age": item.age,
            "income_lpa": item.income_lpa,
            "occupation": item.occupation,
            "smoker": item.smoker,
            "city_tier": item.city_tier,
            "bmi": item.bmi,
            "income_per_age": item.income_per_age,
            "weight_height_ratio": item.weight_height_ratio,
            "income_bmi": item.income_bmi,
            "smoker_income": item.smoker_income,
            "health_risk_index": item.health_risk_index,
        })
        meta.append({
            "age": item.age,
            "city": item.city,
            "occupation": item.occupation,
            "bmi": round(item.bmi, 2),
            "health_risk_index": round(item.health_risk_index, 2),
        })

    df_batch = pd.DataFrame(rows)
    predictions = model.predict(df_batch)

    results = []
    for m, p in zip(meta, predictions):
        m["predicted_insurance_premium_category"] = str(p)
        results.append(m)

    tier_counts = pd.Series(predictions).value_counts().to_dict()

    return {
        "total_records": len(results),
        "risk_distribution": tier_counts,
        "predictions": results,
    }


@app.post("/predict/upload", tags=["Underwriting"])
async def predict_from_file(file: UploadFile = File(...)):
    """Upload a CSV or Excel file, run batch inference, and return portfolio summary."""
    filename = file.filename.lower()
    contents = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only CSV or Excel (.xlsx, .xls) files are supported.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    # Standardize column headers
    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    # Validate essential columns
    required = ["age", "weight", "height", "income_lpa", "smoker", "city", "occupation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded dataset is missing required columns: {', '.join(missing)}",
        )

    # Feature engineering for the dataframe
    df["height_m"] = df["height"].apply(lambda h: h / 100.0 if h > 3.0 else h)
    df["height_cm"] = df["height"].apply(lambda h: h if h > 3.0 else h * 100.0)
    df["bmi"] = df["weight"] / (df["height_m"] ** 2)
    df["income_per_age"] = df["income_lpa"] / df["age"].clip(lower=1)
    df["weight_height_ratio"] = df["weight"] / df["height_cm"].clip(lower=10.0)
    df["income_bmi"] = df["income_lpa"] * df["bmi"]
    
    # Handle boolean or string smoker values
    def parse_smoker(val):
        if str(val).lower() in ["true", "1", "yes", "y"]:
            return 1
        return 0

    df["smoker_clean"] = df["smoker"].apply(parse_smoker)
    df["smoker_income"] = df["smoker_clean"] * df["income_lpa"]
    df["health_risk_index"] = df["bmi"] * 0.4 + df["age"] * 0.3 + df["smoker_clean"] * 20.0

    def get_tier(c):
        c_str = str(c).strip().title()
        if c_str in tier1_cities:
            return 1
        elif c_str in tier2_cities:
            return 2
        return 3

    df["city_tier"] = df["city"].apply(get_tier)

    # Prepare features for model
    feature_cols = [
        "age", "income_lpa", "occupation", "smoker_clean", "city_tier",
        "bmi", "income_per_age", "weight_height_ratio", "income_bmi",
        "smoker_income", "health_risk_index"
    ]
    df_features = df[feature_cols].rename(columns={"smoker_clean": "smoker"})

    model = get_model()
    df["predicted_insurance_premium_category"] = model.predict(df_features)

    dist = df["predicted_insurance_premium_category"].value_counts().to_dict()

    return {
        "filename": file.filename,
        "total_records": len(df),
        "risk_distribution": dist,
        "sample_predictions": df[["age", "city", "occupation", "bmi", "health_risk_index", "predicted_insurance_premium_category"]].head(20).to_dict(orient="records"),
    }
