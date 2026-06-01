### ==========================================================
### Project : Insurance Premium Prediction API & Serving System
### Author  : Omi Kalix
### GitHub  : https://github.com/OMI-KALIX
### ==========================================================

from fastapi import FastAPI,Path,Query,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Annotated,Literal
import numpy as np
import pandas as pd
from pickle import load

tier1_cities = [
    "Delhi",
    "Mumbai",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Ahmedabad"
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
    "Ghaziabad"
]

tier3_cities = [
    "Agra",
    "Allahabad",
    "Srinagar",
    "Meerut",
    "Varanasi",
    "Amritsar",
    "Ranchi"
]

app = FastAPI(title="Insurance Premium Prediction API", version="1.0")
"city	occupation	insurance_premium_category"
class InsurancePremiumInput(BaseModel):
    age: int = Annotated[int, Field(..., ge=0, description="Age of the individual")]
    weight: float = Annotated[float, Field(..., ge=0, description="Weight in kg")]
    height: float = Annotated[float, Field(..., ge=0, description="Height in m")]
    income_lpa: float = Annotated[float, Field(..., ge=0, description="Annual income in lakhs")]
    smoker: int = Annotated[int, Field(..., ge=0, le=1, description="Smoker status (0 or 1)")]
    city: str = Annotated[str, Field(..., description="City of residence")]
    occupation: str = Annotated[str, Literal['Factory Worker', 'Businessman', 'Sales Manager', 'Banker',
                                            'Marketing Manager', 'Insurance Agent', 'HR Manager', 'Pharmacist', 'Teacher',
                                            'Software Engineer', 'Consultant', 'Driver', 'Shop Owner', 'Nurse',
                                            'Accountant', 'Government Employee', 'Architect', 'Engineer',
                                            'Real Estate Agent', 'Civil Servant', 'Plumber', 'Retail Manager', 'Chef',
                                            'Electrician', 'Carpenter', 'Doctor', 'Lab Technician', 'Data Analyst','Lawyer', 'Content Writer'], 
                                            Field(..., description="Occupation of the individual")]

    @field_validator("city")
    @classmethod
    def validate_city(cls, value):
        if value not in tier1_cities + tier2_cities + tier3_cities:
            raise ValueError(f"City must be one of the following: {', '.join(tier1_cities + tier2_cities + tier3_cities)}")
        return value
    
    @field_validator("occupation")
    @classmethod
    def validate_occupation(cls, value):
        if value not in ['Factory Worker', 'Businessman', 'Sales Manager', 'Banker',
                                            'Marketing Manager', 'Insurance Agent', 'HR Manager', 'Pharmacist', 'Teacher',
                                            'Software Engineer', 'Consultant', 'Driver', 'Shop Owner', 'Nurse',
                                            'Accountant', 'Government Employee', 'Architect', 'Engineer',
                                            'Real Estate Agent', 'Civil Servant', 'Plumber', 'Retail Manager', 'Chef',
                                            'Electrician', 'Carpenter', 'Doctor', 'Lab Technician', 'Data Analyst','Lawyer', 'Content Writer']:
            raise ValueError("Invalid occupation. Must be one of the predefined occupations.")
        return value
    
    @computed_field
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)
    
    @computed_field
    def income_per_age(self) -> float:
        return self.income_lpa / self.age
    
    @computed_field
    def weight_height_ratio(self) -> float:
        return self.weight / self.height
    
    @computed_field
    def income_bmi(self) -> float:
        return self.income_lpa * self.bmi
    @computed_field
    def smoker_income(self) -> float:
        return self.smoker * self.income_lpa
    @computed_field
    def health_risk_index(self) -> float:
        return self.bmi * 0.4 + self.age * 0.3 + self.smoker * 20
    
    @computed_field
    def city_tier(self) -> int:
        if self.city in tier1_cities:
            return 1
        elif self.city in tier2_cities:
            return 2
        else:
            return 3
def load_model():   
    with open("insurance_premium_model.pkl", "rb") as f:
        model = load(f)
    return model
@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Premium Prediction API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
def predict_insurance_premium(input_data: InsurancePremiumInput):

    input_dict = pd.DataFrame([{
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
                                "health_risk_index": input_data.health_risk_index
                            }])
    model = load_model()
    prediction = model.predict(input_dict)[0]
    return JSONResponse(status_code=200, content={"predicted_insurance_premium_category": prediction})


