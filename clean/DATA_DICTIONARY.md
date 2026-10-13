# healthsure_ehr_master_clean.csv  (1200 patients, 1 row per patient, 0 missing values, synthetic)

ID: patient_id
Demographics: age, gender, gender_male, age_group, city, blood_group, height_cm, weight_kg, bmi, obese
Vitals (from encounters): last_systolic_bp, last_diastolic_bp, last_glucose, mean_systolic_bp, mean_diastolic_bp, mean_glucose, bp_category
Labs (latest): last_hba1c_pct, last_total_cholesterol, last_ldl, last_hdl, last_creatinine_mg_dl, last_hemoglobin_g_dl
Utilization: n_encounters, n_emergency, n_inpatient, total_stay_days, total_cost_inr, days_since_last_visit, n_medications
Comorbidity: condition_count, charlson_score, comorbidity_risk (Low/Medium/High)
Diagnosis flags (0/1): has_diabetes, has_hypertension, has_cad, has_ckd, has_asthma, has_obesity_dx, has_anemia, has_hypothyroid, has_pneumonia
Other target: high_utilizer (>=3 emergency+inpatient encounters)

## Suggested models (target -> features), AVOID LEAKAGE
1. Diabetes risk:    has_diabetes  <- age, gender_male, bmi, mean_glucose, last_hba1c_pct, mean_systolic_bp   (drop has_* and charlson_score)
2. CKD risk:         has_ckd       <- age, bmi, last_creatinine_mg_dl, last_hemoglobin_g_dl, mean_systolic_bp
3. Cardiovascular:   has_cad       <- age, bmi, last_total_cholesterol, last_ldl, last_hdl, mean_systolic_bp, mean_diastolic_bp
4. Comorbidity tier: comorbidity_risk <- age, bmi, vitals, labs  (drop condition_count, charlson_score, has_*)
5. High utilizer:    high_utilizer <- age, bmi, vitals, labs, days_since_last_visit (drop n_emergency, n_inpatient, n_encounters, total_cost_inr, total_stay_days)
6. Cost regression:  total_cost_inr <- age, condition_count, charlson_score, n_medications, n_encounters
Use train_test_split(stratify=target) and report ROC-AUC, not just accuracy (classes are imbalanced ~15%).
Other files: clean_patients/conditions/encounters/labs/medications.csv are the cleaned individual tables.
