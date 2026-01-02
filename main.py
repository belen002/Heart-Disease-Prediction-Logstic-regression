from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the specific Logistic Regression model
model = joblib.load("logistic_regression_model.joblib")

class PatientData(BaseModel):
    age: float; trestbps: float; chol: float; thalch: float; oldpeak: float; ca: float
    sex_Male: int; cp_atypical_angina: int; cp_non_anginal: int; cp_typical_angina: int
    restecg_normal: int; restecg_st_t_abnormality: int; slope_flat: int; slope_upsloping: int
    thal_normal: int; thal_reversable_defect: int; fbs_True: int; exang_True: int

@app.get("/", response_class=HTMLResponse)
async def get_form():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(data: PatientData):
    # Map the JSON keys to the exact names used in your notebook training
    dict_data = {
        'age': data.age, 'trestbps': data.trestbps, 'chol': data.chol,
        'thalch': data.thalch, 'oldpeak': data.oldpeak, 'ca': data.ca,
        'cp_atypical angina': data.cp_atypical_angina,
        'cp_non-anginal': data.cp_non_anginal,
        'cp_typical angina': data.cp_typical_angina,
        'restecg_normal': data.restecg_normal,
        'restecg_st-t abnormality': data.restecg_st_t_abnormality,
        'slope_flat': data.slope_flat, 'slope_upsloping': data.slope_upsloping,
        'thal_normal': data.thal_normal, 'thal_reversable defect': data.thal_reversable_defect,
        'sex_Male': data.sex_Male, 'fbs_True': data.fbs_True, 'exang_True': data.exang_True
    }
    df = pd.DataFrame([dict_data])
    prediction = model.predict(df)[0]
    result_text = "Heart Disease Likely" if prediction == 1 else "No Heart Disease Detected"
    return {"prediction": result_text, "color": "#d9534f" if prediction == 1 else "#28a745"}