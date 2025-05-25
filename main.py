from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load models
log_model = joblib.load("models/logistic_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")

# Initialize app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

# Define input data model
class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# Predict endpoint
@app.post("/predict")
def predict(data: HeartData):
    features = np.array([[ 
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    logistic_pred = int(log_model.predict(features)[0])
    rf_pred = int(rf_model.predict(features)[0])

    return {
        "logistic_model_prediction": logistic_pred,
        "random_forest_prediction": rf_pred
    }
