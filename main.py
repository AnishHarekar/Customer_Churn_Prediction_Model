from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load(BASE_DIR / "churn_model.joblib")
    config = joblib.load(BASE_DIR /"churn_config.joblib")
    ml_model["threshold"] = config["threshold"]
    ml_model["columns"] = config["columns"]
    yield
    ml_model.clear()


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerData(BaseModel):
    latitude: float
    longitude: float
    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    tenure_months: int = Field(ge=0, le=100)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: YesNoNoInternet
    online_backup: YesNoNoInternet
    device_protection: YesNoNoInternet
    tech_support: YesNoNoInternet
    streaming_tv: YesNoNoInternet
    streaming_movies: YesNoNoInternet
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"]
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)


# JSON field name -> column name the model was trained on
COLUMN_MAP = {
    "latitude": "Latitude", "longitude": "Longitude", "gender": "Gender",
    "senior_citizen": "Senior Citizen", "partner": "Partner", "dependents": "Dependents",
    "tenure_months": "Tenure Months", "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines", "internet_service": "Internet Service",
    "online_security": "Online Security", "online_backup": "Online Backup",
    "device_protection": "Device Protection", "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV", "streaming_movies": "Streaming Movies",
    "contract": "Contract", "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method", "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
}


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()]).rename(columns=COLUMN_MAP)
    input_df = input_df[ml_model["columns"]]          # same columns, same order as training

    probability = float(ml_model["model"].predict_proba(input_df)[0, 1])
    prediction = int(probability >= ml_model["threshold"])

    return {
        "churn_risk_score": round(probability, 4),
        "churn_prediction": prediction,
        "threshold": ml_model["threshold"],
        "result": "High Risk" if prediction == 1 else "Low Risk",
    }

@app.get("/health")
def health():
    return{"status":"ok", "model_loaded": "model" in ml_model}

# Must stay LAST, because it catches every path not matched above
app.mount("/", StaticFiles(directory="static", html=True), name="static")