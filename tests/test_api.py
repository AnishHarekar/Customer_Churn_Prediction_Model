import joblib
import pytest
from fastapi.testclient import TestClient

from main import BASE_DIR, app

VALID = {
    "latitude": 34.05, "longitude": -118.24, "gender": "Male", "senior_citizen": "No",
    "partner": "No", "dependents": "No", "tenure_months": 2, "phone_service": "Yes",
    "multiple_lines": "No", "internet_service": "Fiber optic", "online_security": "No",
    "online_backup": "No", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "No", "streaming_movies": "No", "contract": "Month-to-month",
    "paperless_billing": "Yes", "payment_method": "Electronic check",
    "monthly_charges": 85.0, "total_charges": 170.0,
}

SAFE = {
    **VALID, "contract": "Two year", "tenure_months": 60, "internet_service": "DSL",
    "payment_method": "Credit card (automatic)", "paperless_billing": "No",
    "monthly_charges": 60.0, "total_charges": 3600.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs startup, which loads the model
        yield c


def test_config_is_sane():
    cfg = joblib.load(BASE_DIR / "churn_config.joblib")
    assert 0 < cfg["threshold"] < 1
    assert "CLTV" not in cfg["columns"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_expected_fields(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_risk_score"] <= 1
    assert body["churn_prediction"] in (0, 1)
    assert body["result"] in ("High Risk", "Low Risk")


def test_risky_customer_scores_higher_than_safe_customer(client):
    risky = client.post("/predict", json=VALID).json()["churn_risk_score"]
    safe = client.post("/predict", json=SAFE).json()["churn_risk_score"]
    assert risky > safe


def test_bad_category_is_rejected(client):
    assert client.post("/predict", json={**VALID, "contract": "month to month"}).status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in VALID.items() if k != "tenure_months"}
    assert client.post("/predict", json=bad).status_code == 422