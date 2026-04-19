from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint():
    payload = {
        "age": 72,
        "prior_admissions": 3,
        "medication_complexity": 4,
        "length_of_stay": 6.0,
        "comorbidity_score": 4,
        "diagnosis_group": "circulatory",
        "discharge_disposition": "rehab",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["readmission_risk"] <= 1
    assert body["risk_label"] in {"Low Risk", "Moderate Risk", "High Risk"}
