# Hospital Readmission Prediction (XGBoost)

## Project Description
This project develops a machine learning (ML) prototype to predict 30-day hospital readmission risk using patient data. The goal is to help healthcare providers identify high-risk patients before discharge.

The system uses XGBoost as the main model and includes an interactive HTML dashboard.

## Technologies Used
- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- HTML


## Team Members
- Brenda Franco
- Cheikh Abdoul
- Taiwo Mudah
- Sidharth Gadgil
- Rohan Barai

## Disclaimer
This project is for educational purposes only.


## What this project includes

- Sample batch ingestion pipeline
- Data validation and feature engineering
- Logistic Regression baseline and XGBoost primary model
- MLflow experiment tracking
- Model version metadata
- FastAPI inference service
- Dockerized local deployment
- Basic monitoring via health check and prediction logging
- GitHub Actions CI

## Project structure

```text
app/            FastAPI application
src/            Data pipeline and training code
data/           Sample and processed datasets
models/         Saved model artifacts and metadata
tests/          API and feature pipeline tests
.github/        CI workflow
```

## Quick start

### 1. Create environment and install deps

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Train models and generate artifacts

```bash
python -m src.train
```

This creates:
- `data/sample/hospital_readmissions_sample.csv`
- `data/processed/readmission_features_v1.csv`
- `models/baseline_logreg.pkl`
- `models/xgb_readmit_v1.pkl`
- `models/metadata.json`
- `reports/figures/model_metrics.csv`

### 3. Run the FastAPI app

```bash
uvicorn app.main:app --reload
```

Open:
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 4. Example prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 72,
    "prior_admissions": 3,
    "medication_complexity": 4,
    "length_of_stay": 6.0,
    "comorbidity_score": 4,
    "diagnosis_group": "circulatory",
    "discharge_disposition": "rehab"
  }'
```

## Docker deployment

```bash
docker compose up --build
```

## Testing

```bash
pytest
```

## Monitoring approach

This demo includes:
- `/health` endpoint for service status
- prediction logging to `models/prediction_logs.csv`

Potential real-world monitoring extensions:
- data drift detection
- prediction drift dashboards
- recall/calibration monitoring over time
- alerting for latency or failure spikes

## Real-world deployment discussion

For the course project, deployment is mimicked with a containerized FastAPI app running locally. In a real hospital or healthcare analytics environment, this could be deployed:
- on-prem inside a hospital network
- in a private cloud behind authentication
- behind an internal API gateway with logging and monitoring

## Suggested Git workflow

```bash
git init
git add .
git commit -m "Initial hospital readmission ML system"
git branch -M main
git remote add origin <your-github-url>
git push -u origin main
```
