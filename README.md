# Predictive Analysis of Rental Prices in Berlin

This repository builds a reproducible pipeline to predict **monthly rent (EUR)** for Berlin rental listings.

The implementation is intentionally small and auditable:
- Baseline (mean)
- Ridge regression (linear, regularized)
- Random Forest regression (non-linear ensemble)

## Dataset
Download the CSV from Kaggle and save it as:

`data/berlin_listings.csv`

Dataset: *Real Estate Listings Berlin (DE) April 2023* (Immowelt scrape)
- https://www.kaggle.com/datasets/mathisjander/real-estate-listings-berlin-de-april-2023

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Train (writes model + metrics):
```bash
python -m src.train
```

Evaluate (writes plots + optional MAE-by-district):
```bash
python -m src.evaluate
```

## Outputs
- `outputs/metrics.json` — MAE, RMSE, R² for each model and the selected best model
- `models/final_model.joblib` — serialized best pipeline
- `reports/figures/pred_vs_actual.png` — predicted vs actual plot
- `reports/figures/residuals.png` — residual diagnostic
- `outputs/mae_by_district.csv` — optional (only if a district column is detected)

## Notes
- Column names are auto-detected (German/English variants). If the resolver guesses incorrectly, edit `src/schema.py` and hardcode the correct names.
- Cleaning thresholds are in `src/preprocess.py` and should be justified in the report.
