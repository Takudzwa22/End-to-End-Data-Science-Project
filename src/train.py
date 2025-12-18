from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import CONFIG
from .data_load import load_raw
from .preprocess import basic_clean
from .features import build_preprocess_pipeline
from .models import get_model_specs
from .metrics import regression_metrics
from . import schema

def main() -> None:
    df_raw = load_raw()

    # Infer schema mapping from the CSV headers (reduces manual edits).
    resolved = schema.resolve_inplace(df_raw.columns)
    print("Resolved schema:", resolved)

    df = basic_clean(df_raw)

    if schema.TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{schema.TARGET_COL}' not found after schema resolution. "
            "Update src/schema.py or check your dataset."
        )

    y = df[schema.TARGET_COL]
    X = df.drop(columns=[schema.TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG.TEST_SIZE,
        random_state=CONFIG.RANDOM_STATE,
    )

    pre, num_cols, cat_cols = build_preprocess_pipeline(df)

    results = {}
    best_name = None
    best_rmse = float("inf")
    best_pipe = None

    for spec in get_model_specs(CONFIG.RANDOM_STATE):
        pipe = Pipeline(steps=[
            ("preprocess", pre),
            ("model", spec.model),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        m = regression_metrics(y_test, preds)
        results[spec.name] = m

        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_name = spec.name
            best_pipe = pipe

    # Persist
    CONFIG.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "best_model": best_name,
        "results": results,
        "features": {
            "numeric": num_cols,
            "categorical": cat_cols,
        },
        "notes": [
            "Update src/schema.py to match your dataset columns.",
            "Justify any cleaning thresholds used in src/preprocess.py in your report.",
        ],
    }
    CONFIG.METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joblib.dump(best_pipe, CONFIG.MODEL_PATH)

    print("Training complete.")
    print(f"Best model: {best_name} (RMSE={best_rmse:.2f})")
    print(f"Metrics written to: {CONFIG.METRICS_PATH}")
    print(f"Model saved to: {CONFIG.MODEL_PATH}")

if __name__ == "__main__":
    main()
