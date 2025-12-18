from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG
from .data_load import load_raw
from .preprocess import basic_clean
from . import schema

def plot_pred_vs_actual(y_true, y_pred, outpath: Path) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.3)
    minv = min(float(y_true.min()), float(y_pred.min()))
    maxv = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def plot_residuals(y_true, y_pred, outpath: Path) -> None:
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def main() -> None:
    # Load metrics for convenience
    metrics = json.loads(CONFIG.METRICS_PATH.read_text(encoding="utf-8"))

    df_raw = load_raw()

    # Ensure the same inferred schema is used in evaluation.
    resolved = schema.resolve_inplace(df_raw.columns)
    print("Resolved schema:", resolved)

    df = basic_clean(df_raw)

    y = df[schema.TARGET_COL]
    X = df.drop(columns=[schema.TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG.TEST_SIZE,
        random_state=CONFIG.RANDOM_STATE,
    )

    model = joblib.load(CONFIG.MODEL_PATH)
    y_pred = model.predict(X_test)

    CONFIG.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_pred_vs_actual(y_test, y_pred, CONFIG.FIGURES_DIR / "pred_vs_actual.png")
    plot_residuals(y_test, y_pred, CONFIG.FIGURES_DIR / "residuals.png")

    # Optional: error by district
    if schema.DISTRICT_COL in X_test.columns:
        eval_df = X_test.copy()
        eval_df["y_true"] = y_test.values
        eval_df["y_pred"] = y_pred
        eval_df["abs_error"] = (eval_df["y_true"] - eval_df["y_pred"]).abs()
        by_dist = eval_df.groupby(schema.DISTRICT_COL)["abs_error"].mean().sort_values(ascending=False)
        by_dist.to_csv("outputs/mae_by_district.csv")
        print("Wrote outputs/mae_by_district.csv")

    print("Evaluation complete.")
    print("Figures saved to reports/figures/")
    print("Summary metrics in outputs/metrics.json")

if __name__ == "__main__":
    main()
