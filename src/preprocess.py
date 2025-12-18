from __future__ import annotations
import pandas as pd
import numpy as np
from . import schema

def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _dbg(df: pd.DataFrame, msg: str) -> pd.DataFrame:
    print(f"[preprocess] {msg}: shape={df.shape}")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, defensible cleaning. Keep it simple and explain choices in your report."""
    df = df.copy()
    _dbg(df, "before coercion")

    # Coerce expected numeric cols
    for c in [schema.TARGET_COL, schema.AREA_COL, schema.ROOMS_COL, schema.YEAR_BUILT_COL]:
        _coerce_numeric(df, c)
    _dbg(df, "after coercion")

    # Drop rows missing the target (can't train on them)
    df = df.dropna(subset=[schema.TARGET_COL])

    # Remove obviously invalid values (tune thresholds if needed)
    if schema.AREA_COL in df.columns:
        df = df[df[schema.AREA_COL].between(10, 400, inclusive="both")]  # TODO: justify thresholds
    if schema.TARGET_COL in df.columns:
        df = df[df[schema.TARGET_COL].between(30000, 10000000, inclusive="both")]  # Sale price filters (EUR). Defensive, not "magical".

    # Basic feature: property age
    if schema.YEAR_BUILT_COL in df.columns:
        current_year = 2023  # TODO: set based on dataset snapshot/year
        df["property_age"] = current_year - df[schema.YEAR_BUILT_COL]
        df.loc[(df["property_age"] < 0) | (df["property_age"] > 200), "property_age"] = np.nan
    _dbg(df, "after property age")
    return df
