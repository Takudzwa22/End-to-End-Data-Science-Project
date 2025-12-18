from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from . import schema

def build_preprocess_pipeline(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    
    numeric_cols = [c for c in [schema.AREA_COL, schema.ROOMS_COL, "property_age"] if c in df.columns]
    candidate_cats = [schema.DISTRICT_COL, schema.PROPERTY_TYPE_COL]
    categorical_cols = [c for c in candidate_cats if c and c in df.columns]


    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols

