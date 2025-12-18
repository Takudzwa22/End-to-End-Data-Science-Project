from __future__ import annotations
from dataclasses import dataclass
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: object

def get_model_specs(random_state: int) -> list[ModelSpec]:
    """A small, within-scope model set: baseline -> linear -> non-linear ensemble."""
    return [
        ModelSpec("baseline_mean", DummyRegressor(strategy="mean")),
        ModelSpec("ridge", Ridge(alpha=1.0, random_state=random_state)),
        ModelSpec("random_forest", RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )),
    ]
