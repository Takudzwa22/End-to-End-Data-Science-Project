from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Update if your dataset filename differs
    DATA_PATH: Path = Path("data/berlin_listings.csv")

    # Train/test split
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # Output paths
    METRICS_PATH: Path = Path("outputs/metrics.json")
    MODEL_PATH: Path = Path("models/final_model.joblib")
    FIGURES_DIR: Path = Path("reports/figures")

CONFIG = Config()
