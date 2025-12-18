import pandas as pd
from .config import CONFIG

def load_raw() -> pd.DataFrame:
    """Load raw dataset from CONFIG.DATA_PATH.

    You must download the dataset and place the CSV at the configured path.
    """
    df = pd.read_csv(CONFIG.DATA_PATH)
    return df
