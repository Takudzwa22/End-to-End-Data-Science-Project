import pandas as pd
from .config import CONFIG

def load_raw() -> pd.DataFrame:
    
    df = pd.read_csv(CONFIG.DATA_PATH)
    return df
