from pathlib import Path
import pandas as pd


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load raw CSV data from the given path."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    return pd.read_csv(csv_path)
