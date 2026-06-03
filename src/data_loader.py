"""Functions for loading data."""
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load tree census data from a CSV file."""
    data = pd.read_csv(file_path)
    return data.copy()
