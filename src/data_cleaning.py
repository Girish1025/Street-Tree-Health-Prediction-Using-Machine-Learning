"""Data cleaning utilities for the street tree health project."""
import pandas as pd
from .config import SELECTED_COLUMNS


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase snake_case."""
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def select_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep columns used in the notebook analysis when available."""
    existing_cols = [col for col in SELECTED_COLUMNS if col in df.columns]
    return df[existing_cols].copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values and remove extreme diameter records.

    The original notebook identified Stump and Dead trees as having missing
    values in health/species/steward/sidewalk/problem fields. These values are
    treated as not applicable, while a small number of remaining missing values
    are imputed with domain-friendly defaults.
    """
    df = clean_column_names(df)
    df = select_modeling_columns(df)

    # Replace empty strings with missing values.
    df = df.replace("", pd.NA)

    # Missing health-related fields for Dead/Stump status are not applicable.
    if "status" in df.columns:
        mask = df["status"].isin(["Stump", "Dead"])
        df.loc[mask] = df.loc[mask].fillna("Not Applicable")

    fill_values = {
        "problems": "None",
        "health": "Good",
        "spc_latin": "No Observation",
        "sidewalk": "NoDamage",
        "steward": "Not Applicable",
    }
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # Remove identifier because it does not help prediction.
    if "tree_id" in df.columns:
        df = df.drop(columns=["tree_id"])

    # Cap/filter unrealistic diameters following the notebook logic.
    if {"tree_dbh", "stump_diam"}.issubset(df.columns):
        df = df.query("tree_dbh <= 60 or stump_diam <= 60").copy()

    return df.reset_index(drop=True)
