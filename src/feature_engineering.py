"""Feature engineering for tree health prediction."""
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create target and problem-count features."""
    df = df.copy()

    # Combine Fair and Poor into one minority class, matching notebook framing.
    if "health" in df.columns:
        df["health"] = df["health"].replace({"Fair": "Poor"})
        df = df[df["health"].isin(["Good", "Poor"])].copy()
        df["health"] = df["health"].map({"Poor": 0, "Good": 1})

    problem_cols = [
        "root_stone", "root_grate", "root_other", "trunk_wire", "trnk_light",
        "trnk_other", "brch_light", "brch_shoe", "brch_other"
    ]
    existing_problem_cols = [col for col in problem_cols if col in df.columns]

    # Convert Yes/No problem indicators to numeric and create total problem count.
    for col in existing_problem_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    if existing_problem_cols:
        numeric_problem_frame = df[existing_problem_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["total_problems"] = numeric_problem_frame.sum(axis=1)

    return df.reset_index(drop=True)
