"""Preprocessing and train-test split functions."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .config import RANDOM_STATE, TEST_SIZE, TARGET


def prepare_model_data(df: pd.DataFrame, target: str = TARGET):
    """Encode categorical variables, scale numeric variables, and split data."""
    df = df.copy().dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_encoded = pd.get_dummies(X, drop_first=True)

    numeric_cols = [col for col in ["tree_dbh", "stump_diam", "total_problems"] if col in X_encoded.columns]
    if numeric_cols:
        scaler = MinMaxScaler()
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
