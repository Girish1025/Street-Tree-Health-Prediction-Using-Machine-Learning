"""Exploratory data analysis plots."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def save_countplot(df, column, output_dir, title=None, top_n=None):
    """Create and save a count plot for a categorical column."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    order = None
    if top_n:
        order = data[column].value_counts().head(top_n).index
        data = data[data[column].isin(order)]

    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=column, order=order)
    plt.title(title or f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    file_path = output_dir / f"{column}_countplot.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def save_numeric_distributions(df, numeric_columns, output_dir):
    """Create and save histogram plots for numeric columns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for col in numeric_columns:
        if col not in df.columns:
            continue
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        file_path = output_dir / f"{col}_histogram.png"
        plt.savefig(file_path, dpi=300)
        plt.close()
        paths.append(file_path)

    return paths


def run_eda(df, output_dir):
    """Run core EDA visualizations from the notebook."""
    outputs = []
    for col in ["borough", "health", "status", "steward", "sidewalk"]:
        if col in df.columns:
            outputs.append(save_countplot(df, col, output_dir))
    if "spc_latin" in df.columns:
        outputs.append(save_countplot(df, "spc_latin", output_dir, top_n=20))
    outputs.extend(save_numeric_distributions(df, ["tree_dbh", "stump_diam"], output_dir))
    return outputs
