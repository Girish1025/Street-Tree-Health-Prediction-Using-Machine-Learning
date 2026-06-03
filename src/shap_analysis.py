"""SHAP interpretation functions."""
from pathlib import Path
import matplotlib.pyplot as plt
import shap


def run_shap_analysis(model, X_test, output_dir, model_name="model"):
    """Generate SHAP summary plots for tree-based models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    bar_path = output_dir / f"{model_name}_shap_bar.png"
    summary_path = output_dir / f"{model_name}_shap_summary.png"

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    return bar_path, summary_path
