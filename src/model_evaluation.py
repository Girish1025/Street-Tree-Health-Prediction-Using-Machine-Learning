"""Model evaluation utilities."""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .config import RANDOM_STATE


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model using common classification metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_value = roc_auc_score(y_test, y_prob)
    else:
        auc_value = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc": auc_value,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def evaluate_models(models, X_test, y_test):
    """Evaluate multiple models and return a summary table plus full results."""
    rows = []
    full_results = {}

    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test)
        full_results[name] = result
        rows.append({
            "Model": name,
            "Accuracy": result["accuracy"],
            "Precision": result["precision_weighted"],
            "Recall": result["recall_weighted"],
            "F1-Score": result["f1_weighted"],
            "AUC": result["auc"],
        })

    return pd.DataFrame(rows).sort_values("F1-Score", ascending=False), full_results


def cross_validate_models(models, X_train, y_train):
    """Run 5-fold stratified cross-validation using weighted F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
        rows.append({"Model": name, "Mean F1 Score": scores.mean(), "Std F1 Score": scores.std()})
    return pd.DataFrame(rows).sort_values("Mean F1 Score", ascending=False)
