"""Hyperparameter tuning functions."""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .config import RANDOM_STATE


def tune_models(X_train, y_train):
    """Tune selected models with compact grids suitable for local execution."""
    grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "solver": ["liblinear"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"max_depth": [5, 10, 15, None], "min_samples_split": [2, 10, 25]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [10, 20, None], "min_samples_split": [2, 10]},
        ),
        "XGBoost": (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
        ),
    }

    tuned_models = {}
    best_params = {}
    for name, (model, param_grid) in grids.items():
        search = GridSearchCV(model, param_grid, scoring="f1_weighted", cv=3, n_jobs=-1)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        best_params[name] = search.best_params_

    return tuned_models, best_params
