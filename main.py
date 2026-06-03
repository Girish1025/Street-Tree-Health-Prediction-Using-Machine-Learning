"""Main workflow for Street Tree Health Prediction."""
from src.config import DATA_PATH, FIGURES_DIR, MODELS_DIR
from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.eda import run_eda
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_model_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models, cross_validate_models
from src.hyperparameter_tuning import tune_models
from src.shap_analysis import run_shap_analysis


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw_data = load_data(DATA_PATH)

    print("Cleaning data...")
    cleaned_data = clean_data(raw_data)

    print("Running EDA plots...")
    run_eda(cleaned_data, FIGURES_DIR)

    print("Engineering features...")
    final_data = engineer_features(cleaned_data)

    print("Preparing model data...")
    X_train, X_test, y_train, y_test = prepare_model_data(final_data)

    print("Training baseline models...")
    models = train_models(X_train, y_train)

    print("Evaluating baseline models...")
    summary, full_results = evaluate_models(models, X_test, y_test)
    print(summary)

    print("Running cross-validation...")
    cv_summary = cross_validate_models(models, X_train, y_train)
    print(cv_summary)

    print("Running hyperparameter tuning...")
    tuned_models, best_params = tune_models(X_train, y_train)
    print("Best parameters:")
    print(best_params)

    print("Evaluating tuned models...")
    tuned_summary, tuned_full_results = evaluate_models(tuned_models, X_test, y_test)
    print(tuned_summary)

    # SHAP is best suited to tree-based models such as Random Forest or XGBoost.
    if "Random Forest" in tuned_models:
        print("Generating SHAP plots for Random Forest...")
        run_shap_analysis(tuned_models["Random Forest"], X_test, FIGURES_DIR, "random_forest")

    print("Project run complete.")


if __name__ == "__main__":
    main()
