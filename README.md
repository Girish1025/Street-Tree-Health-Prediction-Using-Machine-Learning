# Street Tree Health Prediction Using Machine Learning

## Project Overview
This project predicts the health condition of street trees using machine learning techniques. The analysis is based on tree census data and focuses on identifying factors that influence tree health, such as tree diameter, stump diameter, species, borough, stewardship, sidewalk condition, and root, trunk and branch problems.

The target variable is `health`, which is converted into a binary classification problem:

- `Good`
- `Poor`

The project includes exploratory data analysis, data cleaning, outlier handling, feature engineering, model building, model evaluation, hyperparameter tuning, cross-validation, and SHAP-based model interpretation.

## Objective
The main objective of this project is to build machine learning models that can predict whether a street tree is in good or poor health.

Key questions explored in this project include:

- Which boroughs have the highest and lowest proportions of poor-health trees?
- Which tree species are most vulnerable?
- How do root, trunk, branch, and sidewalk problems vary across boroughs?
- Which features have the strongest influence on tree health?
- Which machine learning model performs best for predicting tree health?
- How does hyperparameter tuning improve model performance?

## Project Structure

```text
Street-Tree-Health-Prediction/
│
├── data/
│   └── Tree_census.csv
│
├── notebooks/
│   └── code.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── hyperparameter_tuning.py
│   └── shap_analysis.py
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
## Author

**Girish S Chandrappa**
