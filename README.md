# Machine Learning Classification Models and Streamlit Demo

## Problem statement
Build and compare multiple machine learning classification models on a single dataset, evaluate them with standard metrics, and deploy an interactive Streamlit app that lets users upload test data and view model performance (metrics, confusion matrix, classification report).

## Dataset description
**Dataset:** Mushroom Classification  
**Original source:** UCI Machine Learning Repository  
**Task:** Binary classification (poisonous vs edible) using 22 binary/categorical features.

This project downloads the mushroom dataset from UCI repository (dataset_full.csv) for reproducibility and offline usage.

## Models used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (kNN)
- Naive Bayes (GaussianNB)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Comparison table
Metrics below are from a single hold-out test split (`test_size=0.2`, `random_state=42`). Re-run `model/model_training.py` to reproduce.

Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC
---|---:|---:|---:|---:|---:|---:
Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000
Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000
kNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000
Naive Bayes | 0.9508 | 0.9975 | 0.9073 | 1.0000 | 0.9514 | 0.9062
Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000
XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000

### Observations
Model Name | Observation about model performance
---|---
Logistic Regression | Perfect performance across all metrics; well-suited for this linearly separable dataset.
Decision Tree | Achieves perfect classification on test set; dataset features enable clean splits.
kNN | Perfect scores on all metrics; distance-based approach works exceptionally well after feature scaling.
Naive Bayes | Lowest performance with 95.08% accuracy and 90.73% precision; probabilistic assumptions may not fully capture feature relationships despite perfect recall.
Random Forest (Ensemble) | Perfect performance through ensemble voting; robustness from multiple decision trees.
XGBoost (Ensemble) | Perfect performance with gradient boosting; handles complex patterns effectively.

## Repository structure
```
classification_models/
│-- streamlit_app.py
│-- requirements.txt
│-- README.md
│-- model/
│   │-- __init__.py
│   │-- model_training.py
│   │-- training_utils.py
│   └-- artifacts/            # created by training (model pkl files + metrics)
```

## How to run
1) Install dependencies:
```bash
cd "classification_models"
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
```

2) Train models and generate metrics (train and save the models before running Streamlit app):
```bash
python model/model_training.py
```

3) Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

4) Quick demo upload:
- Click on `Download sample test CSV` button to download the `sample_test.csv` file locally.
- Use `sample_test.csv` (includes the target column `poisonous`) to see predictions, metrics, confusion matrix, and classification report.

## Streamlit app features
- Full dataset preview
- Baseline metrics
- Download the test dataset `sample_test.csv` file
- Upload test dataset (`sample_test.csv`); target column is automatically selected.
- Model selection dropdown (6 models)
- Predictions and Metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix + classification report
