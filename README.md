# Fraud Scoring Service

## Overview  
This project builds a reusable machine learning pipeline that detects fraudulent transactions from any structured CSV file. Whether the data comes from PayPal, Stripe, or internal logs, the system validates the input, engineers meaningful features, runs multiple models, and outputs fraud risk scores.

---

## What It Does  
- Accepts any transaction CSV file  
- Validates column structure, data types, and missing values  
- Cleans and transforms the data (encoding, scaling, feature creation)  
- Trains and compares four ML models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
- Outputs fraud probability per transaction  
- Summarizes model performance using Accuracy, Precision, Recall, and F1

---

## How It Works  
1. **Input Validation**  
   - Checks schema, nulls, types, duplicates  
2. **Feature Engineering**  
   - Encodes categoricals, scales numerics, creates derived features  
3. **Model Training & Evaluation**  
   - Benchmarks four classifiers  
   - Compares metrics across models  
4. **Scoring & Output**  
   - Generates fraud scores  
   - Produces model comparison summary  
5. **Optional Deployment**  
   - Streamlit or Flask interface for CSV upload and scoring

---

## Deliverables  
- Modular Python pipeline  
- Fraud scores per transaction  
- Model comparison dashboard  
- Ready-to-integrate output for analysts or systems

---

## Why Machine Learning  
Fraud patterns evolve. Static rules fail. ML adapts. This project turns raw data into actionable insight—fast, scalable, and production-ready.

## Repo folder Tree
```
fraud_scoring_service/
├── data/
│   └── raw/
│        ├──synthetic_fraud_dataset.csv
├── notebooks/
│   ├── 01_topic_selection.ipynb
│   └── 02_data_preparation.ipynb
├── src/
│   ├── validate_input.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   └── model_evaluation.py
├── README.md
└── .gitignore
└── .config.yaml
└── pyproject.toml
└── uv.lock
```
## Kanban 
- Mauricio Write README.md with Project goal: fraud scoring from any CSV, Business logic: why fraud matters, why ML is needed, Deliverables: pipeline, scores, model comparison.
- Download Kaggle dataset and save to data/raw/

## To do
- Create inside 01_topic_selection.ipynb: a write and check the theory and labs with short Markdown cells explaining use case of supervised learning, classification, metrics and how could be convinient or not base o this keggle dataset columns that we have.
- Model plan: Logistic Regression, Decision Tree, Random Forest, KNN Output: this is a Clear onboarding doc for repo and collaborators
- in 01_topic_selection.ipynb, write: ML lifecycle steps: data → model → evaluation → deployment
- Feature engineering theory: selection, transformation, creation
- Model constraints: KNN needs scaled numeric, Trees tolerate raw but benefit from encoding, Logistic Regression needs clean, linear features
- Plan evaluation metrics: Accuracy, Precision, Recall, F1
- Create function feature_engineering.py:
- Encode categoricals (e.g. merchant type, location)
- Scale numerics (e.g. amount, time gaps)
- Create derived features: TotalSpend = sum of spend columns, TimeSinceLast = difference between timestamps
- Save transformed DataFrame
Document logic in notebook 02_data_preparation.ipynb Output: Model-ready features aligned with algorithm needs

## To do (II)
- Dataset Acquisition & Profiling: load and config dataset from data/raw/
- Load into 01_topic_selection.ipynb and inspect: .shape, .dtypes, .isnull().sum(), .describe()
- Check for duplicates, outliers, skewed distributions
- Document findings: Which columns are usable, Which need cleaning or transformation
- Output: Data profile ready for Day 3 cleaning

- Create in src a function ===> validate_input.py: Check required columns: amount, timestamp, merchant, etc., Validate types: numeric, categorical, datetime, Flag missing values and duplicates.

- Create another function ===> clean_data.py: Drop or fill nulls (mean/mode), Convert types (e.g. timestamp to datetime), Save cleaned DataFrame

- document the logic of these functions here in notebook 02_data_preparation.ipynb Output: Clean, validated dataset ready for feature engineering

## tomorrow day 3
- create the function model_evaluation.py:

- Train/test split with random_state=42

- Define model dictionary:
models = {
  'LogisticRegression': LogisticRegression(),
  'DecisionTree': DecisionTreeClassifier(),
  'RandomForest': RandomForestClassifier(),
  'KNN': KNeighborsClassifier()
}
- Define metric functions: accuracy, precision, recall, F1

document setup in 02_data_preparation.ipynb
