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

## Repo Folder Tree
```
fraud_scoring_service/
├── data/
│ └── raw/
│ └── synthetic_fraud_dataset.csv   # the untouched CSV of transactions from keggle
├── notebooks/
│ ├── _main_.ipynb            	              
│ ├── 01_knn_baseline.ipynb         		  
│ ├── 02_knn_with_scaling_and_onehot.ipynb   
│ ├── 03_data_preparation.ipynb     # logic of functions validate schema/types, clean data, engineer features                 
│ ├── 04_model_training.ipynb       # train classifiers and compare metrics        
│ └── with_functions.ipynb                 
├── lib/
│ ├── functions.py                  # load functions
│ ├── validate_input.py             # check for required columns, dtypes, nulls/duplicates
│ ├── clean_data.py                 # fill missing values, convert/fix Timestamp, drop duplicates
│ ├── feature_engineering.py        # encode categoricals, scale numerics, derive new features
├── README.md 
├── .gitattributes                  # paths 
├── .gitignore                      # files and folders excluded from Git commits
├── .config.yaml                    # configuration file (data paths, model parameters)
├── pyproject.toml                  # project metadata and dependency definitions
└── uv.lock                         # locked versions of all dependencies
```
## Slides

- [Presentation]
([https://docs.google.com/presentation/d/1DCRTmxcjTngXTsZA_19D8t6RzMZyaBRoeajUfCeMsb8/edit?usp=sharing](https://docs.google.com/presentation/d/17YETCwV5f-wDzrivk7pBB3f1i-GaUiZQ3jUsCZw9u-w/edit?slide=id.p#slide=id.p))

## Streamlit App

(https://w7pjprojectfraudscoringservice-rrp8ex7co5rxczfmvvjnkx.streamlit.app/)
