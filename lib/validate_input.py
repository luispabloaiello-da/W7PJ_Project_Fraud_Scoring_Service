# lib/validate_input.py

import pandas as pd

REQUIRED_COLUMNS = [
    'Transaction_Amount',
    'Transaction_Type',
    'Timestamp',
    'Location',
    'Merchant_Category',
    'Risk_Score',
    'Fraud_Label'
]

def validate_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True

def validate_types(df):
    issues = {}
    # only verify that Timestamp is datetime
    if 'Timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            issues['Timestamp'] = 'Expected datetime, got object'
    return issues

def check_nulls_and_duplicates(df):
    nulls = df.isnull().sum()
    duplicates = df.duplicated().sum()
    return nulls, duplicates