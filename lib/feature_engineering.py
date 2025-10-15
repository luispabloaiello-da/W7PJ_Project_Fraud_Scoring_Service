# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    # Remove ID columns since they only name rows and do not help the model learn
    for col in ['Transaction_ID', 'User_ID']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Turn every text column into numeric flags
    # This makes categories like “Device_Type” usable by models
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Standardize all numbers so each feature has equal weight
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    # Don’t scale the label column, only the inputs
    numeric_cols = [
        col for col in numeric_cols 
        if col.lower() not in ['fraud_label', 'is_fraud']
    ]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # If we still have a Timestamp, build a “time since last” feature
    # This captures how recent each transaction is
    if 'Timestamp' in df.columns:
        df = df.sort_values(by='Timestamp')
        df['TimeSinceLast'] = (
            df['Timestamp'].diff().dt.total_seconds().fillna(0)
        )
        # Drop the raw Timestamp once we’ve extracted the interval
        df = df.drop('Timestamp', axis=1)

    return df