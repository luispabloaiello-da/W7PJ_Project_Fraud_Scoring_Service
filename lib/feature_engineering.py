# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['merchant_type', 'location'], drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    df[['amount', 'time_gap']] = scaler.fit_transform(df[['amount', 'time_gap']])

    # Create derived features
    df['TotalSpend'] = df[['amount1', 'amount2', 'amount3']].sum(axis=1)
    df['TimeSinceLast'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    return df
