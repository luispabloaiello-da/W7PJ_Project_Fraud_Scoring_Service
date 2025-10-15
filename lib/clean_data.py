# lib/clean_data.py

import pandas as pd

def clean_data(df):
    # Fill nulls for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill nulls for categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Convert timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Timestamp'] = df['Timestamp'].ffill()

    # Drop duplicates
    df = df.drop_duplicates()

    return df