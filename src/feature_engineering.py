def engineer_features(df):
    # Encode categoricals
    df = pd.get_dummies(df, columns=['merchant_type', 'location'], drop_first=True)

    # Scale numerics
    scaler = StandardScaler()
    df[['amount', 'time_gap']] = scaler.fit_transform(df[['amount', 'time_gap']])

    # Create derived features
    df['TotalSpend'] = df[['amount1', 'amount2', 'amount3']].sum(axis=1)
    df['TimeSinceLast'] = df['timestamp'].diff().fillna(0)

    return df