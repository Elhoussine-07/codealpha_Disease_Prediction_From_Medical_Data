import pandas as pd

def clean_data(df):
    df = df.dropna()
    return df

def preprocess_features(df):
    return pd.get_dummies(df)
