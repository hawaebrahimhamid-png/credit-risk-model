import pandas as pd

df = pd.read_csv("data/processed/processed_with_target.csv")

X = df.drop(columns=["is_high_risk"])

X = X.select_dtypes(include=["number"])

print(X.columns.tolist())