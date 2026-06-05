import pandas as pd

df = pd.read_csv("data/processed/processed_with_target.csv")

print(df.columns.tolist())