import joblib
import pandas as pd

model = joblib.load("best_model.pkl")


FEATURE_ORDER = [
    "CountryCode",
    "Amount",
    "Value",
    "PricingStrategy",
    "FraudResult",
    "TotalTransactionAmount",
    "AverageTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount",
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionYear"
]


def predict_risk(data: dict):

    df = pd.DataFrame([data])

    # ensure all required columns exist
    df = df.reindex(columns=FEATURE_ORDER)

    probability = model.predict_proba(df)[0][1]

    print("RAW PROBABILITY:", probability)

    return float(probability)
