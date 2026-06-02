import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load raw transaction data."""
    return pd.read_csv(filepath)


def create_aggregate_features(df):
    """Create customer-level aggregate features."""

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AverageTransactionAmount=("Amount", "mean"),
            TransactionCount=("Amount", "count"),
            StdTransactionAmount=("Amount", "std"),
        )
        .reset_index()
    )

    return agg_df


def extract_time_features(df):
    """Extract time-based features."""

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["TransactionHour"] = (
        df["TransactionStartTime"].dt.hour
    )
    df["TransactionDay"] = (
        df["TransactionStartTime"].dt.day
    )
    df["TransactionMonth"] = (
        df["TransactionStartTime"].dt.month
    )
    df["TransactionYear"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


if __name__ == "__main__":

    # Load data
    df = load_data("data/raw/data.csv")
    print(df.head())

    # Aggregate features
    agg_df = create_aggregate_features(df)
    print(agg_df.head())

    # Time features
    df = extract_time_features(df)

    print(
        df[
            [
                "TransactionHour",
                "TransactionDay",
                "TransactionMonth",
                "TransactionYear",
            ]
        ].head()
    )

    # Merge aggregate features
    df = df.merge(
        agg_df,
        on="CustomerId",
        how="left",
    )

    print(df.shape)

    # Feature lists
    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
    ]

    numerical_features = [
        "Amount",
        "Value",
        "CountryCode",
        "PricingStrategy",
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear",
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount",
    ]

    print("Feature lists created successfully")

    # Numerical preprocessing
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    print("Transformers created successfully")

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    print("Preprocessor created successfully")

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            )
        ]
    )

    print("Pipeline created successfully")

    # Transform data
    X = pipeline.fit_transform(df)

    print(
        "Final transformed shape:",
        X.shape,
    )

    # Save processed data
    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    processed_df = pd.DataFrame(
        X.toarray()
        if hasattr(X, "toarray")
        else X
    )

    processed_df.to_csv(
        "data/processed/processed_data.csv",
        index=False,
    )

    print("Processed data saved successfully")
