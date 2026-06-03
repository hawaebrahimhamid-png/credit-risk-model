import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans


# =========================
# LOAD DATA
# =========================
def load_data(filepath):
    return pd.read_csv(filepath)


# =========================
# AGGREGATE FEATURES
# =========================
def create_aggregate_features(df):
    return (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AverageTransactionAmount=("Amount", "mean"),
            TransactionCount=("Amount", "count"),
            StdTransactionAmount=("Amount", "std"),
        )
        .reset_index()
    )


# =========================
# TIME FEATURES
# =========================
def extract_time_features(df):
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year

    return df


# =========================
# RFM FEATURES
# =========================
def create_rfm_features(df):
    snapshot_date = pd.to_datetime(
        df["TransactionStartTime"]
    ).max() + pd.Timedelta(days=1)

    return (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date
                    - pd.to_datetime(x).max()
                ).days,
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":

    # Load data
    df = load_data("data/raw/data.csv")
    print(df.head())

    # =========================
    # TASK 3: FEATURE ENGINEERING
    # =========================
    agg_df = create_aggregate_features(df)
    df = df.merge(agg_df, on="CustomerId", how="left")

    df = extract_time_features(df)

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

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    X = pipeline.fit_transform(df)

    print("Final transformed shape:", X.shape)

    os.makedirs("data/processed", exist_ok=True)

    processed_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X)

    processed_df.to_csv("data/processed/processed_data.csv", index=False)

    print("Processed data saved")

    # =========================
    # TASK 4: RFM + KMEANS
    # =========================

    rfm_df = create_rfm_features(df)

    # Step 1: Scale RFM
    scaler = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(
        rfm_df[["Recency", "Frequency", "Monetary"]]
    )

    # Step 2: KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)

    print("\nCluster counts:")
    print(rfm_df["Cluster"].value_counts())

    # Step 3: Cluster analysis
    print("\nCluster Means:")
    print(
        rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    )

    # Step 4: Assign HIGH RISK cluster (TEMP RULE: adjust after inspection)
    high_risk_cluster = rfm_df.groupby("Cluster")["Recency"].mean().idxmax()

    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == high_risk_cluster).astype(
        int
    )

    print("\nHigh risk distribution:")
    print(rfm_df["is_high_risk"].value_counts())

    # Step 5: Merge back
    df = df.merge(
        rfm_df[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left",
    )

    print("\nFinal dataset shape with target:", df.shape)

    # Step 6: Save final dataset
    df.to_csv("data/processed/processed_with_target.csv", index=False)

    print("Task 4 completed successfully")
