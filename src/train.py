import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("🔥 train.py is running")

df = pd.read_csv("data/processed/processed_with_target.csv")

FEATURES = [
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

X = df[FEATURES]
X = X.fillna(0)

# Fill missing values
X = X.fillna(0)

y = df["is_high_risk"]

print("Total missing values:", X.isnull().sum().sum())


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

lr = LogisticRegression(
    random_state=42,
    max_iter=1000
)

lr.fit(X_train, y_train)


rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print("Precision:", precision_score(y_test, lr_preds))
print("Recall:", recall_score(y_test, lr_preds))
print("F1:", f1_score(y_test, lr_preds))
print("ROC-AUC:", roc_auc_score(y_test, lr_probs))

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall:", recall_score(y_test, rf_preds))
print("F1:", f1_score(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))

print("Total missing values:", X.isnull().sum().sum())

lr_auc = roc_auc_score(y_test, lr_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

if rf_auc > lr_auc:
    best_model = rf
    print("\n🏆 Best Model: Random Forest")
else:
    best_model = lr
    print("\n🏆 Best Model: Logistic Regression")

mlflow.set_experiment("credit-risk-model")

with mlflow.start_run(run_name="LogisticRegression"):

    mlflow.log_param("model", "LogisticRegression")

    mlflow.log_metric("accuracy", accuracy_score(y_test, lr_preds))
    mlflow.log_metric("precision", precision_score(y_test, lr_preds))
    mlflow.log_metric("recall", recall_score(y_test, lr_preds))
    mlflow.log_metric("f1", f1_score(y_test, lr_preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, lr_probs))

    mlflow.sklearn.log_model(lr, "model")

with mlflow.start_run(run_name="RandomForest"):

    mlflow.log_param("model", "RandomForest")

    mlflow.log_metric("accuracy", accuracy_score(y_test, rf_preds))
    mlflow.log_metric("precision", precision_score(y_test, rf_preds))
    mlflow.log_metric("recall", recall_score(y_test, rf_preds))
    mlflow.log_metric("f1", f1_score(y_test, rf_preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, rf_probs))

    mlflow.sklearn.log_model(rf, "model")

joblib.dump(best_model, "best_model.pkl")

print("Best model saved as best_model.pkl")
