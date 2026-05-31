# Credit Risk Probability Model for Alternative Data

## Project Overview

This project aims to build an end-to-end Credit Risk Scoring System for Bati Bank using alternative transaction data from an eCommerce platform. The system will estimate customer credit risk, generate risk probability scores, and support Buy-Now-Pay-Later (BNPL) lending decisions through a production-ready machine learning pipeline and API.

---

## Business Problem

Bati Bank is partnering with an eCommerce platform to offer Buy-Now-Pay-Later services. Since the available dataset does not contain traditional credit history or loan default information, alternative transaction data must be used to assess customer creditworthiness.

The goal is to identify high-risk and low-risk customers, estimate their probability of default, and provide automated credit risk predictions that support lending decisions.

---

## Credit Scoring Business Understanding

### Basel II and Interpretability

The Basel II Accord requires financial institutions to use risk models that are transparent, explainable, and well documented. Credit risk models must allow regulators, auditors, and business stakeholders to understand how predictions are generated.

Therefore, model interpretability is important because it:

* Supports regulatory compliance
* Enables model validation and auditing
* Improves trust in lending decisions
* Ensures risk assessments can be explained to stakeholders

A well-documented and interpretable model helps satisfy Basel II requirements for risk measurement and governance.

### Need for a Proxy Target Variable

The dataset does not contain a direct default label indicating whether a customer failed to repay a loan. Because supervised machine learning requires a target variable, a proxy target must be created.

Customer behavioral patterns such as Recency, Frequency, and Monetary (RFM) metrics can be used to identify potentially high-risk customers.

However, proxy-based prediction introduces several risks:

* The proxy may not perfectly represent actual default behavior
* Some low-risk customers may be incorrectly classified as high-risk
* Some high-risk customers may be incorrectly classified as low-risk
* Model performance depends on the quality of the proxy definition

Therefore, the proxy variable should be viewed as an approximation of credit risk rather than true default behavior.

### Model Trade-offs

| Logistic Regression  | Gradient Boosting         |
| -------------------- | ------------------------- |
| Easy to interpret    | Harder to interpret       |
| Basel II friendly    | Less transparent          |
| Faster training      | Higher predictive power   |
| Easier validation    | More complex tuning       |
| Easier documentation | More difficult to explain |

#### Trade-off Discussion

Logistic Regression provides high interpretability and is easier to validate in regulated financial environments. Gradient Boosting models often achieve better predictive performance but can be more difficult to explain and document.

In practice, financial institutions must balance predictive accuracy with regulatory requirements, model transparency, and business trust.

---

## Project Structure

credit-risk-model/

├── .github/workflows/ci.yml

├── data/

│ ├── raw/

│ └── processed/

├── notebooks/

│ └── eda.ipynb

├── src/

│ ├── data_processing.py

│ ├── train.py

│ ├── predict.py

│ └── api/

├── tests/

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── .gitignore

└── README.md
