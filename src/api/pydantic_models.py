from pydantic import BaseModel


class PredictionRequest(BaseModel):
    CountryCode: int
    Amount: float
    Value: float
    PricingStrategy: int
    FraudResult: int
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    StdTransactionAmount: float
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int


class PredictionResponse(BaseModel):
    risk_probability: float
