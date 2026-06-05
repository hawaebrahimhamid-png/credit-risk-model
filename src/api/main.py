from fastapi import FastAPI

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.predict import predict_risk

app = FastAPI(title="Credit Risk API")


@app.get("/")
def root():
    return {"message": "Credit Risk Model API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    probability = predict_risk(data.dict())

    print("INPUT:", data.dict())
    print("PROB:", probability)

    return {
        "risk_probability": round(float(probability), 6)
    }
