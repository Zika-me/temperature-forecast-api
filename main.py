from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_utils import SEQ_LENGTH, load_or_train_model, predict_next_temperature

app = FastAPI(
    title="LSTM Daily Minimum Temperature Forecast API",
    description="Microservice exposing a PyTorch LSTM model trained to forecast the next daily minimum temperature from the previous 30 days.",
    version="1.0.0",
)

model, scaler = load_or_train_model()


class TemperatureRequest(BaseModel):
    last_30_temperatures: List[float] = Field(
        ...,
        min_length=SEQ_LENGTH,
        max_length=SEQ_LENGTH,
        description="The previous 30 daily minimum temperatures in degrees Celsius, ordered from oldest to newest.",
        examples=[[13.4, 14.6, 12.5, 11.8, 13.1, 15.0, 14.2, 13.8, 12.9, 11.5, 10.8, 12.0, 13.7, 14.1, 15.3, 16.0, 14.9, 13.6, 12.7, 11.9, 10.5, 9.8, 11.1, 12.6, 13.3, 14.0, 15.1, 14.4, 13.2, 12.4]],
    )


class PredictionResponse(BaseModel):
    predicted_next_min_temperature_celsius: float
    input_days_used: int
    model: str
    note: str


@app.get("/")
def root():
    return {
        "message": "LSTM temperature forecasting service is running.",
        "try_docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TemperatureRequest):
    try:
        prediction = predict_next_temperature(model, scaler, request.last_30_temperatures)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return PredictionResponse(
        predicted_next_min_temperature_celsius=prediction,
        input_days_used=len(request.last_30_temperatures),
        model="PyTorch LSTM time-series forecasting model from previous assignment notebook",
        note="Input temperatures are interpreted as degrees Celsius and must be ordered from oldest to newest.",
    )
