from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predict import predict_from_weather_row

app = FastAPI(title="Weather Energy Forecaster API")


class WeatherFeatures(BaseModel):
    temp_mean: float
    temp_max: float
    temp_min: float
    precip_sum: float
    wind_max: float


class PredictionResponse(BaseModel):
    predicted_energy_index: float


@app.get("/")
def root():
    return {"message": "Weather Energy Forecaster API. See /docs for Swagger UI."}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WeatherFeatures):
    try:
        y_hat = predict_from_weather_row(features.dict())
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return PredictionResponse(predicted_energy_index=y_hat)
