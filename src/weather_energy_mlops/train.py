from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import DEFAULT_LOCATION, DEFAULT_START_DATE, DEFAULT_END_DATE
from .data_ingestion import fetch_weather_history, add_synthetic_energy_target
from .features import make_supervised

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def train_model(
    latitude: float = DEFAULT_LOCATION.latitude,
    longitude: float = DEFAULT_LOCATION.longitude,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    model_path: Path = MODELS_DIR / "model_latest.joblib",
) -> Dict[str, float]:
    """Train a RandomForest model on weather-based synthetic energy data."""
    # Fetch and prepare data
    df_weather = fetch_weather_history(latitude, longitude, start_date, end_date)
    df = add_synthetic_energy_target(df_weather)

    df_sup, feature_cols = make_supervised(df, target_col="energy_index")
    X = df_sup[feature_cols]
    y = df_sup["target_next"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        model_path,
    )

    metrics = {
        "mae": mae,
        "r2": r2,
    }
    return metrics
