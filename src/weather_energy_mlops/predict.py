from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd


def load_model(model_path: Path = Path("models/model_latest.joblib")) -> Tuple[object, List[str]]:
    """Load trained model and feature list from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Train a model first (e.g. `python -m src.main`)."
        )
    bundle: Dict = joblib.load(model_path)
    return bundle["model"], bundle["feature_cols"]


def predict_from_weather_row(
    row: Dict,
    model_path: Path = Path("models/model_latest.joblib"),
) -> float:
    """Run a single prediction given a dict of features."""
    model, feature_cols = load_model(model_path)
    X = pd.DataFrame([row])[feature_cols]
    y_hat = model.predict(X)[0]
    return float(y_hat)
