import requests
import pandas as pd
import numpy as np

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "Europe/Athens",
) -> pd.DataFrame:
    """Fetch daily historical weather data from Open-Meteo.

    Parameters
    ----------
    latitude, longitude : float
        Coordinates of the location.
    start_date, end_date : str
        ISO format dates YYYY-MM-DD.
    timezone : str
        Timezone string, default Europe/Athens.

    Returns
    -------
    DataFrame with columns:
    - date
    - temp_max, temp_min, temp_mean
    - precip_sum
    - wind_max
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
        ],
        "timezone": timezone,
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    if not daily:
        raise ValueError("No 'daily' data returned from Open-Meteo API")

    df = pd.DataFrame({
        "date": daily["time"],
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "temp_mean": daily["temperature_2m_mean"],
        "precip_sum": daily["precipitation_sum"],
        "wind_max": daily["windspeed_10m_max"],
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_synthetic_energy_target(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic 'energy_index' target column.

    This mimics an energy demand pattern influenced by weather.
    You can later replace this with real consumption data.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["energy_index"] = (
        10.0
        + 0.8 * df["temp_mean"]
        + 0.3 * df["precip_sum"]
        + 0.1 * df["wind_max"]
        + rng.normal(scale=3.0, size=len(df))
    )
    return df
