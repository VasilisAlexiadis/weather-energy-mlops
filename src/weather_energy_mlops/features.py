import pandas as pd
from typing import List, Tuple


def make_supervised(
    df: pd.DataFrame,
    target_col: str = "energy_index",
    horizon_days: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Turn time series into supervised learning dataset.

    We predict the next-day (or next `horizon_days`) energy index
    using today's weather features.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df = df.copy()
    df["target_next"] = df[target_col].shift(-horizon_days)
    df = df.dropna(subset=["target_next"])

    feature_cols = ["temp_mean", "temp_max", "temp_min", "precip_sum", "wind_max"]
    cols = ["date"] + feature_cols + ["target_next"]
    return df[cols], feature_cols
