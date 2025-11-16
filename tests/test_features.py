from weather_energy_mlops.data_ingestion import fetch_weather_history, add_synthetic_energy_target
from weather_energy_mlops.features import make_supervised


def test_make_supervised_shapes():
    df = fetch_weather_history(
        latitude=37.98,
        longitude=23.72,
        start_date="2024-01-01",
        end_date="2024-01-20",
    )
    df = add_synthetic_energy_target(df)
    df_sup, feature_cols = make_supervised(df)
    assert not df_sup.empty
    assert "target_next" in df_sup.columns
    for col in feature_cols:
        assert col in df_sup.columns
