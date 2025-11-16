from weather_energy_mlops.data_ingestion import fetch_weather_history, add_synthetic_energy_target


def test_fetch_weather_history_basic():
    df = fetch_weather_history(
        latitude=37.98,
        longitude=23.72,
        start_date="2024-01-01",
        end_date="2024-01-05",
    )
    assert not df.empty
    assert {"date", "temp_mean", "temp_max", "temp_min", "precip_sum", "wind_max"}.issubset(df.columns)


def test_add_synthetic_energy_target():
    df = fetch_weather_history(
        latitude=37.98,
        longitude=23.72,
        start_date="2024-01-01",
        end_date="2024-01-05",
    )
    df2 = add_synthetic_energy_target(df)
    assert "energy_index" in df2.columns
    assert df2["energy_index"].notna().all()
