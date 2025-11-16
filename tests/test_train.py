from pathlib import Path

from weather_energy_mlops.train import train_model


def test_train_model_creates_file(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    metrics = train_model(
        start_date="2024-01-01",
        end_date="2024-02-01",
        model_path=model_path,
    )
    assert model_path.exists()
    assert "mae" in metrics and "r2" in metrics
