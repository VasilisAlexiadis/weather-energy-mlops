from pathlib import Path

from weather_energy_mlops.train import train_model


def main():
    metrics = train_model()
    print("Training finished.")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"R^2: {metrics['r2']:.3f}")
    print(f"Model saved to: {Path('models/model_latest.joblib').resolve()}")


if __name__ == "__main__":
    main()
