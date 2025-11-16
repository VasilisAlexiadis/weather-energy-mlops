# Weather Energy MLOps

End-to-end example MLOps project built around a simple use case:

- Fetch historical daily weather data from the free Open-Meteo API
- Create a synthetic "energy demand index" target
- Train a RandomForest regressor to predict next-day energy index
- Serve predictions via a FastAPI web service
- Package everything in Docker
- Run tests and Docker builds via GitHub Actions CI

## Quickstart

```bash
# create and activate venv (example on Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

pip install -r requirements.txt

# train model (Athens coordinates and sample dates)
python -m src.main

# run API
uvicorn weather_energy_mlops.api:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to try the interactive Swagger UI.

## Docker

```bash
docker build -t weather-energy-mlops:latest .
docker run -p 8000:8000 weather-energy-mlops:latest
```

## GitHub Actions

A simple CI workflow is provided in `.github/workflows/ci.yml`:

- Install dependencies
- Run pytest
- Build Docker image

You can extend this with CD (deployment) as needed.
