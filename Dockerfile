FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "weather_energy_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
