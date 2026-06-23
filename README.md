# LSTM Temperature Forecasting Microservice

This project exposes the PyTorch LSTM model from the uploaded notebook as a FastAPI microservice.

## Model
The model forecasts the next daily minimum temperature using the previous 30 daily minimum temperatures. It uses the same LSTM architecture from the notebook:

- input size: 1
- hidden size: 50
- LSTM layers: 2
- output: 1 next-day temperature prediction

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Test locally

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"last_30_temperatures":[13.4,14.6,12.5,11.8,13.1,15.0,14.2,13.8,12.9,11.5,10.8,12.0,13.7,14.1,15.3,16.0,14.9,13.6,12.7,11.9,10.5,9.8,11.1,12.6,13.3,14.0,15.1,14.4,13.2,12.4]}'
```

## Deploy on Render

1. Upload these files to a GitHub repository.
2. Go to Render and create a new Web Service.
3. Connect the GitHub repository.
4. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

5. After deployment, Render will provide a URL like:

```text
https://lstm-temperature-service.onrender.com
```

Use that URL in the submission document.

## API endpoints

- `GET /` confirms the service is running.
- `GET /health` confirms model status.
- `POST /predict` returns the next temperature prediction.
