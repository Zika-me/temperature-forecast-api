# Microservice Assignment Submission

## Service Overview
This microservice exposes a PyTorch LSTM time-series forecasting model from my previous assignment notebook. The model predicts the next daily minimum temperature using the previous 30 daily minimum temperatures.

## General Input
The service accepts a JSON object containing one field named `last_30_temperatures`. This field must contain exactly 30 numeric temperature values in degrees Celsius. The values should be ordered from oldest to newest.

General input format:

```json
{
  "last_30_temperatures": [30 numeric temperature values]
}
```

## General Output
The service returns a JSON response containing the predicted next daily minimum temperature in Celsius, the number of input days used, the model name, and a short note about the input format.

General output format:

```json
{
  "predicted_next_min_temperature_celsius": 12.85,
  "input_days_used": 30,
  "model": "PyTorch LSTM time-series forecasting model from previous assignment notebook",
  "note": "Input temperatures are interpreted as degrees Celsius and must be ordered from oldest to newest."
}
```

## Specific Input Example
Use this example to invoke the service:

```json
{
  "last_30_temperatures": [13.4, 14.6, 12.5, 11.8, 13.1, 15.0, 14.2, 13.8, 12.9, 11.5, 10.8, 12.0, 13.7, 14.1, 15.3, 16.0, 14.9, 13.6, 12.7, 11.9, 10.5, 9.8, 11.1, 12.6, 13.3, 14.0, 15.1, 14.4, 13.2, 12.4]
}
```

## Specific Output Example
A successful invocation will return a response similar to this. The exact predicted value may vary slightly depending on deployment and model training.

```json
{
  "predicted_next_min_temperature_celsius": 12.75,
  "input_days_used": 30,
  "model": "PyTorch LSTM time-series forecasting model from previous assignment notebook",
  "note": "Input temperatures are interpreted as degrees Celsius and must be ordered from oldest to newest."
}
```

## Service URL
Replace the placeholder below with the deployed Render URL:

```text
https://YOUR-RENDER-SERVICE-NAME.onrender.com
```

## Easy Test Instructions for Instructor
The prediction endpoint is:

```text
https://YOUR-RENDER-SERVICE-NAME.onrender.com/predict
```

The interactive API documentation is available at:

```text
https://YOUR-RENDER-SERVICE-NAME.onrender.com/docs
```

The service can be tested with this `curl` command:

```bash
curl -X POST "https://YOUR-RENDER-SERVICE-NAME.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"last_30_temperatures":[13.4,14.6,12.5,11.8,13.1,15.0,14.2,13.8,12.9,11.5,10.8,12.0,13.7,14.1,15.3,16.0,14.9,13.6,12.7,11.9,10.5,9.8,11.1,12.6,13.3,14.0,15.1,14.4,13.2,12.4]}'
```

A successful response will be returned in JSON format with a predicted next-day minimum temperature.
