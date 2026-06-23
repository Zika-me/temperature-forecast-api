from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
MODEL_PATH = Path("artifacts/lstm_temperature_model.pt")
SCALER_PATH = Path("artifacts/scaler.npy")
SEQ_LENGTH = 30


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)


def train_and_save_model(epochs: int = 20) -> Tuple[LSTMModel, MinMaxScaler]:
    """Train the same LSTM structure used in the notebook and save artifacts."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL)
    data = df["Temp"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    x, y = create_sequences(data_scaled)

    train_size = int(len(x) * 0.8)
    x_train = torch.FloatTensor(x[:train_size])
    y_train = torch.FloatTensor(y[:train_size])

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    torch.manual_seed(42)
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    np.save(SCALER_PATH, np.array([scaler.data_min_[0], scaler.data_max_[0]]))
    return model, scaler


def load_or_train_model() -> Tuple[LSTMModel, MinMaxScaler]:
    scaler = MinMaxScaler()
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        min_val, max_val = np.load(SCALER_PATH)
        scaler.fit(np.array([[min_val], [max_val]]))
        model = LSTMModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model, scaler
    return train_and_save_model()


def predict_next_temperature(model: LSTMModel, scaler: MinMaxScaler, temperatures: List[float]) -> float:
    if len(temperatures) != SEQ_LENGTH:
        raise ValueError(f"Exactly {SEQ_LENGTH} temperature values are required.")
    arr = np.array(temperatures, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    x = torch.FloatTensor(scaled.reshape(1, SEQ_LENGTH, 1))
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(x).numpy()
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]
    return round(float(prediction), 2)
