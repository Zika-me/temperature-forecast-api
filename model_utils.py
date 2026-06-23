from __future__ import annotations

from typing import List, Tuple

SEQ_LENGTH = 30


def load_or_train_model() -> Tuple[None, None]:
    return None, None


def predict_next_temperature(model, scaler, temperatures: List[float]) -> float:
    if len(temperatures) != SEQ_LENGTH:
        raise ValueError(f"Exactly {SEQ_LENGTH} temperature values are required.")

    weights = list(range(1, SEQ_LENGTH + 1))
    weighted_sum = sum(temp * weight for temp, weight in zip(temperatures, weights))
    prediction = weighted_sum / sum(weights)

    return round(float(prediction), 2)
