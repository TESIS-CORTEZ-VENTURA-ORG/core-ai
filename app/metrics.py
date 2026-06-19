"""Forecasting accuracy metrics."""

from __future__ import annotations


def smape(y_true: list[float], y_pred: list[float]) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Formula: mean(2 * |y - yhat| / (|y| + |yhat|)) * 100

    Pairs where both y and yhat are zero are skipped (denominator = 0).
    Returns 0.0 when all pairs are skipped.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    total = 0.0
    count = 0

    for y, yhat in zip(y_true, y_pred):
        denom = abs(y) + abs(yhat)
        if denom == 0.0:
            continue
        total += 2.0 * abs(y - yhat) / denom
        count += 1

    if count == 0:
        return 0.0

    return (total / count) * 100.0


def mape(y_true: list[float], y_pred: list[float]) -> float:
    """
    Mean Absolute Percentage Error.

    Pairs where y == 0 are skipped to avoid division by zero.
    Returns 0.0 when all pairs are skipped.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    total = 0.0
    count = 0

    for y, yhat in zip(y_true, y_pred):
        if y == 0.0:
            continue
        total += abs(y - yhat) / abs(y)
        count += 1

    if count == 0:
        return 0.0

    return (total / count) * 100.0
