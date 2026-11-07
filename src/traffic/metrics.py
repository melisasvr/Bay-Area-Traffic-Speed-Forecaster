import numpy as np


def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p, min_speed=1.0):
    """MAPE in %, ignoring near-zero speeds to avoid division blow-ups."""
    m = y > min_speed
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)


def evaluate(y, p) -> dict:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p)}
