"""Tiny PEMS-BAY-shaped dataset for smoke-testing the pipeline offline."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(0)
T, S = 42 * 288, 20
idx = pd.date_range("2017-01-02", periods=T, freq="5min")
h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
wd = (idx.dayofweek.to_numpy() < 5).astype(float)
dip = lambda c, w: np.exp(-((h - c) / w) ** 2)

common = np.zeros(T)
for t in range(1, T):
    common[t] = 0.97 * common[t - 1] + rng.normal(0, 0.8)
sev = rng.uniform(0.5, 1.2, S)
X = np.zeros((T, S))
for s in range(S):
    ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = 0.9 * ar[t - 1] + rng.normal(0, 1.0)
    X[:, s] = 66 - wd * sev[s] * (28 * dip(8, 1.1) + 30 * dip(17, 1.4)) + common + ar
    for _ in range(25):                       # incidents
        a = rng.integers(0, T - 30); X[a:a + rng.integers(6, 20), s] -= rng.uniform(10, 25)
X = np.clip(X, 5, 80)
X[rng.random((T, S)) < 0.005] = 0             # random dropouts
for _ in range(15):                           # sensor outages
    a, s = rng.integers(0, T - 100), rng.integers(0, S); X[a:a + rng.integers(30, 100), s] = 0

ids = [str(400000 + i) for i in range(S)]
out = Path("data/synthetic"); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(X.astype("float32"), index=idx, columns=ids).to_parquet(out / "pems-bay.parquet")
pd.DataFrame({"i": range(S), "sensor_id": ids, "latitude": 37.3 + 0.01 * np.arange(S) + rng.normal(0, 1e-3, S),
              "longitude": -122.0 + 0.008 * np.arange(S)}).to_csv(out / "graph_sensor_locations_bay.csv", index=False, header=False)
print("wrote", out)
