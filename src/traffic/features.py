"""Feature engineering.

Design: keep everything as (T, S) arrays (time x sensor), then materialise rows only for
sampled forecast origins `t`. One global model is trained across all sensors, with one
model per horizon (direct multi-horizon strategy -> no error accumulation).

Every feature at origin t uses data from <= t, except the explicitly *known-in-advance*
ones (time-of-day / day-of-week of the target and the train-fitted profile at target time).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import haversine_matrix, STEPS_PER_DAY


@dataclass
class Batch:
    X: pd.DataFrame
    y: np.ndarray      # true speed at t+h
    base: np.ndarray   # speed at t (persistence forecast; delta-target anchor)
    t: np.ndarray      # origin position in the timeline
    s: np.ndarray      # sensor position


class FeatureBuilder:
    def __init__(self, speeds: pd.DataFrame, valid: pd.DataFrame, profile: np.ndarray,
                 sensors: pd.DataFrame, train_end: int, cfg, adj=None):
        self.cfg, self.profile, self.adj = cfg, profile, adj
        self.x = speeds.to_numpy(np.float32)
        self.valid = valid.to_numpy(bool)
        self.T, self.S = self.x.shape
        idx = speeds.index
        self.tod = ((idx.hour * 60 + idx.minute) // 5).to_numpy()
        self.dow = idx.dayofweek.to_numpy()
        self._precompute(sensors, train_end)

    def _precompute(self, sensors, train_end):
        cfg, x = self.cfg, self.x
        xdf = pd.DataFrame(x)
        self.roll = {}
        for w in cfg.roll_windows:
            r = xdf.rolling(w, min_periods=max(2, w // 2))
            self.roll[f"roll_mean_{w}"] = r.mean().to_numpy(np.float32)
            self.roll[f"roll_std_{w}"] = r.std().to_numpy(np.float32)
        self.roll["valid_frac_12"] = (pd.DataFrame(self.valid.astype(np.float32))
                                      .rolling(12, min_periods=1).mean().to_numpy(np.float32))

        # Spatial: mean speed of the k geographically nearest sensors (swap for
        # road-network neighbours later). Self is excluded.
        lat, lon = sensors.latitude.to_numpy(), sensors.longitude.to_numpy()
        d = np.nan_to_num(haversine_matrix(lat, lon), nan=1e9)
        nbr = np.argsort(d, axis=1)[:, 1 : cfg.n_neighbors + 1]
        nb = np.zeros_like(x)
        for j in range(nbr.shape[1]):
            nb += x[:, nbr[:, j]]
        self.nb = nb / nbr.shape[1]
        self.net = x.mean(axis=1)                              # network-wide mean speed

        # Directed road-network neighbours from the adjacency matrix: weighted mean speed of the
        # top-k "downstream" (A[s, j]) and "upstream" (A[i, s]) sensors.
        self.dir_means = {}
        if self.adj is not None:
            A = self.adj.copy()
            np.fill_diagonal(A, 0)
            k = cfg.n_adj_neighbors
            for name, M in (("down", A), ("up", A.T)):
                idx = np.argsort(-M, axis=1)[:, :k]
                w = np.take_along_axis(M, idx, axis=1)
                num = np.zeros_like(x)
                for j in range(k):
                    num += x[:, idx[:, j]] * w[:, j]
                wsum = w.sum(axis=1)
                m = np.where(wsum[None, :] > 0, num / np.maximum(wsum, 1e-9), np.nan)
                self.dir_means[name] = m.astype(np.float32)

        tr = x[:train_end]                                      # static stats: train only
        self.static = {
            "lat": lat.astype(np.float32), "lon": lon.astype(np.float32),
            "sensor_mean": tr.mean(0), "sensor_std": tr.std(0),
        }

    def sample_idx(self, h, lo, hi, stride):
        """Origins whose target t+h stays inside [lo, hi) -> no leakage across split borders."""
        return np.arange(max(lo, self.cfg.max_lag), hi - h, stride)

    def make(self, h: int, t_idx: np.ndarray, require_target: bool = True) -> Batch:
        cfg, x, S, n = self.cfg, self.x, self.S, len(t_idx)
        tgt = t_idx + h
        # Train/eval only on really observed targets; live inference keeps every sensor.
        ok = self.valid[tgt] if require_target else np.ones((n, S), bool)
        flat = lambda a: np.broadcast_to(a, (n, S))[ok]
        f = {}

        lag = {k: x[t_idx - k] for k in cfg.lags}
        for k, v in lag.items():
            f[f"lag{k}"] = flat(v)
        for k in (3, 6, 12):
            if k in lag:
                f[f"diff_{k}"] = flat(lag[0] - lag[k])          # momentum
        for name, a in self.roll.items():
            f[name] = flat(a[t_idx])

        f["nb_mean_lag0"] = flat(self.nb[t_idx])
        f["nb_diff_3"] = flat(self.nb[t_idx] - self.nb[t_idx - 3])
        f["net_mean_lag0"] = flat(self.net[t_idx][:, None])
        for name, m in self.dir_means.items():
            f[f"{name}_mean_lag0"] = flat(m[t_idx])
            f[f"{name}_diff_3"] = flat(m[t_idx] - m[t_idx - 3])

        f["profile_at_target"] = flat(self.profile[self.dow[tgt], self.tod[tgt]])
        prof_now = self.profile[self.dow[t_idx], self.tod[t_idx]]
        f["dev_from_profile"] = flat(lag[0] - prof_now)         # how abnormal is traffic now
        f["yday_at_target"] = flat(x[tgt - STEPS_PER_DAY])      # same time yesterday

        ang = 2 * np.pi * self.tod[tgt] / STEPS_PER_DAY
        f["tod_sin"], f["tod_cos"] = flat(np.sin(ang)[:, None]), flat(np.cos(ang)[:, None])
        f["dow"] = flat(self.dow[tgt][:, None])
        f["is_weekend"] = flat((self.dow[tgt] >= 5)[:, None].astype(np.float32))
        for name, v in self.static.items():
            f[name] = flat(v[None, :])

        X = pd.DataFrame({k: v.astype(np.float32) for k, v in f.items()})
        return Batch(X=X, y=flat(x[tgt]), base=f["lag0"].copy(),
                     t=flat(t_idx[:, None]), s=flat(np.arange(S)[None, :]))
