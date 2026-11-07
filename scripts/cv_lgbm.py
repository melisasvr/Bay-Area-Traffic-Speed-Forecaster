"""Rolling-origin cross-validation (expanding train window, leakage-safe)."""
import _path  # noqa
import argparse
import pandas as pd

from src.traffic.config import Config
from src.traffic.cv import expanding_window_folds
from src.traffic.data import load_speeds, load_sensors, load_adjacency
from src.traffic.pipeline import fit_eval

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/raw")
ap.add_argument("--folds", type=int, default=3)
ap.add_argument("--val-days", type=int, default=10)
ap.add_argument("--test-days", type=int, default=14)
ap.add_argument("--min-train-days", type=int, default=28)
ap.add_argument("--train-stride", type=int, default=None)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--no-adj", action="store_true")
a = ap.parse_args()

cfg = Config(data_dir=a.data_dir, train_stride=a.train_stride or (24 if a.fast else 6))
if a.fast:
    cfg.make_fast()
raw = load_speeds(cfg.data_dir); sensors = load_sensors(cfg.data_dir, raw.columns)
adj = None if a.no_adj else load_adjacency(cfg.data_dir, raw.columns)
print("adjacency features:", "ON" if adj is not None else "OFF")
D = 288
out = []
for i, (tr, va, te) in enumerate(expanding_window_folds(
        len(raw), a.folds, a.val_days * D, a.test_days * D, a.min_train_days * D)):
    print(f"fold {i}: train<{raw.index[tr]:%Y-%m-%d} val<{raw.index[va]:%Y-%m-%d} test<={raw.index[te-1]:%Y-%m-%d}")
    m, _ = fit_eval(raw, sensors, cfg, tr, va, te, adj=adj)
    out.append(m.assign(fold=i))
res = pd.concat(out)
res.to_csv("reports/cv_metrics.csv", index=False)
print(res.groupby(["model", "horizon_min"])[["MAE", "RMSE", "MAPE"]].agg(["mean", "std"]).round(3))