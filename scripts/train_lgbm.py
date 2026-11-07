"""Train + evaluate the LightGBM baseline on a single chronological split."""
import _path  # noqa
import argparse
from pathlib import Path

from src.traffic.config import Config
from src.traffic.data import load_speeds, load_sensors, load_adjacency, chronological_bounds
from src.traffic.pipeline import fit_eval

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/raw")
ap.add_argument("--out-dir", default="reports")
ap.add_argument("--model-dir", default="models")
ap.add_argument("--train-stride", type=int, default=None)
ap.add_argument("--fast", action="store_true", help="fewer/bigger trees: ~5x faster, slightly less accurate")
ap.add_argument("--no-adj", action="store_true", help="skip adjacency features (for ablation)")
a = ap.parse_args()

cfg = Config(data_dir=a.data_dir, out_dir=a.out_dir, model_dir=a.model_dir,
             train_stride=a.train_stride or (24 if a.fast else 6))
if a.fast:
    cfg.make_fast()
raw = load_speeds(cfg.data_dir)
sensors = load_sensors(cfg.data_dir, raw.columns)
adj = None if a.no_adj else load_adjacency(cfg.data_dir, raw.columns)
print(f"{raw.shape[0]} steps x {raw.shape[1]} sensors | {raw.index[0]} -> {raw.index[-1]}")
print(f"raw missing/zero: {((raw <= 0) | raw.isna()).values.mean():.2%}")
print("adjacency features:", "ON" if adj is not None else "OFF (no adj_mx_bay.pkl found or --no-adj)")

tr_end, va_end, te_end = chronological_bounds(len(raw), cfg.train_frac, cfg.val_frac)
metrics, imp = fit_eval(raw, sensors, cfg, tr_end, va_end, te_end, model_dir=cfg.model_dir, adj=adj)

Path(cfg.out_dir).mkdir(exist_ok=True)
metrics.to_csv(f"{cfg.out_dir}/metrics.csv", index=False)
imp.to_csv(f"{cfg.out_dir}/feature_importance.csv", index=False)
print("\n", metrics.round(3).to_string(index=False))
top = imp.groupby("feature").gain.sum()
top = (top / top.sum()).sort_values(ascending=False)
print("\nTop features (share of total gain):\n", top.head(10).round(3).to_string())