"""Inference helpers shared by the Streamlit app (and any future API)."""
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config
from .data import load_speeds, load_sensors, chronological_bounds, clean_speeds, load_adjacency
from .features import FeatureBuilder


@dataclass
class Context:
    cfg: Config
    raw: pd.DataFrame
    clean: pd.DataFrame
    sensors: pd.DataFrame
    fb: FeatureBuilder
    boosters: dict
    train_end: int
    val_end: int
    test_end: int


def load_context(data_dir: str, model_dir: str = "models") -> Context:
    cfg = Config(data_dir=data_dir, model_dir=model_dir)
    raw = load_speeds(data_dir)
    sensors = load_sensors(data_dir, raw.columns)
    tr, va, te = chronological_bounds(len(raw), cfg.train_frac, cfg.val_frac)   # same split as training
    clean, valid, profile = clean_speeds(raw, tr, cfg.max_ffill)
    fb = FeatureBuilder(clean, valid, profile, sensors, tr, cfg, adj=load_adjacency(data_dir, raw.columns))

    boosters = {}
    for h in cfg.horizons:
        path = Path(model_dir) / f"lgbm_h{h * 5}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/train_lgbm.py first")
        # Load via Python (not a file path) so non-ASCII folders like 'Masaüstü' work on Windows.
        boosters[h] = lgb.Booster(model_str=path.read_text(encoding="utf-8"))

    if len(boosters) != len(cfg.horizons):
        raise RuntimeError(f"Expected {len(cfg.horizons)} models, loaded {len(boosters)}")
    return Context(cfg, raw, clean, sensors, fb, boosters, tr, va, te)


def forecast(ctx: Context, t: int) -> pd.DataFrame:
    """Forecast every sensor from origin position t. Rows=sensors, cols=minutes ahead."""
    out = {}
    for h, booster in ctx.boosters.items():
        b = ctx.fb.make(h, np.array([t]), require_target=False)
        out[h * 5] = np.clip(b.base + booster.predict(b.X[booster.feature_name()]), 0, 100)
    return pd.DataFrame(out, index=ctx.clean.columns)


def sensor_metrics(ctx: Context, stride: int = 6) -> pd.DataFrame:
    """Per-sensor, per-horizon test-set errors for LightGBM vs persistence."""
    rows = []
    for h, booster in ctx.boosters.items():
        te = ctx.fb.make(h, ctx.fb.sample_idx(h, ctx.val_end, ctx.test_end, stride))
        p = np.clip(te.base + booster.predict(te.X[booster.feature_name()]), 0, 100)
        d = pd.DataFrame({"s": te.s, "y": te.y, "ae": np.abs(te.y - p), "se": (te.y - p) ** 2,
                          "pae": np.abs(te.y - te.base)})
        d["ape"] = np.where(d.y > 1, d.ae / d.y.clip(lower=1) * 100, np.nan)
        g = d.groupby("s").agg(MAE=("ae", "mean"), MSE=("se", "mean"),
                               MAPE=("ape", "mean"), persistence_MAE=("pae", "mean"))
        g["RMSE"] = np.sqrt(g.pop("MSE"))
        g["horizon_min"], g["sensor_id"] = h * 5, ctx.clean.columns[g.index]
        rows.append(g.reset_index(drop=True))
    return pd.concat(rows)