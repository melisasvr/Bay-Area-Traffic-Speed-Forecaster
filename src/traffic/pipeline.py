"""One split -> clean, build features, train per-horizon LightGBM, evaluate vs baselines."""
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import clean_speeds
from .features import FeatureBuilder
from .metrics import evaluate

warnings.filterwarnings("ignore", message=".*eval_set.*")


def fit_eval(raw, sensors, cfg, train_end, val_end, test_end, horizons=None, model_dir=None, adj=None):
    horizons = horizons or cfg.horizons
    t0 = time.time()
    print("Cleaning data...", flush=True)
    clean, valid, profile = clean_speeds(raw, train_end, cfg.max_ffill)   # train-only stats
    print(f"Building features (rolling stats, neighbours)... [{time.time()-t0:.0f}s]", flush=True)
    fb = FeatureBuilder(clean, valid, profile, sensors, train_end, cfg, adj=adj)
    print(f"Feature builder ready [{time.time()-t0:.0f}s]", flush=True)

    rows, imps, pooled = [], [], {}
    for h in horizons:
        print(f"\n=== Horizon {h*5} min: assembling data [{time.time()-t0:.0f}s]", flush=True)
        tr = fb.make(h, fb.sample_idx(h, 0, train_end, cfg.train_stride))
        va = fb.make(h, fb.sample_idx(h, train_end, val_end, cfg.test_stride))
        te = fb.make(h, fb.sample_idx(h, val_end, test_end, cfg.test_stride))
        print(f"train rows={len(tr.X):,} val rows={len(va.X):,} test rows={len(te.X):,}; "
              f"training LightGBM [{time.time()-t0:.0f}s]", flush=True)

        # Predict the CHANGE from current speed: easier target, strong persistence anchor.
        model = lgb.LGBMRegressor(**cfg.lgbm)
        model.fit(tr.X, tr.y - tr.base, eval_set=[(va.X, va.y - va.base)],
                  eval_metric="l1",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)])
        pred = np.clip(te.base + model.predict(te.X), 0, 100)

        preds = {"persistence": te.base,
                 "historical_profile": te.X["profile_at_target"].to_numpy(),
                 "lightgbm": pred}
        for name, p in preds.items():
            rows.append({"model": name, "horizon_min": h * 5, **evaluate(te.y, p)})
            pooled.setdefault(name, []).append((te.y, p))
        imps.append(pd.DataFrame({"feature": tr.X.columns, "horizon_min": h * 5,
                                  "gain": model.booster_.feature_importance("gain")}))
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            (Path(model_dir) / f"lgbm_h{h * 5}.txt").write_text(
                model.booster_.model_to_string(), encoding="utf-8")
        print(f"  h={h*5:>2}min  LGBM MAE={rows[-1]['MAE']:.3f} "
              f"(persist {rows[-3]['MAE']:.3f}) trees={model.best_iteration_} [{time.time()-t0:.0f}s]",
              flush=True)

    for name, parts in pooled.items():   # overall = pooled across horizons
        y = np.concatenate([a for a, _ in parts]); p = np.concatenate([b for _, b in parts])
        rows.append({"model": name, "horizon_min": "all", **evaluate(y, p)})
    return pd.DataFrame(rows), pd.concat(imps)