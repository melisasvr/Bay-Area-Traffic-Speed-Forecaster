"""Single source of truth for pipeline settings (1 step = 5 minutes)."""
from dataclasses import dataclass, field


@dataclass
class Config:
    data_dir: str = "data/raw"
    out_dir: str = "reports"
    model_dir: str = "models"

    # Forecast horizons in steps: 3/6/9/12 -> 15/30/45/60 min
    horizons: tuple = (3, 6, 9, 12)

    # Feature settings
    lags: tuple = (0, 1, 2, 3, 6, 12, 24, 288)   # 288 = same time yesterday
    roll_windows: tuple = (6, 12, 36)            # 30 min, 1 h, 3 h
    n_neighbors: int = 5
    n_adj_neighbors: int = 3                     # upstream/downstream sensors from adjacency matrix

    # Missing-data handling
    max_ffill: int = 6                           # forward-fill gaps up to 30 min

    # Chronological split (fractions of the timeline)
    train_frac: float = 0.70
    val_frac: float = 0.10                       # remainder = test
    train_stride: int = 6                        # subsample train timestamps (RAM/speed)
    test_stride: int = 2

    lgbm: dict = field(default_factory=lambda: dict(
        objective="l1",            # optimises MAE directly; robust to incident spikes
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42,
    ))

    def make_fast(self):
        """Quicker, slightly less accurate LightGBM settings (for CV / iteration)."""
        self.lgbm.update(n_estimators=500, learning_rate=0.1, num_leaves=63)
        return self
    @property
    def max_lag(self) -> int:
        return max(self.lags)
