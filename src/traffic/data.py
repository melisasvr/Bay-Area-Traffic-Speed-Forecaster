"""Loading, cleaning and graph construction for PEMS-BAY.

Expected files in data_dir (either set works):
  PEMS-BAY.csv + PEMS-BAY-META.csv                 (Zenodo)
  pems-bay.h5 + graph_sensor_locations_bay.csv     (DCRNN original)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288


# ----------------------------------------------------------------- loading
SPEED_FILES = ("pems-bay.h5", "pems-bay.parquet", "PEMS-BAY.csv", "pems-bay.csv")
SENSOR_FILES = ("graph_sensor_locations_bay.csv", "PEMS-BAY-META.csv", "sensor_locations_bay.csv")


def _find(d: Path, names):
    for n in names:
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(f"None of {names} found in {d.resolve()} (see README, 'Data' section)")


def _read_speed_csv(path) -> pd.DataFrame:
    """Speed CSV: one timestamp column + one numeric column per sensor."""
    df = pd.read_csv(path)
    ts_col = None
    for c in df.columns:                                   # first text column that parses as dates
        if not pd.api.types.is_numeric_dtype(df[c]) and pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.99:
            ts_col = c
            break
    if ts_col is None:                                     # fall back: first column is the timestamp
        ts_col = df.columns[0]
    df.index = pd.to_datetime(df.pop(ts_col))
    return df.select_dtypes("number")


def load_speeds(data_dir) -> pd.DataFrame:
    path = _find(Path(data_dir), SPEED_FILES)
    if path.suffix == ".h5":
        df = pd.read_hdf(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = _read_speed_csv(path)
    df.columns = [str(int(float(c))) if str(c).replace(".", "", 1).isdigit() else str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df.sort_index().astype("float32")


def load_sensors(data_dir, columns) -> pd.DataFrame:
    """Sensor metadata (lat/lon) aligned to the speed-matrix column order.
    Handles: DCRNN's headerless graph_sensor_locations_bay.csv and Zenodo's PEMS-BAY-META.csv
    (columns are detected by name, so extra metadata columns are fine)."""
    path = _find(Path(data_dir), SENSOR_FILES)
    df = pd.read_csv(path)
    low = {c: str(c).lower() for c in df.columns}
    lat = next((c for c, l in low.items() if "lat" in l), None)
    lon = next((c for c, l in low.items() if l.startswith("lon") or "lng" in l), None)
    if lat and lon:
        pref = ("sensor_id", "sensorid", "id", "sensor", "station", "station_id", "user_id_1")
        idc = next((c for p in pref for c, l in low.items() if l == p), df.columns[0])
        df = df[[idc, lat, lon]]
    else:                                                  # headerless: ..., id, lat, lon
        df = pd.read_csv(path, header=None).iloc[:, -3:]
    df.columns = ["sensor_id", "latitude", "longitude"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df["sensor_id"] = df["sensor_id"].astype(int).astype(str)
    out = df.drop_duplicates("sensor_id").set_index("sensor_id").reindex(columns)
    if out.latitude.notna().mean() < 0.5:
        raise ValueError(f"Sensor ids in {path.name} do not match the speed file's columns. "
                         f"Speed columns look like {list(columns[:3])}; metadata ids like {list(df.sensor_id[:3])}.")
    return out

def load_adjacency(data_dir, columns):
    """Load DCRNN's adj_mx_bay.pkl (directed, Gaussian-kernel road-distance weights) aligned to
    the speed columns. Returns None if the file is absent. Only unpickle files you trust."""
    path = Path(data_dir) / "adj_mx_bay.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    ids = None
    if isinstance(obj, (list, tuple)):                      # DCRNN: [sensor_ids, id_to_ind, adj_mx]
        ids = obj[0] if len(obj) == 3 else None
        obj = next(o for o in reversed(obj) if hasattr(o, "shape") and len(o.shape) == 2)
    adj = obj.toarray() if hasattr(obj, "toarray") else np.asarray(obj)
    adj = adj.astype("float32")
    cols = list(columns)
    if ids is not None:
        pos = {str(int(i)): k for k, i in enumerate(ids)}
        if all(c in pos for c in cols):
            order = [pos[c] for c in cols]
            return adj[np.ix_(order, order)]
    if adj.shape[0] == len(cols):
        return adj                                          # assume same sensor order
    raise ValueError(f"adj_mx_bay.pkl shape {adj.shape} does not match {len(cols)} sensors")

# ------------------------------------------------------------ graph / space
def haversine_matrix(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    dlat, dlon = lat[:, None] - lat[None], lon[:, None] - lon[None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))   # km


def build_adjacency(data_dir, sensors: pd.DataFrame, sigma=None, eps=0.1) -> np.ndarray:
    """Gaussian-kernel adjacency (DCRNN-style). Uses road distances if available,
    otherwise haversine. Not used by LightGBM; this is the input for the GNN step."""
    ids = list(sensors.index)
    path = Path(data_dir) / "distances_bay_2017.csv"
    if path.exists():
        d = pd.read_csv(path)
        d = d.iloc[:, -3:]
        dist = np.full((len(ids), len(ids)), np.inf)
        pos = {s: i for i, s in enumerate(ids)}
        for a, b, c in d.itertuples(index=False):
            a, b = str(int(a)), str(int(b))
            if a in pos and b in pos:
                dist[pos[a], pos[b]] = c
        np.fill_diagonal(dist, 0)
    else:
        dist = haversine_matrix(sensors.latitude.to_numpy(), sensors.longitude.to_numpy())
    finite = dist[np.isfinite(dist)]
    sigma = sigma or finite.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < eps] = 0.0
    return adj.astype("float32")


# ---------------------------------------------------------------- cleaning
def _tod_dow(index):
    return ((index.hour * 60 + index.minute) // 5).to_numpy(), index.dayofweek.to_numpy()


def build_profile(x_train: pd.DataFrame) -> np.ndarray:
    """Median speed per (day-of-week, time-of-day, sensor) -> array (7, 288, S).
    Fitted on TRAIN ONLY, so using it as a feature / fill value leaks nothing."""
    tod, dow = _tod_dow(x_train.index)
    keys = pd.MultiIndex.from_arrays([dow, tod])
    g = x_train.set_axis(keys, axis=0).groupby(level=[0, 1]).median()
    g = g.reindex(pd.MultiIndex.from_product([range(7), range(STEPS_PER_DAY)]))
    g = g.fillna(x_train.median())               # slots never seen in train
    return g.to_numpy(np.float32).reshape(7, STEPS_PER_DAY, -1)


def clean_speeds(raw: pd.DataFrame, train_end: int, max_ffill: int = 6):
    """Returns (clean, valid_mask, profile).

    1. 0 / NaN = missing (PEMS convention) -> valid_mask False (kept for evaluation).
    2. Forward-fill short gaps (past values only -> causal).
    3. Remaining gaps -> train-only weekly profile.
    Training/eval rows use valid_mask at target time, so imputed values are never
    used as ground truth.
    """
    valid = raw.notna() & (raw > 0)
    x = raw.where(valid)
    profile = build_profile(x.iloc[:train_end])
    x = x.ffill(limit=max_ffill)
    tod, dow = _tod_dow(x.index)
    x = x.fillna(pd.DataFrame(profile[dow, tod], index=x.index, columns=x.columns))
    return x.astype("float32"), valid, profile


def chronological_bounds(n: int, train_frac: float, val_frac: float):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return train_end, val_end, n