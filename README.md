# Bay Area Traffic Speed Forecaster

A robust machine learning pipeline for multi-horizon highway speed forecasting (15, 30, 45, and 60 minutes ahead) using the **PEMS-BAY** dataset (325 sensors, 5-minute frequency, January–June 2017).

---

## Architecture & Project Structure

```
src/traffic/
  config.py         # Global settings (horizons, lags, split ratios, LightGBM parameters)
  data.py           # Data loading, cleaning, train-only weekly profiles, and adjacency matrix
  features.py       # FeatureEngineering: lags, momentum, rolling statistics, network/adjacency means
  cv.py             # Rolling-origin (expanding window) cross-validation folds
  metrics.py        # Error metrics (MAE, RMSE, MAPE)
  pipeline.py       # End-to-end pipeline: cleaning -> features -> per-horizon LightGBM -> evaluation
scripts/
  train_lgbm.py     # Train & evaluate LightGBM on a single chronological split
  cv_lgbm.py        # Rolling-origin cross-validation script
  make_synthetic.py # Generator for smoke-testing without real data
app/
  streamlit_app.py  # Interactive Streamlit visualization dashboard
```

## Key Design Decisions

- **Direct Multi-Horizon Strategy:** Trains one global LightGBM model per horizon across all sensors, avoiding error accumulation.
- **Delta-Target Formulation:** Predicts the speed change `Δ = speed(t + h) - speed(t)`, strongly anchored on persistence.
- **Leakage Prevention:** Strict chronological splitting. Historical profiles, imputation statistics, and sensor means are derived strictly from the training split.
- **Robust Imputation:** `0`/`NaN` values are treated as missing; forward-filled up to 30 minutes, falling back to train-only weekly profiles.
- **Meaningful Baselines:** Evaluated against both Persistence and Historical Profile baselines.

## Setup & Installation

- Clone the repository and set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

- Place the required files inside `data/raw/`:

- `pems-bay.h5` — Speed matrix from DCRNN / TorchSpatiotemporal
- `graph_sensor_locations_bay.csv` — Sensor metadata
- `distances_bay_2017.csv` — Optional road distances for adjacency

> **Tip:** For smoke-testing without raw data, run `python scripts/make_synthetic.py` and pass `--data-dir data/synthetic`.

## Running the Pipeline

### 1. Training (Single Split)

Run the LightGBM model in fast mode:

```
python scripts/train_lgbm.py --fast
```

**With Adjacency Features (Default):**

```text
Model Performance Summary (MAE / RMSE / MAPE):
- 15 min: LightGBM MAE = 1.350  (vs. Persistence: 1.586)
- 30 min: LightGBM MAE = 1.688  (vs. Persistence: 2.169)
- 45 min: LightGBM MAE = 1.865  (vs. Persistence: 2.628)
- 60 min: LightGBM MAE = 1.961  (vs. Persistence: 3.037)
- Overall: LightGBM MAE = 1.716 (vs. Persistence: 2.355)
```

**Without Adjacency Features (`--no-adj`):**

```bash
python scripts/train_lgbm.py --fast --no-adj --out-dir reports_noadj --model-dir models_noadj
```

### 2. Cross-Validation

- Evaluate using rolling-origin expanding-window cross-validation:

```bash
python scripts/cv_lgbm.py --folds 3 --fast
```

### 3. Interactive Streamlit Dashboard

- Launch the visualization dashboard (requires trained model artifacts):

```bash
python scripts/train_lgbm.py --fast
streamlit run app/streamlit_app.py
```

- Open `http://localhost:8501` in your browser to inspect real-time forecasts, historical trends, and per-sensor test errors.

## Contributing
- Contributions are welcome! Please follow these steps to contribute:

1. Fork the project.
2. Create your feature branch:

   ```
   git checkout -b feature/AmazingFeature
   ```

3. Commit your changes:

   ```
   git commit -m 'Add some AmazingFeature'
   ```

4. Push to the branch:

   ```
   git push origin feature/AmazingFeature
   ```

5. Open a Pull Request.
- Please ensure your code follows the existing style guidelines and includes appropriate comments.

## License
- Distributed under the MIT License. 
- MIT License:
```
Copyright (c) 2026 Melisa Sever

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
