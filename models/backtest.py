import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# SETTINGS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sales_train_validation.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "backtest_forecasts.csv"

HISTORY_DAYS = 150     # days of history used (training + holdout)
HOLDOUT_DAYS = 28      # evaluation window (4 forecast origins x 7 days)
HORIZON = 7            # forecast horizon, same as the dashboard
WINDOW = 30            # longest feature window (rolling_30)

M5_START = pd.Timestamp("2011-01-29")   # date of d_1 in M5


# ============================================================
# LOAD
# ============================================================
print("Loading sales data...")
sales = pd.read_csv(DATA_PATH)

day_cols = [c for c in sales.columns if c.startswith("d_")][-HISTORY_DAYS:]
S = sales[day_cols].values.astype(np.float32)        # (series, days)
day_nums = np.array([int(c[2:]) for c in day_cols])

item_ids = sales["item_id"].values
store_ids = sales["store_id"].values
item_codes = sales["item_id"].astype("category").cat.codes.values
store_codes = sales["store_id"].astype("category").cat.codes.values

n_days = S.shape[1]
train_end = n_days - HOLDOUT_DAYS    # first holdout day index
print("Series:", S.shape[0], "| days:", n_days, "| train until index", train_end)


# ============================================================
# FEATURES (same as your future_forecasting.py, minus day_number)
# ============================================================
FEATURES = [
    "item_code", "store_code", "day_of_week",
    "lag_1", "lag_7", "rolling_7", "rolling_30",
]


def make_features(window, day_number):
    """window = last WINDOW days of sales for every series, shape (n, WINDOW)."""
    return pd.DataFrame({
        "item_code": item_codes,
        "store_code": store_codes,
        "day_of_week": np.full(len(item_codes), day_number % 7),
        "lag_1": window[:, -1],
        "lag_7": window[:, -7],
        "rolling_7": window[:, -7:].mean(axis=1),
        "rolling_30": window.mean(axis=1),
    })


# ============================================================
# TRAIN (only on data BEFORE the holdout window)
# ============================================================
print("Building training set...")
frames = []
for t in range(WINDOW, train_end):
    f = make_features(S[:, t - WINDOW:t], day_nums[t])
    f["target"] = S[:, t]
    frames.append(f)

train = pd.concat(frames, ignore_index=True)
print("Training rows:", len(train))

model = HistGradientBoostingRegressor(
    max_iter=100, learning_rate=0.08, max_leaf_nodes=31, random_state=42
)
model.fit(train[FEATURES], train["target"])
print("Model trained.")


# ============================================================
# ROLLING-ORIGIN BACKTEST (recursive 7-day forecasts, like production)
# ============================================================
print("Running backtest...")
records = []

for origin in range(train_end, n_days, HORIZON):
    window = S[:, origin - WINDOW:origin].copy()
    moving_avg = S[:, origin - 28:origin].mean(axis=1)    # flat 28-day average

    for h in range(HORIZON):
        t = origin + h
        if t >= n_days:
            break

        pred = np.maximum(model.predict(make_features(window, day_nums[t])[FEATURES]), 0)

        records.append(pd.DataFrame({
            "date": M5_START + pd.to_timedelta(day_nums[t] - 1, unit="D"),
            "item_id": item_ids,
            "store_id": store_ids,
            "actual": S[:, t],
            "predicted": np.round(pred, 3),
            "seasonal_naive": S[:, t - 7],     # same weekday last week (known at origin)
            "moving_avg": np.round(moving_avg, 3),
        }))

        # recursive step: feed the prediction back in
        window = np.column_stack([window[:, 1:], pred])

bt = pd.concat(records, ignore_index=True)
bt.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH, "| rows:", len(bt))


# ============================================================
# QUICK SANITY CHECK (item-store level)
# ============================================================
print("\n=== Item-store level accuracy ===")
for name, col in [("ML model", "predicted"),
                  ("Seasonal naive", "seasonal_naive"),
                  ("Moving average", "moving_avg")]:
    err = bt["actual"] - bt[col]
    print(f"{name:15s} MAE={err.abs().mean():.3f}  RMSE={np.sqrt((err**2).mean()):.3f}")