import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "sales_train_validation.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "future_forecasts.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Walmart sales dataset...")

sales = pd.read_csv(DATA_PATH)

print("Dataset shape:", sales.shape)


# ============================================================
# DAILY SALES COLUMNS
# ============================================================

day_columns = [
    col for col in sales.columns
    if col.startswith("d_")
]

print("Total historical days:", len(day_columns))


# Use latest 60 days
recent_days = day_columns[-60:]


# ============================================================
# PREPARE DATA
# ============================================================

data = sales[
    ["item_id", "store_id"] + recent_days
].copy()


data = data.melt(
    id_vars=["item_id", "store_id"],
    var_name="day",
    value_name="sales"
)


# ============================================================
# CREATE CONSISTENT CODES
# ============================================================

item_categories = sales["item_id"].astype("category")

item_code_map = dict(
    zip(
        item_categories.cat.categories,
        range(len(item_categories.cat.categories))
    )
)

store_categories = sales["store_id"].astype("category")

store_code_map = dict(
    zip(
        store_categories.cat.categories,
        range(len(store_categories.cat.categories))
    )
)


data["item_code"] = data["item_id"].map(item_code_map)

data["store_code"] = data["store_id"].map(store_code_map)


data["day_number"] = (
    data["day"]
    .str.replace("d_", "")
    .astype(int)
)


# ============================================================
# SORT
# ============================================================

data = data.sort_values(
    ["item_id", "store_id", "day_number"]
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

group = data.groupby(
    ["item_id", "store_id"],
    group_keys=False
)


data["lag_1"] = (
    group["sales"]
    .shift(1)
)


data["lag_7"] = (
    group["sales"]
    .shift(7)
)


data["rolling_7"] = (
    group["sales"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(7)
        .mean()
    )
)


data["rolling_30"] = (
    group["sales"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(30)
        .mean()
    )
)


data["day_of_week"] = (
    data["day_number"] % 7
)


data = data.dropna()


# ============================================================
# MODEL FEATURES
# ============================================================

features = [
    "item_code",
    "store_code",
    "day_number",
    "day_of_week",
    "lag_1",
    "lag_7",
    "rolling_7",
    "rolling_30"
]


X = data[features]

y = data["sales"]


# ============================================================
# TRAIN MODEL
# ============================================================

print("Training demand forecasting model...")


model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42
)


model.fit(
    X,
    y
)


print("Model trained successfully.")


# ============================================================
# CREATE CURRENT STATE
# ============================================================

print("Preparing future forecasting state...")


# Keep the latest 30 days for every item/store
history = sales[
    ["item_id", "store_id"] + day_columns[-30:]
].copy()


# Convert sales history to numpy array
history_values = history[
    day_columns[-30:]
].values.astype(float)


item_ids = history["item_id"].values

store_ids = history["store_id"].values


item_codes = np.array([
    item_code_map[x]
    for x in item_ids
])


store_codes = np.array([
    store_code_map[x]
    for x in store_ids
])


# Last historical day
last_day_number = int(
    day_columns[-1].replace("d_", "")
)


# ============================================================
# FUTURE FORECAST
# ============================================================

print("Generating 7-day future forecasts...")


all_forecasts = []


for step in range(1, 8):

    future_day = (
        last_day_number + step
    )


    # --------------------------------------------------------
    # Calculate features for ALL products at once
    # --------------------------------------------------------

    lag_1 = history_values[:, -1]

    lag_7 = history_values[:, -7]

    rolling_7 = (
        history_values[:, -7:]
        .mean(axis=1)
    )

    rolling_30 = (
        history_values[:, -30:]
        .mean(axis=1)
    )

    day_of_week = (
        future_day % 7
    )


    # --------------------------------------------------------
    # Create prediction dataframe
    # --------------------------------------------------------

    prediction_data = pd.DataFrame({

        "item_code": item_codes,

        "store_code": store_codes,

        "day_number":
            np.full(
                len(item_ids),
                future_day
            ),

        "day_of_week":
            np.full(
                len(item_ids),
                day_of_week
            ),

        "lag_1": lag_1,

        "lag_7": lag_7,

        "rolling_7": rolling_7,

        "rolling_30": rolling_30
    })


    # --------------------------------------------------------
    # Predict ALL products together
    # --------------------------------------------------------

    predictions = model.predict(
        prediction_data[features]
    )


    # Prevent negative demand
    predictions = np.maximum(
        predictions,
        0
    )


    predictions = np.round(
        predictions,
        2
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    day_forecast = pd.DataFrame({

        "item_id": item_ids,

        "store_id": store_ids,

        "day_number":
            future_day,

        "forecast_day":
            f"d_{future_day}",

        "predicted_sales":
            predictions
    })


    all_forecasts.append(
        day_forecast
    )


    # --------------------------------------------------------
    # Add today's predictions to history
    # --------------------------------------------------------

    history_values = np.column_stack(
        [
            history_values[:, 1:],
            predictions
        ]
    )


    print(
        f"Day {step}/7 forecast completed."
    )


# ============================================================
# COMBINE FORECASTS
# ============================================================

future_forecasts = pd.concat(
    all_forecasts,
    ignore_index=True
)


# ============================================================
# SAVE FILE
# ============================================================

future_forecasts.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("============================================")
print("FUTURE FORECAST COMPLETED")
print("============================================")

print(
    "Forecast rows:",
    len(future_forecasts)
)

print(
    "Saved to:",
    OUTPUT_PATH
)


# ============================================================
# SAMPLE
# ============================================================

print()
print("Sample forecasts:")

print(
    future_forecasts.head(20)
)