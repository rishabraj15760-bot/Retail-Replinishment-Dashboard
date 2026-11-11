import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. LOAD DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "sales_train_validation.csv"

print("Loading sales data...")

sales = pd.read_csv(DATA_PATH)

print("Dataset loaded!")
print("Shape:", sales.shape)


# ============================================================
# 2. IDENTIFY DAILY SALES COLUMNS
# ============================================================

day_columns = [
    col for col in sales.columns
    if col.startswith("d_")
]

print("Number of days:", len(day_columns))


# ============================================================
# 3. USE RECENT SALES
# ============================================================

# We use the latest 60 days for the first forecasting model.
recent_days = day_columns[-60:]

data = sales[
    ["item_id", "store_id"] + recent_days
].copy()


# ============================================================
# 4. CONVERT DATA FROM WIDE → LONG FORMAT
# ============================================================

data = data.melt(
    id_vars=["item_id", "store_id"],
    var_name="day",
    value_name="sales"
)

print("Long-format data:", data.shape)


# ============================================================
# 5. CREATE NUMERIC IDs
# ============================================================

data["item_code"] = data["item_id"].astype("category").cat.codes
data["store_code"] = data["store_id"].astype("category").cat.codes

data["day_number"] = (
    data["day"]
    .str.replace("d_", "")
    .astype(int)
)


# ============================================================
# 6. CREATE TIME-SERIES FEATURES
# ============================================================

data = data.sort_values(
    ["item_id", "store_id", "day_number"]
)

group = data.groupby(
    ["item_id", "store_id"],
    group_keys=False
)

data["lag_1"] = group["sales"].shift(1)

data["lag_7"] = group["sales"].shift(7)

data["rolling_7"] = group["sales"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)

data["rolling_30"] = group["sales"].transform(
    lambda x: x.shift(1).rolling(30).mean()
)

# Day of week approximation
data["day_of_week"] = data["day_number"] % 7


# ============================================================
# 7. REMOVE ROWS WITH MISSING FEATURES
# ============================================================

data = data.dropna()

print("Training rows:", len(data))


# ============================================================
# 8. DEFINE FEATURES AND TARGET
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
# 9. TRAIN / TEST SPLIT
# ============================================================

# Last 7 days are kept for validation.

split_day = data["day_number"].max() - 7

train = data[data["day_number"] <= split_day]

test = data[data["day_number"] > split_day]

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. TRAIN AI MODEL
# ============================================================

print("Training forecasting model...")

model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")


# ============================================================
# 11. EVALUATE MODEL
# ============================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

print()
print("===================================")
print("MODEL PERFORMANCE")
print("===================================")
print(f"MAE: {mae:.2f} units")
print("===================================")


# ============================================================
# 12. SHOW SAMPLE PREDICTIONS
# ============================================================

results = test[
    ["item_id", "store_id", "day", "sales"]
].copy()

results["predicted_sales"] = predictions

print()
print("Sample predictions:")
print(results.head(10))

# Save forecast predictions
results.to_csv(
    "data/demand_forecasts.csv",
    index=False
)

print("Forecast predictions saved successfully!")