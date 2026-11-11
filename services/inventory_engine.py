import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import ndtr   # standard normal CDF (scipy comes with scikit-learn)


# ============================================================
# ASSUMPTIONS (state these in your report)
# ============================================================
LEAD_TIME_DAYS = 7            # days between placing and receiving an order
SIM_COVER_RANGE = (2, 12)     # simulated stock = 2-12 days of average demand
SIM_SEED = 42                 # keeps the simulated stock identical on every run

HIGH_PROB = 0.50              # stock-out probability >= 50%  -> High
MEDIUM_PROB = 0.20            # stock-out probability >= 20%  -> Medium


def calculate_inventory_metrics(df):
    """
    Calculate inventory planning metrics using ML demand forecasts
    and stock on hand.

    Stock on hand is taken from, in this order:
      1. a 'current_stock' column already in df
      2. data/inventory.csv  (columns: item_id, store_id, current_stock)
      3. a reproducible simulation (the M5 dataset has no inventory data)
    """

    df = df.copy()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    # ========================================================
    # 7-DAY ML FORECAST
    # ========================================================
    forecast_df = pd.read_csv(data_dir / "future_forecasts.csv")

    forecast_summary = (
        forecast_df
        .groupby(["item_id", "store_id"])["predicted_sales"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_sales": "ml_forecast_7_day"})
    )

    df = df.drop(columns=["ml_forecast_7_day"], errors="ignore")
    df = df.merge(forecast_summary, on=["item_id", "store_id"], how="left")

    df["ml_forecast_7_day"] = (
        df["ml_forecast_7_day"]
        .fillna(df["avg_last_30_days"] * 7)
        .round(2)
    )

    # ========================================================
    # STOCK ON HAND
    # ========================================================
    inventory_path = data_dir / "inventory.csv"

    if "current_stock" in df.columns:
        pass
    elif inventory_path.exists():
        inv = pd.read_csv(inventory_path)[["item_id", "store_id", "current_stock"]]
        df = df.merge(inv, on=["item_id", "store_id"], how="left")
        df["current_stock"] = df["current_stock"].fillna(0)
    else:
        rng = np.random.default_rng(SIM_SEED)
        cover_days = rng.uniform(*SIM_COVER_RANGE, size=len(df))
        df["current_stock"] = (df["avg_last_30_days"] * cover_days).round()

    df["current_stock"] = df["current_stock"].round().astype(int)

    # ========================================================
    # DEMAND OVER THE LEAD TIME
    # ========================================================
    lead_time_demand = df["ml_forecast_7_day"] * LEAD_TIME_DAYS / 7
    lead_time_std = df["daily_std"] * np.sqrt(LEAD_TIME_DAYS)

    # ========================================================
    # REORDER POINT AND RECOMMENDED ORDER
    # ========================================================
    df["reorder_point"] = (lead_time_demand + df["safety_stock"]).round().astype(int)

    df["recommended_order"] = (
        (df["reorder_point"] - df["current_stock"]).clip(lower=0).astype(int)
    )

    # ========================================================
    # STOCK-OUT RISK (stock vs predicted demand)
    # ========================================================
    daily_forecast = (df["ml_forecast_7_day"] / 7).clip(lower=0.01)

    df["days_of_cover"] = (df["current_stock"] / daily_forecast).round(1)

    df["projected_shortfall"] = (
        (lead_time_demand - df["current_stock"]).clip(lower=0).round().astype(int)
    )

    sigma = lead_time_std.clip(lower=1e-6)
    z = (df["current_stock"] - lead_time_demand) / sigma
    df["stockout_probability"] = (1 - ndtr(z)).round(3)

    df["stockout_risk"] = np.select(
        [
            df["stockout_probability"] >= HIGH_PROB,
            df["stockout_probability"] >= MEDIUM_PROB,
        ],
        ["High", "Medium"],
        default="Low",
    )

    # Demand variability (coefficient of variation), kept for reference
    df["demand_risk"] = (
        df["daily_std"] / df["avg_last_30_days"].replace(0, 1)
    ).round(2)

    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================
    df["recommended_action"] = np.select(
        [df["stockout_risk"] == "High", df["stockout_risk"] == "Medium"],
        ["Urgent Reorder", "Plan Reorder"],
        default="Monitor",
    )

    return df