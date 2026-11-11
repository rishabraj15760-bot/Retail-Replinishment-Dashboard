import sys
import os
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from services.inventory_engine import calculate_inventory_metrics


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Retail Replenishment Dashboard",
    page_icon="📦",
    layout="wide",
)

DATA_DIR = BASE_DIR / "data"

# Optional columns the app will look for as "stock on hand"
STOCK_COLUMN_CANDIDATES = [
    "current_stock", "stock_on_hand", "on_hand", "inventory", "current_inventory",
]


# ============================================================
# DATA LOADING (cached)
# ============================================================
def add_hierarchy(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive M5 hierarchy (category, dept) from item_id, e.g. FOODS_3_090."""
    parts = frame["item_id"].astype(str).str.split("_")
    if "cat_id" not in frame.columns:
        frame["category"] = parts.str[0]
    else:
        frame["category"] = frame["cat_id"]
    if "dept_id" not in frame.columns:
        frame["dept"] = parts.str[0] + "_" + parts.str[1]
    else:
        frame["dept"] = frame["dept_id"]
    return frame


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "replenishment_recommendations.csv")
    forecast = pd.read_csv(DATA_DIR / "future_forecasts.csv")
    df = calculate_inventory_metrics(df)
    return add_hierarchy(df), add_hierarchy(forecast)


@st.cache_data
def load_backtest():
    """
    Optional file: data/backtest_forecasts.csv
    Required columns: date, item_id, store_id, actual, predicted
    Optional columns: seasonal_naive, moving_avg (computed here if missing)
    """
    path = DATA_DIR / "backtest_forecasts.csv"
    if not path.exists():
        return None

    bt = pd.read_csv(path, parse_dates=["date"])
    bt = add_hierarchy(bt).sort_values(["item_id", "store_id", "date"])
    grp = bt.groupby(["item_id", "store_id"])["actual"]

    if "seasonal_naive" not in bt.columns:
        bt["seasonal_naive"] = grp.shift(7)
    if "moving_avg" not in bt.columns:
        bt["moving_avg"] = grp.transform(
            lambda s: s.shift(1).rolling(28, min_periods=7).mean()
        )
    return bt


df, future_forecast_df = load_data()
backtest_df = load_backtest()


# ============================================================
# METRIC HELPERS
# ============================================================
def aggregate(bt: pd.DataFrame, keys: list) -> pd.DataFrame:
    cols = ["actual", "predicted", "seasonal_naive", "moving_avg"]
    return bt.groupby(keys + ["date"], as_index=False)[cols].sum(min_count=1)


def accuracy_table(bt: pd.DataFrame, keys: list) -> pd.DataFrame:
    """MAE / RMSE / MASE for model and baselines at a hierarchy level.
    MASE scale = in-sample MAE of the lag-7 seasonal naive forecast,
    computed on the actuals of the evaluation window."""
    models = {
        "ML model": "predicted",
        "Seasonal naive": "seasonal_naive",
        "Moving average": "moving_avg",
    }
    agg = aggregate(bt, keys).dropna(subset=list(models.values()))
    if keys:
        series = list(agg.groupby(keys))
    else:
        series = [("all", agg)]

    rows = {name: {"MAE": [], "RMSE": [], "MASE": []} for name in models}
    for _, g in series:
        g = g.sort_values("date")
        scale = g["actual"].diff(7).abs().mean()
        for name, col in models.items():
            err = g["actual"] - g[col]
            mae = err.abs().mean()
            rows[name]["MAE"].append(mae)
            rows[name]["RMSE"].append(np.sqrt((err ** 2).mean()))
            rows[name]["MASE"].append(
                mae / scale if scale and not np.isnan(scale) else np.nan
            )

    out = pd.DataFrame(
        {name: {m: np.nanmean(v) for m, v in vals.items()} for name, vals in rows.items()}
    ).T
    return out.round(3)


# ============================================================
# TITLE
# ============================================================
st.title("📦 AI Retail Replenishment Dashboard")
st.markdown(
    """
    **AI-powered inventory intelligence system**

    Hierarchical demand forecasts, accuracy vs baselines, stock-out risk,
    inventory cost impact and replenishment recommendations.
    """
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🔎 Inventory Filters")

risk_options = sorted(df["stockout_risk"].unique())
risk_filter = st.sidebar.multiselect("Stock-out Risk", risk_options, default=risk_options)

store_options = sorted(df["store_id"].unique())
store_filter = st.sidebar.multiselect("Store", store_options, default=store_options)

cat_options = sorted(df["category"].unique())
cat_filter = st.sidebar.multiselect("Category", cat_options, default=cat_options)

st.sidebar.header("💰 Cost Assumptions")
stockout_cost = st.sidebar.number_input(
    "Stock-out cost per unit (lost margin)", min_value=0.0, value=2.0, step=0.5
)
holding_cost = st.sidebar.number_input(
    "Holding cost per unit per day", min_value=0.0, value=0.1, step=0.05
)

filtered_df = df[
    df["stockout_risk"].isin(risk_filter)
    & df["store_id"].isin(store_filter)
    & df["category"].isin(cat_filter)
]

if filtered_df.empty:
    st.warning("No products match the selected filters.")
    st.stop()

filtered_keys = filtered_df[["item_id", "store_id"]].drop_duplicates()


# ============================================================
# KPI CARDS (overall, then filtered)
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Products", f"{len(df):,}")
c2.metric("Recommended Order", f"{int(df['recommended_order'].sum()):,}")
c3.metric("High Stock-out Risk", f"{int((df['stockout_risk'] == 'High').sum()):,}")
c4.metric("Medium Stock-out Risk", f"{int((df['stockout_risk'] == 'Medium').sum()):,}")

st.divider()
st.subheader("📊 Inventory Summary (filtered)")
s1, s2, s3 = st.columns(3)
s1.metric("Filtered Products", f"{len(filtered_df):,}")
s2.metric("Recommended Units", f"{int(filtered_df['recommended_order'].sum()):,}")
s3.metric(
    "Urgent Reorders",
    f"{int((filtered_df['recommended_action'] == 'Urgent Reorder').sum()):,}",
)
s4 = st.columns(1)[0]
s4.metric("Projected Shortfall (units)", f"{int(filtered_df['projected_shortfall'].sum()):,}")
st.divider()


# ============================================================
# FORECAST VS ACTUAL + ACCURACY (needs backtest file)
# ============================================================
st.subheader("📈 Forecast vs Actual Demand")

LEVELS = {
    "Total": [],
    "Category": ["category"],
    "Department": ["dept"],
    "Store": ["store_id"],
    "Item-Store": ["item_id", "store_id"],
}

if backtest_df is None:
    st.info(
        "Add `data/backtest_forecasts.csv` with columns "
        "`date, item_id, store_id, actual, predicted` "
        "(optional: `seasonal_naive`, `moving_avg`) to enable forecast-vs-actual, "
        "accuracy metrics and cost impact."
    )
else:
    bt = backtest_df[
        backtest_df["store_id"].isin(store_filter)
        & backtest_df["category"].isin(cat_filter)
    ]

    level_name = st.selectbox("Hierarchy level", list(LEVELS.keys()))
    keys = LEVELS[level_name]

    agg = aggregate(bt, keys)
    if keys:
        labels = agg[keys].astype(str).agg(" | ".join, axis=1)
        choice = st.selectbox(f"Select {level_name}", sorted(labels.unique()))
        plot_df = agg[labels == choice]
    else:
        plot_df = agg

    plot_df = plot_df.groupby("date")[
        ["actual", "predicted", "seasonal_naive", "moving_avg"]
    ].sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["actual"], label="Actual", linewidth=2)
    ax.plot(plot_df.index, plot_df["predicted"], label="ML forecast", linewidth=2)
    ax.plot(plot_df.index, plot_df["seasonal_naive"], label="Seasonal naive",
            linestyle="--", alpha=0.6)
    ax.plot(plot_df.index, plot_df["moving_avg"], label="Moving average",
            linestyle=":", alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- accuracy ----------------
    st.subheader("🎯 Forecast Accuracy vs Baselines")
    acc = accuracy_table(bt, keys)
    st.dataframe(acc, use_container_width=True)
    if not acc.empty:
        best = acc["MASE"].idxmin()
        st.caption(
            f"Lowest MASE at **{level_name}** level: **{best}**. "
            "MASE < 1 means better than the in-sample seasonal naive."
        )

    # ---------------- cost impact ----------------
    st.subheader("💰 Inventory Cost Impact of Forecast Error")
    st.caption(
        "Under-forecast units × stock-out cost, plus over-forecast units × "
        "holding cost (one day of holding per unit)."
    )

    err = bt.copy()
    err["under_units"] = (err["actual"] - err["predicted"]).clip(lower=0)
    err["over_units"] = (err["predicted"] - err["actual"]).clip(lower=0)
    err["stockout_cost"] = err["under_units"] * stockout_cost
    err["holding_cost"] = err["over_units"] * holding_cost

    cost = err.groupby("category")[["stockout_cost", "holding_cost"]].sum()
    cost["total_cost"] = cost.sum(axis=1)

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.dataframe(cost.round(0), use_container_width=True)
    with cc2:
        fig, ax = plt.subplots()
        cost[["stockout_cost", "holding_cost"]].plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Cost")
        ax.set_xlabel("Category")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

st.divider()


# ============================================================
# CHARTS
# ============================================================
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("⚠️ Stock-out Risk Distribution")
    fig, ax = plt.subplots()
    filtered_df["stockout_risk"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Products")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    st.subheader("🏪 Replenishment by Store")
    fig, ax = plt.subplots()
    (
        filtered_df.groupby("store_id")["recommended_order"]
        .sum()
        .sort_values(ascending=False)
        .plot(kind="bar", ax=ax)
    )
    ax.set_xlabel("Store")
    ax.set_ylabel("Recommended Units")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("📦 Replenishment by Category")
cat_reorder = filtered_df.groupby("category")["recommended_order"].sum()
st.bar_chart(cat_reorder)


# ============================================================
# TOP REPLENISHMENT / URGENT
# ============================================================
st.subheader("🚨 Products Needing Attention")

table_cols = [
    "item_id", "category", "store_id", "avg_last_30_days", "ml_forecast_7_day",
    "current_stock", "days_of_cover", "projected_shortfall", "stockout_probability",
    "safety_stock", "reorder_point", "recommended_order",
    "stockout_risk", "recommended_action",
]
table_cols = [c for c in table_cols if c in filtered_df.columns]

st.dataframe(
    filtered_df.sort_values("recommended_order", ascending=False)[table_cols].head(20),
    use_container_width=True,
)

st.subheader("🔴 Urgent Reorder Items")
urgent_df = (
    filtered_df[filtered_df["recommended_action"] == "Urgent Reorder"]
    .sort_values("recommended_order", ascending=False)
    .head(10)
)
if len(urgent_df) > 0:
    st.dataframe(urgent_df[table_cols], use_container_width=True)
else:
    st.success("No urgent reorder items found for the selected filters.")


# ============================================================
# 7-DAY ML FORECAST (respects filters)
# ============================================================
st.divider()
st.subheader("🔮 7-Day ML Demand Forecast")

ff = future_forecast_df.merge(filtered_keys, on=["item_id", "store_id"], how="inner")

f1, f2, f3 = st.columns(3)
f1.metric("Forecast Horizon", "7 Days")
f2.metric("Forecasted Units", f"{int(ff['predicted_sales'].sum()):,}")
f3.metric("Products Forecasted", f"{ff[['item_id', 'store_id']].drop_duplicates().shape[0]:,}")

daily = ff.groupby("forecast_day")["predicted_sales"].sum()
st.line_chart(daily)

st.dataframe(
    ff[["item_id", "store_id", "forecast_day", "predicted_sales"]].head(50),
    use_container_width=True,
)


# ============================================================
# DOWNLOAD
# ============================================================
st.divider()
st.subheader("📥 Download Replenishment Report")
st.download_button(
    label="⬇️ Download Replenishment Report",
    data=filtered_df.to_csv(index=False),
    file_name="replenishment_report.csv",
    mime="text/csv",
)


# ============================================================
# WHAT-IF SIMULATOR (all filtered products)
# ============================================================
st.divider()
st.subheader("🧪 What-If Replenishment Simulator")
st.markdown("Simulate changes in demand and safety stock across the filtered products.")

sim1, sim2, sim3 = st.columns(3)
with sim1:
    demand_change = st.slider("📈 Demand Change (%)", -30, 50, 0, 5)
with sim2:
    safety_change = st.slider("🛡️ Safety Stock Change (%)", 0, 50, 0, 5)
with sim3:
    forecast_days = st.selectbox("📅 Forecast Horizon (days)", [7, 14, 30], index=0)

base_forecast = filtered_df["ml_forecast_7_day"].sum()
scenario_forecast = (base_forecast / 7) * forecast_days * (1 + demand_change / 100)

base_safety_total = filtered_df["safety_stock"].sum()
scenario_safety_total = base_safety_total * (1 + safety_change / 100)

scenario_replenishment = scenario_forecast + scenario_safety_total
base_replenishment = base_forecast + base_safety_total
change_units = scenario_replenishment - base_replenishment

r1, r2, r3, r4 = st.columns(4)
r1.metric("Base Demand", f"{int(base_forecast):,}")
r2.metric("Scenario Demand", f"{int(scenario_forecast):,}")
r3.metric("Scenario Replenishment", f"{int(scenario_replenishment):,}")
r4.metric("Change in Units", f"{int(change_units):+,}")

if demand_change > 0:
    st.warning(
        f"⚠️ Demand simulated to increase by {demand_change}%. Prepare for about "
        f"{int(scenario_replenishment):,} units."
    )
elif demand_change < 0:
    st.success(
        f"📉 Demand simulated to decrease by {abs(demand_change)}%. "
        "Replenishment requirement decreases accordingly."
    )
else:
    st.info("ℹ️ Current scenario uses the baseline ML forecast.")


# ============================================================
# PRODUCT-LEVEL WHAT-IF
# ============================================================
st.divider()
st.subheader("🎯 Product-Level What-If Analysis")
st.markdown("Select a product and store to see how demand changes affect its order.")

p1, p2 = st.columns(2)
with p1:
    selected_item = st.selectbox("📦 Select Product", sorted(filtered_df["item_id"].unique()))
with p2:
    available_stores = sorted(
        filtered_df[filtered_df["item_id"] == selected_item]["store_id"].unique()
    )
    selected_store = st.selectbox("🏪 Select Store", available_stores)

selected_product = filtered_df[
    (filtered_df["item_id"] == selected_item)
    & (filtered_df["store_id"] == selected_store)
].iloc[0]

st.markdown("### ⚙️ Scenario Settings")
sc1, sc2 = st.columns(2)
with sc1:
    product_demand_change = st.slider(
        "📈 Product Demand Change (%)", -30, 50, 20, 5, key="product_demand_change"
    )
with sc2:
    product_safety_change = st.slider(
        "🛡️ Safety Stock Change (%)", 0, 50, 10, 5, key="product_safety_change"
    )

base_demand = float(selected_product["ml_forecast_7_day"])
base_safety = float(selected_product["safety_stock"])
base_rop = float(selected_product["reorder_point"])
base_order = float(selected_product["recommended_order"])

scenario_demand = base_demand * (1 + product_demand_change / 100)
scenario_safety = base_safety * (1 + product_safety_change / 100)

# Reorder point = demand over horizon + safety stock. Scale the base ROP's
# demand part by the same change so both sides are computed consistently.
scenario_rop = base_rop + (scenario_demand - base_demand) + (scenario_safety - base_safety)

# Order = reorder point minus stock on hand (never negative)
stock_col = next((c for c in STOCK_COLUMN_CANDIDATES if c in filtered_df.columns), None)
if stock_col is not None:
    stock_on_hand = float(selected_product[stock_col])
    scenario_order = max(scenario_rop - stock_on_hand, 0.0)
else:
    # No stock column: shift the existing order by the change in reorder point
    scenario_order = max(base_order + (scenario_rop - base_rop), 0.0)

st.info(f"📦 Product: **{selected_item}**  |  🏪 Store: **{selected_store}**")

st.markdown("### 📊 Current vs Scenario")
comparison_df = pd.DataFrame({
    "Metric": ["7-Day ML Demand", "Safety Stock", "Reorder Point", "Recommended Order"],
    "Current": [round(base_demand), round(base_safety), round(base_rop), round(base_order)],
    "Scenario": [round(scenario_demand), round(scenario_safety), round(scenario_rop),
                 round(scenario_order)],
    "Change": [
        round(scenario_demand - base_demand),
        round(scenario_safety - base_safety),
        round(scenario_rop - base_rop),
        round(scenario_order - base_order),
    ],
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.markdown("### 📈 Scenario Impact")
i1, i2, i3 = st.columns(3)
i1.metric("Scenario Demand", f"{round(scenario_demand):,}",
          delta=f"{round(scenario_demand - base_demand):+,}")
i2.metric("Scenario Reorder Point", f"{round(scenario_rop):,}",
          delta=f"{round(scenario_rop - base_rop):+,}")
i3.metric("Scenario Order", f"{round(scenario_order):,}",
          delta=f"{round(scenario_order - base_order):+,}")

if product_demand_change >= 30:
    st.warning("🚨 High-demand scenario: consider more replenishment capacity and monitor closely.")
elif product_demand_change > 0:
    st.info("📈 Increased-demand scenario: additional replenishment recommended.")
elif product_demand_change < 0:
    st.success("📉 Lower-demand scenario: replenishment requirements are reduced.")
else:
    st.info("ℹ️ Baseline scenario: no demand adjustment applied.")