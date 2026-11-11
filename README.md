# 🛒 Retail Replenishment Dashboard
### AI-Powered Demand Forecasting & Inventory Optimization

A Python-based retail analytics application designed to help retailers make smarter inventory decisions using demand forecasting, data analysis, and automated replenishment recommendations.

## 📌 Project Overview

The Retail Replenishment Dashboard helps retailers analyze product demand, forecast future sales, monitor inventory requirements, and generate replenishment recommendations.

The system uses historical sales data and forecasting models to support data-driven inventory planning, reduce potential stockouts, and improve inventory management.

## 🚀 Key Features

- **Demand Forecasting:** Predict future product demand using historical sales patterns.
- **Inventory Optimization:** Calculate inventory requirements using demand and safety-stock information.
- **Replenishment Recommendations:** Generate recommendations for products requiring restocking.
- **Sales Analysis:** Analyze historical sales and demand trends.
- **Forecast Comparison:** Compare actual demand with predicted demand.
- **Interactive Dashboard:** Visualize inventory and forecasting insights through Streamlit.
- **Data-Driven Decisions:** Support better inventory planning using analytical insights.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Interactive dashboard |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Jupyter Notebook | Data exploration |

## 📂 Project Structure

```text
Retail_Replenishment_Project/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── backtest_forecasts.csv
│   ├── calendar.csv
│   ├── demand_forecasts.csv
│   ├── future_forecasts.csv
│   └── replenishment_recommendations.csv
│
├── models/
│   ├── backtest.py
│   ├── demand_forecasting.py
│   └── future_forecasting.py
│
├── notebooks/
│
├── services/
│   ├── __init__.py
│   └── inventory_engine.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/rishabraj15760-bot/Retail-Replenishment-Dashboard.git
```

### 2. Navigate to the project directory

```bash
cd Retail-Replenishment-Dashboard
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open your browser at:

`http://localhost:8501`

## 📊 Dataset Information

The project uses historical retail sales and supporting datasets for demand forecasting and inventory analysis.

Important dataset fields include:

- `item_id` – Product identifier
- `store_id` – Store identifier
- `avg_last_30_days` – Average recent demand
- `expected_7_day_demand` – Forecasted seven-day demand
- `daily_std` – Daily demand variability
- `safety_stock` – Additional stock maintained to reduce stockout risk

**Note:** The original `sales_train_validation.csv` and `sell_prices.csv` datasets are excluded from this GitHub repository because they exceed GitHub's 100 MB file limit. Place the required datasets inside the `data/` directory locally before running workflows that depend on them.

## 🎯 Project Objectives

1. Improve retail inventory planning.
2. Forecast future product demand.
3. Generate intelligent replenishment recommendations.
4. Reduce the risk of overstocking and stockouts.
5. Provide an interactive platform for retail data analysis.

## 🔮 Future Enhancements

- Real-time inventory tracking.
- Advanced machine learning forecasting models.
- Automated stock replenishment alerts.
- Cloud-based dashboard deployment.
- Integration with retail management systems.

## 👨‍💻 Developer

**Rishab Raj**

GitHub: [@rishabraj15760-bot](https://github.com/rishabraj15760-bot)

---

⭐ If you find this project interesting, consider starring the repository!

**Built with Python, Machine Learning & Streamlit.**
