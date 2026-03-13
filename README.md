# 📈 Investment Bank Stock Analytics Dashboard
## Top 10 S&P 500 Stocks | Yahoo Finance | 2021–2026

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. App opens at `http://localhost:8501`

---

## How the Excel File Works

The app **automatically fetches live data from Yahoo Finance** on first run and:
- Saves it to `stock_prices_sp500_top10.xlsx` in the same folder
- Reuses the Excel file if Yahoo Finance is unreachable
- You can also manually regenerate it from the **Download Data** page

The Excel file contains:
| Sheet | Contents |
|-------|----------|
| `Overview` | All 10 tickers summary table |
| `AAPL` | Apple OHLCV + Daily Return % + Cumulative Return % + Chart |
| `MSFT` | Microsoft OHLCV + returns + chart |
| `NVDA` | NVIDIA … |
| `AMZN` | Amazon … |
| `GOOGL` | Alphabet … |
| `META` | Meta … |
| `TSLA` | Tesla … |
| `BRK-B` | Berkshire Hathaway B … |
| `JPM` | JPMorgan … |
| `UNH` | UnitedHealth … |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Market Overview | Normalised prices, candlestick, correlation heatmap, KPI cards, rolling volatility |
| 🔍 EDA Deep Dive | Return distributions, drawdowns, seasonality, outlier & volume anomaly detection |
| ⚙️ Feature Engineering | RSI, MACD, Bollinger Bands, OBV, feature correlation heatmap, data table |
| 🤖 Model Training | Linear, OLS, Ridge, Lasso, ElasticNet, Random Forest, XGBoost — metrics + charts |
| 💡 Investment Signals | Composite scoring, BUY/HOLD/AVOID, Sharpe/Sortino, portfolio allocation |
| 📥 Download Data | Excel regeneration, CSV exports, model metrics Excel |

---

## Models Used

| Model | Type | Key Feature |
|-------|------|-------------|
| Linear Regression | Baseline | No regularisation |
| OLS (statsmodels) | Statistical | p-values, R², F-stat |
| Ridge | L2 regularisation | Prevents overfitting |
| Lasso | L1 regularisation | Automatic feature selection |
| Elastic Net | L1+L2 | Best of both worlds |
| Random Forest | Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | Best accuracy |

**All models use strict time-series train/test split (80/20) — no lookahead bias.**

---

## Target Variable
`Next_5d_Return` = (Close[t+5] / Close[t]) - 1

Predicts the 5-day forward return for each stock.

---

*Disclaimer: For informational purposes only. Not financial advice.*
