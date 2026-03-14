# IB Stock Analytics Dashboard v6 — LSTM + Multi-Timeframe

## New in v6
- Extended history from 1993 for long-term charts (Yahoo Finance)
- **Multi-Timeframe Analysis page** with 7 sub-tabs:
  - Weekly candlestick + EMA 10/20/50 + RSI/MACD/Stochastic
  - Monthly candlestick + EMA 6/12/24 + RSI/MACD/ADX
  - Quarterly candlestick + EMA 4/8/12 + RSI
  - Monthly Return Calendar Heatmap (Year × Month)
  - Yearly Returns bar chart + heatmap (all 10 stocks)
  - Support & Resistance (pivot-based, Daily/Weekly/Monthly)
  - Cross-Timeframe Signal Table (Weekly/Monthly/Quarterly aligned)
- Deep Drill-Down enhanced with monthly/quarterly long-term charts
- Full yearly return history in Deep Drill-Down

## Data
| Source | Period | Use |
|--------|--------|-----|
| Yahoo Finance (daily) | 2015 → today | LSTM training, daily TA |
| Yahoo Finance (extended) | 1993 → today | MTF charts, long-term context |

## LSTM
- Single-layer LSTM hidden=48, seq_len=20
- Trained via BPTT + Adam, 10 epochs, gradient clipping
- RobustScaler on train-only (zero leakage)
- Ensemble: LSTM hidden states → HistGradientBoosting

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
