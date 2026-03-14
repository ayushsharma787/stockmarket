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
## Troubleshooting: `TypeError` from `px.histogram(..., **PLOT_DARK)`

If you see a Streamlit red error box around a line like:

```python
fig = px.histogram(..., **PLOT_DARK)
```

the issue is usually that `PLOT_DARK` contains **layout keys** (for example `paper_bgcolor`,
`plot_bgcolor`, `font`, `legend`) that are not valid constructor arguments for `plotly.express`.

### Fix

Create the figure with `px.histogram` first, then apply layout styling with `update_layout`:

```python
fig = px.histogram(
    fdf,
    x="Age",
    color="Edu Label",
    nbins=20,
    barmode="stack",
    color_discrete_sequence=["#3B82F6", "#D4A853", "#00D4AA"],
)
fig.update_layout(**PLOT_DARK)
```

This separates **data/trace arguments** (for `px.histogram`) from **layout/theme arguments**
(`update_layout`) and avoids the `TypeError`.pip install -r requirements.txt
streamlit run app.py
```
