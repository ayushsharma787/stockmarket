# 🏗️ Kwality Construction — Financial Dashboard

Forensic bank statement analysis for M/S. Kwality Construction Company  
FY 2024-25 · HDFC Bank A/c 27718630000017 · Mussoorie Branch

## Files
```
main.py            → Streamlit dashboard (run this)
bank_data.py       → All extracted & categorized transaction data
requirements.txt   → Python dependencies
README.md          → This file
```

## Run
```bash
pip install -r requirements.txt
streamlit run main.py
```
Opens at http://localhost:8501

## 4 Dashboard Tabs

| Tab | What's in it |
|-----|-------------|
| 📊 Expense Deep Dive | Donut + bar charts for 10 categories, vendor drill-down, CA insights |
| 📈 Cash Flow & Revenue | Monthly inflows/outflows, net surplus/deficit, balance trend, revenue sources |
| 🔴 Suspicious Transactions | 8 flagged entities, severity ratings, Noorkhan pattern analysis |
| 🚀 Loss → Profit Roadmap | Waterfall profit bridge, savings donut, 17 recommendations, 90-day plan |
