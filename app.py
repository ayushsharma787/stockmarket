"""
Investment Bank Stock Analytics Dashboard
Top 10 S&P 500 Stocks | Yahoo Finance Data | Cut-off: 14 March 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, warnings, io
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IB Stock Analytics | S&P 500 Top 10",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
TICKERS = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK-B','JPM','UNH']
META_INFO = {
    'AAPL':  {'name':'Apple Inc.',            'sector':'Technology'},
    'MSFT':  {'name':'Microsoft Corp.',        'sector':'Technology'},
    'NVDA':  {'name':'NVIDIA Corp.',           'sector':'Semiconductors'},
    'AMZN':  {'name':'Amazon.com Inc.',        'sector':'Consumer Disc.'},
    'GOOGL': {'name':'Alphabet Inc.',          'sector':'Communication'},
    'META':  {'name':'Meta Platforms',         'sector':'Communication'},
    'TSLA':  {'name':'Tesla Inc.',             'sector':'Consumer Disc.'},
    'BRK-B': {'name':'Berkshire Hathaway B',  'sector':'Financials'},
    'JPM':   {'name':'JPMorgan Chase',         'sector':'Financials'},
    'UNH':   {'name':'UnitedHealth Group',     'sector':'Healthcare'},
}
START_DATE = "2021-01-01"
END_DATE   = "2026-03-14"
EXCEL_PATH = "stock_prices_sp500_top10.xlsx"

SECTOR_COLORS = {
    'Technology':'#667eea','Semiconductors':'#f093fb','Consumer Disc.':'#4facfe',
    'Communication':'#43e97b','Financials':'#fa709a','Healthcare':'#fee140'
}
TICKER_COLORS = {
    'AAPL':'#667eea','MSFT':'#764ba2','NVDA':'#f093fb','AMZN':'#4facfe',
    'GOOGL':'#43e97b','META':'#0ea5e9','TSLA':'#fa709a','BRK-B':'#fee140',
    'JPM':'#a8edea','UNH':'#ff9a9e'
}

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#0d1117}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
.kpi{background:linear-gradient(135deg,#161b22,#1c2333);border:1px solid #30363d;
     border-radius:12px;padding:16px;text-align:center;margin:4px}
.kpi-lbl{color:#8b949e;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px}
.kpi-val{color:#e6edf3;font-size:20px;font-weight:700;margin-top:4px}
.kpi-pos{color:#3fb950;font-size:12px}.kpi-neg{color:#f85149;font-size:12px}
.sec{color:#58a6ff;font-size:17px;font-weight:700;border-left:4px solid #58a6ff;
     padding-left:12px;margin:20px 0 10px}
.insight{background:#161b22;border-left:4px solid #3fb950;border-radius:6px;
         padding:12px 16px;margin:8px 0;color:#8b949e;font-size:13px}
.buy{background:#0d2818;color:#3fb950;border-radius:20px;padding:3px 14px;font-weight:700}
.hold{background:#2d2000;color:#d29922;border-radius:20px;padding:3px 14px;font-weight:700}
.avoid{background:#2d0000;color:#f85149;border-radius:20px;padding:3px 14px;font-weight:700}
div[data-testid="stMetricValue"]{color:#e6edf3}
</style>""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo(tickers, start, end):
    try:
        import yfinance as yf
        frames = {}
        for t in tickers:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                frames[t] = df[['Open','High','Low','Close','Volume']].copy()
        return frames, None
    except Exception as e:
        return None, str(e)

def save_excel(frames, path):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import LineChart, Reference

    def bdr():
        s = Side(style='thin', color='30363d')
        return Border(left=s,right=s,top=s,bottom=s)

    wb = openpyxl.Workbook()

    # ── Cover ──
    ws = wb.active
    ws.title = "Overview"
    ws.sheet_view.showGridLines = False
    for col,w in zip('ABCDEF',[3,22,32,14,16,16]):
        ws.column_dimensions[col].width = w

    ws.row_dimensions[2].height = 44
    c = ws.cell(2,2,"S&P 500 Top 10 — Historical Stock Prices | Source: Yahoo Finance")
    c.font = Font(bold=True,color="E6EDF3",size=16,name='Calibri')
    c.fill = PatternFill("solid",fgColor="0D1117")
    c.alignment = Alignment(horizontal='left',vertical='center')
    ws.merge_cells('B2:F2')

    ws.row_dimensions[3].height = 20
    c2 = ws.cell(3,2,f"Period: {start} to {end}  |  Generated: {datetime.today().strftime('%d %b %Y %H:%M')}  |  Tickers: {len(frames)}")
    c2.font = Font(italic=True,color="D29922",size=10,name='Calibri')
    c2.fill = PatternFill("solid",fgColor="0D1117")
    c2.alignment = Alignment(horizontal='left',vertical='center')
    ws.merge_cells('B3:F3')

    ws.row_dimensions[5].height = 22
    for ci,h in enumerate(['Ticker','Company','Sector','Records','From','To'],2):
        cell = ws.cell(5,ci,h)
        cell.font = Font(bold=True,color="E6EDF3",size=10,name='Calibri')
        cell.fill = PatternFill("solid",fgColor="161B22")
        cell.alignment = Alignment(horizontal='center',vertical='center')
        cell.border = bdr()

    for i,ticker in enumerate(TICKERS):
        df = frames.get(ticker, pd.DataFrame())
        r = 6+i
        ws.row_dimensions[r].height = 18
        bg = "0D1117" if i%2==0 else "161B22"
        row_vals = [
            ticker, META_INFO[ticker]['name'], META_INFO[ticker]['sector'],
            len(df),
            df.index.min().strftime('%Y-%m-%d') if not df.empty else 'N/A',
            df.index.max().strftime('%Y-%m-%d') if not df.empty else 'N/A'
        ]
        for ci,v in enumerate(row_vals,2):
            cell = ws.cell(r,ci,v)
            cell.font = Font(name='Calibri',size=9,color="C9D1D9")
            cell.fill = PatternFill("solid",fgColor=bg)
            cell.alignment = Alignment(horizontal='center',vertical='center')
            cell.border = bdr()

    nr = 6+len(TICKERS)+1
    note = ws.cell(nr,2,"Prices are split & dividend adjusted. Data pulled via yfinance from Yahoo Finance. Not financial advice.")
    note.font = Font(name='Calibri',size=9,color="8B949E",italic=True)
    note.fill = PatternFill("solid",fgColor="161B22")
    ws.merge_cells(f'B{nr}:F{nr}')

    # ── Per-ticker sheets ──
    headers = ['Date','Open (USD)','High (USD)','Low (USD)','Close (USD)','Volume','Daily Return %','Cumulative Return %']
    col_ws  = [14,13,13,13,13,16,16,18]

    for ticker, df in frames.items():
        ws2 = wb.create_sheet(ticker)
        ws2.sheet_view.showGridLines = False
        ws2.freeze_panes = 'A3'

        # Banner
        ws2.row_dimensions[1].height = 28
        bn = ws2.cell(1,1,f"{ticker}  |  {META_INFO[ticker]['name']}  |  {META_INFO[ticker]['sector']}  |  Yahoo Finance  |  {start} → {end}")
        bn.font = Font(bold=True,color="E6EDF3",size=11,name='Calibri')
        bn.fill = PatternFill("solid",fgColor="0D1117")
        bn.alignment = Alignment(horizontal='left',vertical='center')
        ws2.merge_cells(f'A1:{get_column_letter(len(headers))}1')

        # Headers
        ws2.row_dimensions[2].height = 24
        for ci,(h,w) in enumerate(zip(headers,col_ws),1):
            cell = ws2.cell(2,ci,h)
            cell.font = Font(bold=True,color="E6EDF3",size=9,name='Calibri')
            cell.fill = PatternFill("solid",fgColor="161B22")
            cell.alignment = Alignment(horizontal='center',vertical='center')
            cell.border = bdr()
            ws2.column_dimensions[get_column_letter(ci)].width = w

        df_s = df.sort_index().copy()
        df_s['dr'] = df_s['Close'].pct_change() * 100
        first_close = df_s['Close'].iloc[0]
        df_s['cr'] = (df_s['Close'] / first_close - 1) * 100

        for ri,(idx,row) in enumerate(df_s.iterrows()):
            r = ri+3
            bg = "0D1117" if ri%2==0 else "0D1117"
            bg = "0D1117" if ri%2==0 else "111827"
            dr_val = round(float(row['dr']),3) if not pd.isna(row['dr']) else 0
            cr_val = round(float(row['cr']),3)
            row_vals = [
                idx.strftime('%Y-%m-%d'),
                round(float(row['Open']),2),
                round(float(row['High']),2),
                round(float(row['Low']),2),
                round(float(row['Close']),2),
                int(row['Volume']),
                dr_val,
                cr_val
            ]
            for ci,v in enumerate(row_vals,1):
                cell = ws2.cell(r,ci,v)
                cell.font = Font(name='Calibri',size=8,color="C9D1D9")
                cell.fill = PatternFill("solid",fgColor=bg)
                cell.alignment = Alignment(horizontal='center',vertical='center')
                cell.border = bdr()
                # Color daily return column
                if ci == 7 and isinstance(v,(int,float)):
                    if v > 0:
                        cell.font = Font(name='Calibri',size=8,color="3FB950")
                    elif v < 0:
                        cell.font = Font(name='Calibri',size=8,color="F85149")

        # Highlight last 5 rows (most recent)
        last_start = len(df_s) - 4
        for ri in range(5):
            r = last_start + ri + 3
            if r > 2:
                for ci in range(1, len(headers)+1):
                    cell = ws2.cell(r,ci)
                    cell.fill = PatternFill("solid",fgColor="1C2333")
                    cell.font = Font(name='Calibri',size=8,color="D29922",bold=True)

        # Price chart — sample data for performance
        step = max(1, len(df_s)//250)
        chart_col = len(headers)+2
        ws2.cell(2,chart_col,"Close_Sampled").font = Font(size=1,color="000000")
        for ci2,src_r in enumerate(range(3,3+len(df_s),step)):
            ws2.cell(3+ci2, chart_col, ws2.cell(src_r,5).value)

        n_chart = len(range(3,3+len(df_s),step))
        lc = LineChart()
        lc.title = f"{ticker} — Adjusted Close Price (USD) | {start} → {end}"
        lc.y_axis.title = "Price (USD)"
        lc.x_axis.title = "Trading Sessions"
        lc.style = 10
        lc.width = 22; lc.height = 12
        dr = Reference(ws2, min_col=chart_col, max_col=chart_col, min_row=2, max_row=2+n_chart)
        lc.add_data(dr, titles_from_data=True)
        lc.series[0].graphicalProperties.line.solidFill = "58A6FF"
        lc.series[0].graphicalProperties.line.width = 18000
        ws2.add_chart(lc, f"J4")

    wb.save(path)

@st.cache_data(ttl=3600, show_spinner=False)
def load_excel(path):
    frames = {}
    try:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            if sheet in TICKERS:
                df = pd.read_excel(path, sheet_name=sheet, skiprows=1)
                df = df.rename(columns={
                    'Date':'Date','Open (USD)':'Open','High (USD)':'High',
                    'Low (USD)':'Low','Close (USD)':'Close','Volume':'Volume'
                })
                df = df[['Date','Open','High','Low','Close','Volume']].dropna()
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                frames[sheet] = df
        return frames, None
    except Exception as e:
        return {}, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def get_data():
    frames, err = fetch_yahoo(TICKERS, START_DATE, END_DATE)
    if frames and len(frames) >= 5:
        try:
            save_excel(frames, EXCEL_PATH)
        except:
            pass
        return frames, "🟢 Yahoo Finance (Live)", None
    if os.path.exists(EXCEL_PATH):
        frames2, err2 = load_excel(EXCEL_PATH)
        if frames2:
            return frames2, f"🟡 Excel Cache ({EXCEL_PATH})", f"Yahoo Finance unavailable — using saved data"
    return {}, "🔴 No Data", f"Cannot reach Yahoo Finance. No Excel cache found.\nRun: pip install yfinance"

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engineer(_frames):
    enriched = {}
    for ticker, df in _frames.items():
        d = df.copy().sort_index()
        c = d['Close']; v = d['Volume']
        d['Return']      = c.pct_change()
        d['Log_Return']  = np.log(c/c.shift(1))
        for w in [5,10,20,50,200]:
            d[f'SMA_{w}'] = c.rolling(w).mean()
            d[f'EMA_{w}'] = c.ewm(span=w,adjust=False).mean()
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI'] = 100 - (100/(1+(gain/loss.replace(0,np.nan))))
        e12 = c.ewm(span=12,adjust=False).mean()
        e26 = c.ewm(span=26,adjust=False).mean()
        d['MACD'] = e12-e26
        d['MACD_Sig'] = d['MACD'].ewm(span=9,adjust=False).mean()
        d['MACD_Hist'] = d['MACD']-d['MACD_Sig']
        sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        d['BB_Up'] = sma20+2*std20; d['BB_Lo'] = sma20-2*std20
        d['BB_Width'] = (d['BB_Up']-d['BB_Lo'])/sma20
        h,l,pc = d['High'],d['Low'],c.shift(1)
        tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        d['ATR'] = tr.rolling(14).mean()
        d['OBV'] = (np.sign(c.diff())*v).cumsum()
        d['Vol_30'] = d['Return'].rolling(30).std()*np.sqrt(252)
        for lag in [1,2,3,5,10]:
            d[f'Lag_{lag}'] = d['Return'].shift(lag)
        d['Target'] = c.shift(-5)/c - 1
        d['Drawdown'] = (c - c.cummax())/c.cummax()
        d['Cum_Return'] = (1+d['Return'].fillna(0)).cumprod()-1
        enriched[ticker] = d
    return enriched

# ── MODELLING ────────────────────────────────────────────────────────────────
FEATS = ['SMA_5','SMA_20','EMA_5','EMA_20','RSI','MACD','MACD_Sig',
         'MACD_Hist','BB_Width','ATR','Vol_30','Lag_1','Lag_2','Lag_3','Lag_5','Lag_10']

@st.cache_data(show_spinner=False)
def run_models(ticker, _enriched):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    try:
        import xgboost as xgb
        HAS_XGB = True
    except:
        HAS_XGB = False
    try:
        import statsmodels.api as sm
        HAS_SM = True
    except:
        HAS_SM = False

    df = _enriched[ticker].copy()
    feats = [f for f in FEATS if f in df.columns]
    sub = df[feats+['Target']].dropna()
    if len(sub) < 120:
        return None, "Not enough data"

    X = sub[feats].values
    y = sub['Target'].values
    dates = sub.index
    split = int(len(X)*0.8)
    Xtr,Xte = X[:split],X[split:]
    ytr,yte = y[:split],y[split:]
    dte = dates[split:]

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    def mets(yt,yp,name):
        mae  = mean_absolute_error(yt,yp)
        rmse = np.sqrt(mean_squared_error(yt,yp))
        r2   = r2_score(yt,yp)
        mape = np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100
        da   = np.mean(np.sign(yt)==np.sign(yp))*100
        return {'Model':name,'MAE':round(mae,5),'RMSE':round(rmse,5),
                'R²':round(r2,4),'MAPE%':round(mape,2),'Dir_Acc%':round(da,1)}

    results = {}

    # 1. Linear
    m = LinearRegression().fit(Xtr_s,ytr)
    p = m.predict(Xte_s)
    results['Linear'] = {'metrics':mets(yte,p,'Linear Regression'),'pred':p,
                         'feat_imp':dict(zip(feats,np.abs(m.coef_)))}

    # 2. OLS
    if HAS_SM:
        import statsmodels.api as sm
        Xtr_c = sm.add_constant(Xtr_s)
        Xte_c = sm.add_constant(Xte_s)
        ols = sm.OLS(ytr,Xtr_c).fit()
        p2 = ols.predict(Xte_c)
        results['OLS'] = {'metrics':mets(yte,p2,'OLS (statsmodels)'),'pred':p2,
                          'r2_in':round(ols.rsquared,4),'pvals':dict(zip(['const']+feats, ols.pvalues)),
                          'feat_imp':dict(zip(feats,np.abs(ols.params[1:])))}
    else:
        m2 = LinearRegression().fit(Xtr_s,ytr)
        p2 = m2.predict(Xte_s)
        results['OLS'] = {'metrics':mets(yte,p2,'OLS (statsmodels)'),'pred':p2,
                          'feat_imp':dict(zip(feats,np.abs(m2.coef_)))}

    # 3. Ridge
    from sklearn.linear_model import RidgeCV
    alphas = [0.01,0.1,1,10,100]
    m3 = RidgeCV(alphas=alphas,cv=5).fit(Xtr_s,ytr)
    p3 = m3.predict(Xte_s)
    results['Ridge'] = {'metrics':mets(yte,p3,f'Ridge (α={m3.alpha_:.3g})'),'pred':p3,
                        'alpha':m3.alpha_,'feat_imp':dict(zip(feats,np.abs(m3.coef_)))}

    # 4. Lasso
    from sklearn.linear_model import LassoCV
    m4 = LassoCV(cv=5,max_iter=5000).fit(Xtr_s,ytr)
    p4 = m4.predict(Xte_s)
    zero_feats = [f for f,c in zip(feats,m4.coef_) if abs(c)<1e-8]
    results['Lasso'] = {'metrics':mets(yte,p4,f'Lasso (α={m4.alpha_:.3g})'),'pred':p4,
                        'alpha':m4.alpha_,'zeroed':zero_feats,
                        'feat_imp':dict(zip(feats,np.abs(m4.coef_)))}

    # 5. Elastic Net
    from sklearn.linear_model import ElasticNetCV
    m5 = ElasticNetCV(cv=5,max_iter=5000,l1_ratio=[0.1,0.3,0.5,0.7,0.9]).fit(Xtr_s,ytr)
    p5 = m5.predict(Xte_s)
    results['ElasticNet'] = {'metrics':mets(yte,p5,f'ElasticNet (α={m5.alpha_:.3g},l1={m5.l1_ratio_:.2f})'),'pred':p5,
                             'feat_imp':dict(zip(feats,np.abs(m5.coef_)))}

    # 6. Random Forest
    m6 = RandomForestRegressor(n_estimators=200,max_depth=6,random_state=42,n_jobs=-1)
    m6.fit(Xtr,ytr)
    p6 = m6.predict(Xte)
    results['RF'] = {'metrics':mets(yte,p6,'Random Forest (n=200)'),'pred':p6,
                     'feat_imp':dict(zip(feats,m6.feature_importances_))}

    # 7. XGBoost
    if HAS_XGB:
        import xgboost as xgb
        m7 = xgb.XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,
                               subsample=0.8,random_state=42,verbosity=0)
        m7.fit(Xtr,ytr)
        p7 = m7.predict(Xte)
        results['XGB'] = {'metrics':mets(yte,p7,'XGBoost (n=200)'),'pred':p7,
                          'feat_imp':dict(zip(feats,m7.feature_importances_))}
    else:
        results['XGB'] = results['RF'].copy()
        results['XGB']['metrics']['Model'] = 'XGBoost (fallback=RF)'

    return {'results':results,'y_test':yte,'dates_test':dte,
            'split':split,'n_train':split,'n_test':len(yte),
            'features':feats}, None

# ── INVESTMENT SCORING ────────────────────────────────────────────────────────
def compute_scores(enriched, model_results):
    scores = {}
    for ticker in TICKERS:
        d = enriched.get(ticker)
        if d is None or len(d) < 30:
            scores[ticker] = {'score':0,'signal':'Insufficient Data','breakdown':{}}
            continue
        last = d.dropna().iloc[-1]
        score = 0; breakdown = {}

        # RSI signal (20%)
        rsi = last.get('RSI', 50)
        if rsi < 35:      rsi_s = 20
        elif rsi < 50:    rsi_s = 15
        elif rsi < 65:    rsi_s = 10
        else:             rsi_s = 5
        score += rsi_s; breakdown['RSI Signal'] = rsi_s

        # MACD (20%)
        macd_s = 20 if last.get('MACD',0) > last.get('MACD_Sig',0) else 5
        score += macd_s; breakdown['MACD Cross'] = macd_s

        # Price vs 200 SMA (15%)
        close = last['Close']
        sma200 = last.get('SMA_200', close)
        sma_s = 15 if close > sma200 else 5
        score += sma_s; breakdown['vs 200-SMA'] = sma_s

        # Volatility (15%) — lower is better
        vol = last.get('Vol_30', 0.3)
        if vol < 0.2:     vol_s = 15
        elif vol < 0.35:  vol_s = 10
        else:             vol_s = 5
        score += vol_s; breakdown['Low Volatility'] = vol_s

        # Model prediction (30%)
        mr = model_results.get(ticker)
        if mr and mr[0]:
            best_r = max(mr[0]['results'].values(), key=lambda x: x['metrics']['Dir_Acc%'])
            da = best_r['metrics']['Dir_Acc%']
            pred_last = best_r['pred'][-1] if len(best_r['pred']) > 0 else 0
            model_s = int(30 * (da/100) * (1 if pred_last > 0 else 0.4))
        else:
            model_s = 10
        score += model_s; breakdown['Model Prediction'] = model_s

        if score >= 65:     signal = 'BUY'
        elif score >= 45:   signal = 'HOLD'
        else:               signal = 'AVOID'

        # Sharpe / Sortino
        ret = d['Return'].dropna()
        sharpe  = (ret.mean()/ret.std())*np.sqrt(252) if ret.std()>0 else 0
        neg_ret = ret[ret<0]
        sortino = (ret.mean()/neg_ret.std())*np.sqrt(252) if len(neg_ret)>0 and neg_ret.std()>0 else 0

        scores[ticker] = {
            'score': score, 'signal': signal, 'breakdown': breakdown,
            'sharpe': round(sharpe,2), 'sortino': round(sortino,2),
            'rsi': round(rsi,1), 'vol': round(vol*100,1),
            'close': round(close,2)
        }
    return scores

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 IB Analytics Dashboard")
    st.markdown("**S&P 500 Top 10 | Yahoo Finance**")
    st.markdown(f"*Data through: {END_DATE}*")
    st.divider()

    page = st.radio("Navigate", [
        "📊 Market Overview",
        "🔍 EDA Deep Dive",
        "⚙️ Feature Engineering",
        "🤖 Model Training",
        "💡 Investment Signals",
        "📥 Download Data"
    ])
    st.divider()

    selected_tickers = st.multiselect("Stocks", TICKERS, default=TICKERS[:5])
    if not selected_tickers:
        selected_tickers = TICKERS[:5]

    st.divider()
    st.caption("Disclaimer: For informational purposes only. Not financial advice.")

# ── LOAD DATA ──────────────────────────────────────────────────────────────
with st.spinner("Loading stock data from Yahoo Finance..."):
    frames, data_src, data_err = get_data()

if data_err:
    st.warning(f"⚠️ {data_err}")

if not frames:
    st.error("No data available. Please install yfinance: `pip install yfinance`")
    st.stop()

enriched = engineer(frames)

# ── MODEL CACHE (background) ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_all_models(_enriched):
    res = {}
    for t in TICKERS:
        res[t] = run_models(t, _enriched)
    return res

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("# 📊 Market Overview")
    st.markdown(f"<small>Data source: {data_src}</small>", unsafe_allow_html=True)

    # KPI cards
    st.markdown('<div class="sec">Key Performance Indicators</div>', unsafe_allow_html=True)
    cols = st.columns(len(selected_tickers))
    for i, t in enumerate(selected_tickers):
        df = frames[t]
        cur = df['Close'].iloc[-1]
        prev_yr = df['Close'].iloc[max(0, len(df)-252)]
        ytd_ret = (cur/prev_yr - 1)*100
        hi52  = df['Close'].rolling(252).max().iloc[-1]
        lo52  = df['Close'].rolling(252).min().iloc[-1]
        with cols[i]:
            delta_cls = "kpi-pos" if ytd_ret >= 0 else "kpi-neg"
            delta_sym = "▲" if ytd_ret >= 0 else "▼"
            st.markdown(f"""<div class="kpi">
                <div class="kpi-lbl">{t}</div>
                <div class="kpi-val">${cur:.2f}</div>
                <div class="{delta_cls}">{delta_sym} {abs(ytd_ret):.1f}% 1Y</div>
                <div style="color:#8b949e;font-size:11px;margin-top:4px">
                52W: ${lo52:.0f} – ${hi52:.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Normalised price chart
    st.markdown('<div class="sec">Normalised Price Performance (Base = 100)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for t in selected_tickers:
        df = frames[t]
        norm = df['Close'] / df['Close'].iloc[0] * 100
        fig.add_trace(go.Scatter(x=df.index, y=norm, name=t, mode='lines',
                                  line=dict(width=2)))
    fig.update_layout(template='plotly_dark', height=420,
                      xaxis_title='Date', yaxis_title='Indexed Price (Base=100)',
                      legend=dict(orientation='h', y=1.02, x=0),
                      margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Candlestick
        st.markdown('<div class="sec">Candlestick Chart</div>', unsafe_allow_html=True)
        cs_ticker = st.selectbox("Select ticker", selected_tickers, key='cs')
        df_cs = frames[cs_ticker].last('180D')
        fig2 = go.Figure(data=[go.Candlestick(
            x=df_cs.index, open=df_cs['Open'], high=df_cs['High'],
            low=df_cs['Low'], close=df_cs['Close'], name=cs_ticker)])
        fig2.add_trace(go.Bar(x=df_cs.index, y=df_cs['Volume'], name='Volume',
                              yaxis='y2', marker_color='rgba(88,166,255,0.2)'))
        fig2.update_layout(template='plotly_dark', height=380,
                           yaxis2=dict(overlaying='y', side='right', showgrid=False),
                           margin=dict(l=40,r=40,t=30,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Correlation heatmap
        st.markdown('<div class="sec">Return Correlation Heatmap</div>', unsafe_allow_html=True)
        ret_df = pd.DataFrame({t: frames[t]['Close'].pct_change() for t in TICKERS}).dropna()
        corr = ret_df.corr().round(3)
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1, aspect='auto')
        fig3.update_layout(template='plotly_dark', height=380,
                           margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig3, use_container_width=True)

    # Rolling volatility
    st.markdown('<div class="sec">30-Day Rolling Volatility (Annualised)</div>', unsafe_allow_html=True)
    fig4 = go.Figure()
    for t in selected_tickers:
        vol = frames[t]['Close'].pct_change().rolling(30).std()*np.sqrt(252)*100
        fig4.add_trace(go.Scatter(x=frames[t].index, y=vol, name=t, mode='lines', line=dict(width=1.5)))
    fig4.update_layout(template='plotly_dark', height=320,
                       yaxis_title='Annualised Volatility (%)',
                       margin=dict(l=40,r=20,t=30,b=40))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="insight">💡 <b>Insight:</b> NVDA and TSLA consistently show highest volatility (>50% annualised), while BRK-B and JPM remain the most stable. High correlation clusters (Tech group: AAPL/MSFT/NVDA/GOOGL/META) suggest limited diversification benefit within that group.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA Deep Dive":
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    st.markdown("# 🔍 EDA Deep Dive")

    eda_ticker = st.selectbox("Select Stock", selected_tickers, key='eda_t')
    df = enriched[eda_ticker].dropna(subset=['Return'])

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📉 Drawdown & Trends", "📅 Seasonality", "🚨 Anomalies"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec">Daily Return Distribution</div>', unsafe_allow_html=True)
            ret = df['Return'].dropna()*100
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ret, nbinsx=80, name='Daily Returns',
                                        marker_color='#58a6ff', opacity=0.7,
                                        histnorm='probability density'))
            mu, sig = ret.mean(), ret.std()
            x_norm = np.linspace(ret.min(), ret.max(), 200)
            y_norm = (1/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_norm-mu)/sig)**2)
            fig.add_trace(go.Scatter(x=x_norm, y=y_norm, name='Normal Fit',
                                      line=dict(color='#f85149', width=2)))
            fig.update_layout(template='plotly_dark', height=340,
                               xaxis_title='Daily Return (%)', yaxis_title='Density',
                               margin=dict(l=40,r=20,t=30,b=40))
            st.plotly_chart(fig, use_container_width=True)
            skew = ret.skew(); kurt = ret.kurtosis()
            st.markdown(f'<div class="insight">Mean: <b>{mu:.3f}%</b> | Std: <b>{sig:.3f}%</b> | Skew: <b>{skew:.3f}</b> | Kurtosis: <b>{kurt:.2f}</b> (fat tails = crash risk)</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="sec">Box Plot — All Stocks Returns</div>', unsafe_allow_html=True)
            ret_all = pd.DataFrame({t: frames[t]['Close'].pct_change()*100 for t in TICKERS}).dropna()
            fig2 = go.Figure()
            for t in TICKERS:
                fig2.add_trace(go.Box(y=ret_all[t], name=t, boxpoints='outliers'))
            fig2.update_layout(template='plotly_dark', height=340,
                                yaxis_title='Daily Return (%)',
                                margin=dict(l=40,r=20,t=30,b=40))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec">Drawdown from Peak</div>', unsafe_allow_html=True)
            fig3 = go.Figure()
            for t in selected_tickers:
                dd = enriched[t]['Drawdown']*100
                fig3.add_trace(go.Scatter(x=dd.index, y=dd, name=t,
                                           fill='tozeroy', mode='lines', line=dict(width=1)))
            fig3.update_layout(template='plotly_dark', height=340,
                                yaxis_title='Drawdown (%)',
                                margin=dict(l=40,r=20,t=30,b=40))
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.markdown('<div class="sec">Price vs Moving Averages</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close',
                                       line=dict(color='#e6edf3', width=1.5)))
            for w, col in [(20,'#58a6ff'),(50,'#f6ad55'),(200,'#3fb950')]:
                col_name = f'SMA_{w}'
                if col_name in df.columns:
                    fig4.add_trace(go.Scatter(x=df.index, y=df[col_name], name=f'SMA {w}',
                                               line=dict(color=col, width=1.2, dash='dot')))
            fig4.update_layout(template='plotly_dark', height=340,
                                yaxis_title='Price (USD)',
                                margin=dict(l=40,r=20,t=30,b=40))
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.markdown('<div class="sec">Average Monthly Return (%)</div>', unsafe_allow_html=True)
        df['Month'] = df.index.month
        monthly_avg = df.groupby('Month')['Return'].mean()*100
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig5 = go.Figure(go.Bar(
            x=[months[m-1] for m in monthly_avg.index],
            y=monthly_avg.values,
            marker_color=['#3fb950' if v >= 0 else '#f85149' for v in monthly_avg.values]
        ))
        fig5.update_layout(template='plotly_dark', height=320,
                            yaxis_title='Avg Monthly Return (%)',
                            margin=dict(l=40,r=20,t=30,b=40))
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        st.markdown('<div class="sec">Outlier Days (|Return| > 3σ)</div>', unsafe_allow_html=True)
        ret = df['Return'].dropna()
        mu, sig = ret.mean(), ret.std()
        outliers = df[np.abs(df['Return']-mu) > 3*sig][['Close','Return','Vol_30']].copy()
        outliers['Return%'] = (outliers['Return']*100).round(3)
        outliers['Type'] = outliers['Return'].apply(lambda x: '🟢 Surge' if x > 0 else '🔴 Crash')
        st.dataframe(
            outliers[['Close','Return%','Type']].sort_values('Return%'),
            use_container_width=True, height=300
        )
        st.markdown(f'<div class="insight">Found <b>{len(outliers)}</b> extreme days (>3σ) for {eda_ticker}. These events represent tail risk and require position sizing discipline.</div>', unsafe_allow_html=True)

        # Volume anomalies
        st.markdown('<div class="sec">Volume Anomaly Days (> 2× 30-Day Avg)</div>', unsafe_allow_html=True)
        df2 = frames[eda_ticker].copy()
        vol_avg = df2['Volume'].rolling(30).mean()
        vol_anom = df2[df2['Volume'] > 2*vol_avg][['Close','Volume']].copy()
        vol_anom['Vol_Multiplier'] = (df2['Volume'] / vol_avg).round(2)
        st.dataframe(vol_anom.sort_values('Vol_Multiplier', ascending=False).head(20),
                     use_container_width=True, height=250)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Feature Engineering":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("# ⚙️ Feature Engineering & Technical Indicators")

    fe_ticker = st.selectbox("Select Stock", selected_tickers, key='fe_t')
    df = enriched[fe_ticker].dropna()

    tab1, tab2, tab3 = st.tabs(["📐 Indicators", "🔗 Feature Correlations", "📊 Feature Data Table"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec">RSI (14)</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#f6ad55',width=1.5)))
            fig.add_hline(y=70, line_dash='dash', line_color='#f85149', annotation_text='Overbought')
            fig.add_hline(y=30, line_dash='dash', line_color='#3fb950', annotation_text='Oversold')
            fig.update_layout(template='plotly_dark', height=280, margin=dict(l=40,r=20,t=30,b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="sec">MACD</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#58a6ff',width=1.5)))
            fig2.add_trace(go.Scatter(x=df.index, y=df['MACD_Sig'], name='Signal', line=dict(color='#f85149',width=1.2,dash='dot')))
            hist_c = ['#3fb950' if v>=0 else '#f85149' for v in df['MACD_Hist']]
            fig2.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
                                   marker_color=hist_c, opacity=0.6))
            fig2.update_layout(template='plotly_dark', height=280, margin=dict(l=40,r=20,t=30,b=30))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="sec">Bollinger Bands</div>', unsafe_allow_html=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], name='Upper Band',
                                       line=dict(color='rgba(88,166,255,0.4)',width=1)))
            fig3.add_trace(go.Scatter(x=df.index, y=df['BB_Lo'], name='Lower Band',
                                       fill='tonexty', fillcolor='rgba(88,166,255,0.05)',
                                       line=dict(color='rgba(88,166,255,0.4)',width=1)))
            fig3.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close',
                                       line=dict(color='#e6edf3',width=1.2)))
            fig3.update_layout(template='plotly_dark', height=280, margin=dict(l=40,r=20,t=30,b=30))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown('<div class="sec">OBV (On Balance Volume)</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV',
                                       fill='tozeroy', line=dict(color='#a371f7',width=1.5)))
            fig4.update_layout(template='plotly_dark', height=280, margin=dict(l=40,r=20,t=30,b=30))
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.markdown('<div class="sec">Feature Correlation with Target (Next 5-Day Return)</div>', unsafe_allow_html=True)
        feat_cols = [f for f in FEATS if f in df.columns] + ['Target']
        corr_df = df[feat_cols].dropna().corr()
        target_corr = corr_df['Target'].drop('Target').sort_values()
        fig5 = go.Figure(go.Bar(
            x=target_corr.values, y=target_corr.index, orientation='h',
            marker_color=['#3fb950' if v >= 0 else '#f85149' for v in target_corr.values]
        ))
        fig5.update_layout(template='plotly_dark', height=420,
                            xaxis_title='Pearson Correlation with Target',
                            margin=dict(l=140,r=20,t=30,b=40))
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown('<div class="sec">Full Feature Correlation Matrix</div>', unsafe_allow_html=True)
        fig6 = px.imshow(corr_df.round(2), text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig6.update_layout(template='plotly_dark', height=500, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig6, use_container_width=True)

    with tab3:
        show_cols = ['Close','Return','RSI','MACD','BB_Width','ATR','Vol_30','Lag_1','Lag_2','Lag_3','Target']
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].tail(100).round(5), use_container_width=True, height=400)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    st.markdown("# 🤖 Model Training & Comparison")

    model_ticker = st.selectbox("Select Stock to Model", TICKERS, key='mod_t')

    with st.spinner(f"Training 7 models on {model_ticker}..."):
        model_out, model_err = run_models(model_ticker, enriched)

    if model_err or model_out is None:
        st.error(f"Model error: {model_err}")
        st.stop()

    results  = model_out['results']
    y_test   = model_out['y_test']
    dates_te = model_out['dates_test']
    features = model_out['features']

    st.success(f"✅ Trained on {model_out['n_train']} days | Tested on {model_out['n_test']} days | Features: {len(features)}")

    # Metrics table
    st.markdown('<div class="sec">Model Performance Comparison</div>', unsafe_allow_html=True)
    metrics_rows = [v['metrics'] for v in results.values()]
    mdf = pd.DataFrame(metrics_rows).set_index('Model')
    best_da_model = mdf['Dir_Acc%'].idxmax()
    st.dataframe(
        mdf.style
            .highlight_max(subset=['R²','Dir_Acc%'], color='#0d2818')
            .highlight_min(subset=['MAE','RMSE','MAPE%'], color='#0d2818')
            .format(precision=4),
        use_container_width=True
    )
    st.markdown(f'<div class="insight">🏆 Best model by Directional Accuracy: <b>{best_da_model}</b> — {mdf.loc[best_da_model,"Dir_Acc%"]:.1f}% correct direction prediction on unseen test data.</div>', unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown('<div class="sec">Actual vs Predicted — Test Period</div>', unsafe_allow_html=True)
    sel_model = st.selectbox("Select model to inspect", list(results.keys()), key='sel_m')
    pred = results[sel_model]['pred']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_te, y=y_test*100, name='Actual 5D Return (%)',
                              line=dict(color='#e6edf3', width=1.5)))
    fig.add_trace(go.Scatter(x=dates_te, y=pred*100, name='Predicted',
                              line=dict(color='#58a6ff', width=1.5, dash='dot')))
    fig.add_hline(y=0, line_dash='dash', line_color='#30363d')
    fig.update_layout(template='plotly_dark', height=350,
                       yaxis_title='5-Day Forward Return (%)',
                       margin=dict(l=40,r=20,t=30,b=40))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Residuals
        st.markdown('<div class="sec">Residuals Plot</div>', unsafe_allow_html=True)
        resid = y_test - pred
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates_te, y=resid*100, mode='markers',
                                   marker=dict(color='#f6ad55', size=3, opacity=0.6), name='Residuals'))
        fig2.add_hline(y=0, line_dash='dash', line_color='#f85149')
        fig2.update_layout(template='plotly_dark', height=300,
                            yaxis_title='Residual (%)', margin=dict(l=40,r=20,t=30,b=30))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Feature importance
        st.markdown('<div class="sec">Feature Importance</div>', unsafe_allow_html=True)
        fi = results[sel_model].get('feat_imp', {})
        if fi:
            fi_s = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
            fig3 = go.Figure(go.Bar(
                y=list(fi_s.keys()), x=list(fi_s.values()), orientation='h',
                marker_color='#58a6ff'
            ))
            fig3.update_layout(template='plotly_dark', height=300,
                                margin=dict(l=100,r=20,t=30,b=30))
            st.plotly_chart(fig3, use_container_width=True)

    # Lasso zero features
    if 'Lasso' in results and results['Lasso'].get('zeroed'):
        zeroed = results['Lasso']['zeroed']
        if zeroed:
            st.markdown(f'<div class="insight">🔍 <b>Lasso</b> eliminated {len(zeroed)} features (coefficient → 0): <b>{", ".join(zeroed)}</b>. These have near-zero predictive power for {model_ticker}.</div>', unsafe_allow_html=True)

    # Radar chart
    st.markdown('<div class="sec">Model Comparison Radar</div>', unsafe_allow_html=True)
    categories = ['Dir_Acc%','R² (scaled)','Low MAE','Low RMSE']
    max_da = mdf['Dir_Acc%'].max() or 1
    max_r2 = max(mdf['R²'].max(), 0.01)
    min_mae = mdf['MAE'].min() or 1
    min_rmse= mdf['RMSE'].min() or 1
    fig4 = go.Figure()
    for model_name, row in mdf.iterrows():
        vals = [
            row['Dir_Acc%']/max_da*100,
            max(row['R²'],0)/max_r2*100,
            (1 - (row['MAE']-min_mae)/(mdf['MAE'].max()-min_mae+1e-9))*100,
            (1 - (row['RMSE']-min_rmse)/(mdf['RMSE'].max()-min_rmse+1e-9))*100
        ]
        vals += [vals[0]]
        fig4.add_trace(go.Scatterpolar(r=vals, theta=categories+[categories[0]],
                                        fill='toself', name=model_name, opacity=0.6))
    fig4.update_layout(polar=dict(bgcolor='#161b22',
                                   radialaxis=dict(visible=True, range=[0,100])),
                        template='plotly_dark', height=400,
                        margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — INVESTMENT SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💡 Investment Signals":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("# 💡 Investment Signals & Scoring")
    st.markdown("*Composite signal engine: Model prediction (30%) + RSI (20%) + MACD (20%) + vs 200-SMA (15%) + Volatility (15%)*")

    with st.spinner("Computing scores across all 10 stocks..."):
        all_models = get_all_models(enriched)
        scores = compute_scores(enriched, all_models)

    # Sort by score
    sorted_tickers = sorted(TICKERS, key=lambda t: scores[t]['score'], reverse=True)

    # Top 3 / Hold / Avoid
    buys   = [t for t in sorted_tickers if scores[t]['signal']=='BUY'][:3]
    holds  = [t for t in sorted_tickers if scores[t]['signal']=='HOLD'][:3]
    avoids = [t for t in sorted_tickers if scores[t]['signal']=='AVOID'][:3]

    st.markdown('<div class="sec">📌 Top Picks</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    def render_picks(col, tickers_list, badge_cls, label):
        with col:
            st.markdown(f"**{label}**")
            for t in tickers_list:
                s = scores[t]
                st.markdown(f"""<div class="kpi" style="margin-bottom:10px">
                    <div class="kpi-lbl">{t} — {META_INFO[t]['name']}</div>
                    <div class="kpi-val">${s['close']:.2f}</div>
                    <span class="{badge_cls}">{s['signal']}</span>
                    <div style="color:#8b949e;font-size:12px;margin-top:6px">
                    Score: <b style="color:#e6edf3">{s['score']}/100</b> &nbsp;|&nbsp;
                    Sharpe: <b style="color:#e6edf3">{s['sharpe']}</b> &nbsp;|&nbsp;
                    RSI: <b style="color:#e6edf3">{s['rsi']}</b></div>
                </div>""", unsafe_allow_html=True)

    render_picks(col1, buys,   'buy',   '🟢 BUY Recommendations')
    render_picks(col2, holds,  'hold',  '🟡 HOLD')
    render_picks(col3, avoids, 'avoid', '🔴 AVOID / Watch')

    st.divider()

    # Full leaderboard
    st.markdown('<div class="sec">Full Investment Score Leaderboard</div>', unsafe_allow_html=True)
    lb_rows = []
    for t in sorted_tickers:
        s = scores[t]
        lb_rows.append({
            'Rank': sorted_tickers.index(t)+1,
            'Ticker': t,
            'Company': META_INFO[t]['name'],
            'Sector': META_INFO[t]['sector'],
            'Score': s['score'],
            'Signal': s['signal'],
            'Price': f"${s['close']:.2f}",
            'Sharpe': s['sharpe'],
            'Sortino': s['sortino'],
            'RSI': s['rsi'],
            'Volatility%': s['vol'],
        })
    lb_df = pd.DataFrame(lb_rows).set_index('Rank')
    st.dataframe(lb_df, use_container_width=True)

    # Score bar chart
    st.markdown('<div class="sec">Investment Score Comparison</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=[t for t in sorted_tickers],
        y=[scores[t]['score'] for t in sorted_tickers],
        marker_color=['#3fb950' if scores[t]['signal']=='BUY'
                      else '#d29922' if scores[t]['signal']=='HOLD'
                      else '#f85149' for t in sorted_tickers],
        text=[f"{scores[t]['score']}" for t in sorted_tickers],
        textposition='outside'
    ))
    fig.add_hline(y=65, line_dash='dash', line_color='#3fb950', annotation_text='BUY threshold')
    fig.add_hline(y=45, line_dash='dash', line_color='#d29922', annotation_text='HOLD threshold')
    fig.update_layout(template='plotly_dark', height=360,
                       yaxis_title='Composite Score (0-100)',
                       margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Sharpe vs Sortino scatter
        st.markdown('<div class="sec">Risk-Adjusted Return: Sharpe vs Sortino</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for t in TICKERS:
            s = scores[t]
            col_map = '#3fb950' if s['signal']=='BUY' else '#d29922' if s['signal']=='HOLD' else '#f85149'
            fig2.add_trace(go.Scatter(x=[s['sharpe']], y=[s['sortino']], mode='markers+text',
                                       text=[t], textposition='top center',
                                       marker=dict(color=col_map, size=14),
                                       name=t))
        fig2.add_vline(x=0, line_dash='dash', line_color='#30363d')
        fig2.add_hline(y=0, line_dash='dash', line_color='#30363d')
        fig2.update_layout(template='plotly_dark', height=360,
                            xaxis_title='Sharpe Ratio', yaxis_title='Sortino Ratio',
                            showlegend=False, margin=dict(l=40,r=20,t=30,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Sector allocation pie
        st.markdown('<div class="sec">Suggested Portfolio Allocation (Top Picks)</div>', unsafe_allow_html=True)
        top5 = sorted_tickers[:5]
        sectors = [META_INFO[t]['sector'] for t in top5]
        sector_counts = pd.Series(sectors).value_counts()
        fig3 = go.Figure(go.Pie(
            labels=sector_counts.index, values=sector_counts.values,
            hole=0.4, textinfo='label+percent'
        ))
        fig3.update_layout(template='plotly_dark', height=360,
                            margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="insight">⚠️ <b>Disclaimer:</b> Investment scores are model outputs based on historical patterns and technical indicators only. They do not constitute financial advice. Past performance does not guarantee future results. Always conduct independent due diligence.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — DOWNLOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📥 Download Data":
    import plotly.graph_objects as go

    st.markdown("# 📥 Download Data & Reports")

    st.markdown('<div class="sec">Excel Stock Price File</div>', unsafe_allow_html=True)
    st.markdown("""
    The Excel file contains:
    - **Overview sheet** — all 10 tickers summary
    - **One sheet per ticker** — full OHLCV data with daily returns, cumulative returns, and a price chart
    - **Source:** Yahoo Finance via `yfinance`
    - **Period:** 2021-01-01 → 2026-03-14
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Regenerate Excel from Yahoo Finance", type='primary'):
            with st.spinner("Fetching fresh data and generating Excel..."):
                fresh_frames, err = fetch_yahoo(TICKERS, START_DATE, END_DATE)
                if fresh_frames:
                    save_excel(fresh_frames, EXCEL_PATH)
                    st.success(f"✅ Excel saved to `{EXCEL_PATH}` with {sum(len(v) for v in fresh_frames.values()):,} total rows")
                else:
                    st.error(f"Could not fetch from Yahoo Finance: {err}")

    with col2:
        if os.path.exists(EXCEL_PATH):
            with open(EXCEL_PATH, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=f.read(),
                    file_name=EXCEL_PATH,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type='primary'
                )
        else:
            st.warning("Excel file not yet generated. Click 'Regenerate' first.")

    st.divider()
    st.markdown('<div class="sec">Download Feature-Engineered Dataset (CSV)</div>', unsafe_allow_html=True)
    dl_ticker = st.selectbox("Select ticker", TICKERS, key='dl_t')
    df_dl = enriched[dl_ticker].copy()
    csv_buf = io.StringIO()
    df_dl.to_csv(csv_buf)
    st.download_button(
        label=f"⬇️ Download {dl_ticker} Features CSV",
        data=csv_buf.getvalue(),
        file_name=f"{dl_ticker}_features_{END_DATE}.csv",
        mime='text/csv'
    )

    st.divider()
    st.markdown('<div class="sec">Model Metrics Summary (All Stocks)</div>', unsafe_allow_html=True)
    if st.button("📊 Generate Model Metrics Excel"):
        with st.spinner("Training models on all tickers..."):
            all_models = get_all_models(enriched)
        rows = []
        for t in TICKERS:
            mr, err = all_models[t]
            if mr:
                for model_name, res in mr['results'].items():
                    row = {'Ticker': t, **res['metrics']}
                    rows.append(row)
        metrics_df = pd.DataFrame(rows)
        metrics_buf = io.BytesIO()
        metrics_df.to_excel(metrics_buf, index=False)
        st.download_button(
            label="⬇️ Download Model Metrics Excel",
            data=metrics_buf.getvalue(),
            file_name=f"model_metrics_sp500_{END_DATE}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    st.divider()
    st.markdown("""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;color:#8b949e;font-size:12px">
    <b>⚠️ Legal Disclaimer</b><br>
    This dashboard and all outputs are for <b>informational and educational purposes only</b>.
    Nothing on this dashboard constitutes financial, investment, legal, or tax advice.
    Historical data is sourced from Yahoo Finance. Past performance is not indicative of future results.
    The investment bank and its affiliates accept no liability for any decisions made based on this tool.
    Always consult a qualified financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
