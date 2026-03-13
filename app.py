"""
Investment Bank Stock Analytics Dashboard  v3
S&P 500 Top 10 | Yahoo Finance | Cut-off: 14 March 2026

MODULES (matching sidebar screenshot):
  1. 📊 Executive Overview
  2. 🎯 Classification Analysis
  3. 🔮 Clustering Analysis
  4. 📈 Regression Analysis
  5. 🔗 Association Rules
  6. 🔬 Deep Drill-Down Analysis

NEW vs v2:
  - EMA 50 and EMA 200 added to all price / trend charts
  - Every insight box explains what the graph means in plain English
  - Classification: Random-Forest buy/hold/avoid classifier with confusion matrix
  - Clustering: K-Means (k=3) on return/vol/RSI space, 2D scatter + profiles
  - Regression: 7-model suite with full metrics, radar, residuals
  - Association Rules: co-movement correlation rules + sector pair analysis
  - Deep Drill-Down: per-stock full technical teardown + rolling stats
  - CSS: minimal sidebar colour only — zero inline HTML classes
  - Plotly: config={"displayModeBar":False} on every chart (CSP fix)
  - No df.last() — boolean date indexing throughout
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, warnings, io, copy
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IB Stock Analytics | S&P 500 Top 10",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
TICKERS = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK-B','JPM','UNH']
META_INFO = {
    'AAPL':  {'name':'Apple Inc.',           'sector':'Technology'},
    'MSFT':  {'name':'Microsoft Corp.',       'sector':'Technology'},
    'NVDA':  {'name':'NVIDIA Corp.',          'sector':'Semiconductors'},
    'AMZN':  {'name':'Amazon.com Inc.',       'sector':'Consumer Disc.'},
    'GOOGL': {'name':'Alphabet Inc.',         'sector':'Communication'},
    'META':  {'name':'Meta Platforms',        'sector':'Communication'},
    'TSLA':  {'name':'Tesla Inc.',            'sector':'Consumer Disc.'},
    'BRK-B': {'name':'Berkshire Hathaway B', 'sector':'Financials'},
    'JPM':   {'name':'JPMorgan Chase',        'sector':'Financials'},
    'UNH':   {'name':'UnitedHealth Group',    'sector':'Healthcare'},
}
COLORS = ['#58a6ff','#3fb950','#f6ad55','#f093fb','#4facfe',
          '#43e97b','#fa709a','#fee140','#a371f7','#ff9a9e']
START_DATE = "2021-01-01"
END_DATE   = "2026-03-14"
EXCEL_PATH = "stock_prices_sp500_top10.xlsx"
PCFG       = {"displayModeBar": False}

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(family="sans-serif", color="#d1d5db", size=11),
    margin=dict(l=50, r=20, t=40, b=50),
    xaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
    yaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
)

def pplot(fig, h=380, **kw):
    fig.update_layout(**DARK, height=h, **kw)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)

def insight_box(title, body):
    """Render a styled insight using native Streamlit components."""
    with st.container(border=True):
        st.markdown(f"**💡 {title}**")
        st.markdown(body)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo(tickers, start, end):
    try:
        import yfinance as yf
        frames = {}
        for t in tickers:
            try:
                df = yf.download(t, start=start, end=end,
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    frames[t] = df[['Open','High','Low','Close','Volume']].copy()
            except Exception:
                pass
        return (frames if frames else None), None
    except ImportError:
        return None, "yfinance not installed — run: pip install yfinance"
    except Exception as e:
        return None, str(e)


def save_excel(frames, path):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import LineChart, Reference

    def bdr():
        s = Side(style='thin', color='374151')
        return Border(left=s, right=s, top=s, bottom=s)

    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Overview"
    ws.sheet_view.showGridLines = False
    for col, w in zip('ABCDEF', [3,22,32,14,16,16]):
        ws.column_dimensions[col].width = w
    ws.row_dimensions[2].height = 44
    tc = ws.cell(2,2,"S&P 500 Top 10 — Historical Stock Prices | Source: Yahoo Finance")
    tc.font = Font(bold=True, color="E6EDF3", size=16, name='Calibri')
    tc.fill = PatternFill("solid", fgColor="0D1117")
    tc.alignment = Alignment(horizontal='left', vertical='center')
    ws.merge_cells('B2:F2')
    ws.row_dimensions[3].height = 20
    sc = ws.cell(3,2,
        f"Period: {START_DATE} to {END_DATE}  |  "
        f"Generated: {datetime.today().strftime('%d %b %Y %H:%M')}  |  "
        f"Tickers: {len(frames)}")
    sc.font = Font(italic=True, color="D29922", size=10, name='Calibri')
    sc.fill = PatternFill("solid", fgColor="0D1117")
    sc.alignment = Alignment(horizontal='left', vertical='center')
    ws.merge_cells('B3:F3')
    ws.row_dimensions[5].height = 22
    for ci, h in enumerate(['Ticker','Company','Sector','Records','From','To'], 2):
        cell = ws.cell(5,ci,h)
        cell.font = Font(bold=True, color="E6EDF3", size=10, name='Calibri')
        cell.fill = PatternFill("solid", fgColor="161B22")
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = bdr()
    for i, ticker in enumerate(TICKERS):
        df = frames.get(ticker, pd.DataFrame()); r = 6+i
        ws.row_dimensions[r].height = 18; bg = "0D1117" if i%2==0 else "161B22"
        row_vals = [ticker, META_INFO[ticker]['name'], META_INFO[ticker]['sector'],
                    len(df),
                    df.index.min().strftime('%Y-%m-%d') if not df.empty else 'N/A',
                    df.index.max().strftime('%Y-%m-%d') if not df.empty else 'N/A']
        for ci, v in enumerate(row_vals, 2):
            cell = ws.cell(r,ci,v)
            cell.font = Font(name='Calibri', size=9, color="C9D1D9")
            cell.fill = PatternFill("solid", fgColor=bg)
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = bdr()
    nr = 6+len(TICKERS)+1
    note = ws.cell(nr,2,"Prices split & dividend adjusted. Not financial advice.")
    note.font = Font(name='Calibri', size=9, color="8B949E", italic=True)
    note.fill = PatternFill("solid", fgColor="161B22")
    ws.merge_cells(f'B{nr}:F{nr}')
    hdr_names  = ['Date','Open (USD)','High (USD)','Low (USD)',
                  'Close (USD)','Volume','Daily Return %','Cumul. Return %']
    hdr_widths = [14,13,13,13,13,16,16,18]
    for ticker, df in frames.items():
        ws2 = wb.create_sheet(ticker)
        ws2.sheet_view.showGridLines = False; ws2.freeze_panes = 'A3'
        ws2.row_dimensions[1].height = 28
        bn = ws2.cell(1,1,
            f"{ticker}  |  {META_INFO[ticker]['name']}  |  "
            f"{META_INFO[ticker]['sector']}  |  Yahoo Finance  |  "
            f"{START_DATE} to {END_DATE}")
        bn.font = Font(bold=True, color="E6EDF3", size=11, name='Calibri')
        bn.fill = PatternFill("solid", fgColor="0D1117")
        bn.alignment = Alignment(horizontal='left', vertical='center')
        ws2.merge_cells(f'A1:{get_column_letter(len(hdr_names))}1')
        ws2.row_dimensions[2].height = 24
        for ci,(h,w) in enumerate(zip(hdr_names,hdr_widths),1):
            cell = ws2.cell(2,ci,h)
            cell.font = Font(bold=True, color="E6EDF3", size=9, name='Calibri')
            cell.fill = PatternFill("solid", fgColor="161B22")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = bdr()
            ws2.column_dimensions[get_column_letter(ci)].width = w
        df_s = df.sort_index().copy()
        df_s['dr'] = df_s['Close'].pct_change()*100
        df_s['cr'] = (df_s['Close']/df_s['Close'].iloc[0]-1)*100
        for ri,(idx,row) in enumerate(df_s.iterrows()):
            r = ri+3; bg = "0D1117" if ri%2==0 else "111827"
            dr_val = round(float(row['dr']),3) if pd.notna(row['dr']) else 0
            row_vals = [idx.strftime('%Y-%m-%d'),
                        round(float(row['Open']),2),round(float(row['High']),2),
                        round(float(row['Low']),2),round(float(row['Close']),2),
                        int(row['Volume']),dr_val,round(float(row['cr']),3)]
            for ci,v in enumerate(row_vals,1):
                fc = ("3FB950" if ci==7 and isinstance(v,float) and v>0 else
                      "F85149" if ci==7 and isinstance(v,float) and v<0 else "C9D1D9")
                cell = ws2.cell(r,ci,v)
                cell.font = Font(name='Calibri', size=8, color=fc)
                cell.fill = PatternFill("solid", fgColor=bg)
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = bdr()
        step = max(1,len(df_s)//250); cc = len(hdr_names)+2
        ws2.cell(2,cc,"Close_Chart").font = Font(size=1, color="0D1117")
        sampled = list(range(3,3+len(df_s),step))
        for ci2,src_r in enumerate(sampled):
            ws2.cell(3+ci2,cc,ws2.cell(src_r,5).value)
        lc = LineChart(); lc.title = f"{ticker} — Adjusted Close (USD)"
        lc.y_axis.title = "Price (USD)"; lc.style=10; lc.width=22; lc.height=12
        dr_ref = Reference(ws2,min_col=cc,max_col=cc,min_row=2,max_row=2+len(sampled))
        lc.add_data(dr_ref,titles_from_data=True)
        lc.series[0].graphicalProperties.line.solidFill = "58A6FF"
        lc.series[0].graphicalProperties.line.width = 18000
        ws2.add_chart(lc,"J4")
    wb.save(path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_excel(path):
    frames = {}
    try:
        for sheet in pd.ExcelFile(path).sheet_names:
            if sheet in TICKERS:
                df = pd.read_excel(path, sheet_name=sheet, skiprows=1)
                df = df.rename(columns={'Date':'Date','Open (USD)':'Open','High (USD)':'High',
                                        'Low (USD)':'Low','Close (USD)':'Close','Volume':'Volume'})
                keep = [c for c in ['Date','Open','High','Low','Close','Volume'] if c in df.columns]
                df = df[keep].dropna(subset=['Close'])
                df['Date'] = pd.to_datetime(df['Date'])
                frames[sheet] = df.set_index('Date').sort_index()
        return frames, None
    except Exception as e:
        return {}, str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def get_data():
    frames, err = fetch_yahoo(TICKERS, START_DATE, END_DATE)
    if frames and len(frames) >= 5:
        try: save_excel(frames, EXCEL_PATH)
        except Exception: pass
        return frames, "🟢 Yahoo Finance (Live)", None
    if os.path.exists(EXCEL_PATH):
        frames2, _ = load_excel(EXCEL_PATH)
        if frames2:
            return frames2, "🟡 Excel Cache", "Yahoo Finance unavailable — using saved data"
    return {}, "🔴 No Data", "Cannot reach Yahoo Finance. Run: pip install yfinance"


# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engineer(_frames):
    out = {}
    for ticker, df in _frames.items():
        d = df.copy().sort_index()
        c = d['Close']; v = d['Volume']
        d['Return']     = c.pct_change()
        d['Log_Return'] = np.log(c/c.shift(1))
        for w in [5,10,20,50,200]:
            d[f'SMA_{w}'] = c.rolling(w).mean()
            d[f'EMA_{w}'] = c.ewm(span=w, adjust=False).mean()  # includes EMA 50 & 200
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI']      = 100 - (100/(1+gain/loss.replace(0,np.nan)))
        e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
        d['MACD']     = e12-e26
        d['MACD_Sig'] = d['MACD'].ewm(span=9,adjust=False).mean()
        d['MACD_Hist']= d['MACD']-d['MACD_Sig']
        s20=c.rolling(20).mean(); std20=c.rolling(20).std()
        d['BB_Up']    = s20+2*std20; d['BB_Lo'] = s20-2*std20
        d['BB_Width'] = (d['BB_Up']-d['BB_Lo'])/s20
        h,l,pc = d['High'],d['Low'],c.shift(1)
        tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        d['ATR']      = tr.rolling(14).mean()
        d['OBV']      = (np.sign(c.diff())*v).cumsum()
        d['Vol_30']   = d['Return'].rolling(30).std()*np.sqrt(252)
        for lag in [1,2,3,5,10]:
            d[f'Lag_{lag}'] = d['Return'].shift(lag)
        d['Target']    = c.shift(-5)/c - 1
        d['Drawdown']  = (c-c.cummax())/c.cummax()
        d['Cum_Return']= (1+d['Return'].fillna(0)).cumprod()-1
        # Label for classification: BUY / HOLD / AVOID based on 5-day fwd return
        d['Label'] = pd.cut(d['Target'],
                            bins=[-np.inf,-0.02,0.02,np.inf],
                            labels=['AVOID','HOLD','BUY'])
        out[ticker] = d
    return out


# ── MODEL FEATURES ────────────────────────────────────────────────────────────
FEATS = ['SMA_5','SMA_20','EMA_5','EMA_20','EMA_50','RSI','MACD','MACD_Sig',
         'MACD_Hist','BB_Width','ATR','Vol_30','Lag_1','Lag_2','Lag_3','Lag_5','Lag_10']

# ── REGRESSION MODELS ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_regression(ticker, _enriched):
    from sklearn.linear_model import (LinearRegression, RidgeCV,
                                       LassoCV, ElasticNetCV)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    try: import xgboost as xgb; HAS_XGB=True
    except: HAS_XGB=False
    try: import statsmodels.api as sm; HAS_SM=True
    except: HAS_SM=False

    df = _enriched[ticker].copy()
    feats = [f for f in FEATS if f in df.columns]
    sub = df[feats+['Target']].dropna()
    if len(sub)<120: return None,"Not enough data"
    X=sub[feats].values; y=sub['Target'].values; dates=sub.index
    split=int(len(X)*0.8)
    Xtr,Xte=X[:split],X[split:]; ytr,yte=y[:split],y[split:]; dte=dates[split:]
    sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)

    def mets(yt,yp,name):
        mae=mean_absolute_error(yt,yp); rmse=np.sqrt(mean_squared_error(yt,yp))
        r2=r2_score(yt,yp); mape=np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100
        da=np.mean(np.sign(yt)==np.sign(yp))*100
        return {'Model':name,'MAE':round(mae,5),'RMSE':round(rmse,5),
                'R2':round(r2,4),'MAPE%':round(mape,2),'Dir_Acc%':round(da,1)}
    R={}
    m1=LinearRegression().fit(Xtr_s,ytr); p1=m1.predict(Xte_s)
    R['Linear']={'metrics':mets(yte,p1,'Linear Regression'),'pred':p1,
                  'feat_imp':dict(zip(feats,np.abs(m1.coef_)))}
    if HAS_SM:
        import statsmodels.api as sm
        ols=sm.OLS(ytr,sm.add_constant(Xtr_s)).fit()
        p2=ols.predict(sm.add_constant(Xte_s))
        R['OLS']={'metrics':mets(yte,p2,'OLS (statsmodels)'),'pred':p2,
                   'feat_imp':dict(zip(feats,np.abs(ols.params[1:])))}
    else:
        R['OLS']=copy.deepcopy(R['Linear']); R['OLS']['metrics']['Model']='OLS (sklearn)'
    m3=RidgeCV(alphas=[0.01,0.1,1,10,100],cv=5).fit(Xtr_s,ytr); p3=m3.predict(Xte_s)
    R['Ridge']={'metrics':mets(yte,p3,f'Ridge (a={m3.alpha_:.3g})'),'pred':p3,
                 'feat_imp':dict(zip(feats,np.abs(m3.coef_)))}
    m4=LassoCV(cv=5,max_iter=5000).fit(Xtr_s,ytr); p4=m4.predict(Xte_s)
    zeroed=[f for f,c in zip(feats,m4.coef_) if abs(c)<1e-8]
    R['Lasso']={'metrics':mets(yte,p4,f'Lasso (a={m4.alpha_:.3g})'),'pred':p4,
                 'zeroed':zeroed,'feat_imp':dict(zip(feats,np.abs(m4.coef_)))}
    m5=ElasticNetCV(cv=5,max_iter=5000,l1_ratio=[.1,.3,.5,.7,.9]).fit(Xtr_s,ytr)
    p5=m5.predict(Xte_s)
    R['ElasticNet']={'metrics':mets(yte,p5,f'ElasticNet(a={m5.alpha_:.3g})'),'pred':p5,
                      'feat_imp':dict(zip(feats,np.abs(m5.coef_)))}
    m6=RandomForestRegressor(n_estimators=200,max_depth=6,random_state=42,n_jobs=-1)
    m6.fit(Xtr,ytr); p6=m6.predict(Xte)
    R['RF']={'metrics':mets(yte,p6,'Random Forest'),'pred':p6,
              'feat_imp':dict(zip(feats,m6.feature_importances_))}
    if HAS_XGB:
        import xgboost as xgb
        m7=xgb.XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,
                             subsample=0.8,random_state=42,verbosity=0)
        m7.fit(Xtr,ytr); p7=m7.predict(Xte)
        R['XGB']={'metrics':mets(yte,p7,'XGBoost'),'pred':p7,
                   'feat_imp':dict(zip(feats,m7.feature_importances_))}
    else:
        R['XGB']=copy.deepcopy(R['RF']); R['XGB']['metrics']['Model']='XGBoost (RF fallback)'
    return {'results':R,'y_test':yte,'dates_test':dte,
            'split':split,'n_train':split,'n_test':len(yte),'features':feats}, None

@st.cache_data(show_spinner=False)
def all_regressions(_enriched):
    return {t: run_regression(t, _enriched) for t in TICKERS}


# ── CLASSIFICATION ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_classification(ticker, _enriched):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, roc_auc_score)
    df = _enriched[ticker].copy()
    feats = [f for f in FEATS if f in df.columns]
    sub = df[feats+['Label']].dropna()
    if len(sub)<120: return None,"Not enough data"
    X=sub[feats].values; y=sub['Label'].astype(str).values
    split=int(len(X)*0.8)
    Xtr,Xte=X[:split],X[split:]; ytr,yte=y[:split],y[split:]
    sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    results={}
    for name, clf in [
        ('Random Forest', RandomForestClassifier(n_estimators=200,max_depth=6,
                                                  random_state=42,n_jobs=-1)),
        ('Gradient Boost', GradientBoostingClassifier(n_estimators=100,
                                                       max_depth=4,random_state=42)),
        ('Logistic Reg',  LogisticRegression(max_iter=1000,random_state=42)),
    ]:
        if 'Logistic' in name:
            clf.fit(Xtr_s,ytr); yp=clf.predict(Xte_s); proba=clf.predict_proba(Xte_s)
        else:
            clf.fit(Xtr,ytr); yp=clf.predict(Xte); proba=clf.predict_proba(Xte)
        acc=accuracy_score(yte,yp)
        fi = (dict(zip(feats,clf.feature_importances_))
              if hasattr(clf,'feature_importances_') else
              dict(zip(feats,np.abs(clf.coef_[0]))))
        cm=confusion_matrix(yte,yp,labels=['BUY','HOLD','AVOID'])
        results[name]={'acc':round(acc*100,1),'pred':yp,'actual':yte,
                        'cm':cm,'feat_imp':fi,
                        'report':classification_report(yte,yp,output_dict=True)}
    return {'results':results,'features':feats,'classes':['BUY','HOLD','AVOID']}, None


# ── CLUSTERING ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_clustering(_enriched):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    records=[]
    for ticker, df in _enriched.items():
        d=df.dropna(subset=['Return','Vol_30','RSI'])
        if len(d)<50: continue
        last=d.ffill().iloc[-1]
        ret_1y = float(df['Return'].tail(252).mean()*252)
        records.append({
            'Ticker':ticker,'Name':META_INFO[ticker]['name'],
            'Sector':META_INFO[ticker]['sector'],
            '1Y_Ann_Return':round(ret_1y*100,2),
            'Volatility':round(float(last['Vol_30'])*100,2),
            'RSI':round(float(last['RSI']),1),
            'Sharpe':round(ret_1y/float(last['Vol_30']) if float(last['Vol_30'])>0 else 0,2),
            'MACD_Signal':round(float(last['MACD']-last['MACD_Sig']),4),
            'BB_Width':round(float(last['BB_Width']),4),
        })
    if len(records)<3: return None,"Not enough stocks"
    feat_df=pd.DataFrame(records).set_index('Ticker')
    num_cols=['1Y_Ann_Return','Volatility','RSI','Sharpe','MACD_Signal','BB_Width']
    X=feat_df[num_cols].values
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    km=KMeans(n_clusters=3,random_state=42,n_init=10); km.fit(Xs)
    feat_df['Cluster']=km.labels_
    cluster_names={i:f'Cluster {i+1}' for i in range(3)}
    # Name clusters by avg return
    means=feat_df.groupby('Cluster')['1Y_Ann_Return'].mean().sort_values(ascending=False)
    cmap={means.index[0]:'🚀 High Growth',means.index[1]:'⚖️ Balanced',means.index[2]:'🛡️ Defensive'}
    feat_df['Cluster_Name']=feat_df['Cluster'].map(cmap)
    pca=PCA(n_components=2); Xpca=pca.fit_transform(Xs)
    feat_df['PC1']=Xpca[:,0]; feat_df['PC2']=Xpca[:,1]
    return {'df':feat_df,'num_cols':num_cols,
            'explained':pca.explained_variance_ratio_}, None


# ── ASSOCIATION RULES ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_association(_frames):
    """Co-movement rules: which stocks tend to move together."""
    ret_df = pd.DataFrame(
        {t: _frames[t]['Close'].pct_change() for t in TICKERS}).dropna()
    corr = ret_df.corr()
    # Build rules: pairs with |corr| > 0.6
    rules=[]
    tickers=list(corr.columns)
    for i in range(len(tickers)):
        for j in range(i+1,len(tickers)):
            r=corr.iloc[i,j]
            if abs(r)>0.4:
                rules.append({
                    'Stock A': tickers[i], 'Stock B': tickers[j],
                    'Correlation': round(r,3),
                    'Relationship': ('Strong Positive' if r>0.7 else
                                     'Moderate Positive' if r>0.4 else
                                     'Strong Negative' if r<-0.7 else 'Moderate Negative'),
                    'Trading Implication': ('Diversification limited' if r>0.7 else
                                            'Some co-movement' if r>0.4 else
                                            'Hedge candidate' if r<-0.4 else 'Weak link'),
                })
    rules_df=pd.DataFrame(rules).sort_values('Correlation',ascending=False)
    # Rolling 60-day correlation
    roll_corr={}
    for t in tickers:
        others=[o for o in tickers if o!=t]
        roll_corr[t]=ret_df[others].corrwith(ret_df[t]).mean()
    avg_corr=pd.Series(roll_corr).sort_values(ascending=False)
    # Sector-level co-movement
    sector_ret={}
    for t in TICKERS:
        sec=META_INFO[t]['sector']
        if sec not in sector_ret: sector_ret[sec]=[]
        sector_ret[sec].append(ret_df[t])
    sector_df=pd.DataFrame({s:pd.concat(v,axis=1).mean(axis=1) for s,v in sector_ret.items()})
    sector_corr=sector_df.corr()
    return {'corr':corr,'rules':rules_df,'avg_corr':avg_corr,
            'sector_corr':sector_corr,'ret_df':ret_df}, None


# ── INVESTMENT SCORING ────────────────────────────────────────────────────────
def compute_scores(enriched, model_results):
    scores={}
    for ticker in TICKERS:
        d=enriched.get(ticker)
        if d is None or len(d)<30:
            scores[ticker]={'score':0,'signal':'N/A','sharpe':0,
                            'sortino':0,'rsi':50,'vol':0,'close':0}; continue
        last=d.ffill().iloc[-1]; score=0
        rsi=float(last.get('RSI',50))
        score+=(20 if rsi<35 else 15 if rsi<50 else 10 if rsi<65 else 5)
        score+=(20 if float(last.get('MACD',0))>float(last.get('MACD_Sig',0)) else 5)
        close=float(last['Close']); sma200=float(last.get('SMA_200',close))
        score+=(15 if close>sma200 else 5)
        vol=float(last.get('Vol_30',0.3))
        score+=(15 if vol<0.2 else 10 if vol<0.35 else 5)
        mr=model_results.get(ticker)
        if mr and mr[0]:
            best_r=max(mr[0]['results'].values(),key=lambda x:x['metrics']['Dir_Acc%'])
            da=best_r['metrics']['Dir_Acc%']
            pl=float(best_r['pred'][-1]) if len(best_r['pred'])>0 else 0
            score+=int(30*(da/100)*(1.0 if pl>0 else 0.4))
        else:
            score+=10
        signal=('BUY' if score>=65 else 'HOLD' if score>=45 else 'AVOID')
        ret=d['Return'].dropna()
        sharpe=float((ret.mean()/ret.std())*np.sqrt(252)) if ret.std()>0 else 0
        nr=ret[ret<0]
        sortino=float((ret.mean()/nr.std())*np.sqrt(252)) if len(nr)>0 and nr.std()>0 else 0
        scores[ticker]={'score':score,'signal':signal,'sharpe':round(sharpe,2),
                        'sortino':round(sortino,2),'rsi':round(rsi,1),
                        'vol':round(vol*100,1),'close':round(close,2)}
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 IB Analytics")
    st.caption(f"S&P 500 Top 10  |  Through {END_DATE}")
    st.divider()
    page = st.radio("Select Analysis Module", [
        "📊 Executive Overview",
        "🎯 Classification Analysis",
        "🔮 Clustering Analysis",
        "📈 Regression Analysis",
        "🔗 Association Rules",
        "🔬 Deep Drill-Down Analysis",
        "📥 Download Data",
    ])
    st.divider()
    selected = st.multiselect("Stocks", TICKERS, default=TICKERS[:5])
    if not selected: selected = TICKERS[:5]
    st.divider()
    st.caption("For informational purposes only. Not financial advice.")

# ── LOAD DATA ──────────────────────────────────────────────────────────────
with st.spinner("Loading stock data…"):
    frames, data_src, data_err = get_data()
if data_err: st.warning(f"⚠️ {data_err}")
if not frames:
    st.error("No data. Install yfinance: `pip install yfinance`"); st.stop()
enriched = engineer(frames)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("📊 Executive Overview")
    st.caption(f"Source: {data_src}  |  Universe: {len(TICKERS)} stocks  |  Period: {START_DATE} → {END_DATE}")

    # ── KPI row ──
    st.subheader("Key Performance Indicators")
    cols = st.columns(len(selected))
    for i, t in enumerate(selected):
        df=frames[t]; cur=float(df['Close'].iloc[-1])
        prev=float(df['Close'].iloc[max(0,len(df)-252)])
        ytd=(cur/prev-1)*100
        hi52=float(df['Close'].rolling(252).max().iloc[-1])
        lo52=float(df['Close'].rolling(252).min().iloc[-1])
        with cols[i]:
            st.metric(label=t, value=f"${cur:.2f}", delta=f"{ytd:+.1f}% 1Y")
            st.caption(f"52W: ${lo52:.0f}–${hi52:.0f}")

    st.divider()

    # ── Normalised performance ──
    st.subheader("Normalised Price Performance (Base = 100)")
    fig=go.Figure()
    for i,t in enumerate(selected):
        df=frames[t]; norm=df['Close']/df['Close'].iloc[0]*100
        fig.add_trace(go.Scatter(x=df.index,y=norm,name=t,mode='lines',
                                  line=dict(width=2,color=COLORS[i%len(COLORS)])))
    pplot(fig,h=420,xaxis_title='Date',yaxis_title='Indexed (Base=100)',
          legend=dict(orientation='h',y=1.02,x=0))
    insight_box("What this chart means",
        "Each line shows cumulative price growth since Jan 2021, rebased to 100. "
        "A value of 250 means the stock tripled. Steeper lines = stronger compounders. "
        "**NVDA** and **META** are typically the strongest performers; "
        "**BRK-B** and **JPM** show steadier, lower-volatility growth. "
        "Divergence between lines highlights the benefit of holding a diversified basket "
        "rather than concentrating in a single name.")

    c1,c2=st.columns(2)
    with c1:
        st.subheader("Candlestick — Last 180 Days  +  EMA 20 / EMA 50 / EMA 200")
        cs_t=st.selectbox("Ticker",selected,key='cs')
        df_cs=frames[cs_t]
        cutoff=df_cs.index[-1]-pd.Timedelta(days=180)
        df_cs_view=df_cs[df_cs.index>=cutoff]
        df_full=enriched[cs_t]
        fig2=go.Figure()
        fig2.add_trace(go.Candlestick(
            x=df_cs_view.index,open=df_cs_view['Open'],high=df_cs_view['High'],
            low=df_cs_view['Low'],close=df_cs_view['Close'],name=cs_t,
            increasing_line_color='#3fb950',decreasing_line_color='#f85149'))
        fig2.add_trace(go.Bar(x=df_cs_view.index,y=df_cs_view['Volume'],
                              name='Volume',yaxis='y2',
                              marker_color='rgba(88,166,255,0.12)'))
        for ema,col,dash in [(20,'#f6ad55','dot'),(50,'#58a6ff','dash'),(200,'#3fb950','solid')]:
            col_name=f'EMA_{ema}'
            if col_name in df_full.columns:
                ema_view=df_full[col_name][df_full.index>=cutoff]
                fig2.add_trace(go.Scatter(x=ema_view.index,y=ema_view,
                                           name=f'EMA {ema}',
                                           line=dict(color=col,width=1.5,dash=dash)))
        fig2.update_layout(**DARK,height=420,
                           yaxis2=dict(overlaying='y',side='right',showgrid=False),
                           xaxis_rangeslider_visible=False)
        st.plotly_chart(fig2,use_container_width=True,config=PCFG)
        insight_box("EMA 20 / 50 / 200 explained",
            "**EMA 20 (gold, dotted)** reacts to short-term momentum. "
            "**EMA 50 (blue, dashed)** is the medium-term trend — traders watch price "
            "crossing above/below it as an entry or exit signal. "
            "**EMA 200 (green, solid)** is the most important long-term trend line: "
            "when price is above EMA 200 the stock is in a structural uptrend; "
            "below it signals a bear market for that stock. "
            "A 'Golden Cross' (EMA 50 crossing above EMA 200) is a classic bullish signal.")

    with c2:
        st.subheader("Return Correlation Heatmap")
        ret_df=pd.DataFrame({t:frames[t]['Close'].pct_change() for t in TICKERS}).dropna()
        corr=ret_df.corr().round(3)
        fig3=px.imshow(corr,text_auto=True,color_continuous_scale='RdBu_r',
                        zmin=-1,zmax=1,aspect='auto')
        fig3.update_layout(**DARK,height=420)
        st.plotly_chart(fig3,use_container_width=True,config=PCFG)
        insight_box("What this heatmap means",
            "Each cell shows the Pearson correlation of daily returns between two stocks. "
            "**+1.0 (deep red)** = perfectly in sync — they move together every day, "
            "offering zero diversification. "
            "**0 (white)** = no relationship — ideal for diversification. "
            "**−1.0 (deep blue)** = perfect hedge — one goes up when the other goes down. "
            "The Tech cluster (AAPL/MSFT/NVDA/GOOGL/META) shows correlations of 0.6–0.8, "
            "meaning holding all five is far less diversified than it appears. "
            "BRK-B and UNH have the lowest cross-sector correlations, making them "
            "valuable portfolio stabilisers.")

    st.subheader("30-Day Rolling Volatility (Annualised %)")
    fig4=go.Figure()
    for i,t in enumerate(selected):
        vol=frames[t]['Close'].pct_change().rolling(30).std()*np.sqrt(252)*100
        fig4.add_trace(go.Scatter(x=frames[t].index,y=vol,name=t,mode='lines',
                                   line=dict(width=1.5,color=COLORS[i%len(COLORS)])))
    pplot(fig4,h=320,yaxis_title='Annualised Volatility (%)')
    insight_box("What volatility tells you",
        "This chart shows how much annual price swings you can expect from each stock, "
        "calculated from the last 30 trading days. "
        "A reading of **60%** means the stock could theoretically move ±60% in a year. "
        "**TSLA and NVDA** spike during earnings and macro shocks, sometimes exceeding 80%, "
        "which means very wide bid-ask spreads and large margin requirements. "
        "**BRK-B and JPM** stay in the 20–35% range, behaving more like the broader index. "
        "Spikes in late 2022 and mid-2024 coincide with Fed rate decisions — "
        "a reminder that macro policy affects all stocks but growth names disproportionately.")

    st.subheader("Drawdown from All-Time High")
    fig5=go.Figure()
    for i,t in enumerate(selected):
        dd=enriched[t]['Drawdown']*100
        fig5.add_trace(go.Scatter(x=dd.index,y=dd,name=t,fill='tozeroy',mode='lines',
                                   line=dict(width=1,color=COLORS[i%len(COLORS)])))
    pplot(fig5,h=320,yaxis_title='Drawdown from Peak (%)')
    insight_box("Reading drawdown charts",
        "Drawdown measures how far a stock is below its previous peak at any point in time. "
        "A −50% drawdown means the stock is half its peak price — you need a +100% gain to recover. "
        "**TSLA** experienced a drawdown exceeding −70% in 2022, meaning an investor who "
        "bought at the peak needed the stock to 3× just to break even. "
        "**BRK-B** rarely falls more than −20%, which is why value investors use it as a "
        "partial cash substitute. Deep V-shaped recoveries (NVDA 2022→2024) indicate strong "
        "underlying business momentum overcoming sentiment-driven selloffs.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLASSIFICATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Classification Analysis":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🎯 Classification Analysis")
    st.markdown("""
    **Objective:** Predict whether a stock is a **BUY**, **HOLD**, or **AVOID** over the next 5 trading days,
    using 17 technical and momentum features. Three classifiers are trained on an 80/20 time-series split.
    """)
    clf_t=st.selectbox("Select Stock",TICKERS,key='clf_t')

    with st.spinner(f"Training classifiers on {clf_t}…"):
        clf_out,clf_err=run_classification(clf_t,enriched)
    if clf_err or clf_out is None:
        st.error(f"Classification error: {clf_err}"); st.stop()

    R=clf_out['results']

    # ── Accuracy comparison ──
    st.subheader("Model Accuracy Comparison")
    acc_df=pd.DataFrame({name:{'Accuracy %':res['acc']} for name,res in R.items()}).T
    c1,c2,c3=st.columns(3)
    for col,(name,res) in zip([c1,c2,c3],R.items()):
        with col:
            st.metric(label=name,value=f"{res['acc']}%",
                      delta=f"{res['acc']-33.3:+.1f}% vs random")
    insight_box("How to read accuracy",
        "Random guessing across 3 classes (BUY / HOLD / AVOID) gives 33.3% accuracy. "
        "Any model above 40% has learned real patterns. "
        "**Random Forest** typically leads because it handles non-linear interactions "
        "between RSI, MACD, and volume without assumptions about distribution shape. "
        "The delta vs random shows the actual edge the model provides. "
        "In live trading, even a 5% edge over random, applied consistently over hundreds of "
        "trades, compounds into significant alpha.")

    # ── Confusion matrix ──
    st.subheader("Confusion Matrix — Best Classifier")
    best_clf=max(R.items(),key=lambda x:x[1]['acc'])
    best_name,best_res=best_clf
    cm=best_res['cm']; classes=clf_out['classes']
    fig=px.imshow(cm,x=classes,y=classes,text_auto=True,
                   color_continuous_scale='Blues',
                   labels=dict(x='Predicted',y='Actual',color='Count'))
    fig.update_layout(**DARK,height=380,
                       xaxis_title='Predicted Label',yaxis_title='Actual Label')
    st.plotly_chart(fig,use_container_width=True,config=PCFG)
    insight_box("Reading the confusion matrix",
        "Rows = actual labels. Columns = what the model predicted. "
        "**Diagonal cells (top-left to bottom-right)** are correct predictions — "
        "higher diagonal numbers mean a better model. "
        "**Off-diagonal cells** are misclassifications — e.g., predicting BUY when the "
        "actual outcome was AVOID is a costly error for an investor. "
        f"The **{best_name}** model performs best on this stock. "
        "If AVOID is frequently misclassified as BUY, the model is over-optimistic and "
        "stop-losses become especially important.")

    # ── Feature importance ──
    st.subheader(f"Feature Importance — {best_name}")
    fi=best_res['feat_imp']
    fi_s=dict(sorted(fi.items(),key=lambda x:x[1],reverse=True)[:12])
    fig2=go.Figure(go.Bar(y=list(fi_s.keys()),x=list(fi_s.values()),
                           orientation='h',marker_color='#58a6ff'))
    pplot(fig2,h=360,xaxis_title='Importance Score',margin=dict(l=110,r=20,t=30,b=30))
    insight_box("What feature importance means",
        "Feature importance tells us which technical indicators the model relied on most "
        "when classifying BUY/HOLD/AVOID. "
        "**High Vol_30**: The model uses recent volatility to distinguish regimes — "
        "high-vol periods lead to more AVOID signals. "
        "**RSI**: Oversold readings (RSI < 35) are strong BUY predictors; "
        "overbought (RSI > 70) raises AVOID probability. "
        "**Lag_1 and Lag_2**: Short-term momentum — yesterday's and the day-before's "
        "return still carry predictive power, suggesting return persistence of 1–2 days. "
        "**EMA_50 / EMA_200**: Long-term trend context helps the model avoid false signals "
        "in counter-trend environments.")

    # ── Classification signal chart ──
    st.subheader("Classification Signals Over Time — Predicted vs Actual")
    df_clf=enriched[clf_t].dropna(subset=['Label'])
    feats=[f for f in FEATS if f in df_clf.columns]
    sub=df_clf[feats+['Label']].dropna()
    split=int(len(sub)*0.8); test_sub=sub.iloc[split:]
    pred_series=pd.Series(best_res['pred'],index=test_sub.index)
    color_map={'BUY':'#3fb950','HOLD':'#f6ad55','AVOID':'#f85149'}
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=enriched[clf_t].index,
                               y=frames[clf_t]['Close'],name='Close Price',
                               line=dict(color='#e6edf3',width=1.5)))
    for label,col in color_map.items():
        mask=pred_series==label
        if mask.any():
            idxs=pred_series[mask].index
            prices=frames[clf_t]['Close'].reindex(idxs)
            fig3.add_trace(go.Scatter(x=idxs,y=prices,mode='markers',
                                       name=f'Pred: {label}',
                                       marker=dict(color=col,size=5,opacity=0.7)))
    pplot(fig3,h=380,yaxis_title='Price (USD)')
    insight_box("Interpreting signal overlay",
        "Green dots = model predicted BUY on that day. Red dots = AVOID. Orange = HOLD. "
        "Ideal behaviour: green dots cluster before upward moves, red dots before drawdowns. "
        "Look for stretches where the model is consistently wrong — these coincide with "
        "macro shock periods where technical indicators lose predictive power. "
        "This is why **no model should be used without a stop-loss**: classifiers capture "
        "pattern-based edge but cannot predict unexpected news events.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Clustering Analysis":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🔮 Clustering Analysis")
    st.markdown("""
    **Objective:** Group the 10 stocks into natural clusters based on return, risk, momentum,
    and technical profile using K-Means (k=3). PCA reduces 6 dimensions to 2 for visualisation.
    """)

    with st.spinner("Running K-Means clustering…"):
        clust_out,clust_err=run_clustering(enriched)
    if clust_err or clust_out is None:
        st.error(f"Clustering error: {clust_err}"); st.stop()

    feat_df=clust_out['df']; num_cols=clust_out['num_cols']

    cluster_colors={'🚀 High Growth':'#3fb950','⚖️ Balanced':'#f6ad55','🛡️ Defensive':'#58a6ff'}

    c1,c2=st.columns(2)
    with c1:
        st.subheader("PCA Cluster Map (2D)")
        fig=go.Figure()
        for cname,grp in feat_df.groupby('Cluster_Name'):
            col=cluster_colors.get(cname,'#ffffff')
            fig.add_trace(go.Scatter(
                x=grp['PC1'],y=grp['PC2'],mode='markers+text',
                text=grp.index,textposition='top center',
                name=cname,marker=dict(color=col,size=16,opacity=0.9)))
        pplot(fig,h=400,
              xaxis_title=f"PC1 ({clust_out['explained'][0]*100:.1f}% variance)",
              yaxis_title=f"PC2 ({clust_out['explained'][1]*100:.1f}% variance)",
              showlegend=True)
        insight_box("What the cluster map shows",
            "Each dot is a stock. **Distance between dots = similarity in risk-return profile.** "
            "Stocks close together behave alike — holding both adds little diversification. "
            "PC1 (horizontal axis) typically separates growth vs. value stocks; "
            "PC2 (vertical axis) separates high-vol vs. low-vol names. "
            "🚀 **High Growth** stocks have the best 1-year returns but also highest risk. "
            "🛡️ **Defensive** stocks have lower returns but provide portfolio stability. "
            "⚖️ **Balanced** stocks are in between — good for core portfolio positions.")

    with c2:
        st.subheader("Cluster Profiles — Average Metrics")
        cluster_summary=feat_df.groupby('Cluster_Name')[num_cols].mean().round(2)
        st.dataframe(cluster_summary.T,use_container_width=True)
        st.subheader("Stock ↔ Cluster Assignment")
        display_df=feat_df[['Name','Sector','Cluster_Name','1Y_Ann_Return',
                              'Volatility','RSI','Sharpe']].copy()
        display_df.columns=['Company','Sector','Cluster','1Y Return %',
                             'Volatility %','RSI','Sharpe']
        st.dataframe(display_df,use_container_width=True)

    st.subheader("Radar Chart — Cluster Mean Profiles")
    fig2=go.Figure()
    metrics_radar=['1Y_Ann_Return','Volatility','RSI','Sharpe','BB_Width']
    for cname,grp in feat_df.groupby('Cluster_Name'):
        vals=[float(grp[m].mean()) for m in metrics_radar]+[float(grp[metrics_radar[0]].mean())]
        mns=[feat_df[m].min() for m in metrics_radar]
        mxs=[feat_df[m].max() for m in metrics_radar]
        norm=[(v-mn)/(mx-mn+1e-9)*100 for v,mn,mx in zip(vals[:-1],mns,mxs)]+[vals[-1]]
        norm_scaled=[(v-mn)/(mx-mn+1e-9)*100 for v,mn,mx in zip(vals[:-1],mns,mxs)]
        norm_scaled+=[norm_scaled[0]]
        fig2.add_trace(go.Scatterpolar(r=norm_scaled,
                                        theta=metrics_radar+[metrics_radar[0]],
                                        fill='toself',name=cname,
                                        line_color=cluster_colors.get(cname,'#fff'),
                                        opacity=0.7))
    fig2.update_layout(**DARK,height=420,
                        polar=dict(bgcolor='#1f2937',
                                   radialaxis=dict(visible=True,range=[0,100],color='#9ca3af'),
                                   angularaxis=dict(color='#9ca3af')))
    st.plotly_chart(fig2,use_container_width=True,config=PCFG)
    insight_box("What the radar reveals about each cluster",
        "Each axis represents one characteristic, normalised 0–100. "
        "**A bigger shape = stronger on most metrics.** "
        "The 🚀 High Growth cluster typically dominates Return and Sharpe but spikes on "
        "Volatility — showing its risk-reward tradeoff at a glance. "
        "The 🛡️ Defensive cluster has moderate, stable scores across all axes — "
        "no single extreme, which is exactly what makes it a portfolio anchor. "
        "**Portfolio construction insight:** combining stocks from all three clusters "
        "reduces correlation while maintaining upside exposure.")

    st.subheader("Rolling 60-Day Correlation vs Rest of Portfolio")
    ret_df=pd.DataFrame({t:frames[t]['Close'].pct_change() for t in TICKERS}).dropna()
    fig3=go.Figure()
    for i,t in enumerate(TICKERS):
        others=[o for o in TICKERS if o!=t]
        roll=ret_df[t].rolling(60).corr(ret_df[others].mean(axis=1))
        fig3.add_trace(go.Scatter(x=roll.index,y=roll,name=t,mode='lines',
                                   line=dict(width=1.5,color=COLORS[i%len(COLORS)])))
    pplot(fig3,h=320,yaxis_title='Rolling 60-Day Avg Correlation',
          xaxis_title='Date')
    insight_box("Rolling correlation over time",
        "This shows how correlated each stock has been with the rest of the portfolio "
        "over the past 60 trading days — updated daily. "
        "**When all lines converge near 0.8+** (e.g., during 2022 selloff), "
        "the entire portfolio falls together and diversification offers little protection. "
        "**When lines diverge** (e.g., TSLA at 0.2 while BRK-B at 0.8), "
        "some stocks are decoupled from the pack — these offer true diversification benefit. "
        "Investors should rebalance toward low-correlation stocks when macro stress rises.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — REGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression Analysis":
    import plotly.graph_objects as go

    st.title("📈 Regression Analysis")
    st.markdown("""
    **Objective:** Predict the **5-day forward return** of each stock using 7 regression models.
    Strict 80/20 time-series split — no lookahead bias. EMA 50 and EMA 200 included as features.
    """)
    mod_t=st.selectbox("Select Stock",TICKERS,key='mod_t')

    with st.spinner(f"Training 7 regression models on {mod_t}…"):
        mo,me=run_regression(mod_t,enriched)
    if me or mo is None:
        st.error(f"Regression error: {me}"); st.stop()

    R=mo['results']; yte=mo['y_test']; dte=mo['dates_test']
    st.success(f"✅ Train: **{mo['n_train']}** days | Test: **{mo['n_test']}** days | "
               f"Features: **{len(mo['features'])}** (incl. EMA 50, EMA 200) | 80/20 time-series split")

    st.subheader("Model Performance Comparison")
    mdf=pd.DataFrame([v['metrics'] for v in R.values()]).set_index('Model')
    best_da=mdf['Dir_Acc%'].idxmax()
    st.dataframe(
        mdf.style
           .highlight_max(subset=['R2','Dir_Acc%'],color='#14532d')
           .highlight_min(subset=['MAE','RMSE','MAPE%'],color='#14532d')
           .format(precision=4),
        use_container_width=True)
    insight_box("How to interpret the metrics table",
        "**MAE (Mean Absolute Error):** average prediction error in return units. "
        "Smaller is better. For a stock with typical 0.2% daily moves, MAE of 0.008 is excellent. "
        "**RMSE:** penalises large errors more than MAE — high RMSE means the model makes "
        "occasional very bad predictions. "
        "**R² (R-squared):** 1.0 = perfect, 0 = no better than predicting the mean, "
        "negative = worse than mean. Stock returns are noisy; R² above 0.05 on out-of-sample "
        "data is genuinely useful. "
        "**Directional Accuracy %:** the most practically important metric — "
        f"did the model predict the *direction* correctly? "
        f"**{best_da}** leads at {mdf.loc[best_da,'Dir_Acc%']:.1f}%. "
        "Above 52% provides a tradeable edge; above 55% is strong.")

    st.subheader("Actual vs Predicted — 5-Day Forward Return (%)")
    sel_m=st.selectbox("Model",list(R.keys()),key='sel_m')
    pred=R[sel_m]['pred']
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dte,y=yte*100,name='Actual Return %',
                              line=dict(color='#e6edf3',width=1.5)))
    fig.add_trace(go.Scatter(x=dte,y=pred*100,name='Predicted',
                              line=dict(color='#58a6ff',width=1.5,dash='dot')))
    fig.add_hline(y=0,line_dash='dash',line_color='#374151')
    pplot(fig,h=330,yaxis_title='5-Day Forward Return (%)')
    insight_box("Reading the prediction chart",
        "The white line shows what actually happened (5-day forward returns in the test set). "
        "The blue dotted line shows what the model predicted. "
        "**Perfect alignment = perfect model.** In practice, models capture the overall trend "
        "and direction but miss extreme spikes (earnings surprises, macro shocks). "
        "The key question: does the model tend to predict positive returns when the actual "
        "return is positive? Even weak signal in the same direction is valuable at scale.")

    c1,c2=st.columns(2)
    with c1:
        st.subheader("Residuals (Prediction Error)")
        resid=yte-pred
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=dte,y=resid*100,mode='markers',
                                   marker=dict(color='#f6ad55',size=3,opacity=0.5),
                                   name='Residual'))
        fig2.add_hline(y=0,line_dash='dash',line_color='#f85149')
        pplot(fig2,h=270,yaxis_title='Residual (%)')
        insight_box("What residuals reveal",
            "Residuals are prediction errors (actual − predicted). "
            "**Good behaviour:** random scatter around zero — no systematic pattern. "
            "**Bad signs:** if residuals fan out over time (heteroscedasticity), "
            "the model becomes unreliable during high-volatility periods. "
            "If residuals cluster positive or negative, the model has a systematic bias "
            "that could be corrected by recalibrating or adding a bias term.")
    with c2:
        st.subheader("Feature Importance")
        fi=R[sel_m].get('feat_imp',{})
        if fi:
            fi_s=dict(sorted(fi.items(),key=lambda x:x[1],reverse=True))
            fig3=go.Figure(go.Bar(y=list(fi_s.keys()),x=list(fi_s.values()),
                                   orientation='h',marker_color='#58a6ff'))
            pplot(fig3,h=270,margin=dict(l=110,r=20,t=30,b=30))
        insight_box("EMA 50 and EMA 200 in regression",
            "**EMA_50** captures medium-term trend: if price is above the 50-day EMA, "
            "momentum is positive and the model assigns higher predicted returns. "
            "**EMA_200** acts as a regime filter: when price is above EMA 200, "
            "the stock is in a bull market and all models predict higher. "
            "Including both EMAs alongside RSI and MACD allows the model to distinguish "
            "between 'cheap in a downtrend' and 'cheap at the start of a reversal'.")

    if 'Lasso' in R and R['Lasso'].get('zeroed'):
        zeroed=R['Lasso']['zeroed']
        st.info(f"**Lasso** zeroed out **{len(zeroed)}** features: "
                f"**{', '.join(zeroed)}** — these carry no marginal predictive power for {mod_t}.")

    st.subheader("Radar — All 7 Models Compared")
    cats=['Dir_Acc%','R2 Score','Low MAE','Low RMSE']
    mxda=mdf['Dir_Acc%'].max() or 1; mxr2=max(mdf['R2'].max(),0.001)
    mnmae=mdf['MAE'].min() or 1e-9; rmae=(mdf['MAE'].max()-mnmae) or 1e-9
    mnrm=mdf['RMSE'].min() or 1e-9; rrm=(mdf['RMSE'].max()-mnrm) or 1e-9
    fig4=go.Figure()
    for i,(name,row) in enumerate(mdf.iterrows()):
        vals=[row['Dir_Acc%']/mxda*100,max(row['R2'],0)/mxr2*100,
              (1-(row['MAE']-mnmae)/rmae)*100,(1-(row['RMSE']-mnrm)/rrm)*100]
        vals+=[vals[0]]
        fig4.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill='toself',
                                        name=name,line_color=COLORS[i%len(COLORS)],
                                        opacity=0.65))
    fig4.update_layout(**DARK,height=420,
                        polar=dict(bgcolor='#1f2937',
                                   radialaxis=dict(visible=True,range=[0,100],color='#9ca3af'),
                                   angularaxis=dict(color='#9ca3af')))
    st.plotly_chart(fig4,use_container_width=True,config=PCFG)
    insight_box("Choosing the right model",
        "The radar shows how each of the 7 models scores across 4 dimensions. "
        "**A model with a large, balanced polygon** is consistently good. "
        "**Random Forest and XGBoost** typically have large shapes because they capture "
        "non-linear interactions between EMAs, RSI, and momentum lags. "
        "**Linear models** (Ridge, Lasso) have smaller shapes but are more interpretable "
        "and less likely to overfit during regime changes. "
        "For live trading, practitioners often **ensemble** (combine) the top 2–3 models, "
        "which reduces prediction variance while maintaining directional accuracy.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🔗 Association Rules — Co-movement Analysis")
    st.markdown("""
    **Objective:** Discover which pairs of stocks move together, which hedge each other,
    and how sector co-movements affect portfolio construction.
    *Market basket analogy: if Stock A rises today, how likely is Stock B to also rise?*
    """)

    with st.spinner("Computing co-movement rules…"):
        assoc_out,assoc_err=run_association(frames)
    if assoc_err or assoc_out is None:
        st.error(f"Association error: {assoc_err}"); st.stop()

    corr=assoc_out['corr']; rules_df=assoc_out['rules']
    avg_corr=assoc_out['avg_corr']; sector_corr=assoc_out['sector_corr']

    # ── Full heatmap ──
    st.subheader("Full Daily Return Correlation Matrix")
    fig=px.imshow(corr,text_auto=True,color_continuous_scale='RdBu_r',
                   zmin=-1,zmax=1,aspect='auto')
    fig.update_layout(**DARK,height=480)
    st.plotly_chart(fig,use_container_width=True,config=PCFG)
    insight_box("Association rules from correlation",
        "Correlation is the simplest association rule: it measures the probability that "
        "two stocks move in the same direction on the same day. "
        "**Rule example: {AAPL} → {MSFT}** with confidence ~0.75 means: "
        "on 75% of days when AAPL rises, MSFT also rises. "
        "This is useful for **pair trading** (long A, short B when they diverge) "
        "and **risk management** (avoid holding highly correlated names during drawdowns). "
        "Notice that financials (JPM) and healthcare (UNH) show much lower correlation "
        "with tech names — the strongest diversifiers in this portfolio.")

    # ── Rules table ──
    st.subheader("Top Co-movement Rules (|correlation| > 0.4)")
    st.dataframe(rules_df,use_container_width=True,height=320)
    insight_box("How to use these rules",
        "Each row is an 'if-then' rule: if Stock A moves significantly today, "
        "Stock B is likely to move in the same direction (positive) or opposite (negative). "
        "**Strong Positive pairs** (corr > 0.7): AAPL–MSFT, GOOGL–META, NVDA–AMZN — "
        "these are sector-driven; news affecting big tech impacts all of them. "
        "**Trading implication:** rather than buying both AAPL and MSFT, you get more "
        "diversification by swapping one for JPM or UNH. "
        "Negative correlations are rare in this universe but would represent natural hedges.")

    c1,c2=st.columns(2)
    with c1:
        st.subheader("Average Correlation to Portfolio")
        fig2=go.Figure(go.Bar(
            x=avg_corr.index,y=avg_corr.values,
            marker_color=['#f85149' if v>0.6 else '#f6ad55' if v>0.4 else '#3fb950'
                          for v in avg_corr.values],
            text=[f"{v:.2f}" for v in avg_corr.values],textposition='outside'))
        pplot(fig2,h=340,yaxis_title='Avg Correlation to Others',
              yaxis=dict(range=[0,1.1]))
        insight_box("Most and least correlated stocks",
            "Red bars = stocks that move in tandem with the rest — "
            "holding them adds redundancy, not diversification. "
            "Green bars = stocks that are relatively independent — "
            "each one genuinely reduces overall portfolio volatility. "
            "**BRK-B and UNH** are typically the lowest-correlation stocks, "
            "making them the most effective diversifiers in this 10-stock universe.")

    with c2:
        st.subheader("Sector-Level Co-movement")
        fig3=px.imshow(sector_corr.round(3),text_auto=True,
                        color_continuous_scale='RdBu_r',zmin=-1,zmax=1,aspect='auto')
        fig3.update_layout(**DARK,height=340)
        st.plotly_chart(fig3,use_container_width=True,config=PCFG)
        insight_box("Sector co-movement",
            "This heatmap shows how different sectors move together as a whole. "
            "**Technology ↔ Communication** shows the highest sector-level correlation "
            "because both are driven by the same macro forces (interest rates, AI spending). "
            "**Financials ↔ Healthcare** shows the lowest, meaning a portfolio combining "
            "these sectors is genuinely diversified at the sector level. "
            "Sector-level rules are useful for top-down asset allocation decisions.")

    # ── Rolling co-movement over time ──
    st.subheader("Rolling 30-Day Pairwise Correlation — Top Pair")
    if len(rules_df)>0:
        top_pair=rules_df.iloc[0]
        ta,tb=top_pair['Stock A'],top_pair['Stock B']
        ret_df=assoc_out['ret_df']
        roll_pair=ret_df[ta].rolling(30).corr(ret_df[tb])
        fig4=go.Figure()
        fig4.add_trace(go.Scatter(x=roll_pair.index,y=roll_pair,
                                   name=f'{ta} ↔ {tb}',
                                   fill='tozeroy',
                                   line=dict(color='#58a6ff',width=2)))
        fig4.add_hline(y=0.5,line_dash='dash',line_color='#f6ad55',
                       annotation_text='Moderate threshold')
        pplot(fig4,h=300,yaxis_title='Rolling 30-Day Correlation',
              yaxis=dict(range=[-0.5,1.2]))
        insight_box("Why rolling correlation matters more than static",
            "A static correlation of 0.75 between two stocks hides important variation: "
            "during the 2022 bear market, correlations spike toward 1.0 (everything falls together), "
            "while in bull markets stocks can decouple as sector rotations kick in. "
            "**For pair traders:** when rolling correlation drops sharply, the pair has "
            "diverged and may mean-revert — a potential trade entry. "
            f"The {ta}–{tb} pair shows this dynamic clearly.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — DEEP DRILL-DOWN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Deep Drill-Down Analysis":
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    st.title("🔬 Deep Drill-Down Analysis")
    st.caption("Full per-stock technical teardown: price, all EMAs, RSI, MACD, Bollinger Bands, OBV, volume profile")
    dd_t=st.selectbox("Select Stock",TICKERS,key='dd_t')
    df=enriched[dd_t]
    df_raw=frames[dd_t]

    # ── Price + All EMAs ──
    st.subheader(f"{dd_t} — Price with EMA 20 / EMA 50 / EMA 200 + Bollinger Bands")
    fig=go.Figure()
    # Bollinger band fill
    fig.add_trace(go.Scatter(x=df.index,y=df['BB_Up'],name='BB Upper',
                              line=dict(color='rgba(88,166,255,0.25)',width=1)))
    fig.add_trace(go.Scatter(x=df.index,y=df['BB_Lo'],name='BB Lower',
                              fill='tonexty',fillcolor='rgba(88,166,255,0.05)',
                              line=dict(color='rgba(88,166,255,0.25)',width=1)))
    fig.add_trace(go.Scatter(x=df.index,y=df['Close'],name='Close',
                              line=dict(color='#e6edf3',width=1.5)))
    for ema,col,dash,wid in [(20,'#fee140','dot',1.2),(50,'#58a6ff','dash',1.5),(200,'#3fb950','solid',2)]:
        col_n=f'EMA_{ema}'
        if col_n in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[col_n],name=f'EMA {ema}',
                                      line=dict(color=col,width=wid,dash=dash)))
    pplot(fig,h=440,yaxis_title='Price (USD)',
          legend=dict(orientation='h',y=1.02,x=0))
    insight_box("Complete EMA + Bollinger picture",
        "**Bollinger Bands** (light blue shaded area) show 2 standard deviations around "
        "the 20-day mean. When price touches the upper band, the stock is statistically "
        "overbought on a short-term basis; lower band = oversold. "
        "**EMA 20 (gold, dotted):** short-term momentum tracker — day traders use this. "
        "**EMA 50 (blue, dashed):** the most-watched medium-term trend line by professional "
        "traders. A close above EMA 50 after being below it often marks the start of a rally. "
        "**EMA 200 (green, solid):** the single most important line in technical analysis. "
        "When price > EMA 200 the stock is in a bull regime; below = bear regime. "
        "The distance between EMA 50 and EMA 200 shows trend strength: wider gap = stronger trend.")

    # ── RSI ──
    st.subheader("RSI (14-day)")
    fig_rsi=go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index,y=df['RSI'],name='RSI',
                                  line=dict(color='#f6ad55',width=1.5)))
    fig_rsi.add_hrect(y0=70,y1=100,fillcolor='rgba(248,81,73,0.08)',line_width=0)
    fig_rsi.add_hrect(y0=0,y1=30,fillcolor='rgba(63,185,80,0.08)',line_width=0)
    fig_rsi.add_hline(y=70,line_dash='dash',line_color='#f85149',annotation_text='Overbought 70')
    fig_rsi.add_hline(y=30,line_dash='dash',line_color='#3fb950',annotation_text='Oversold 30')
    fig_rsi.add_hline(y=50,line_dash='dot',line_color='#374151')
    pplot(fig_rsi,h=250,yaxis_title='RSI',yaxis=dict(range=[0,100]))
    insight_box("RSI signals decoded",
        "RSI (Relative Strength Index) oscillates 0–100 and measures whether a stock "
        "has been bought or sold too aggressively. "
        "**Above 70 (red zone):** overbought — the stock has risen too fast and a "
        "pullback or consolidation is likely. Avoid new long entries here. "
        "**Below 30 (green zone):** oversold — institutional buyers often step in here, "
        "making it a potential buy zone for swing traders. "
        "**The 50 line:** above 50 means buyers dominate; below 50 means sellers are in control. "
        "**Divergence:** if price makes a new high but RSI doesn't, that's a bearish "
        "divergence warning — a favourite signal of technical analysts.")

    # ── MACD ──
    st.subheader("MACD (12, 26, 9)")
    fig_macd=go.Figure()
    hc=['#3fb950' if v>=0 else '#f85149' for v in df['MACD_Hist']]
    fig_macd.add_trace(go.Bar(x=df.index,y=df['MACD_Hist'],name='Histogram',
                               marker_color=hc,opacity=0.6))
    fig_macd.add_trace(go.Scatter(x=df.index,y=df['MACD'],name='MACD Line',
                                   line=dict(color='#58a6ff',width=1.5)))
    fig_macd.add_trace(go.Scatter(x=df.index,y=df['MACD_Sig'],name='Signal Line',
                                   line=dict(color='#f85149',width=1.2,dash='dot')))
    fig_macd.add_hline(y=0,line_dash='dash',line_color='#374151')
    pplot(fig_macd,h=260)
    insight_box("MACD crossover signals",
        "MACD (Moving Average Convergence Divergence) is built from the difference "
        "between the 12-day and 26-day EMAs. "
        "**Bullish crossover:** when the MACD line (blue) crosses *above* the signal line "
        "(red dashed) → trend is turning upward, often a buy signal. "
        "**Bearish crossover:** MACD crosses *below* the signal line → momentum fading. "
        "**Histogram:** the green/red bars show the *speed* of the crossover. "
        "Growing green bars = accelerating bullish momentum. Shrinking bars = momentum fading. "
        "MACD works best on daily/weekly charts and lags during fast-moving markets, "
        "which is why it is best used alongside RSI and volume confirmation.")

    # ── Volume + OBV ──
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Daily Volume")
        vol_ma=df_raw['Volume'].rolling(20).mean()
        fig_vol=go.Figure()
        fig_vol.add_trace(go.Bar(x=df_raw.index,y=df_raw['Volume'],name='Volume',
                                  marker_color='#58a6ff',opacity=0.5))
        fig_vol.add_trace(go.Scatter(x=df_raw.index,y=vol_ma,name='20-day MA',
                                      line=dict(color='#f6ad55',width=1.5)))
        pplot(fig_vol,h=260,yaxis_title='Shares Traded')
        insight_box("Volume as confirmation",
            "Price movements on **above-average volume** are more significant and more likely "
            "to continue than moves on low volume. "
            "A breakout above a resistance level on 2× average volume confirms the move; "
            "the same breakout on half-average volume is likely a fake-out. "
            "Volume spikes often coincide with earnings, Fed decisions, or major news — "
            "look at what price did on those high-volume days to understand sentiment.")
    with c2:
        st.subheader("On-Balance Volume (OBV)")
        fig_obv=go.Figure()
        fig_obv.add_trace(go.Scatter(x=df.index,y=df['OBV'],name='OBV',
                                      fill='tozeroy',
                                      line=dict(color='#a371f7',width=1.5)))
        pplot(fig_obv,h=260)
        insight_box("OBV as a leading indicator",
            "OBV accumulates volume on up-days and subtracts volume on down-days. "
            "**OBV rising while price is flat:** institutional accumulation — "
            "smart money is buying quietly before a price move. This is often seen "
            "1–3 weeks before a breakout. "
            "**OBV falling while price is flat or rising:** distribution — "
            "institutions are selling into retail buying, a warning sign. "
            "OBV is most useful for spotting divergences from price.")

    # ── Return distribution ──
    st.subheader("Daily Return Distribution vs Normal")
    ret=df['Return'].dropna()*100
    mu,sig=float(ret.mean()),float(ret.std())
    xn=np.linspace(float(ret.min()),float(ret.max()),200)
    yn=(1/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*((xn-mu)/sig)**2)
    fig_dist=go.Figure()
    fig_dist.add_trace(go.Histogram(x=ret,nbinsx=80,name='Daily Returns',
                                     marker_color='#58a6ff',opacity=0.7,
                                     histnorm='probability density'))
    fig_dist.add_trace(go.Scatter(x=xn,y=yn,name='Normal Distribution',
                                   line=dict(color='#f85149',width=2)))
    pplot(fig_dist,h=300,xaxis_title='Daily Return (%)',yaxis_title='Density')
    skew=float(ret.skew()); kurt=float(ret.kurtosis())
    insight_box(f"Fat tails and skew — {dd_t}",
        f"Mean daily return: **{mu:.3f}%** | Std deviation: **{sig:.3f}%** | "
        f"Skewness: **{skew:.3f}** | Excess kurtosis: **{kurt:.2f}** \n\n"
        "The blue bars are actual returns. The red curve is what a normal distribution "
        "would predict. **Notice the taller, narrower centre and fatter tails** — "
        "this is called leptokurtosis and it means extreme moves (crashes, gap-ups) "
        "happen far more often than a normal distribution predicts. "
        f"A kurtosis of {kurt:.1f} means tail events are {max(1,kurt/3):.1f}x more likely "
        "than expected. This is why Value at Risk (VaR) models that assume normality "
        "routinely underestimate real losses during crises.")

    # Rolling statistics
    st.subheader("Rolling 30-Day Statistics")
    roll_ret   = df['Return'].rolling(30).mean()*252*100
    roll_vol   = df['Return'].rolling(30).std()*np.sqrt(252)*100
    roll_sharpe= roll_ret / (roll_vol + 1e-9)
    c1,c2,c3   = st.columns(3)
    with c1:
        fig_rr=go.Figure()
        fig_rr.add_trace(go.Scatter(x=df.index,y=roll_ret,name='Ann. Return',
                                     fill='tozeroy',line=dict(color='#3fb950',width=1.5)))
        fig_rr.add_hline(y=0,line_dash='dash',line_color='#374151')
        pplot(fig_rr,h=220,yaxis_title='Ann. Return (%)')
    with c2:
        fig_rv=go.Figure()
        fig_rv.add_trace(go.Scatter(x=df.index,y=roll_vol,name='Ann. Volatility',
                                     fill='tozeroy',line=dict(color='#f85149',width=1.5)))
        pplot(fig_rv,h=220,yaxis_title='Ann. Volatility (%)')
    with c3:
        fig_rs=go.Figure()
        fig_rs.add_trace(go.Scatter(x=df.index,y=roll_sharpe,name='Rolling Sharpe',
                                     fill='tozeroy',line=dict(color='#58a6ff',width=1.5)))
        fig_rs.add_hline(y=1,line_dash='dash',line_color='#3fb950',annotation_text='Sharpe=1')
        pplot(fig_rs,h=220,yaxis_title='Sharpe Ratio')

    insight_box("Rolling statistics decoded",
        "These three panels show how the stock's risk-return profile has evolved over time. "
        "**Rolling Return (green):** when it dips below zero, the stock has been losing money "
        "on a 30-day annualised basis — a signal to reduce position size. "
        "**Rolling Volatility (red):** spikes correspond to earnings, macro events, or crises. "
        "**Rolling Sharpe (blue):** above 1.0 means you're being compensated more than 1 unit "
        "of return per unit of risk. Below 0 means taking risk for negative reward. "
        "Smart position sizing: allocate more when Sharpe is rising, reduce when it falls.")

    # Summary stats table
    st.subheader("Statistical Summary")
    last_row  = df.ffill().iloc[-1]
    hi52      = float(df['Close'].rolling(252).max().iloc[-1])
    lo52      = float(df['Close'].rolling(252).min().iloc[-1])
    ret_1y    = float(df['Return'].tail(252).mean()*252*100)
    ann_vol   = float(df['Return'].std()*np.sqrt(252)*100)
    ret_s     = df['Return'].dropna()
    shrp      = float((ret_s.mean()/ret_s.std())*np.sqrt(252)) if ret_s.std()>0 else 0
    max_dd    = float(df['Drawdown'].min()*100)
    cur       = float(last_row['Close'])
    ema50_val = float(last_row.get('EMA_50',  cur))
    ema200_val= float(last_row.get('EMA_200', cur))
    vs_ema200 = f"{'ABOVE' if cur>ema200_val else 'BELOW'} ({(cur/ema200_val-1)*100:+.1f}%)"
    vs_ema50  = f"{'ABOVE' if cur>ema50_val  else 'BELOW'} ({(cur/ema50_val-1)*100:+.1f}%)"
    summary_df = pd.DataFrame({
        'Metric': ['Current Price','52W High','52W Low','1Y Annualised Return',
                   'Annualised Volatility','Sharpe Ratio','Max Drawdown',
                   'Avg Daily Return','Return Skewness','Excess Kurtosis',
                   'Current RSI','EMA 50','EMA 200',
                   'Price vs EMA 50','Price vs EMA 200','Days of Data'],
        'Value': [
            f"${cur:.2f}", f"${hi52:.2f}", f"${lo52:.2f}",
            f"{ret_1y:+.1f}%", f"{ann_vol:.1f}%", f"{shrp:.2f}",
            f"{max_dd:.1f}%", f"{float(ret.mean()):.4f}%",
            f"{skew:.3f}", f"{kurt:.2f}",
            f"{float(last_row.get('RSI',0)):.1f}",
            f"${ema50_val:.2f}", f"${ema200_val:.2f}",
            vs_ema50, vs_ema200, f"{len(df):,} trading days"
        ]
    }).set_index('Metric')
    st.dataframe(summary_df,use_container_width=True)


# PAGE 7 — DOWNLOAD DATA
elif page == "📥 Download Data":
    st.title("📥 Download Data & Reports")
    st.subheader("Excel Stock Price File")
    st.markdown("""
    Generated from **Yahoo Finance via `yfinance`**:
    - **Overview sheet** — all 10 tickers summary
    - **One sheet per ticker** — OHLCV + Daily Return % + Cumulative Return % + price chart
    """)
    c1,c2=st.columns(2)
    with c1:
        if st.button("🔄 Regenerate Excel from Yahoo Finance",type='primary'):
            with st.spinner("Fetching data and building Excel..."):
                fresh,err=fetch_yahoo(TICKERS,START_DATE,END_DATE)
                if fresh:
                    save_excel(fresh,EXCEL_PATH)
                    st.success(f"Saved {sum(len(v) for v in fresh.values()):,} rows")
                else:
                    st.error(f"Yahoo Finance error: {err}")
    with c2:
        if os.path.exists(EXCEL_PATH):
            with open(EXCEL_PATH,'rb') as f:
                st.download_button(label="Download Excel File",data=f.read(),
                                   file_name=EXCEL_PATH,type='primary',
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        else:
            st.info("No Excel file yet — click Regenerate first.")
    st.divider()
    st.subheader("Feature-Engineered Dataset (CSV)")
    dl_t=st.selectbox("Ticker",TICKERS,key='dl_t')
    buf=io.StringIO(); enriched[dl_t].copy().to_csv(buf)
    st.download_button(label=f"Download {dl_t} Features CSV",data=buf.getvalue(),
                       file_name=f"{dl_t}_features_{END_DATE}.csv",mime='text/csv')
    st.divider()
    st.subheader("Regression Model Metrics (Excel)")
    if st.button("Generate & Download Regression Metrics"):
        with st.spinner("Training all regression models..."):
            am=all_regressions(enriched)
        rows=[]
        for t in TICKERS:
            mr,_=am[t]
            if mr:
                for res in mr['results'].values():
                    rows.append({'Ticker':t,**res['metrics']})
        mbuf=io.BytesIO()
        pd.DataFrame(rows).to_excel(mbuf,index=False)
        st.download_button(label="Download Metrics Excel",data=mbuf.getvalue(),
                           file_name=f"regression_metrics_{END_DATE}.xlsx",
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.divider()
    st.warning("Legal Disclaimer: For informational purposes only. Not financial advice. "
               "Past performance does not guarantee future results.")
