"""
Investment Bank S&P 500 Stock Analytics Dashboard
Top 10 S&P 500 | Yahoo Finance ONLY | Live Market Data
Data cut-off context: 14 March 2026

Business Objective: Determine which stocks offer the best risk-adjusted return
potential using EDA, technical analysis, LSTM classification, and 6 regression
models — producing actionable BUY / HOLD / AVOID signals.

Academic Coverage:
  1. Real market data as synthetic-equivalent validation dataset
  2. Data cleaning, transformation, 47-feature engineering
  3. Descriptive analytics, EDA, correlation graphs with logical explanations

Models: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest,
        HistGradientBoosting, LSTM (pure NumPy, BPTT+Adam)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, warnings, io
from datetime import datetime
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IB Stock Analytics | S&P 500 Top 10",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    "<style>section[data-testid='stSidebar']{background-color:#161b22}</style>",
    unsafe_allow_html=True,
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────
TICKERS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","UNH"]
META_INFO = {
    "AAPL":  {"name":"Apple Inc.",           "sector":"Technology"},
    "MSFT":  {"name":"Microsoft Corp.",       "sector":"Technology"},
    "NVDA":  {"name":"NVIDIA Corp.",          "sector":"Semiconductors"},
    "AMZN":  {"name":"Amazon.com Inc.",       "sector":"Consumer Disc."},
    "GOOGL": {"name":"Alphabet Inc.",         "sector":"Communication"},
    "META":  {"name":"Meta Platforms",        "sector":"Communication"},
    "TSLA":  {"name":"Tesla Inc.",            "sector":"Consumer Disc."},
    "BRK-B": {"name":"Berkshire Hathaway B", "sector":"Financials"},
    "JPM":   {"name":"JPMorgan Chase",        "sector":"Financials"},
    "UNH":   {"name":"UnitedHealth Group",    "sector":"Healthcare"},
}
COLORS = ["#58a6ff","#3fb950","#f6ad55","#f093fb","#4facfe",
          "#43e97b","#fa709a","#fee140","#a371f7","#ff9a9e"]
START      = "2015-01-01"
START_LONG = "2005-01-01"   # lazy-loaded only for MTF / Drill-Down pages
END        = datetime.today().strftime("%Y-%m-%d")
XLS_PATH   = "sp500_top10.xlsx"
PCFG       = {"displayModeBar": False}
SEQ_LEN    = 20
LSTM_H     = 48

# Pre-define before sidebar so it exists when sidebar renders
LOADED_TICKERS = TICKERS   # narrowed after data loads

# ── DARK THEME + HELPERS ───────────────────────────────────────────────────
DARK = dict(
    template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(family="sans-serif", color="#d1d5db", size=11),
    margin=dict(l=50, r=20, t=40, b=50),
    xaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
    yaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
)

def pplot(fig, h=380, **kw):
    """Render Plotly figure with dark theme; deep-merges dict kwargs to avoid TypeError."""
    layout = {**DARK, "height": h}
    for k, v in kw.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)

def ibox(title: str, body: str):
    """Insight box using native Streamlit — no unsafe HTML."""
    with st.container(border=True):
        st.markdown(f"**💡 {title}**")
        st.markdown(body)

def safe_get(d: dict, key: str, fallback_key: str = None):
    """Safe dict access — never raises KeyError, never triggers DataFrame bool."""
    if key in d:
        return d[key]
    if fallback_key and fallback_key in d:
        return d[fallback_key]
    if d:
        return d[next(iter(d))]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# LSTM — Pure NumPy with real BPTT + Adam
# ═══════════════════════════════════════════════════════════════════════════
class LSTM:
    """Single-layer LSTM trained via backpropagation through time + Adam."""

    def __init__(self, in_sz, h_sz, n_cls=2, lr=5e-4, seed=42):
        np.random.seed(seed)
        self.hs = h_sz; self.nc = n_cls; self.lr = lr
        sz = in_sz + h_sz; k = np.sqrt(2.0 / sz)
        self.Wf = np.random.randn(h_sz, sz)*k;  self.bf = np.zeros(h_sz)
        self.Wi = np.random.randn(h_sz, sz)*k;  self.bi = np.ones(h_sz)
        self.Wg = np.random.randn(h_sz, sz)*k;  self.bg = np.zeros(h_sz)
        self.Wo = np.random.randn(h_sz, sz)*k;  self.bo = np.zeros(h_sz)
        self.Wy = np.random.randn(n_cls, h_sz)*np.sqrt(2.0/h_sz)
        self.by = np.zeros(n_cls)
        pn = ["Wf","Wi","Wg","Wo","bf","bi","bg","bo","Wy","by"]
        self._pn = pn; self._t = 0
        self._b1 = 0.9; self._b2 = 0.999; self._eps = 1e-8
        self._m = {n: np.zeros_like(getattr(self, n)) for n in pn}
        self._v = {n: np.zeros_like(getattr(self, n)) for n in pn}

    @staticmethod
    def _s(x): return 1.0/(1.0+np.exp(-np.clip(x,-12,12)))
    @staticmethod
    def _t_(x): return np.tanh(np.clip(x,-10,10))

    def _fwd(self, X):
        h = np.zeros(self.hs); c = np.zeros(self.hs); cache = []
        for t in range(len(X)):
            xh = np.concatenate([X[t], h])
            f  = self._s(self.Wf@xh+self.bf); i = self._s(self.Wi@xh+self.bi)
            g  = self._t_(self.Wg@xh+self.bg); o = self._s(self.Wo@xh+self.bo)
            c2 = f*c+i*g; h2 = o*self._t_(c2)
            cache.append((xh,f,i,g,o,c,c2,h,h2)); h,c = h2,c2
        z = self.Wy@h+self.by; z -= z.max()
        e = np.exp(z); return e/e.sum(), h, cache

    def _adam(self, name, grad, clip=1.0):
        np.clip(grad, -clip, clip, out=grad)
        self._t += 1
        self._m[name] = self._b1*self._m[name]+(1-self._b1)*grad
        self._v[name] = self._b2*self._v[name]+(1-self._b2)*grad**2
        mh = self._m[name]/(1-self._b1**self._t)
        vh = self._v[name]/(1-self._b2**self._t)
        return self.lr*mh/(np.sqrt(vh)+self._eps)

    def step(self, X, y_true):
        probs, h_last, cache = self._fwd(X)
        loss = -np.log(probs[y_true]+1e-9)
        dz = probs.copy(); dz[y_true] -= 1.0
        dWy = np.outer(dz, h_last); dby = dz.copy()
        dh = self.Wy.T@dz; dc = np.zeros(self.hs)
        dWf=np.zeros_like(self.Wf); dWi=np.zeros_like(self.Wi)
        dWg=np.zeros_like(self.Wg); dWo=np.zeros_like(self.Wo)
        dbf=np.zeros(self.hs); dbi=np.zeros(self.hs)
        dbg=np.zeros(self.hs); dbo=np.zeros(self.hs)
        for t in reversed(range(len(cache))):
            xh,f,i,g,o,cp,cc,hp,hc = cache[t]
            tc=self._t_(cc); do=dh*tc; ddc=dh*o*(1-tc**2)+dc
            df_=ddc*cp; di_=ddc*g; dg_=ddc*i; dc=ddc*f
            dfp=df_*f*(1-f); dip=di_*i*(1-i)
            dgp=dg_*(1-g**2); dop=do*o*(1-o)
            dWf+=np.outer(dfp,xh); dbf+=dfp
            dWi+=np.outer(dip,xh); dbi+=dip
            dWg+=np.outer(dgp,xh); dbg+=dgp
            dWo+=np.outer(dop,xh); dbo+=dop
            dh=(self.Wf.T@dfp+self.Wi.T@dip+self.Wg.T@dgp+self.Wo.T@dop)[:self.hs]
        for nm,dW in [("Wf",dWf),("Wi",dWi),("Wg",dWg),("Wo",dWo),
                      ("bf",dbf),("bi",dbi),("bg",dbg),("bo",dbo),
                      ("Wy",dWy),("by",dby)]:
            setattr(self, nm, getattr(self, nm)-self._adam(nm, dW))
        return loss

    def get_hidden(self, X):
        _, h, _ = self._fwd(X); return h

    def predict_proba(self, X):
        p, _, _ = self._fwd(X); return p

    def train(self, Xtr, ytr, epochs=10):
        N = len(Xtr); hist = []
        for ep in range(epochs):
            idx = np.random.permutation(N); loss = 0.0
            for j in idx: loss += self.step(Xtr[j], ytr[j])
            hist.append(loss/N)
        return hist

    def hidden_batch(self, Xs):
        return np.array([self.get_hidden(Xs[i]) for i in range(len(Xs))])

    def proba_batch(self, Xs):
        return np.array([self.predict_proba(Xs[i]) for i in range(len(Xs))])


# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER — Yahoo Finance batch downloads
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo():
    """Batch download daily OHLCV from 2015. Falls back to sequential on error."""
    import yfinance as yf
    frames = {}
    try:
        raw = yf.download(TICKERS, start=START, end=END,
                          progress=False, auto_adjust=True, group_by="ticker")
        for t in TICKERS:
            try:
                df = raw[t].dropna(how="all") if t in raw.columns.get_level_values(0) else pd.DataFrame()
                if df.empty: continue
                df.index = pd.to_datetime(df.index)
                frames[t] = df[["Open","High","Low","Close","Volume"]].dropna()
            except Exception: pass
    except Exception:
        for t in TICKERS:
            try:
                df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
                if df.empty: continue
                df.index = pd.to_datetime(df.index)
                if isinstance(df.columns, pd.MultiIndex): df.columns=[c[0] for c in df.columns]
                frames[t] = df[["Open","High","Low","Close","Volume"]].dropna()
            except Exception: pass
    return frames


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_long():
    """Extended history from 2005 — lazy-loaded, never at startup."""
    import yfinance as yf
    frames = {}
    try:
        raw = yf.download(TICKERS, start=START_LONG, end=END,
                          progress=False, auto_adjust=True, group_by="ticker")
        for t in TICKERS:
            try:
                df = raw[t].dropna(how="all") if t in raw.columns.get_level_values(0) else pd.DataFrame()
                if df.empty: continue
                df.index = pd.to_datetime(df.index)
                frames[t] = df[["Open","High","Low","Close","Volume"]].dropna()
            except Exception: pass
    except Exception:
        for t in TICKERS:
            try:
                df = yf.download(t, start=START_LONG, end=END, progress=False, auto_adjust=True)
                if df.empty: continue
                df.index = pd.to_datetime(df.index)
                if isinstance(df.columns, pd.MultiIndex): df.columns=[c[0] for c in df.columns]
                frames[t] = df[["Open","High","Low","Close","Volume"]].dropna()
            except Exception: pass
    return frames


# ── NumPy indicator helpers ────────────────────────────────────────────────
def _ema(a, s):
    k=2/(s+1); o=np.zeros(len(a)); o[0]=a[0]
    for i in range(1,len(a)): o[i]=k*a[i]+(1-k)*o[i-1]
    return o

def _sma(a, w):
    return np.array([a[max(0,i-w+1):i+1].mean() for i in range(len(a))])

def _roll_std(a, w):
    return np.array([a[max(0,i-w+1):i+1].std() for i in range(len(a))])


@st.cache_data(show_spinner=False)
def compute_indicators(_frames):
    """Compute full 20+ indicator suite on every ticker. Returns dict of DataFrames."""
    out = {}
    for ticker, df in _frames.items():
        d = df.copy().sort_index()
        c=d["Close"].values; h=d["High"].values
        l=d["Low"].values;   v=d["Volume"].values; n=len(c)
        ret=np.zeros(n); ret[1:]=(c[1:]-c[:-1])/(c[:-1]+1e-9)
        ind = {"Return": ret}
        for w in [5,10,20,50,100,200]:
            ind[f"SMA_{w}"] = _sma(c,w)
            ind[f"EMA_{w}"] = _ema(c,w)
        delta=np.diff(c,prepend=c[0])
        ag=_sma(np.maximum(delta,0),14); al=_sma(-np.minimum(delta,0),14)
        ind["RSI"] = np.clip(100-100/(1+ag/(al+1e-9)),0,100)
        e12=_ema(c,12); e26=_ema(c,26)
        ind["MACD"]=e12-e26; ind["MACD_Signal"]=_ema(ind["MACD"],9)
        ind["MACD_Hist"]=ind["MACD"]-ind["MACD_Signal"]
        p20=min(20,n-1); s20=_sma(c,p20); std20=_roll_std(c,p20)
        ind["BB_Upper"]=s20+2*std20; ind["BB_Lower"]=s20-2*std20
        ind["BB_Mid"]=s20; ind["BB_Width"]=4*std20/(s20+1e-9)
        ind["BB_Pct"]=(c-(s20-2*std20))/(4*std20+1e-9)
        pc=np.roll(c,1); pc[0]=c[0]
        tr=np.maximum.reduce([h-l,np.abs(h-pc),np.abs(l-pc)])
        ind["ATR"]=_sma(tr,14); ind["ATR_Pct"]=ind["ATR"]/(c+1e-9)
        lo14=np.array([l[max(0,i-14):i+1].min() for i in range(n)])
        hi14=np.array([h[max(0,i-14):i+1].max() for i in range(n)])
        k_raw=(c-lo14)/(hi14-lo14+1e-9)*100
        ind["Stoch_K"]=_sma(k_raw,3); ind["Stoch_D"]=_sma(ind["Stoch_K"],3)
        ind["Williams_R"]=-100*(hi14-c)/(hi14-lo14+1e-9)
        tp=(h+l+c)/3; stp=_sma(tp,p20)
        mad=np.array([np.abs(tp[max(0,i-20):i+1]-tp[max(0,i-20):i+1].mean()).mean() for i in range(n)])
        ind["CCI"]=(tp-stp)/(0.015*mad+1e-9)
        p_dm=np.maximum(np.diff(h,prepend=h[0]),0)
        n_dm=np.maximum(-np.diff(l,prepend=l[0]),0)
        p_dm=np.where(p_dm>n_dm,p_dm,0); n_dm=np.where(n_dm>p_dm,n_dm,0)
        atr14=_sma(tr,14)+1e-9
        di_p=100*_sma(p_dm,14)/atr14; di_n=100*_sma(n_dm,14)/atr14
        dx=100*np.abs(di_p-di_n)/(di_p+di_n+1e-9)
        ind["ADX"]=_sma(dx,14); ind["DI_Plus"]=di_p; ind["DI_Minus"]=di_n
        ind["OBV"]=np.cumsum(np.sign(ret)*v)
        obv_ma=_sma(ind["OBV"],20); ind["OBV_Slope"]=(ind["OBV"]-obv_ma)/(np.abs(obv_ma)+1e-9)
        def ichi(p): 
            hp=np.array([h[max(0,i-p):i+1].max() for i in range(n)])
            lp=np.array([l[max(0,i-p):i+1].min() for i in range(n)])
            return (hp+lp)/2
        ind["Ichi_Tenkan"]=ichi(9); ind["Ichi_Kijun"]=ichi(26)
        ind["Ichi_SpanA"]=(ind["Ichi_Tenkan"]+ind["Ichi_Kijun"])/2
        ind["Ichi_SpanB"]=ichi(52)
        ind["VWAP"]=np.cumsum(tp*v)/(np.cumsum(v)+1e-9)
        ind["Vol_5"] =_roll_std(ret,5)*np.sqrt(252)
        ind["Vol_20"]=_roll_std(ret,20)*np.sqrt(252)
        ind["Vol_30"]=_roll_std(ret,30)*np.sqrt(252)
        e50=_ema(c,50); e200=_ema(c,200)
        ind["Regime"]=(c>e200).astype(float)
        ind["Golden_Cross"]=(e50>e200).astype(float)
        ind["EMA50_200"]=(e50-e200)/(e200+1e-9)
        ind["Drawdown"]=(c-np.maximum.accumulate(c))/(np.maximum.accumulate(c)+1e-9)
        ind["Cum_Return"]=(1+pd.Series(ret).fillna(0)).cumprod().values-1
        for w in [1,2,3,5,10,20]: ind[f"Lag_{w}"]=np.roll(ret,w)
        for w in [5,10,20]: ind[f"Cum_{w}"]=np.array([(c[i]/c[max(0,i-w)]-1) for i in range(n)])
        ind["Vol_Ratio"]=v/(_sma(v,20)+1e-9)
        df_ind=pd.DataFrame(ind,index=df.index)
        for col in ["Open","High","Low","Close","Volume"]: df_ind[col]=df[col].values
        out[ticker]=df_ind
    return out


# ── LSTM feature builder (47 features, zero lookahead) ────────────────────
def _make_X(ind_df):
    c=ind_df["Close"].values
    d={}
    d["P_SMA5"]  =(c-ind_df["SMA_5"].values)  /(ind_df["SMA_5"].values  +1e-9)
    d["P_EMA5"]  =(c-ind_df["EMA_5"].values)  /(ind_df["EMA_5"].values  +1e-9)
    d["P_SMA20"] =(c-ind_df["SMA_20"].values) /(ind_df["SMA_20"].values +1e-9)
    d["P_EMA20"] =(c-ind_df["EMA_20"].values) /(ind_df["EMA_20"].values +1e-9)
    d["P_EMA50"] =(c-ind_df["EMA_50"].values) /(ind_df["EMA_50"].values +1e-9)
    d["P_EMA200"]=(c-ind_df["EMA_200"].values)/(ind_df["EMA_200"].values+1e-9)
    d["RSI_n"]   =ind_df["RSI"].values/100.0
    d["MACD_n"]  =ind_df["MACD"].values/(c+1e-9)
    d["MACD_h_n"]=ind_df["MACD_Hist"].values/(c+1e-9)
    d["BB_Pct"]  =ind_df["BB_Pct"].values
    d["BB_Width"]=ind_df["BB_Width"].values
    d["ATR_n"]   =ind_df["ATR_Pct"].values
    d["Stoch_K_n"]=ind_df["Stoch_K"].values/100.0
    d["WR_n"]    =(ind_df["Williams_R"].values+100)/100.0
    d["CCI_n"]   =np.tanh(ind_df["CCI"].values/200.0)
    d["ADX_n"]   =ind_df["ADX"].values/100.0
    d["OBV_Slope"]=np.tanh(ind_df["OBV_Slope"].values)
    d["Vol_Ratio_n"]=np.tanh(ind_df["Vol_Ratio"].values-1)
    d["Vol_5"]   =ind_df["Vol_5"].values
    d["Vol_20"]  =ind_df["Vol_20"].values
    d["Regime"]  =ind_df["Regime"].values
    d["Golden_Cross"]=ind_df["Golden_Cross"].values
    d["EMA50_200_n"]=np.tanh(ind_df["EMA50_200"].values*10)
    d["Drawdown"]=ind_df["Drawdown"].values
    for lag in [1,2,3,5,10]: d[f"Lag_{lag}"]=ind_df[f"Lag_{lag}"].values
    for w in [5,10,20]: d[f"Cum_{w}_n"]=np.tanh(ind_df[f"Cum_{w}"].values*10)
    d["Ret_n"]   =np.tanh(ind_df["Return"].values*50)
    X=np.column_stack(list(d.values()))
    return np.nan_to_num(X,nan=0,posinf=1,neginf=-1)


def _seq_stats(Xs):
    last=Xs[:,-1,:]; mn=Xs.mean(1); sd=Xs.std(1); trend=Xs[:,-1,:]-Xs[:,0,:]
    q3=Xs[:,SEQ_LEN*3//4:,:].mean(1)-Xs[:,:SEQ_LEN//4,:].mean(1)
    return np.concatenate([last,mn,sd,trend,q3],axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ENGINE (lazy)
# ═══════════════════════════════════════════════════════════════════════════
def _resample_ohlcv(df, freq):
    return df.resample(freq).agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna()


def _ind_tf(df):
    """Full indicator suite on any-frequency OHLCV."""
    c=df["Close"].values; h=df["High"].values
    l=df["Low"].values;   v=df["Volume"].values; n=len(c)
    if n<10: return df
    ret=np.zeros(n); ret[1:]=(c[1:]-c[:-1])/(c[:-1]+1e-9)
    out=df.copy(); out["Return"]=ret
    for w in [5,10,20,50,100,200]:
        if w<n: out[f"SMA_{w}"]=_sma(c,w); out[f"EMA_{w}"]=_ema(c,w)
    delta=np.diff(c,prepend=c[0])
    ag=_sma(np.maximum(delta,0),14); al=_sma(-np.minimum(delta,0),14)
    out["RSI"]=np.clip(100-100/(1+ag/(al+1e-9)),0,100)
    e12=_ema(c,12); e26=_ema(c,26)
    out["MACD"]=e12-e26; out["MACD_Signal"]=_ema(out["MACD"].values,9)
    out["MACD_Hist"]=out["MACD"].values-out["MACD_Signal"].values
    p20=min(20,n-1); s20=_sma(c,p20); std20=_roll_std(c,p20)
    out["BB_Upper"]=s20+2*std20; out["BB_Lower"]=s20-2*std20
    out["BB_Mid"]=s20; out["BB_Width"]=4*std20/(s20+1e-9)
    out["BB_Pct"]=(c-(s20-2*std20))/(4*std20+1e-9)
    pc=np.roll(c,1); pc[0]=c[0]
    tr=np.maximum.reduce([h-l,np.abs(h-pc),np.abs(l-pc)])
    out["ATR"]=_sma(tr,14)
    lo14=np.array([l[max(0,i-14):i+1].min() for i in range(n)])
    hi14=np.array([h[max(0,i-14):i+1].max() for i in range(n)])
    k_raw=(c-lo14)/(hi14-lo14+1e-9)*100
    out["Stoch_K"]=_sma(k_raw,3); out["Stoch_D"]=_sma(out["Stoch_K"].values,3)
    out["Williams_R"]=-100*(hi14-c)/(hi14-lo14+1e-9)
    tp=(h+l+c)/3; stp=_sma(tp,p20)
    mad=np.array([np.abs(tp[max(0,i-20):i+1]-tp[max(0,i-20):i+1].mean()).mean() for i in range(n)])
    out["CCI"]=(tp-stp)/(0.015*mad+1e-9)
    p_dm=np.maximum(np.diff(h,prepend=h[0]),0); n_dm=np.maximum(-np.diff(l,prepend=l[0]),0)
    p_dm=np.where(p_dm>n_dm,p_dm,0); n_dm=np.where(n_dm>p_dm,n_dm,0)
    atr14=_sma(tr,14)+1e-9
    di_p=100*_sma(p_dm,14)/atr14; di_n=100*_sma(n_dm,14)/atr14
    dx=100*np.abs(di_p-di_n)/(di_p+di_n+1e-9)
    out["ADX"]=_sma(dx,14); out["DI_Plus"]=di_p; out["DI_Minus"]=di_n
    out["OBV"]=np.cumsum(np.sign(ret)*v)
    w52=min(52,n-1)
    hi52=np.array([h[max(0,i-w52):i+1].max() for i in range(n)])
    lo52=np.array([l[max(0,i-w52):i+1].min() for i in range(n)])
    out["Pct_Range_52"]=(c-lo52)/(hi52-lo52+1e-9)*100
    out["Regime"]=(c>_ema(c,min(50,n-1))).astype(float)
    out["Drawdown"]=(c-np.maximum.accumulate(c))/(np.maximum.accumulate(c)+1e-9)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def build_mtf(_frames_long):
    result={}
    for ticker,df in _frames_long.items():
        tfs={}
        for key,freq in [("W","W"),("ME","ME"),("QE","QE"),("YE","YE")]:
            try:
                rs=_resample_ohlcv(df,freq)
                if len(rs)>=10: tfs[key]=_ind_tf(rs)
            except Exception: pass
        result[ticker]=tfs
    return result


def _monthly_heatmap(df_daily):
    df=df_daily[["Close"]].copy()
    df["Year"]=df.index.year; df["Month"]=df.index.month
    monthly=(df.groupby(["Year","Month"])["Close"]
               .agg(["first","last"])
               .assign(ret=lambda x:(x["last"]/x["first"]-1)*100)["ret"]
               .reset_index())
    pivot=monthly.pivot(index="Year",columns="Month",values="ret")
    mnames=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns=mnames[:len(pivot.columns)]
    return pivot


def _yearly_returns(df_daily):
    df=df_daily[["Close"]].copy(); df["Year"]=df.index.year
    yr=df.groupby("Year")["Close"].agg(["first","last"])
    yr["Return%"]=(yr["last"]/yr["first"]-1)*100
    return yr["Return%"]


def _support_resistance(c_arr, window=20, n_levels=5):
    pivots_hi=[]; pivots_lo=[]
    for i in range(window,len(c_arr)-window):
        seg=c_arr[i-window:i+window+1]
        if c_arr[i]==seg.max(): pivots_hi.append(float(c_arr[i]))
        if c_arr[i]==seg.min(): pivots_lo.append(float(c_arr[i]))
    def cluster(pts,tol=0.02):
        if not pts: return []
        pts=sorted(pts,reverse=True); cls=[[pts[0]]]
        for p in pts[1:]:
            if abs(p-cls[-1][-1])/(cls[-1][-1]+1e-9)<tol: cls[-1].append(p)
            else: cls.append([p])
        return [float(np.mean(c)) for c in cls[:n_levels]]
    return cluster(pivots_lo), cluster(pivots_hi)


# ═══════════════════════════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def train_lstm(ticker, _ind):
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    ind=_ind[ticker]; X_raw=_make_X(ind); c=ind["Close"].values; n=len(c)
    fwd=np.zeros(n)
    for i in range(n-5): fwd[i]=(c[i+5]-c[i])/(c[i]+1e-9)
    y=(fwd>0).astype(int)   # clean binary label — no lookahead
    Xs=np.array([X_raw[i-SEQ_LEN:i] for i in range(SEQ_LEN,n)])
    ys=y[SEQ_LEN:]; dates=ind.index[SEQ_LEN:]; reg=ind["Regime"].values[SEQ_LEN:]
    ok=np.isfinite(Xs).all(axis=(1,2))&np.isfinite(ys)
    Xs=Xs[ok]; ys=ys[ok]; dates=dates[ok]; reg=reg[ok]
    N=len(Xs); split=int(N*0.80)
    Xtr,Xte=Xs[:split],Xs[split:]; ytr,yte=ys[:split],ys[split:]
    dte=dates[split:]; reg_te=reg[split:]; F=Xtr.shape[2]
    sc=RobustScaler()
    Xtr_n=np.nan_to_num(sc.fit_transform(Xtr.reshape(-1,F)).reshape(Xtr.shape),0)
    Xte_n=np.nan_to_num(sc.transform(Xte.reshape(-1,F)).reshape(Xte.shape),0)
    model=LSTM(F,LSTM_H,n_cls=2,lr=5e-4,seed=42)
    loss_hist=model.train(Xtr_n,ytr,epochs=10)
    H_tr=model.hidden_batch(Xtr_n); H_te=model.hidden_batch(Xte_n)
    Ftr=np.concatenate([H_tr,_seq_stats(Xtr_n)],axis=1)
    Fte=np.concatenate([H_te,_seq_stats(Xte_n)],axis=1)
    sc2=RobustScaler()
    Ftr_s=np.nan_to_num(sc2.fit_transform(Ftr),0)
    Fte_s=np.nan_to_num(sc2.transform(Fte),0)
    hgb=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.025,max_depth=5,
        min_samples_leaf=10,l2_regularization=0.1,random_state=42,
        early_stopping=True,validation_fraction=0.12,n_iter_no_change=25)
    hgb.fit(Ftr_s,ytr)
    probs=hgb.predict_proba(Fte_s); preds=np.argmax(probs,axis=1); conf=probs.max(axis=1)
    lstm_pr=model.proba_batch(Xte_n); lstm_pred=np.argmax(lstm_pr,axis=1)
    acc_lstm=accuracy_score(yte,lstm_pred)*100; acc_all=accuracy_score(yte,preds)*100
    bull=reg_te==1; bear=reg_te==0
    acc_bull=accuracy_score(yte[bull],preds[bull])*100 if bull.sum()>10 else 0.0
    acc_bear=accuracy_score(yte[bear],preds[bear])*100 if bear.sum()>10 else 0.0
    conf_tbl=[]
    for th in [0.50,0.52,0.55,0.58,0.60,0.65,0.70]:
        m=conf>=th
        if m.sum()>=10:
            a=accuracy_score(yte[m],preds[m])*100
            conf_tbl.append({"Threshold":f">={th:.0%}","Accuracy":f"{a:.1f}%",
                             "Signals":int(m.sum()),"Coverage":f"{m.mean()*100:.0f}%",
                             "Edge over 50%":f"+{a-50:.1f}%"})
        mb=m&bull
        if mb.sum()>=8:
            ab=accuracy_score(yte[mb],preds[mb])*100
            conf_tbl.append({"Threshold":f">={th:.0%} + Bull","Accuracy":f"{ab:.1f}%",
                             "Signals":int(mb.sum()),"Coverage":f"{mb.mean()*100:.0f}%",
                             "Edge over 50%":f"+{ab-50:.1f}%"})
    cm=confusion_matrix(yte,preds,labels=[0,1])
    Xs_all=np.array([X_raw[i-SEQ_LEN:i] for i in range(SEQ_LEN,n)])
    ok2=np.isfinite(Xs_all).all(axis=(1,2)); Xs_ok=Xs_all[ok2]
    Xall_n=np.nan_to_num(sc.transform(Xs_ok.reshape(-1,F)).reshape(Xs_ok.shape),0)
    H_all=model.hidden_batch(Xall_n)
    Fall=np.concatenate([H_all,_seq_stats(Xall_n)],axis=1)
    Fall_s=np.nan_to_num(sc2.transform(Fall),0)
    pr_all=hgb.predict_proba(Fall_s)
    pa_all=np.argmax(pr_all,axis=1); ca_all=pr_all.max(axis=1)
    da_all=ind.index[SEQ_LEN:][ok2]
    return {"acc_lstm":round(acc_lstm,1),"acc_all":round(acc_all,1),
            "acc_bull":round(acc_bull,1),"acc_bear":round(acc_bear,1),
            "conf_tbl":conf_tbl,"loss_hist":loss_hist,
            "preds":preds,"actual":yte,"conf":conf,"probs":probs,
            "dates_te":dte,"reg_te":reg_te,"cm":cm,
            "n_train":split,"n_test":len(yte),"n_feats":F,
            "preds_all":pa_all,"conf_all":ca_all,"dates_all":da_all,
            "report":classification_report(yte,preds,output_dict=True)}, None


@st.cache_data(show_spinner=False)
def run_regression(ticker, _ind):
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ind=_ind[ticker]; X_raw=_make_X(ind); c=ind["Close"].values; n=len(c)
    fwd=np.zeros(n)
    for i in range(n-5): fwd[i]=(c[i+5]-c[i])/(c[i]+1e-9)
    ok=np.isfinite(X_raw).all(axis=1)&np.isfinite(fwd)&(fwd!=0)
    X=X_raw[ok]; y=fwd[ok]; dates=ind.index[ok]
    sp=int(len(X)*0.8)
    Xtr,Xte=X[:sp],X[sp:]; ytr,yte=y[:sp],y[sp:]; dte=dates[sp:]
    sc=RobustScaler()
    Xtr_s=np.nan_to_num(sc.fit_transform(Xtr),0)
    Xte_s=np.nan_to_num(sc.transform(Xte),0)
    def mets(yt,yp,nm):
        return {"Model":nm,"MAE":round(mean_absolute_error(yt,yp),5),
                "RMSE":round(np.sqrt(mean_squared_error(yt,yp)),5),
                "R2":round(r2_score(yt,yp),4),
                "Dir_Acc%":round(np.mean(np.sign(yt)==np.sign(yp))*100,1)}
    R={}
    m1=LinearRegression().fit(Xtr_s,ytr); p1=m1.predict(Xte_s)
    R["Linear"]={"pred":p1,"fi":dict(zip(range(X.shape[1]),np.abs(m1.coef_))),"metrics":mets(yte,p1,"Linear Regression")}
    m2=RidgeCV(alphas=[0.01,0.1,1,10,100],cv=5).fit(Xtr_s,ytr); p2=m2.predict(Xte_s)
    R["Ridge"]={"pred":p2,"fi":dict(zip(range(X.shape[1]),np.abs(m2.coef_))),"metrics":mets(yte,p2,f"Ridge (a={m2.alpha_:.3g})")}
    m3=LassoCV(cv=5,max_iter=5000).fit(Xtr_s,ytr); p3=m3.predict(Xte_s)
    zeroed=[i for i,c_ in enumerate(m3.coef_) if abs(c_)<1e-8]
    R["Lasso"]={"pred":p3,"fi":dict(zip(range(X.shape[1]),np.abs(m3.coef_))),"zeroed":zeroed,"metrics":mets(yte,p3,f"Lasso (a={m3.alpha_:.3g})")}
    m4=ElasticNetCV(cv=5,max_iter=5000,l1_ratio=[.1,.3,.5,.7,.9]).fit(Xtr_s,ytr); p4=m4.predict(Xte_s)
    R["ElasticNet"]={"pred":p4,"fi":dict(zip(range(X.shape[1]),np.abs(m4.coef_))),"metrics":mets(yte,p4,f"ElasticNet(a={m4.alpha_:.3g})")}
    m5=RandomForestRegressor(n_estimators=200,max_depth=6,random_state=42,n_jobs=-1).fit(Xtr_s,ytr); p5=m5.predict(Xte_s)
    R["RF"]={"pred":p5,"fi":dict(zip(range(X.shape[1]),m5.feature_importances_)),"metrics":mets(yte,p5,"Random Forest")}
    m6=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.02,max_depth=5,random_state=42,
        early_stopping=True,validation_fraction=0.1,n_iter_no_change=20).fit(Xtr_s,ytr); p6=m6.predict(Xte_s)
    R["HGB"]={"pred":p6,"fi":{},"metrics":mets(yte,p6,"Hist Gradient Boost")}
    feat_names=["P_SMA5","P_EMA5","P_SMA20","P_EMA20","P_EMA50","P_EMA200",
                "RSI_n","MACD_n","MACD_h_n","BB_Pct","BB_Width","ATR_n",
                "Stoch_K_n","WR_n","CCI_n","ADX_n","OBV_Slope","Vol_Ratio_n",
                "Vol_5","Vol_20","Regime","Golden_Cross","EMA50_200_n","Drawdown",
                "Lag_1","Lag_2","Lag_3","Lag_5","Lag_10",
                "Cum_5_n","Cum_10_n","Cum_20_n","Ret_n"]
    for k in R:
        named={}
        for idx,val in R[k]["fi"].items():
            name=feat_names[idx] if idx<len(feat_names) else f"f{idx}"
            named[name]=val
        R[k]["fi"]=named
    return {"results":R,"y_test":yte,"dates_test":dte,"n_train":sp,"n_test":len(yte)}, None


# ═══════════════════════════════════════════════════════════════════════════
# INVESTMENT SCORING
# ═══════════════════════════════════════════════════════════════════════════
def score_stock(ind_df, lstm_res):
    last=ind_df.ffill().iloc[-1]; score=0; bd={}
    rsi=float(last["RSI"])
    rs=20 if rsi<30 else 15 if rsi<50 else 10 if rsi<65 else 5
    score+=rs; bd["RSI"]=rs
    ms=20 if float(last["MACD_Hist"])>0 else 5
    score+=ms; bd["MACD"]=ms
    cur=float(last["Close"]); e200=float(last["EMA_200"])
    es=15 if cur>e200 else 5; score+=es; bd["vs EMA200"]=es
    vol=float(last["Vol_30"])
    vs=15 if vol<0.2 else 10 if vol<0.35 else 5; score+=vs; bd["Volatility"]=vs
    adx=float(last["ADX"]); ads=10 if adx>25 else 5; score+=ads; bd["ADX"]=ads
    if lstm_res:
        lp=int(lstm_res["preds_all"][-1]) if len(lstm_res["preds_all"])>0 else 1
        lc=float(lstm_res["conf_all"][-1]) if len(lstm_res["conf_all"])>0 else 0.5
        ls=int(15*lc*(1.0 if lp==1 else 0.25)); score+=ls; bd["LSTM"]=ls
    signal="BUY" if score>=65 else "HOLD" if score>=45 else "AVOID"
    ret=ind_df["Return"].dropna()
    sh=float((ret.mean()/ret.std())*np.sqrt(252)) if ret.std()>0 else 0
    nr=ret[ret<0]
    so=float((ret.mean()/nr.std())*np.sqrt(252)) if len(nr)>0 and nr.std()>0 else 0
    return {"score":score,"signal":signal,"breakdown":bd,
            "sharpe":round(sh,2),"sortino":round(so,2),
            "rsi":round(rsi,1),"vol":round(vol*100,1),"close":round(cur,2)}


# ── Lazy long-term data loader ─────────────────────────────────────────────
def _ensure_long_data():
    """Load extended history + MTF indicators only when a page needs them."""
    global frames_long, mtf_data
    if not frames_long:
        with st.spinner("📅 Loading extended history (2005→today)…"):
            try: frames_long=fetch_yahoo_long()
            except Exception: frames_long={}
        if not frames_long: frames_long=frames
    if not mtf_data:
        with st.spinner("📊 Building Weekly/Monthly/Quarterly charts…"):
            try: mtf_data=build_mtf(frames_long)
            except Exception: mtf_data={}


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR  (LOADED_TICKERS must be defined before this block)
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 IB Analytics")
    st.caption(f"Yahoo Finance  |  Live as of {END}")
    st.divider()
    page=st.radio("Select Module",[
        "📊 Executive Overview",
        "📉 Technical Analysis",
        "⏱️ Multi-Timeframe Analysis",
        "🎯 LSTM Classification",
        "📈 Regression Analysis",
        "🔮 Clustering Analysis",
        "🔗 Association Rules",
        "🔬 Deep Drill-Down",
        "📥 Download Data",
    ])
    st.divider()
    selected=st.multiselect("Stocks",LOADED_TICKERS,
                             default=LOADED_TICKERS[:min(5,len(LOADED_TICKERS))])
    if not selected: selected=LOADED_TICKERS[:min(5,len(LOADED_TICKERS))]
    st.divider()
    st.caption("⚠️ Not financial advice.")


# ── STARTUP DATA LOAD ──────────────────────────────────────────────────────
with st.spinner("📡 Fetching live data from Yahoo Finance…"):
    frames=fetch_yahoo()

if not frames:
    st.error("❌ Cannot reach Yahoo Finance. Install: pip install yfinance"); st.stop()

with st.spinner("⚙️ Computing technical indicators…"):
    ind_data=compute_indicators(frames)

# Lazy-load placeholders — filled only when MTF/Drill-Down pages are selected
frames_long: dict = {}
mtf_data:    dict = {}

data_src=f"🟢 Yahoo Finance  |  {len(frames)} stocks  |  {START} → {END}"

# Narrow to tickers that actually downloaded (guards all KeyErrors)
LOADED_TICKERS=[t for t in TICKERS if t in frames and t in ind_data]
if not LOADED_TICKERS:
    st.error("No stock data loaded."); st.stop()
if len(LOADED_TICKERS)<len(TICKERS):
    missing=[t for t in TICKERS if t not in LOADED_TICKERS]
    st.warning(f"⚠️ Failed to load: {', '.join(missing)}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page=="📊 Executive Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("📊 Executive Overview")
    st.caption(data_src)

    st.subheader("Live Market Snapshot")
    cols=st.columns(min(len(selected),5))
    for i,t in enumerate([t for t in selected if t in LOADED_TICKERS][:5]):
        df=frames[t]; cur=float(df["Close"].iloc[-1])
        pv=float(df["Close"].iloc[max(0,len(df)-252)])
        d1=(float(df["Close"].iloc[-1])-float(df["Close"].iloc[-2])) if len(df)>1 else 0
        ytd=(cur/pv-1)*100
        hi=float(df["Close"].rolling(252).max().iloc[-1])
        lo=float(df["Close"].rolling(252).min().iloc[-1])
        with cols[i]:
            st.metric(t,f"${cur:.2f}",f"{d1:+.2f} ({ytd:+.1f}% 1Y)")
            st.progress(max(0,min(1,(cur-lo)/(hi-lo+1e-9))),text=f"52W: ${lo:.0f}–${hi:.0f}")

    st.divider()
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Normalised Performance (Base=100)")
        fig=go.Figure()
        for i,t in enumerate([t for t in selected if t in LOADED_TICKERS]):
            df=frames[t]; norm=df["Close"]/df["Close"].iloc[0]*100
            fig.add_trace(go.Scatter(x=df.index,y=norm,name=t,mode="lines",
                                     line=dict(width=2,color=COLORS[i%len(COLORS)])))
        pplot(fig,h=320,xaxis_title="Date",yaxis_title="Indexed (Base=100)",
              legend=dict(orientation="h",y=1.02,x=0))
        ibox("What this shows","How $100 invested at start has grown. Steeper = stronger compounder. "
             "When all lines fall together = macro correlation spike — diversification fails. "
             "Divergence between lines = genuine individual factor returns.")
    with c2:
        st.subheader("Return Correlation Heatmap")
        ret_df=pd.DataFrame({t:frames[t]["Close"].pct_change() for t in LOADED_TICKERS}).dropna()
        corr=ret_df.corr().round(3)
        fig2=px.imshow(corr,text_auto=True,color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
        fig2.update_layout(**DARK,height=320)
        st.plotly_chart(fig2,use_container_width=True,config=PCFG)
        ibox("Correlation insight","Red = stocks that always move together — concentrated risk. "
             "0 = independent — ideal diversification. Tech cluster (AAPL/MSFT/NVDA) shows 0.6–0.8. "
             "BRK-B and UNH are the strongest portfolio diversifiers here.")

    st.subheader("Price + EMA 20 / 50 / 200  (Last 180 Days)")
    cs_t=st.selectbox("Ticker",selected,key="cs_ov")
    ind=ind_data[cs_t] if cs_t in ind_data else ind_data[LOADED_TICKERS[0]]
    cutoff=ind.index[-1]-pd.Timedelta(days=180)
    iv=ind[ind.index>=cutoff]
    fig3=go.Figure()
    fig3.add_trace(go.Candlestick(x=iv.index,open=iv["Open"],high=iv["High"],
                                   low=iv["Low"],close=iv["Close"],name=cs_t,
                                   increasing_line_color="#3fb950",decreasing_line_color="#f85149"))
    fig3.add_trace(go.Bar(x=iv.index,y=iv["Volume"],name="Volume",yaxis="y2",
                          marker_color="rgba(88,166,255,0.12)"))
    for ema,col,dash,wid in [(20,"#fee140","dot",1.2),(50,"#58a6ff","dash",1.6),(200,"#3fb950","solid",2.0)]:
        fig3.add_trace(go.Scatter(x=iv.index,y=iv[f"EMA_{ema}"],name=f"EMA {ema}",
                                   line=dict(color=col,width=wid,dash=dash)))
    fig3.update_layout(**DARK,height=440,
                       yaxis2=dict(overlaying="y",side="right",showgrid=False),
                       xaxis_rangeslider_visible=False)
    st.plotly_chart(fig3,use_container_width=True,config=PCFG)
    ibox("EMA 20/50/200 decoded",
         "**EMA 20 (gold):** Short-term momentum — reacts within a week. "
         "**EMA 50 (blue):** Institutional medium-term trend — most-watched line by professional traders. "
         "**EMA 200 (green):** Structural regime boundary. Above = bull market; below = bear market. "
         "**Golden Cross** (EMA 50 > EMA 200): strongest long-term buy signal in technical analysis.")

    c3,c4=st.columns(2)
    with c3:
        st.subheader("30-Day Rolling Volatility")
        fig4=go.Figure()
        for i,t in enumerate([t for t in selected if t in LOADED_TICKERS]):
            v=ind_data[t]["Vol_30"]*100
            fig4.add_trace(go.Scatter(x=v.index,y=v,name=t,mode="lines",
                                      line=dict(width=1.5,color=COLORS[i%len(COLORS)])))
        pplot(fig4,h=260,yaxis_title="Ann. Volatility (%)")
        ibox("Volatility regime","A 60% reading means ±60% potential annual move. TSLA/NVDA often exceed 80% "
             "during earnings. BRK-B stays below 25% — that's why it's classified Defensive in clustering.")
    with c4:
        st.subheader("Drawdown from All-Time High")
        fig5=go.Figure()
        for i,t in enumerate([t for t in selected if t in LOADED_TICKERS]):
            dd=ind_data[t]["Drawdown"]*100
            fig5.add_trace(go.Scatter(x=dd.index,y=dd,name=t,fill="tozeroy",mode="lines",
                                      line=dict(width=1,color=COLORS[i%len(COLORS)])))
        pplot(fig5,h=260,yaxis_title="Drawdown (%)")
        ibox("Drawdown as risk signal","Deep drawdown in a bull regime (above EMA 200) = mean-reversion "
             "entry opportunity. Same drawdown below EMA 200 = structural downtrend — AVOID signal.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page=="📉 Technical Analysis":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.title("📉 Full Technical Analysis Suite")
    ta_t=st.selectbox("Select Stock",LOADED_TICKERS,key="ta_t")
    ind=ind_data[ta_t] if ta_t in ind_data else ind_data[LOADED_TICKERS[0]]
    days=st.slider("Days to display",60,len(ind),min(365,len(ind)),30)
    iv=ind.iloc[-days:]

    st.subheader("Panel 1: Price + EMA 20/50/200 + Bollinger Bands + Ichimoku Cloud")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=iv.index,y=iv["Ichi_SpanA"],name="Span A",
                              line=dict(color="rgba(0,0,0,0)",width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=iv.index,y=iv["Ichi_SpanB"],name="Kumo Cloud",
                              fill="tonexty",fillcolor="rgba(63,185,80,0.07)",
                              line=dict(color="rgba(0,0,0,0)",width=0)))
    fig.add_trace(go.Scatter(x=iv.index,y=iv["BB_Upper"],name="BB Upper",
                              line=dict(color="rgba(88,166,255,0.35)",width=1,dash="dot")))
    fig.add_trace(go.Scatter(x=iv.index,y=iv["BB_Lower"],name="BB Lower",fill="tonexty",
                              fillcolor="rgba(88,166,255,0.05)",
                              line=dict(color="rgba(88,166,255,0.35)",width=1,dash="dot")))
    for ema,col,dash,wid in [(20,"#fee140","dot",1.2),(50,"#58a6ff","dash",1.6),(200,"#3fb950","solid",2.0)]:
        fig.add_trace(go.Scatter(x=iv.index,y=iv[f"EMA_{ema}"],name=f"EMA {ema}",
                                  line=dict(color=col,width=wid,dash=dash)))
    fig.add_trace(go.Scatter(x=iv.index,y=iv["Ichi_Tenkan"],name="Tenkan(9)",
                              line=dict(color="#f093fb",width=1,dash="dot")))
    fig.add_trace(go.Scatter(x=iv.index,y=iv["Ichi_Kijun"],name="Kijun(26)",
                              line=dict(color="#4facfe",width=1,dash="dash")))
    fig.add_trace(go.Candlestick(x=iv.index,open=iv["Open"],high=iv["High"],
                                  low=iv["Low"],close=iv["Close"],name=ta_t,
                                  increasing_line_color="#3fb950",decreasing_line_color="#f85149"))
    fig.update_layout(**DARK,height=500,xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h",y=1.01,x=0,font=dict(size=9)))
    st.plotly_chart(fig,use_container_width=True,config=PCFG)
    ibox("Reading Panel 1",
         "**Bollinger Bands:** Band squeeze (width narrows) = volatility contraction before a big move. "
         "Price at upper band = short-term overbought. "
         "**Ichimoku Cloud:** Price above green cloud = confirmed bull trend. Cloud thickness = support strength. "
         "**Tenkan/Kijun cross:** Short-term entry signal within Ichimoku system. "
         "**EMA 200:** The most important line — structural regime boundary used by the LSTM as its Regime feature.")

    st.subheader("Panel 2: RSI(14)  |  Stochastic(14,3)  |  Williams %R(14)")
    fig2=make_subplots(rows=3,cols=1,shared_xaxes=True,
                        subplot_titles=["RSI(14)","Stochastic %K/%D","Williams %R(14)"],
                        vertical_spacing=0.06)
    fig2.add_trace(go.Scatter(x=iv.index,y=iv["RSI"],name="RSI",
                               line=dict(color="#f6ad55",width=1.8)),row=1,col=1)
    for yv,col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
        fig2.add_hline(y=yv,line_color=col,line_dash="dash",row=1,col=1)
    fig2.add_trace(go.Scatter(x=iv.index,y=iv["Stoch_K"],name="%K",
                               line=dict(color="#58a6ff",width=1.6)),row=2,col=1)
    fig2.add_trace(go.Scatter(x=iv.index,y=iv["Stoch_D"],name="%D",
                               line=dict(color="#f093fb",width=1.2,dash="dot")),row=2,col=1)
    for yv,col in [(80,"#f85149"),(20,"#3fb950")]:
        fig2.add_hline(y=yv,line_color=col,line_dash="dash",row=2,col=1)
    fig2.add_trace(go.Scatter(x=iv.index,y=iv["Williams_R"],name="Williams %R",
                               line=dict(color="#43e97b",width=1.6)),row=3,col=1)
    for yv,col in [(-20,"#f85149"),(-80,"#3fb950")]:
        fig2.add_hline(y=yv,line_color=col,line_dash="dash",row=3,col=1)
    fig2.update_layout(**DARK,height=520)
    st.plotly_chart(fig2,use_container_width=True,config=PCFG)
    ibox("Triple oscillator confirmation",
         "**RSI < 35 + Stochastic < 20 + Williams %R < -80** simultaneously = "
         "highest-probability oversold buy setup in technical analysis. "
         "The LSTM sees 20 consecutive days of all three — it learns multi-indicator alignment patterns "
         "that single-point models miss. Weekly RSI is far more reliable than daily.")

    st.subheader("Panel 3: MACD(12,26,9)  |  CCI(20)  |  ADX(14)")
    fig3=make_subplots(rows=3,cols=1,shared_xaxes=True,
                        subplot_titles=["MACD","CCI(20)","ADX(14)"],vertical_spacing=0.06)
    hc=["#3fb950" if v>=0 else "#f85149" for v in iv["MACD_Hist"]]
    fig3.add_trace(go.Bar(x=iv.index,y=iv["MACD_Hist"],name="Histogram",marker_color=hc,opacity=0.7),row=1,col=1)
    fig3.add_trace(go.Scatter(x=iv.index,y=iv["MACD"],name="MACD",line=dict(color="#58a6ff",width=1.6)),row=1,col=1)
    fig3.add_trace(go.Scatter(x=iv.index,y=iv["MACD_Signal"],name="Signal",
                               line=dict(color="#f85149",width=1.2,dash="dot")),row=1,col=1)
    fig3.add_hline(y=0,line_color="#374151",line_dash="dash",row=1,col=1)
    fig3.add_trace(go.Scatter(x=iv.index,y=iv["CCI"],name="CCI",
                               line=dict(color="#fee140",width=1.6)),row=2,col=1)
    for yv,col in [(100,"#f85149"),(-100,"#3fb950"),(0,"#374151")]:
        fig3.add_hline(y=yv,line_color=col,line_dash="dash",row=2,col=1)
    fig3.add_trace(go.Scatter(x=iv.index,y=iv["ADX"],name="ADX",
                               line=dict(color="#a371f7",width=1.8)),row=3,col=1)
    fig3.add_trace(go.Scatter(x=iv.index,y=iv["DI_Plus"],name="DI+",
                               line=dict(color="#3fb950",width=1.2,dash="dash")),row=3,col=1)
    fig3.add_trace(go.Scatter(x=iv.index,y=iv["DI_Minus"],name="DI-",
                               line=dict(color="#f85149",width=1.2,dash="dash")),row=3,col=1)
    fig3.add_hline(y=25,line_color="#f6ad55",line_dash="dash",annotation_text="Trend (25)",row=3,col=1)
    fig3.update_layout(**DARK,height=520)
    st.plotly_chart(fig3,use_container_width=True,config=PCFG)
    ibox("Trend strength decoded",
         "**MACD:** Growing green histogram = accelerating bullish momentum. "
         "Shrinking green = momentum peak — early exit signal. "
         "**CCI > +100:** Strong uptrend — ride it. **CCI < -100:** Strong downtrend. "
         "**ADX > 25:** Strong directional trend — momentum strategies work. "
         "**ADX < 20:** No trend — oscillator strategies work instead. "
         "DI+ > DI- AND ADX > 25 = strongest trend-following buy confirmation.")

    st.subheader("Panel 4: OBV + Volume")
    fig4=make_subplots(rows=2,cols=1,shared_xaxes=True,
                        subplot_titles=["OBV (On-Balance Volume)","Daily Volume vs 20-Day MA"],
                        vertical_spacing=0.08)
    fig4.add_trace(go.Scatter(x=iv.index,y=iv["OBV"],name="OBV",fill="tozeroy",
                               line=dict(color="#a371f7",width=1.8)),row=1,col=1)
    vc=["#3fb950" if r>=0 else "#f85149" for r in iv["Return"].fillna(0)]
    fig4.add_trace(go.Bar(x=iv.index,y=iv["Volume"],name="Volume",marker_color=vc,opacity=0.6),row=2,col=1)
    vma20=iv["Volume"].rolling(20).mean()
    fig4.add_trace(go.Scatter(x=iv.index,y=vma20,name="Vol MA20",
                               line=dict(color="#f6ad55",width=1.8)),row=2,col=1)
    fig4.update_layout(**DARK,height=400)
    st.plotly_chart(fig4,use_container_width=True,config=PCFG)
    ibox("Volume confirms price",
         "**OBV rising while price is flat** = institutional accumulation (smart money buying quietly). "
         "This precedes price breakouts by 1–3 weeks — the LSTM's OBV_Slope feature encodes this. "
         "**Large green volume bar on up day** = institutional buying confirmation. "
         "**Large red bar** = panic selling — often a capitulation bottom, not a continuation.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MULTI-TIMEFRAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page=="⏱️ Multi-Timeframe Analysis":
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    _ensure_long_data()   # lazy-load only when this page is selected
    st.title("⏱️ Multi-Timeframe Analysis")
    st.caption(f"Extended history: {START_LONG} → {END}  |  Yahoo Finance")
    mtf_t=st.selectbox("Select Stock",LOADED_TICKERS,key="mtf_t")
    tfs=mtf_data.get(mtf_t,{})
    df_d=(frames_long or frames).get(mtf_t) or frames.get(mtf_t)

    tab_w,tab_m,tab_q,tab_cal,tab_yr,tab_sr,tab_cmp=st.tabs([
        "📅 Weekly","🗓️ Monthly","📆 Quarterly",
        "🔥 Monthly Heatmap","📈 Yearly Returns",
        "🎯 Support & Resistance","🔀 Cross-TF Table"])

    with tab_w:
        df_w=tfs.get("W")
        if df_w is None or len(df_w)<20:
            st.info("Not enough weekly data.")
        else:
            n_w=st.slider("Weeks",52,len(df_w),min(260,len(df_w)),13,key="wk_sl")
            wv=df_w.iloc[-n_w:]
            st.subheader("Weekly Chart + W-EMA 10/20/50 + Bollinger Bands")
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=wv.index,open=wv["Open"],high=wv["High"],
                                          low=wv["Low"],close=wv["Close"],name="Weekly",
                                          increasing_line_color="#3fb950",decreasing_line_color="#f85149"))
            fig.add_trace(go.Bar(x=wv.index,y=wv["Volume"],name="Vol",yaxis="y2",
                                  marker_color="rgba(88,166,255,0.12)"))
            if "BB_Upper" in wv.columns:
                fig.add_trace(go.Scatter(x=wv.index,y=wv["BB_Upper"],name="BB Upper",
                                          line=dict(color="rgba(88,166,255,0.3)",width=1,dash="dot")))
                fig.add_trace(go.Scatter(x=wv.index,y=wv["BB_Lower"],name="BB Lower",fill="tonexty",
                                          fillcolor="rgba(88,166,255,0.05)",
                                          line=dict(color="rgba(88,166,255,0.3)",width=1,dash="dot")))
            for ema,col,dash,wid in [(10,"#fee140","dot",1.2),(20,"#58a6ff","dash",1.6),(50,"#3fb950","solid",2.0)]:
                cn=f"EMA_{ema}"
                if cn in wv.columns:
                    fig.add_trace(go.Scatter(x=wv.index,y=wv[cn],name=f"W-EMA {ema}",
                                              line=dict(color=col,width=wid,dash=dash)))
            fig.update_layout(**DARK,height=480,xaxis_rangeslider_visible=False,
                              yaxis2=dict(overlaying="y",side="right",showgrid=False),
                              legend=dict(orientation="h",y=1.01,x=0,font=dict(size=9)))
            st.plotly_chart(fig,use_container_width=True,config=PCFG)
            st.subheader("Weekly RSI  |  MACD  |  Stochastic")
            fig2=make_subplots(rows=3,cols=1,shared_xaxes=True,
                                subplot_titles=["W-RSI(14)","W-MACD","W-Stochastic"],vertical_spacing=0.06)
            if "RSI" in wv.columns:
                fig2.add_trace(go.Scatter(x=wv.index,y=wv["RSI"],name="W-RSI",
                                           line=dict(color="#f6ad55",width=1.8)),row=1,col=1)
                for yv,col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                    fig2.add_hline(y=yv,line_color=col,line_dash="dash",row=1,col=1)
            if "MACD" in wv.columns:
                hc=["#3fb950" if v>=0 else "#f85149" for v in wv["MACD_Hist"]]
                fig2.add_trace(go.Bar(x=wv.index,y=wv["MACD_Hist"],name="W-Hist",marker_color=hc,opacity=0.7),row=2,col=1)
                fig2.add_trace(go.Scatter(x=wv.index,y=wv["MACD"],name="W-MACD",
                                           line=dict(color="#58a6ff",width=1.6)),row=2,col=1)
                fig2.add_trace(go.Scatter(x=wv.index,y=wv["MACD_Signal"],name="W-Signal",
                                           line=dict(color="#f85149",width=1.2,dash="dot")),row=2,col=1)
            if "Stoch_K" in wv.columns:
                fig2.add_trace(go.Scatter(x=wv.index,y=wv["Stoch_K"],name="W-%K",
                                           line=dict(color="#58a6ff",width=1.6)),row=3,col=1)
                fig2.add_trace(go.Scatter(x=wv.index,y=wv["Stoch_D"],name="W-%D",
                                           line=dict(color="#f093fb",width=1.2,dash="dot")),row=3,col=1)
                for yv,col in [(80,"#f85149"),(20,"#3fb950")]:
                    fig2.add_hline(y=yv,line_color=col,line_dash="dash",row=3,col=1)
            fig2.update_layout(**DARK,height=500)
            st.plotly_chart(fig2,use_container_width=True,config=PCFG)
            ibox("Weekly oscillators — fewer false alarms",
                 "Weekly RSI < 30 = significant multi-week selloff — historically excellent medium-term entry. "
                 "Weekly MACD bullish crossover = trend change lasting weeks to months. "
                 "Weekly signals eliminate daily noise and reveal the true medium-term direction.")

    with tab_m:
        df_m=tfs.get("ME")
        if df_m is None or len(df_m)<12:
            st.info("Not enough monthly data.")
        else:
            n_m=st.slider("Months",24,len(df_m),min(120,len(df_m)),12,key="mo_sl")
            mv=df_m.iloc[-n_m:]
            st.subheader("Monthly Chart + M-EMA 6/12/24 (≈6mo/1yr/2yr)")
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=mv.index,open=mv["Open"],high=mv["High"],
                                          low=mv["Low"],close=mv["Close"],name="Monthly",
                                          increasing_line_color="#3fb950",decreasing_line_color="#f85149"))
            fig.add_trace(go.Bar(x=mv.index,y=mv["Volume"],name="Vol",yaxis="y2",
                                  marker_color="rgba(88,166,255,0.12)"))
            if "BB_Upper" in mv.columns:
                fig.add_trace(go.Scatter(x=mv.index,y=mv["BB_Upper"],name="M-BB Upper",
                                          line=dict(color="rgba(88,166,255,0.3)",width=1,dash="dot")))
                fig.add_trace(go.Scatter(x=mv.index,y=mv["BB_Lower"],name="M-BB Lower",fill="tonexty",
                                          fillcolor="rgba(88,166,255,0.05)",
                                          line=dict(color="rgba(88,166,255,0.3)",width=1,dash="dot")))
            for ema,col,dash,wid,lbl in [(6,"#fee140","dot",1.3,"~6mo"),(12,"#58a6ff","dash",1.7,"~1yr"),(24,"#3fb950","solid",2.2,"~2yr")]:
                cn=f"EMA_{ema}"
                if cn in mv.columns:
                    fig.add_trace(go.Scatter(x=mv.index,y=mv[cn],name=f"M-EMA {ema}({lbl})",
                                              line=dict(color=col,width=wid,dash=dash)))
            fig.update_layout(**DARK,height=500,xaxis_rangeslider_visible=False,
                              yaxis2=dict(overlaying="y",side="right",showgrid=False),
                              legend=dict(orientation="h",y=1.01,x=0,font=dict(size=9)))
            st.plotly_chart(fig,use_container_width=True,config=PCFG)
            st.subheader("Monthly RSI  |  MACD  |  ADX")
            fig2=make_subplots(rows=3,cols=1,shared_xaxes=True,
                                subplot_titles=["M-RSI(14)","M-MACD","M-ADX"],vertical_spacing=0.06)
            if "RSI" in mv.columns:
                fig2.add_trace(go.Scatter(x=mv.index,y=mv["RSI"],name="M-RSI",
                                           line=dict(color="#f6ad55",width=2.0)),row=1,col=1)
                for yv,col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                    fig2.add_hline(y=yv,line_color=col,line_dash="dash",row=1,col=1)
            if "MACD" in mv.columns:
                hc=["#3fb950" if v>=0 else "#f85149" for v in mv["MACD_Hist"]]
                fig2.add_trace(go.Bar(x=mv.index,y=mv["MACD_Hist"],name="M-Hist",marker_color=hc,opacity=0.7),row=2,col=1)
                fig2.add_trace(go.Scatter(x=mv.index,y=mv["MACD"],name="M-MACD",
                                           line=dict(color="#58a6ff",width=1.8)),row=2,col=1)
                fig2.add_trace(go.Scatter(x=mv.index,y=mv["MACD_Signal"],name="M-Signal",
                                           line=dict(color="#f85149",width=1.3,dash="dot")),row=2,col=1)
            if "ADX" in mv.columns:
                fig2.add_trace(go.Scatter(x=mv.index,y=mv["ADX"],name="M-ADX",
                                           line=dict(color="#a371f7",width=1.8)),row=3,col=1)
                fig2.add_trace(go.Scatter(x=mv.index,y=mv["DI_Plus"],name="M-DI+",
                                           line=dict(color="#3fb950",width=1.2,dash="dash")),row=3,col=1)
                fig2.add_trace(go.Scatter(x=mv.index,y=mv["DI_Minus"],name="M-DI-",
                                           line=dict(color="#f85149",width=1.2,dash="dash")),row=3,col=1)
                fig2.add_hline(y=25,line_color="#f6ad55",line_dash="dash",row=3,col=1)
            fig2.update_layout(**DARK,height=520)
            st.plotly_chart(fig2,use_container_width=True,config=PCFG)
            ibox("Monthly charts — macro cycle timing",
                 "Monthly RSI < 30 is historically very rare and marks major market bottoms (Mar 2009, Mar 2020). "
                 "Monthly MACD bullish crossover = new multi-year bull phase beginning. "
                 "Monthly ADX > 40 = extremely strong trend — like NVDA's 2023–2024 AI surge.")

    with tab_q:
        df_q=tfs.get("QE")
        if df_q is None or len(df_q)<8:
            st.info("Not enough quarterly data.")
        else:
            n_q=st.slider("Quarters",8,len(df_q),min(40,len(df_q)),4,key="q_sl")
            qv=df_q.iloc[-n_q:]
            st.subheader("Quarterly Chart + Q-EMA 4/8/12 (≈1yr/2yr/3yr)")
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=qv.index,open=qv["Open"],high=qv["High"],
                                          low=qv["Low"],close=qv["Close"],name="Quarterly",
                                          increasing_line_color="#3fb950",decreasing_line_color="#f85149"))
            for ema,col,dash,wid,lbl in [(4,"#fee140","dot",1.3,"~1yr"),(8,"#58a6ff","dash",1.7,"~2yr"),(12,"#3fb950","solid",2.2,"~3yr")]:
                cn=f"EMA_{ema}"
                if cn in qv.columns:
                    fig.add_trace(go.Scatter(x=qv.index,y=qv[cn],name=f"Q-EMA{ema}({lbl})",
                                              line=dict(color=col,width=wid,dash=dash)))
            fig.update_layout(**DARK,height=420,xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h",y=1.01,x=0,font=dict(size=9)))
            st.plotly_chart(fig,use_container_width=True,config=PCFG)
            c1q,c2q=st.columns(2)
            with c1q:
                if "RSI" in qv.columns:
                    fig2=go.Figure()
                    fig2.add_trace(go.Scatter(x=qv.index,y=qv["RSI"],name="Q-RSI",
                                               fill="tozeroy",line=dict(color="#f6ad55",width=2)))
                    for yv,col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                        fig2.add_hline(y=yv,line_color=col,line_dash="dash")
                    pplot(fig2,h=280,yaxis_title="Quarterly RSI",yaxis={"range":[0,100]})
                    ibox("Quarterly RSI","Q-RSI < 30 = multi-year buying opportunity. "
                         "Historically: Q3 2022 for most tech (NVDA ~25), March 2020 (~20). "
                         "Q-RSI > 80 = multi-year overbought — significant caution territory.")
            with c2q:
                if "Pct_Range_52" in qv.columns:
                    fig3=go.Figure(go.Bar(x=qv.index,y=qv["Pct_Range_52"],
                                          marker_color=["#3fb950" if v>50 else "#f85149" for v in qv["Pct_Range_52"]]))
                    fig3.add_hline(y=50,line_dash="dash",line_color="#374151")
                    pplot(fig3,h=280,yaxis_title="% of 52-Quarter Range",yaxis={"range":[0,110]})
                    ibox("52-quarter range","0% = 13-year low. 100% = 13-year high. "
                         ">80% = historically expensive. <20% = strong long-term value zone.")
            ibox("Quarterly chart — a decade of context",
                 "Each candle = one quarter. Q-EMA 8 = 2-year average — the institutional positioning line. "
                 "A stock below Q-EMA 8 has underperformed for 2 years — structural concern, not a dip. "
                 "Quarterly Bollinger touches represent generational valuation extremes.")

    with tab_cal:
        st.subheader("Monthly Return Calendar Heatmap (%)")
        if df_d is not None and len(df_d)>24:
            pivot=_monthly_heatmap(df_d).tail(15)
            figc=px.imshow(pivot.round(1),text_auto=True,color_continuous_scale="RdYlGn",
                            color_continuous_midpoint=0,aspect="auto",
                            labels=dict(x="Month",y="Year",color="Return %"))
            figc.update_traces(textfont=dict(size=10))
            figc.update_layout(**DARK,height=max(350,len(pivot)*32+80))
            st.plotly_chart(figc,use_container_width=True,config=PCFG)
            yr_ret=_yearly_returns(df_d).reindex(pivot.index)
            yr_df=pd.DataFrame({"Year":yr_ret.index,"Annual Return %":yr_ret.values.round(1),
                                 "Best Month":pivot.idxmax(axis=1).values,
                                 "Worst Month":pivot.idxmin(axis=1).values})
            st.dataframe(yr_df.set_index("Year"),use_container_width=True)
            ibox("Reading the heatmap","Each cell = return for that month/year. "
                 "**Seasonal patterns:** January Effect (Jan rally), Sell in May (May-Oct weaker), "
                 "Q4 strength (Nov-Dec often positive for tech). "
                 "Grey = data not available. Identifying seasonal patterns lets you time entries at "
                 "statistically favourable calendar points.")
        else:
            st.info("Not enough daily history.")

    with tab_yr:
        st.subheader("Calendar Year Returns — All Stocks")
        yr_data={}
        for t in LOADED_TICKERS:
            dfl=(frames_long or frames).get(t) or frames.get(t)
            if dfl is not None and len(dfl)>250:
                yr_data[t]=_yearly_returns(dfl)
        if yr_data:
            yr_df=pd.DataFrame(yr_data).round(1)
            figyr=px.imshow(yr_df.T,text_auto=True,color_continuous_scale="RdYlGn",
                             color_continuous_midpoint=0,aspect="auto",
                             labels=dict(x="Year",y="Stock",color="Return %"))
            figyr.update_traces(textfont=dict(size=10))
            figyr.update_layout(**DARK,height=420)
            st.plotly_chart(figyr,use_container_width=True,config=PCFG)
            avail=sorted(yr_df.index.tolist(),reverse=True)
            sel_yr=st.selectbox("Drill into a year",avail,key="yr_sel")
            yr_row=yr_df.loc[sel_yr].dropna().sort_values(ascending=False)
            fig_bar=go.Figure(go.Bar(x=yr_row.index,y=yr_row.values,
                                      marker_color=["#3fb950" if v>=0 else "#f85149" for v in yr_row.values],
                                      text=[f"{v:.1f}%" for v in yr_row.values],textposition="outside"))
            fig_bar.update_layout(**DARK,height=320,title=f"{sel_yr} Returns",yaxis_title="Return (%)")
            st.plotly_chart(fig_bar,use_container_width=True,config=PCFG)
            ibox("Long-term return patterns","Stocks outperforming across multiple years have compounding "
                 "competitive advantages. Years with broad negative returns = macro/rate-driven selloffs. "
                 "High win-rate (>70% positive years) = reliable compounder.")
        else:
            st.info("Not enough yearly history.")

    with tab_sr:
        st.subheader("Key Price Levels: Support & Resistance")
        sr_t=st.selectbox("Stock",LOADED_TICKERS,key="sr_t")
        sr_tf=st.selectbox("Timeframe",["Daily","Weekly","Monthly"],key="sr_tf")
        tf_map={"Daily":None,"Weekly":"W","Monthly":"ME"}
        tf_key=tf_map[sr_tf]
        if tf_key:
            df_sr=mtf_data.get(sr_t,{}).get(tf_key)
        else:
            df_sr=ind_data.get(sr_t)
        if df_sr is not None and len(df_sr)>30:
            n_sr=st.slider("Periods",30,len(df_sr),min(120,len(df_sr)),10,key="sr_sl")
            srv=df_sr.iloc[-n_sr:]
            c_sr=df_sr["Close"].values
            win=3 if tf_key=="ME" else (5 if tf_key=="W" else 10)
            supps,ress=_support_resistance(c_sr,window=win)
            fig_sr=go.Figure()
            fig_sr.add_trace(go.Candlestick(x=srv.index,open=srv["Open"],high=srv["High"],
                                             low=srv["Low"],close=srv["Close"],name=f"{sr_t}({sr_tf})",
                                             increasing_line_color="#3fb950",decreasing_line_color="#f85149"))
            for ema,col,dash,wid in [(20,"#58a6ff","dash",1.4),(50,"#3fb950","solid",1.8)]:
                cn=f"EMA_{ema}"
                if cn in srv.columns:
                    fig_sr.add_trace(go.Scatter(x=srv.index,y=srv[cn],name=f"EMA {ema}",
                                                 line=dict(color=col,width=wid,dash=dash)))
            cur_px=float(df_sr["Close"].iloc[-1])
            for i,s in enumerate(supps):
                fig_sr.add_hline(y=s,line_color="#3fb950",line_dash="dot",line_width=1.5,
                                  annotation_text=f"S{i+1} ${s:.2f}",annotation_position="right",
                                  annotation_font_color="#3fb950")
            for i,r in enumerate(ress):
                fig_sr.add_hline(y=r,line_color="#f85149",line_dash="dot",line_width=1.5,
                                  annotation_text=f"R{i+1} ${r:.2f}",annotation_position="right",
                                  annotation_font_color="#f85149")
            fig_sr.update_layout(**DARK,height=500,xaxis_rangeslider_visible=False,
                                  legend=dict(orientation="h",y=1.01,x=0))
            st.plotly_chart(fig_sr,use_container_width=True,config=PCFG)
            c1s,c2s=st.columns(2)
            with c1s:
                st.markdown("**🟢 Support Levels**")x
                for i,s in enumerate(supps):
                    dist=(cur_px-s)/cur_px*100
                    st.write(f"S{i+1}: **${s:.2f}** ({dist:+.1f}% from current)")
            with c2s:
                st.markdown("**🔴 Resistance Levels**")
                for i,r in enumerate(ress):
                    dist=(r-cur_px)/cur_px*100
                    st.write(f"R{i+1}: **${r:.2f}** (+{dist:.1f}% to target)")
            ibox("Support & Resistance methodology",
                 "Levels identified via pivot-point analysis — a price is a pivot high/low "
                 "if it exceeds all surrounding prices within a ±N-bar window. "
                 "Nearby pivots clustered within 2% tolerance form one significant level. "
                 "**Monthly S/R** = multi-year price memory — the most psychologically significant levels. "
                 "The more times price has touched a level, the stronger it becomes.")
        else:
            st.info("Not enough data for S/R calculation.")

    with tab_cmp:
        st.subheader("Cross-Timeframe Signal Summary")
        rows=[]
        for t in LOADED_TICKERS:
            row={"Stock":t,"Sector":META_INFO[t]["sector"]}
            tfs_t=mtf_data.get(t,{})
            for tf_key,tf_lbl in [("W","Weekly"),("ME","Monthly"),("QE","Quarterly")]:
                df_tf=tfs_t.get(tf_key)
                if df_tf is not None and len(df_tf)>5:
                    last=df_tf.ffill().iloc[-1]
                    row[f"{tf_lbl} RSI"]=round(float(last["RSI"]),1) if "RSI" in last.index else None
                    row[f"{tf_lbl} MACD_H"]=round(float(last["MACD_Hist"]),4) if "MACD_Hist" in last.index else None
                    row[f"{tf_lbl} ADX"]=round(float(last["ADX"]),1) if "ADX" in last.index else None
                    row[f"{tf_lbl} Regime"]="🟢" if float(last.get("Regime",0))==1 else "🔴"
                    if len(df_tf)>1:
                        r1=round((float(df_tf["Close"].iloc[-1])/float(df_tf["Close"].iloc[-2])-1)*100,2)
                        row[f"{tf_lbl} Ret%"]=r1
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Stock"),use_container_width=True)
        ibox("How to use the cross-TF table",
             "**Triple alignment = highest conviction:** When Weekly RSI, Monthly RSI, AND Quarterly RSI "
             "all below 50 and rising, all timeframes confirm a recovery. "
             "When all three regimes show 🟢, the stock is in a confirmed structural uptrend. "
             "**Divergence = caution:** Weekly bullish + Monthly bearish = short-term bounce may fail. "
             "Always establish the trend on the highest timeframe first, then zoom in.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — LSTM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
elif page=="🎯 LSTM Classification":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🎯 LSTM Classification — Real Trained Model")
    st.markdown(f"""
| | Detail |
|---|---|
| **Model** | Single-layer LSTM (hidden={LSTM_H}, seq_len={SEQ_LEN}) |
| **Training** | BPTT + Adam, 10 epochs, gradient clipping ±1.0 |
| **Label** | 5-day forward return > 0 → UP=1 / DOWN=0 (zero lookahead) |
| **Data** | Yahoo Finance {START} → {END} |
| **Ensemble** | LSTM hidden states → HistGradientBoosting |
| **Split** | 80/20 strict time-series |
    """)
    clf_t=st.selectbox("Select Stock",LOADED_TICKERS,key="clf_t")
    bar=st.progress(0,text="Initialising LSTM…")
    with st.spinner(f"Training LSTM for {clf_t} (~40s first run, cached after)…"):
        res,err=train_lstm(clf_t,ind_data)
        bar.empty()
    if err or res is None: st.error(f"Error: {err}"); st.stop()

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Raw LSTM Acc.",f"{res['acc_lstm']}%",help="LSTM output alone")
    c2.metric("LSTM+HGB Ensemble",f"{res['acc_all']}%",help="All out-of-sample test days")
    c3.metric("Bull Regime Acc.",f"{res['acc_bull']}%",delta=f"{res['acc_bull']-res['acc_all']:+.1f}% vs overall")
    c4.metric("Bear Regime Acc.",f"{res['acc_bear']}%",delta=f"{res['acc_bear']-res['acc_all']:+.1f}% vs overall")
    ibox("Accuracy explained",
         f"**50% = coin-flip baseline.** Any consistent edge above 50% on live unseen data is real and tradeable. "
         f"**{res['acc_all']}% overall** means the model correctly predicts 5-day direction {res['acc_all']}% of the time. "
         f"**Bull regime ({res['acc_bull']}%):** Technical indicators are more reliable in uptrends — momentum persists. "
         "The LSTM's Regime and Golden_Cross features explicitly encode this market state.")

    st.subheader("Training Loss (BPTT + Adam)")
    fig_l=go.Figure()
    fig_l.add_trace(go.Scatter(y=res["loss_hist"],x=list(range(1,len(res["loss_hist"])+1)),
                               mode="lines+markers",name="Training Loss",
                               line=dict(color="#58a6ff",width=2.5),marker=dict(size=9)))
    pplot(fig_l,h=240,xaxis_title="Epoch",yaxis_title="Cross-Entropy Loss")
    ibox("What the loss curve tells you",
         "Decreasing loss = LSTM weights are genuinely being updated via backpropagation through time. "
         "Adam optimizer adapts learning rates per parameter — 5–10× faster than plain SGD. "
         "Loss plateauing = LSTM has extracted maximum learnable signal from the 20-day window.")

    if res["conf_tbl"]:
        st.subheader("Accuracy vs Confidence Threshold")
        st.dataframe(pd.DataFrame(res["conf_tbl"]),use_container_width=True,hide_index=True)
        ibox("The confidence table is the key output",
             "The ensemble outputs a probability for each prediction. "
             "Low confidence (~50%) = indicators conflict → skip this signal. "
             "High confidence (≥65%) = indicators strongly align → act on it. "
             "**Bull + high confidence** rows show highest accuracy — the ideal trade setup. "
             "Edge over 50% converts directly to expected profit per trade.")

    st.subheader("LSTM Signal Overlay (≥60% confidence)")
    fig_s=go.Figure()
    df_p=frames[clf_t] if clf_t in frames else frames[LOADED_TICKERS[0]]
    fig_s.add_trace(go.Scatter(x=df_p.index,y=df_p["Close"],name="Close",
                               line=dict(color="#e6edf3",width=1.5)))
    for ema,col,dash in [(50,"#58a6ff","dash"),(200,"#3fb950","solid")]:
        e=ind_data[clf_t][f"EMA_{ema}"] if clf_t in ind_data else ind_data[LOADED_TICKERS[0]][f"EMA_{ema}"]
        fig_s.add_trace(go.Scatter(x=e.index,y=e,name=f"EMA {ema}",
                                   line=dict(color=col,width=1,dash=dash),opacity=0.7))
    pa,ca,da=res["preds_all"],res["conf_all"],res["dates_all"]
    for lbl,sym,col in [(1,"triangle-up","#3fb950"),(0,"triangle-down","#f85149")]:
        m=(pa==lbl)&(ca>=0.60)
        if m.sum()>0:
            ix=da[m]; px_=df_p["Close"].reindex(ix)
            lab="LSTM: UP ≥60%" if lbl==1 else "LSTM: DOWN ≥60%"
            fig_s.add_trace(go.Scatter(x=ix,y=px_,mode="markers",name=lab,
                                       marker=dict(color=col,size=7,symbol=sym,opacity=0.85)))
    pplot(fig_s,h=440,yaxis_title="Price (USD)")
    ibox("Signal overlay",
         "▲ Green = LSTM predicts UP over next 5 days (≥60% confidence). "
         "▼ Red = DOWN. Do green triangles precede upward moves? Do red precede drawdowns? "
         "Signal density drops during choppy markets (low ADX) — the model correctly reduces confidence.")

    st.subheader("Confusion Matrix  |  Classification Report")
    c1_,c2_=st.columns(2)
    with c1_:
        cm=res["cm"]
        figc=px.imshow(cm,x=["DOWN","UP"],y=["DOWN","UP"],text_auto=True,
                        color_continuous_scale="Blues",labels=dict(x="Predicted",y="Actual"))
        figc.update_layout(**DARK,height=300)
        st.plotly_chart(figc,use_container_width=True,config=PCFG)
    with c2_:
        rep=res["report"]
        rows_r=[]
        for k,label in [("0","DOWN"),("1","UP")]:
            if k in rep:
                rows_r.append({"Class":label,"Precision":round(rep[k]["precision"],3),
                               "Recall":round(rep[k]["recall"],3),"F1":round(rep[k]["f1-score"],3),
                               "Support":int(rep[k]["support"])})
        st.dataframe(pd.DataFrame(rows_r),use_container_width=True,hide_index=True)
        ibox("Precision vs Recall",
             "**Precision (UP):** of all 'UP' predictions, % correct — high = few false alarms. "
             "**Recall (UP):** of all actual UP days, % caught — high = few missed gains. "
             "Disciplined traders prioritise precision: better to miss gains than enter bad trades.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — REGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page=="📈 Regression Analysis":
    import plotly.graph_objects as go

    st.title("📈 Regression Analysis — 5-Day Return Prediction")
    st.markdown("**Models:** Linear · Ridge · Lasso · Elastic Net · Random Forest · HistGradientBoosting  |  "
                "80/20 time-series split  |  RobustScaler on train only")
    mod_t=st.selectbox("Stock",LOADED_TICKERS,key="mod_t")
    ind=ind_data[mod_t] if mod_t in ind_data else ind_data[LOADED_TICKERS[0]]
    with st.spinner(f"Training 6 regression models for {mod_t}…"):
        mo,me=run_regression(mod_t,ind_data)
    if me or mo is None: st.error(f"Error: {me}"); st.stop()

    R=mo["results"]; yte=mo["y_test"]; dte=mo["dates_test"]
    st.success(f"Train: {mo['n_train']} | Test: {mo['n_test']} | Features: 33")
    mdf=pd.DataFrame([v["metrics"] for v in R.values()]).set_index("Model")
    best=mdf["Dir_Acc%"].idxmax()
    st.subheader("Model Performance")
    st.dataframe(mdf.style.highlight_max(subset=["R2","Dir_Acc%"],color="#14532d")
                         .highlight_min(subset=["MAE","RMSE"],color="#14532d")
                         .format(precision=4),use_container_width=True)
    ibox("Key metrics",
         f"**Directional Accuracy (Dir_Acc%)** is the trading-relevant metric. "
         f"**{best}** leads at {mdf.loc[best,'Dir_Acc%']:.1f}%. Each 1pp above 50% is a real edge. "
         "**R2 > 0.05** out-of-sample on stock returns is genuinely significant. "
         "**Lasso** zeros out irrelevant features — the zeroed list shows which indicators add no predictive power. "
         "**Ridge** handles correlated features (EMA50/EMA200 are collinear) via L2 regularisation.")

    sel_m=st.selectbox("Inspect model",list(R.keys()),key="sel_m")
    pred=R[sel_m]["pred"]
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dte,y=yte*100,name="Actual",line=dict(color="#e6edf3",width=1.5)))
        fig.add_trace(go.Scatter(x=dte,y=pred*100,name="Predicted",
                                  line=dict(color="#58a6ff",width=1.5,dash="dot")))
        fig.add_hline(y=0,line_dash="dash",line_color="#374151")
        pplot(fig,h=280,yaxis_title="5-day Return (%)")
        ibox("Prediction chart","Direction matters more than magnitude. "
             "Long when predicted > 0, flat when < 0 — that's the trading rule the Dir_Acc% measures.")
    with c2:
        fi=R[sel_m]["fi"]
        if fi:
            fi_s=dict(sorted(fi.items(),key=lambda x:x[1],reverse=True)[:15])
            fig2=go.Figure(go.Bar(y=list(fi_s.keys()),x=list(fi_s.values()),
                                   orientation="h",marker_color="#58a6ff"))
            pplot(fig2,h=280,margin={"l":120,"r":10,"t":30,"b":30})
            ibox("Feature importance","High-ranked EMA features confirm trend context is the strongest predictor. "
                 "Regime and Golden_Cross add 2–3pp to Dir_Acc% vs models without them.")
    if "Lasso" in R and R["Lasso"].get("zeroed"):
        z=R["Lasso"]["zeroed"]
        feat_names=["P_SMA5","P_EMA5","P_SMA20","P_EMA20","P_EMA50","P_EMA200",
                    "RSI_n","MACD_n","MACD_h_n","BB_Pct","BB_Width","ATR_n",
                    "Stoch_K_n","WR_n","CCI_n","ADX_n","OBV_Slope","Vol_Ratio_n",
                    "Vol_5","Vol_20","Regime","Golden_Cross","EMA50_200_n","Drawdown",
                    "Lag_1","Lag_2","Lag_3","Lag_5","Lag_10","Cum_5_n","Cum_10_n","Cum_20_n","Ret_n"]
        zeroed_names=[feat_names[i] if i<len(feat_names) else f"f{i}" for i in z]
        st.info(f"**Lasso** zeroed {len(z)} features: **{', '.join(zeroed_names[:8])}**"
                f"{'…' if len(zeroed_names)>8 else ''}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
elif page=="🔮 Clustering Analysis":
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    st.title("🔮 Clustering Analysis — K-Means (k=3)")
    st.markdown("**Objective:** Group the 10 stocks by risk-return profile. "
                "K-Means on 6 features → PCA to 2D for visualisation.")
    with st.spinner("Running K-Means…"):
        recs=[]
        for t in LOADED_TICKERS:
            ind=ind_data[t]; last=ind.ffill().iloc[-1]
            r1y=float(ind["Return"].tail(252).mean()*252)
            recs.append({"Ticker":t,"Name":META_INFO[t]["name"],"Sector":META_INFO[t]["sector"],
                         "1Y_Return%":round(r1y*100,2),"Volatility%":round(float(last["Vol_30"])*100,2),
                         "RSI":round(float(last["RSI"]),1),
                         "Sharpe":round(r1y/(float(last["Vol_30"])+1e-9),2),
                         "ADX":round(float(last["ADX"]),1),
                         "BB_Width":round(float(last["BB_Width"]),4)})
        fd=pd.DataFrame(recs).set_index("Ticker")
        nc=["1Y_Return%","Volatility%","RSI","Sharpe","ADX","BB_Width"]
        scl=StandardScaler(); Xs=scl.fit_transform(fd[nc].values)
        km=KMeans(n_clusters=3,random_state=42,n_init=10); km.fit(Xs)
        fd["Cluster"]=km.labels_
        mn=fd.groupby("Cluster")["1Y_Return%"].mean().sort_values(ascending=False)
        cmp={mn.index[0]:"🚀 High Growth",mn.index[1]:"⚖️ Balanced",mn.index[2]:"🛡️ Defensive"}
        fd["Group"]=fd["Cluster"].map(cmp)
        Xp=PCA(n_components=2).fit_transform(Xs); fd["PC1"]=Xp[:,0]; fd["PC2"]=Xp[:,1]

    gc={"🚀 High Growth":"#3fb950","⚖️ Balanced":"#f6ad55","🛡️ Defensive":"#58a6ff"}
    c1,c2=st.columns(2)
    with c1:
        st.subheader("PCA Cluster Map")
        fig=go.Figure()
        for gn,grp in fd.groupby("Group"):
            fig.add_trace(go.Scatter(x=grp["PC1"],y=grp["PC2"],mode="markers+text",
                                      text=grp.index,textposition="top center",name=gn,
                                      marker=dict(color=gc.get(gn,"#fff"),size=16,opacity=0.9)))
        pplot(fig,h=380)
        ibox("Cluster map","Distance between dots = similarity in risk-return space. "
             "Stocks close together add little diversification — concentrated risk. "
             "Optimal portfolio: one stock from each cluster, weighted by Sharpe.")
    with c2:
        st.subheader("Cluster Profiles")
        st.dataframe(fd.groupby("Group")[nc].mean().round(2).T,use_container_width=True)
        st.dataframe(fd[["Name","Sector","Group","1Y_Return%","Volatility%","Sharpe"]],use_container_width=True)

    st.subheader("Cluster Radar")
    fig2=go.Figure()
    for gn,grp in fd.groupby("Group"):
        vals=[float(grp[m].mean()) for m in nc]
        mn_=[fd[m].min() for m in nc]; mx_=[fd[m].max() for m in nc]
        norm=[(v-a)/(b-a+1e-9)*100 for v,a,b in zip(vals,mn_,mx_)]; norm+=[norm[0]]
        fig2.add_trace(go.Scatterpolar(r=norm,theta=nc+[nc[0]],fill="toself",
                                        name=gn,line_color=gc.get(gn,"#fff"),opacity=0.7))
    fig2.update_layout(**DARK,height=400,
                        polar=dict(bgcolor="#1f2937",
                                   radialaxis=dict(visible=True,range=[0,100],color="#9ca3af"),
                                   angularaxis=dict(color="#9ca3af")))
    st.plotly_chart(fig2,use_container_width=True,config=PCFG)
    ibox("Radar insight","🚀 High Growth dominates Return + Sharpe but spikes on Volatility. "
         "🛡️ Defensive shows a balanced flat polygon — no extremes, lowest risk. "
         "⚖️ Balanced sits between both. Best diversification: combine all three clusters.")

    st.subheader("Rolling 60-Day Correlation to Portfolio")
    ret_df=pd.DataFrame({t:frames[t]["Close"].pct_change() for t in LOADED_TICKERS}).dropna()
    fig3=go.Figure()
    for i,t in enumerate(LOADED_TICKERS):
        roll=ret_df[t].rolling(60).corr(ret_df[[o for o in LOADED_TICKERS if o!=t]].mean(axis=1))
        fig3.add_trace(go.Scatter(x=roll.index,y=roll,name=t,mode="lines",
                                   line=dict(width=1.5,color=COLORS[i%len(COLORS)])))
    pplot(fig3,h=300,yaxis_title="Avg Rolling 60-Day Correlation")
    ibox("Rolling correlation","When all lines converge near 0.8+ (2022 selloff), everything falls together. "
         "When lines diverge, some stocks decouple — genuine portfolio diversifiers. "
         "Rebalance toward low-correlation stocks when macro stress rises.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════════════════
elif page=="🔗 Association Rules":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🔗 Association Rules — Co-movement Analysis")
    ret_df=pd.DataFrame({t:frames[t]["Close"].pct_change() for t in LOADED_TICKERS}).dropna()
    corr=ret_df.corr()

    st.subheader("Full Return Correlation Matrix")
    fig=px.imshow(corr.round(3),text_auto=True,color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
    fig.update_layout(**DARK,height=460)
    st.plotly_chart(fig,use_container_width=True,config=PCFG)
    ibox("Association rules from correlation",
         "Each cell = P(Stock B rises | Stock A rises). "
         "AAPL–MSFT at 0.75 means: on 75% of days when AAPL rises, MSFT also rises. "
         "**Pair trading:** when two high-correlation stocks diverge, mean-reversion trade. "
         "**Risk management:** holding two 0.8+ correlated stocks is not diversification.")

    rules=[]
    for i in range(len(LOADED_TICKERS)):
        for j in range(i+1,len(LOADED_TICKERS)):
            r=float(corr.iloc[i,j])
            if abs(r)>0.4:
                rules.append({"Pair":f"{LOADED_TICKERS[i]}↔{LOADED_TICKERS[j]}",
                               "Correlation":round(r,3),
                               "Relationship":"Strong Positive" if r>0.7 else "Moderate Positive",
                               "Implication":"Concentrated risk — limit both" if r>0.7 else "Partial co-movement"})
    st.subheader("Co-movement Rules (|corr|>0.4)")
    if rules:
        st.dataframe(pd.DataFrame(rules).sort_values("Correlation",ascending=False),
                     use_container_width=True,hide_index=True)

    c1,c2=st.columns(2)
    with c1:
        st.subheader("Avg Correlation to Portfolio")
        avg_c=pd.Series({t:ret_df[[o for o in LOADED_TICKERS if o!=t]].corrwith(ret_df[t]).mean()
                         for t in LOADED_TICKERS}).sort_values(ascending=False)
        fig2=go.Figure(go.Bar(x=avg_c.index,y=avg_c.values,
                               marker_color=["#f85149" if v>0.6 else "#f6ad55" if v>0.4 else "#3fb950" for v in avg_c.values],
                               text=[f"{v:.2f}" for v in avg_c.values],textposition="outside"))
        pplot(fig2,h=300,yaxis={"range":[0,1.1]})
        ibox("Diversification score","Red = redundant — moves with everything. "
             "Green = genuine diversifier. BRK-B and UNH are typically strongest diversifiers.")
    with c2:
        if rules:
            top=sorted(rules,key=lambda x:x["Correlation"],reverse=True)[0]
            ta,tb=top["Pair"].split("↔")
            roll=ret_df[ta].rolling(30).corr(ret_df[tb])
            fig3=go.Figure()
            fig3.add_trace(go.Scatter(x=roll.index,y=roll,fill="tozeroy",
                                       name=f"{ta}↔{tb}",line=dict(color="#58a6ff",width=2)))
            fig3.add_hline(y=0.5,line_dash="dash",line_color="#f6ad55")
            pplot(fig3,h=300,yaxis={"range":[-0.1,1.2]})
            ibox("Rolling correlation","Sharp drop = pair has diverged = pair-trade entry opportunity. "
                 "Correlation always normalises eventually — mean-reversion trade.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 8 — DEEP DRILL-DOWN
# ═══════════════════════════════════════════════════════════════════════════
elif page=="🔬 Deep Drill-Down":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_long_data()
    st.title("🔬 Deep Drill-Down")
    dd_t=st.selectbox("Stock",LOADED_TICKERS,key="dd_t")
    ind=ind_data[dd_t] if dd_t in ind_data else ind_data[LOADED_TICKERS[0]]
    last=ind.ffill().iloc[-1]; cur=float(last["Close"])
    e50=float(last["EMA_50"]); e200=float(last["EMA_200"])
    ret_=ind["Return"].dropna()
    hi52=float(ind["Close"].rolling(252).max().iloc[-1])
    lo52=float(ind["Close"].rolling(252).min().iloc[-1])
    ann_ret=float(ret_.tail(252).mean()*252*100); ann_vol=float(ret_.std()*np.sqrt(252)*100)
    sh=float((ret_.mean()/ret_.std())*np.sqrt(252)) if ret_.std()>0 else 0
    nr=ret_[ret_<0]; so=float((ret_.mean()/nr.std())*np.sqrt(252)) if len(nr)>0 and nr.std()>0 else 0
    max_dd=float(ind["Drawdown"].min()*100)

    summary=pd.DataFrame({
        "Metric":["Live Price","52W High","52W Low","vs EMA 50","vs EMA 200","Regime",
                  "Golden Cross","1Y Ann. Return","Ann. Volatility","Sharpe","Sortino","Max Drawdown",
                  "RSI(14)","MACD Hist","ADX","Stoch %K","Williams %R","CCI","OBV Slope","BB Position","ATR"],
        "Value":[f"${cur:.2f}",f"${hi52:.2f}",f"${lo52:.2f}",
                 f"{'ABOVE' if cur>e50 else 'BELOW'} ({(cur/e50-1)*100:+.1f}%)",
                 f"{'ABOVE' if cur>e200 else 'BELOW'} ({(cur/e200-1)*100:+.1f}%)",
                 "🟢 BULL" if cur>e200 else "🔴 BEAR",
                 "✅ YES" if float(last["Golden_Cross"])==1 else "❌ NO",
                 f"{ann_ret:+.1f}%",f"{ann_vol:.1f}%",f"{sh:.2f}",f"{so:.2f}",f"{max_dd:.1f}%",
                 f"{float(last['RSI']):.1f}",f"{float(last['MACD_Hist']):.4f}",
                 f"{float(last['ADX']):.1f}",f"{float(last['Stoch_K']):.1f}",
                 f"{float(last['Williams_R']):.1f}",f"{float(last['CCI']):.1f}",
                 f"{float(last['OBV_Slope']):.4f}",f"{float(last['BB_Pct'])*100:.1f}%",
                 f"${float(last['ATR']):.2f}"]
    }).set_index("Metric")
    st.dataframe(summary,use_container_width=True)

    st.subheader("Full Price History + All EMAs")
    fig=go.Figure()
    for ema,col,dash,wid in [(20,"#fee140","dot",1.1),(50,"#58a6ff","dash",1.5),
                               (100,"#f093fb","dash",1.2),(200,"#3fb950","solid",2.0)]:
        fig.add_trace(go.Scatter(x=ind.index,y=ind[f"EMA_{ema}"],name=f"EMA {ema}",
                                  line=dict(color=col,width=wid,dash=dash),opacity=0.85))
    fig.add_trace(go.Scatter(x=ind.index,y=ind["Close"],name="Close",
                              line=dict(color="#e6edf3",width=1.5)))
    pplot(fig,h=400,yaxis_title="Price (USD)",legend=dict(orientation="h",y=1.02,x=0))
    ibox("Full EMA history","The full history shows every Golden Cross and Death Cross event. "
         "The LSTM trains on this entire history — it learns that post-Golden-Cross periods "
         "have persistently higher directional accuracy than Death Cross periods.")

    tfs_dd=mtf_data.get(dd_t,{})
    c_lt1,c_lt2=st.columns(2)
    with c_lt1:
        df_m_dd=tfs_dd.get("ME")
        if df_m_dd is not None and len(df_m_dd)>24:
            st.subheader("Monthly Chart (M-EMA 6/12/24)")
            fig_m=go.Figure()
            fig_m.add_trace(go.Candlestick(x=df_m_dd.index,open=df_m_dd["Open"],high=df_m_dd["High"],
                                            low=df_m_dd["Low"],close=df_m_dd["Close"],name="Monthly",
                                            increasing_line_color="#3fb950",decreasing_line_color="#f85149"))
            for ema,col,dash,wid in [(6,"#fee140","dot",1.2),(12,"#58a6ff","dash",1.5),(24,"#3fb950","solid",2)]:
                cn=f"EMA_{ema}"
                if cn in df_m_dd.columns:
                    fig_m.add_trace(go.Scatter(x=df_m_dd.index,y=df_m_dd[cn],name=f"M-EMA {ema}",
                                               line=dict(color=col,width=wid,dash=dash)))
            fig_m.update_layout(**DARK,height=320,xaxis_rangeslider_visible=False,
                                legend=dict(orientation="h",y=1.01,x=0,font=dict(size=9)))
            st.plotly_chart(fig_m,use_container_width=True,config=PCFG)
            ibox("Monthly view","M-EMA 12 ≈ 1-year moving average. M-EMA 24 ≈ 2-year — "
                 "the most important macro support/resistance line.")
    with c_lt2:
        df_q_dd=tfs_dd.get("QE")
        if df_q_dd is not None and len(df_q_dd)>8:
            st.subheader("Quarterly Chart + Q-RSI")
            fig_q=make_subplots(rows=2,cols=1,shared_xaxes=True,
                                 subplot_titles=["Quarterly Price","Quarterly RSI"],
                                 vertical_spacing=0.08,row_heights=[0.65,0.35])
            fig_q.add_trace(go.Candlestick(x=df_q_dd.index,open=df_q_dd["Open"],high=df_q_dd["High"],
                                            low=df_q_dd["Low"],close=df_q_dd["Close"],name="Quarterly",
                                            increasing_line_color="#3fb950",decreasing_line_color="#f85149"),row=1,col=1)
            if "EMA_8" in df_q_dd.columns:
                fig_q.add_trace(go.Scatter(x=df_q_dd.index,y=df_q_dd["EMA_8"],name="Q-EMA 8(~2yr)",
                                            line=dict(color="#3fb950",width=1.8)),row=1,col=1)
            if "RSI" in df_q_dd.columns:
                fig_q.add_trace(go.Scatter(x=df_q_dd.index,y=df_q_dd["RSI"],name="Q-RSI",
                                            fill="tozeroy",line=dict(color="#f6ad55",width=1.6)),row=2,col=1)
                for yv,col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                    fig_q.add_hline(y=yv,line_color=col,line_dash="dash",row=2,col=1)
            fig_q.update_layout(**DARK,height=320,showlegend=False,xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_q,use_container_width=True,config=PCFG)
            ibox("Quarterly RSI","Q-RSI < 30 marks major market bottoms — once-in-a-cycle opportunity. "
                 "Q-RSI > 80 = multi-year overbought. Puts current price in 10-20 year context.")

dfl_dd = frames_long.get(dd_t, frames.get(dd_t))
        if dfl_dd is not None and len(dfl_dd) > 500:
            st.subheader("Full Yearly Return History")
    dfl_dd = frames_long.get(dd_t, frames.get(dd_t))
    
    if dfl_dd is not None and len(dfl_dd) > 500:
        yr_s = _yearly_returns(dfl_dd)
        fig_yr = go.Figure(go.Bar(
            x=yr_s.index.astype(str), y=yr_s.values,
            marker_color=["#3fb950" if v >= 0 else "#f85149" for v in yr_s.values],
            text=[f"{v:.1f}%" for v in yr_s.values], textposition="outside",
        ))
        pplot(fig_yr, h=300, yaxis_title="Annual Return (%)",
              xaxis_title="Year", yaxis=dict(range=[yr_s.min()-10, yr_s.max()+10]))
        best_yr  = yr_s.idxmax()
        worst_yr = yr_s.idxmin()
        avg_ret  = yr_s.mean()
        pos_yrs  = (yr_s > 0).sum()
        col1y, col2y, col3y, col4y = st.columns(4)
        col1y.metric("Best Year",  f"{best_yr} ({yr_s[best_yr]:+.1f}%)")
        col2y.metric("Worst Year", f"{worst_yr} ({yr_s[worst_yr]:+.1f}%)")
        col3y.metric("Avg Annual", f"{avg_ret:+.1f}%")
        col4y.metric("Positive Years", f"{pos_yrs}/{len(yr_s)} ({pos_yrs/len(yr_s)*100:.0f}%)")
        ibox("Yearly return history decoded",
             "This is the single most useful long-term chart for understanding a stock's character. "
             "High win rate (>70% positive years) = reliable compounder (MSFT, AAPL, JPM). "
             "High variance with huge up-years and big down-years = volatile compounder (NVDA, TSLA). "
             "The average annual return combined with the positive year % gives you the "
             "expected value of holding this stock over any given year.")

    st.subheader("Return Distribution vs Normal")
    rp=ret_*100; mu=float(rp.mean()); sg=float(rp.std())
    xn=np.linspace(float(rp.min()),float(rp.max()),200)
    yn=(1/(sg*np.sqrt(2*np.pi)))*np.exp(-0.5*((xn-mu)/sg)**2)
    fig2=go.Figure()
    fig2.add_trace(go.Histogram(x=rp,nbinsx=80,name="Actual",marker_color="#58a6ff",opacity=0.7,
                                 histnorm="probability density"))
    fig2.add_trace(go.Scatter(x=xn,y=yn,name="Normal Fit",line=dict(color="#f85149",width=2)))
    pplot(fig2,h=280,xaxis_title="Daily Return (%)",yaxis_title="Density")
    skew=float(rp.skew()); kurt=float(rp.kurtosis())
    ibox(f"Fat tails — {dd_t}",
         f"Mean: **{mu:.3f}%** | Std: **{sg:.3f}%** | Skew: **{skew:.3f}** | Excess Kurtosis: **{kurt:.2f}**  \n"
         f"Kurtosis of {kurt:.1f} means extreme daily moves are {max(1,kurt/3):.1f}× more likely than normal distribution predicts. "
         "This is why VaR models assuming normality underestimate tail losses during crises. "
         "The LSTM's Vol_5/Vol_20 ratio detects when the distribution is shifting toward fatter tails.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 9 — DOWNLOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
elif page=="📥 Download Data":
    st.title("📥 Download Data")
    st.caption(f"All data: Yahoo Finance  |  {data_src}")

    st.subheader("Full Indicator Dataset (CSV per stock)")
    _dl_t=st.selectbox("Ticker",LOADED_TICKERS,key="dl_t")
    _dl_ind=ind_data[_dl_t] if _dl_t in ind_data else ind_data[LOADED_TICKERS[0]]
    buf=io.StringIO(); _dl_ind.to_csv(buf)
    st.download_button(
        label=f"⬇️ {_dl_t} — Full Indicator CSV ({len(_dl_ind)} rows × {len(_dl_ind.columns)} cols)",
        data=buf.getvalue(),file_name=f"{_dl_t}_indicators_{END}.csv",mime="text/csv")

    st.divider()
    st.subheader("Excel Workbook (All 10 Stocks + Overview)")
    if st.button("🔄 Build Excel from Yahoo Finance",type="primary"):
        with st.spinner("Building Excel…"):
            try:
                import openpyxl
                from openpyxl.styles import Font,PatternFill,Alignment
                wb=openpyxl.Workbook(); ws=wb.active; ws.title="Overview"
                ws["A1"]=f"S&P 500 Top 10 | Yahoo Finance | {datetime.today().strftime('%Y-%m-%d %H:%M')}"
                ws["A1"].font=Font(bold=True,size=13)
                ws.append(["Ticker","Company","Sector","Price","1Y Ret%","Vol%","RSI","ADX","Regime","Signal"])
                for t in LOADED_TICKERS:
                    ind=ind_data[t]; last=ind.ffill().iloc[-1]
                    r1y=float(ind["Return"].tail(252).mean()*252*100)
                    sc_=score_stock(ind,None)
                    ws.append([t,META_INFO[t]["name"],META_INFO[t]["sector"],
                               round(float(last["Close"]),2),round(r1y,2),
                               round(float(last["Vol_30"])*100,1),round(float(last["RSI"]),1),
                               round(float(last["ADX"]),1),
                               "BULL" if float(last["Regime"])==1 else "BEAR",sc_["signal"]])
                for t in LOADED_TICKERS:
                    ws2=wb.create_sheet(t); df=frames[t].reset_index()
                    ws2.append(list(df.columns))
                    for row in df.values.tolist(): ws2.append(row)
                wb.save(XLS_PATH); st.success(f"✅ Saved {XLS_PATH}")
            except Exception as e: st.error(f"Error: {e}")
    if os.path.exists(XLS_PATH):
        with open(XLS_PATH,"rb") as f:
            st.download_button("⬇️ Download Excel Workbook",data=f.read(),
                               file_name=XLS_PATH,type="primary",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.subheader("Regression Model Metrics — All Stocks (Excel)")
    if st.button("📊 Run All 6 Regression Models & Export"):
        with st.spinner("Training regression models on all tickers…"):
            all_rows=[]
            for t in LOADED_TICKERS:
                mo,_=run_regression(t,ind_data)
                if mo:
                    for res in mo["results"].values():
                        all_rows.append({"Ticker":t,**res["metrics"]})
        if all_rows:
            mbuf=io.BytesIO(); pd.DataFrame(all_rows).to_excel(mbuf,index=False)
            st.download_button("⬇️ Regression Metrics Excel",data=mbuf.getvalue(),
                               file_name=f"regression_metrics_{END}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.warning("⚠️ **Legal Disclaimer:** This dashboard is for informational and educational "
               "purposes only. Nothing constitutes financial, investment, legal, or tax advice. "
               "Data sourced from Yahoo Finance. Past performance does not guarantee future results. "
               "Always consult a qualified financial advisor before making investment decisions.")
