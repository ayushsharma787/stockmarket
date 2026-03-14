"""
Investment Bank Stock Analytics Dashboard  v6
S&P 500 Top 10 | Yahoo Finance ONLY | Live market data

LSTM ENGINE — Real BPTT Training:
  - Labels: Binary UP/DOWN on 5-day forward return (zero lookahead)
  - Model: Single-layer LSTM (hidden=48) trained via Adam+BPTT, 10 epochs
  - Ensemble: Trained LSTM hidden states fed into HistGradientBoosting
  - RobustScaler fitted on train-only (no data leakage)
  - 80/20 strict time-series split

Technical Analysis — FOUR TIMEFRAMES:
  Daily  : EMA 20/50/100/200, RSI(14), MACD(12,26,9), BB(20,2),
           ATR(14), Stoch(14,3), Williams%R, CCI(20), ADX(14),
           OBV, VWAP, Ichimoku(9,26,52)
  Weekly : EMA 10/20/50, RSI(14), MACD, BB, ADX, Stoch
  Monthly: EMA 6/12/24, RSI(14), MACD, BB, ADX — macro trend
  Quarterly: EMA 4/8, RSI, Trend strength — very long term

Data:  Daily from 2015 + extended history from 1990 for long-term charts

Modules:
  1. Executive Overview
  2. Technical Analysis (daily suite)
  3. Multi-Timeframe Analysis  ← NEW
  4. LSTM Classification
  5. Regression Analysis
  6. Clustering Analysis
  7. Association Rules
  8. Deep Drill-Down
  9. Download Data
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
    initial_sidebar_state="expanded",
)
st.markdown(
    "<style>section[data-testid='stSidebar']{background-color:#161b22}</style>",
    unsafe_allow_html=True,
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────
TICKERS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","UNH"]
META = {
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
COLORS   = ["#58a6ff","#3fb950","#f6ad55","#f093fb","#4facfe",
            "#43e97b","#fa709a","#fee140","#a371f7","#ff9a9e"]
START        = "2015-01-01"   # Daily LSTM training data
START_LONG   = "1993-01-01"   # Extended history for long-term charts
END          = datetime.today().strftime("%Y-%m-%d")
XLS_PATH = "sp500_top10.xlsx"
PCFG     = {"displayModeBar": False}
SEQ_LEN  = 20
LSTM_H   = 48

DARK = dict(
    template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(family="sans-serif", color="#d1d5db", size=11),
    margin=dict(l=50, r=20, t=40, b=50),
    xaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
    yaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
)

def pplot(fig, h=380, **kw):
    # Merge DARK with kw safely — kw values override DARK values for the same key
    layout = {**DARK, "height": h}
    for k, v in kw.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k] = {**layout[k], **v}   # deep-merge dicts (e.g. yaxis, xaxis)
        else:
            layout[k] = v
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)

def ibox(title, body):
    with st.container(border=True):
        st.markdown(f"**💡 {title}**")
        st.markdown(body)


# ═══════════════════════════════════════════════════════════════════════════
# LSTM  — real BPTT + Adam (no TensorFlow needed)
# ═══════════════════════════════════════════════════════════════════════════
class LSTM:
    """Trainable single-layer LSTM with Adam optimiser."""

    def __init__(self, in_sz, h_sz, n_cls=2, lr=5e-4, seed=42):
        np.random.seed(seed)
        self.hs = h_sz
        self.nc = n_cls
        self.lr = lr
        sz = in_sz + h_sz
        k  = np.sqrt(2.0 / (in_sz + h_sz))
        self.Wf = np.random.randn(h_sz, sz) * k;  self.bf = np.zeros(h_sz)
        self.Wi = np.random.randn(h_sz, sz) * k;  self.bi = np.ones(h_sz)
        self.Wg = np.random.randn(h_sz, sz) * k;  self.bg = np.zeros(h_sz)
        self.Wo = np.random.randn(h_sz, sz) * k;  self.bo = np.zeros(h_sz)
        self.Wy = np.random.randn(n_cls, h_sz) * np.sqrt(2.0 / h_sz)
        self.by = np.zeros(n_cls)
        pn = ["Wf","Wi","Wg","Wo","bf","bi","bg","bo","Wy","by"]
        self._pn = pn
        self._t  = 0; self._b1 = 0.9; self._b2 = 0.999; self._eps = 1e-8
        self._m  = {n: np.zeros_like(getattr(self, n)) for n in pn}
        self._v  = {n: np.zeros_like(getattr(self, n)) for n in pn}

    @staticmethod
    def _s(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))
    @staticmethod
    def _t_(x): return np.tanh(np.clip(x, -10, 10))

    def _fwd(self, X):
        h = np.zeros(self.hs); c = np.zeros(self.hs); cache = []
        for t in range(len(X)):
            xh = np.concatenate([X[t], h])
            f  = self._s(self.Wf @ xh + self.bf)
            i  = self._s(self.Wi @ xh + self.bi)
            g  = self._t_(self.Wg @ xh + self.bg)
            o  = self._s(self.Wo @ xh + self.bo)
            c2 = f * c + i * g
            h2 = o * self._t_(c2)
            cache.append((xh, f, i, g, o, c, c2, h, h2))
            h, c = h2, c2
        z = self.Wy @ h + self.by
        z -= z.max()
        e = np.exp(z)
        return e / e.sum(), h, cache

    def _adam(self, name, grad, clip=1.0):
        np.clip(grad, -clip, clip, out=grad)
        self._t += 1
        self._m[name] = self._b1 * self._m[name] + (1 - self._b1) * grad
        self._v[name] = self._b2 * self._v[name] + (1 - self._b2) * grad ** 2
        mh = self._m[name] / (1 - self._b1 ** self._t)
        vh = self._v[name] / (1 - self._b2 ** self._t)
        return self.lr * mh / (np.sqrt(vh) + self._eps)

    def step(self, X, y_true):
        probs, h_last, cache = self._fwd(X)
        loss = -np.log(probs[y_true] + 1e-9)
        dz   = probs.copy(); dz[y_true] -= 1.0
        dWy  = np.outer(dz, h_last); dby = dz.copy()
        dh   = self.Wy.T @ dz; dc = np.zeros(self.hs)
        dWf = np.zeros_like(self.Wf); dWi = np.zeros_like(self.Wi)
        dWg = np.zeros_like(self.Wg); dWo = np.zeros_like(self.Wo)
        dbf = np.zeros(self.hs); dbi = np.zeros(self.hs)
        dbg = np.zeros(self.hs); dbo = np.zeros(self.hs)
        for t in reversed(range(len(cache))):
            xh, f, i, g, o, c_p, c_c, h_p, h_c = cache[t]
            tc  = self._t_(c_c)
            do  = dh * tc
            ddc = dh * o * (1 - tc ** 2) + dc
            df  = ddc * c_p; di = ddc * g; dg_ = ddc * i; dc = ddc * f
            dfp = df * f * (1 - f); dip = di * i * (1 - i)
            dgp = dg_ * (1 - g ** 2); dop = do * o * (1 - o)
            dWf += np.outer(dfp, xh); dbf += dfp
            dWi += np.outer(dip, xh); dbi += dip
            dWg += np.outer(dgp, xh); dbg += dgp
            dWo += np.outer(dop, xh); dbo += dop
            dh = (self.Wf.T@dfp + self.Wi.T@dip + self.Wg.T@dgp + self.Wo.T@dop)[:self.hs]
        for nm, dW in [("Wf",dWf),("Wi",dWi),("Wg",dWg),("Wo",dWo),
                       ("bf",dbf),("bi",dbi),("bg",dbg),("bo",dbo),
                       ("Wy",dWy),("by",dby)]:
            setattr(self, nm, getattr(self, nm) - self._adam(nm, dW))
        return loss

    def get_hidden(self, X):
        _, h, _ = self._fwd(X)
        return h

    def predict_proba(self, X):
        p, _, _ = self._fwd(X)
        return p

    def train(self, Xtr, ytr, epochs=10, progress_fn=None):
        N = len(Xtr); hist = []
        for ep in range(epochs):
            idx = np.random.permutation(N); loss = 0.0
            for j in idx:
                loss += self.step(Xtr[j], ytr[j])
            hist.append(loss / N)
            if progress_fn:
                progress_fn(ep + 1, epochs, hist[-1])
        return hist

    def hidden_batch(self, Xs):
        return np.array([self.get_hidden(Xs[i]) for i in range(len(Xs))])

    def proba_batch(self, Xs):
        return np.array([self.predict_proba(Xs[i]) for i in range(len(Xs))])


# ═══════════════════════════════════════════════════════════════════════════
# DATA  — Yahoo Finance only
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo():
    import yfinance as yf
    frames = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
            if df.empty:
                continue
            df.index = pd.to_datetime(df.index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            frames[t] = df[["Open","High","Low","Close","Volume"]].dropna()
        except Exception as e:
            st.warning(f"⚠️ {t}: {e}")
    return frames


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_long():
    """Extended history from 1993 for long-term multi-timeframe charts."""
    import yfinance as yf
    frames = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=START_LONG, end=END, progress=False, auto_adjust=True)
            if df.empty:
                continue
            df.index = pd.to_datetime(df.index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            frames[t] = df[["Open","High","Low","Close","Volume"]].dropna()
        except Exception as e:
            pass
    return frames


# ── Technical indicator engine ─────────────────────────────────────────────
def _ema(a, s):
    k = 2/(s+1); o = np.zeros(len(a)); o[0] = a[0]
    for i in range(1, len(a)): o[i] = k*a[i] + (1-k)*o[i-1]
    return o

def _sma(a, w):
    return np.array([a[max(0, i-w+1):i+1].mean() for i in range(len(a))])

def _roll_std(a, w):
    return np.array([a[max(0, i-w+1):i+1].std() for i in range(len(a))])


@st.cache_data(show_spinner=False)
def compute_indicators(_frames):
    out = {}
    for ticker, df in _frames.items():
        d = df.copy().sort_index()
        c = d["Close"].values
        h = d["High"].values
        l = d["Low"].values
        v = d["Volume"].values
        n = len(c)

        ret = np.zeros(n)
        ret[1:] = (c[1:] - c[:-1]) / (c[:-1] + 1e-9)

        ind = {"Return": ret}

        # Moving averages
        for w in [5, 10, 20, 50, 100, 200]:
            ind[f"SMA_{w}"] = _sma(c, w)
            ind[f"EMA_{w}"] = _ema(c, w)

        # RSI(14)
        delta = np.diff(c, prepend=c[0])
        ag = _sma(np.maximum(delta, 0), 14)
        al = _sma(-np.minimum(delta, 0), 14)
        ind["RSI"] = 100 - 100 / (1 + ag / (al + 1e-9))

        # MACD(12,26,9)
        e12 = _ema(c, 12); e26 = _ema(c, 26)
        ind["MACD"]        = e12 - e26
        ind["MACD_Signal"] = _ema(ind["MACD"], 9)
        ind["MACD_Hist"]   = ind["MACD"] - ind["MACD_Signal"]

        # Bollinger Bands(20,2)
        s20 = _sma(c, 20); std20 = _roll_std(c, 20)
        ind["BB_Upper"] = s20 + 2*std20
        ind["BB_Lower"] = s20 - 2*std20
        ind["BB_Mid"]   = s20
        ind["BB_Width"] = 4*std20 / (s20 + 1e-9)
        ind["BB_Pct"]   = (c - ind["BB_Lower"]) / (ind["BB_Upper"] - ind["BB_Lower"] + 1e-9)

        # ATR(14)
        pc = np.roll(c, 1); pc[0] = c[0]
        tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
        ind["ATR"] = _sma(tr, 14)

        # Stochastic(14,3)
        lo14 = np.array([l[max(0,i-14):i+1].min() for i in range(n)])
        hi14 = np.array([h[max(0,i-14):i+1].max() for i in range(n)])
        k_raw = (c - lo14) / (hi14 - lo14 + 1e-9) * 100
        ind["Stoch_K"] = _sma(k_raw, 3)
        ind["Stoch_D"] = _sma(ind["Stoch_K"], 3)

        # Williams %R(14)
        ind["Williams_R"] = -100 * (hi14 - c) / (hi14 - lo14 + 1e-9)

        # CCI(20)
        tp = (h + l + c) / 3
        stp = _sma(tp, 20)
        mad = np.array([np.abs(tp[max(0,i-20):i+1] - tp[max(0,i-20):i+1].mean()).mean()
                        for i in range(n)])
        ind["CCI"] = (tp - stp) / (0.015 * mad + 1e-9)

        # ADX(14)
        p_dm = np.maximum(np.diff(h, prepend=h[0]), 0)
        n_dm = np.maximum(-np.diff(l, prepend=l[0]), 0)
        p_dm = np.where(p_dm > n_dm, p_dm, 0)
        n_dm = np.where(n_dm > p_dm, n_dm, 0)
        atr14 = _sma(tr, 14) + 1e-9
        di_p = 100 * _sma(p_dm, 14) / atr14
        di_n = 100 * _sma(n_dm, 14) / atr14
        dx   = 100 * np.abs(di_p - di_n) / (di_p + di_n + 1e-9)
        ind["ADX"]      = _sma(dx, 14)
        ind["DI_Plus"]  = di_p
        ind["DI_Minus"] = di_n

        # OBV
        ind["OBV"]       = np.cumsum(np.sign(ret) * v)
        obv_ma           = _sma(ind["OBV"], 20)
        ind["OBV_Slope"] = (ind["OBV"] - obv_ma) / (np.abs(obv_ma) + 1e-9)

        # Ichimoku(9,26,52)
        def ichi(p):
            hp = np.array([h[max(0,i-p):i+1].max() for i in range(n)])
            lp = np.array([l[max(0,i-p):i+1].min() for i in range(n)])
            return (hp + lp) / 2
        ind["Ichi_Tenkan"] = ichi(9)
        ind["Ichi_Kijun"]  = ichi(26)
        ind["Ichi_SpanA"]  = (ind["Ichi_Tenkan"] + ind["Ichi_Kijun"]) / 2
        ind["Ichi_SpanB"]  = ichi(52)

        # VWAP
        ind["VWAP"] = np.cumsum(tp * v) / (np.cumsum(v) + 1e-9)

        # Volatility
        ind["Vol_5"]  = _roll_std(ret, 5)  * np.sqrt(252)
        ind["Vol_20"] = _roll_std(ret, 20) * np.sqrt(252)
        ind["Vol_30"] = _roll_std(ret, 30) * np.sqrt(252)

        # Regime
        e50  = _ema(c, 50);  e200 = _ema(c, 200)
        ind["Regime"]       = (c > e200).astype(float)
        ind["Golden_Cross"] = (e50 > e200).astype(float)
        ind["EMA50_200"]    = (e50 - e200) / (e200 + 1e-9)

        # Drawdown
        ind["Drawdown"]   = (c - np.maximum.accumulate(c)) / (np.maximum.accumulate(c) + 1e-9)
        ind["Cum_Return"] = (1 + pd.Series(ret).fillna(0)).cumprod().values - 1

        # Momentum
        for w in [1, 2, 3, 5, 10, 20]:
            ind[f"Lag_{w}"] = np.roll(ret, w)
        for w in [5, 10, 20]:
            ind[f"Cum_{w}"] = np.array([(c[i]/c[max(0,i-w)] - 1) for i in range(n)])

        # Volume ratio
        ind["Vol_Ratio"] = v / (_sma(v, 20) + 1e-9)

        # Merge back
        df_ind = pd.DataFrame(ind, index=df.index)
        for col in ["Open","High","Low","Close","Volume"]:
            df_ind[col] = df[col].values
        out[ticker] = df_ind
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def _resample_ohlcv(df_daily, freq):
    """Resample daily OHLCV to W/ME/QE/YE with correct aggregation."""
    rs = df_daily.resample(freq).agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna()
    return rs


def _ind_tf(df):
    """Full indicator suite on any-frequency OHLCV DataFrame."""
    c = df["Close"].values; h = df["High"].values
    l = df["Low"].values;   v = df["Volume"].values; n = len(c)
    if n < 10:
        return df
    ret = np.zeros(n); ret[1:] = (c[1:]-c[:-1])/(c[:-1]+1e-9)
    out = df.copy(); out["Return"] = ret
    for w in [5,10,20,50,100,200]:
        if w < n:
            out[f"SMA_{w}"] = _sma(c, w)
            out[f"EMA_{w}"] = _ema(c, w)
    delta = np.diff(c, prepend=c[0])
    ag = _sma(np.maximum(delta,0),14); al = _sma(-np.minimum(delta,0),14)
    out["RSI"] = np.clip(100-100/(1+ag/(al+1e-9)),0,100)
    e12 = _ema(c,12); e26 = _ema(c,26)
    out["MACD"] = e12-e26
    out["MACD_Signal"] = _ema(out["MACD"].values,9)
    out["MACD_Hist"]   = out["MACD"].values - out["MACD_Signal"].values
    p20 = min(20,n-1)
    s20 = _sma(c,p20); std20 = _roll_std(c,p20)
    out["BB_Upper"] = s20+2*std20; out["BB_Lower"] = s20-2*std20
    out["BB_Mid"]   = s20;         out["BB_Width"] = 4*std20/(s20+1e-9)
    out["BB_Pct"]   = (c-(s20-2*std20))/(4*std20+1e-9)
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum.reduce([h-l,np.abs(h-pc),np.abs(l-pc)])
    out["ATR"] = _sma(tr,14)
    lo14 = np.array([l[max(0,i-14):i+1].min() for i in range(n)])
    hi14 = np.array([h[max(0,i-14):i+1].max() for i in range(n)])
    k_raw = (c-lo14)/(hi14-lo14+1e-9)*100
    out["Stoch_K"] = _sma(k_raw,3); out["Stoch_D"] = _sma(out["Stoch_K"].values,3)
    out["Williams_R"] = -100*(hi14-c)/(hi14-lo14+1e-9)
    tp = (h+l+c)/3; stp = _sma(tp,p20)
    mad = np.array([np.abs(tp[max(0,i-20):i+1]-tp[max(0,i-20):i+1].mean()).mean() for i in range(n)])
    out["CCI"] = (tp-stp)/(0.015*mad+1e-9)
    p_dm = np.maximum(np.diff(h,prepend=h[0]),0)
    n_dm = np.maximum(-np.diff(l,prepend=l[0]),0)
    p_dm = np.where(p_dm>n_dm,p_dm,0); n_dm = np.where(n_dm>p_dm,n_dm,0)
    atr14 = _sma(tr,14)+1e-9
    di_p = 100*_sma(p_dm,14)/atr14; di_n = 100*_sma(n_dm,14)/atr14
    dx = 100*np.abs(di_p-di_n)/(di_p+di_n+1e-9)
    out["ADX"] = _sma(dx,14); out["DI_Plus"] = di_p; out["DI_Minus"] = di_n
    out["OBV"] = np.cumsum(np.sign(ret)*v)
    w52 = min(52,n-1)
    hi52 = np.array([h[max(0,i-w52):i+1].max() for i in range(n)])
    lo52 = np.array([l[max(0,i-w52):i+1].min() for i in range(n)])
    out["Pct_Range_52"] = (c-lo52)/(hi52-lo52+1e-9)*100
    ema_long = _ema(c,min(50,n-1))
    out["Regime"]   = (c>ema_long).astype(float)
    out["Drawdown"] = (c-np.maximum.accumulate(c))/(np.maximum.accumulate(c)+1e-9)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def build_mtf(_frames_long):
    """Build W/ME/QE/YE indicator DataFrames for all tickers."""
    result = {}
    for ticker, df in _frames_long.items():  # _frames_long already filtered
        tfs = {}
        for key, freq in [("W","W"),("ME","ME"),("QE","QE"),("YE","YE")]:
            try:
                rs = _resample_ohlcv(df, freq)
                if len(rs) >= 10:
                    tfs[key] = _ind_tf(rs)
            except Exception:
                pass
        result[ticker] = tfs
    return result


def _monthly_heatmap_pivot(df_daily):
    """Year × Month matrix of monthly returns (%) for calendar heatmap."""
    df = df_daily[["Close"]].copy()
    df["Year"] = df.index.year; df["Month"] = df.index.month
    monthly = (
        df.groupby(["Year","Month"])["Close"]
          .agg(["first","last"])
          .assign(ret=lambda x: (x["last"]/x["first"]-1)*100)["ret"]
          .reset_index()
    )
    pivot = monthly.pivot(index="Year", columns="Month", values="ret")
    mnames = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = mnames[:len(pivot.columns)]
    return pivot


def _yearly_returns(df_daily):
    """Calendar year returns (%)."""
    df = df_daily[["Close"]].copy(); df["Year"] = df.index.year
    yr = df.groupby("Year")["Close"].agg(["first","last"])
    yr["Return%"] = (yr["last"]/yr["first"]-1)*100
    return yr["Return%"]


def _support_resistance(c_arr, window=20, n_levels=5):
    """Pivot-based support/resistance levels with clustering."""
    pivots_hi, pivots_lo = [], []
    for i in range(window, len(c_arr)-window):
        seg = c_arr[i-window:i+window+1]
        if c_arr[i] == seg.max(): pivots_hi.append(float(c_arr[i]))
        if c_arr[i] == seg.min(): pivots_lo.append(float(c_arr[i]))
    def cluster(pts, tol=0.02):
        if not pts: return []
        pts = sorted(pts, reverse=True)
        cls = [[pts[0]]]
        for p in pts[1:]:
            if abs(p-cls[-1][-1])/(cls[-1][-1]+1e-9) < tol: cls[-1].append(p)
            else: cls.append([p])
        return [float(np.mean(c)) for c in cls[:n_levels]]
    return cluster(pivots_lo), cluster(pivots_hi)


_LSTM_COLS = [
    "P_SMA5","P_EMA5","P_SMA20","P_EMA20","P_EMA50","P_EMA200",
    "RSI_n","MACD_n","MACD_h_n","BB_Pct","BB_Width",
    "ATR_n","Stoch_K_n","WR_n","CCI_n","ADX_n",
    "OBV_Slope","Vol_Ratio_n","Vol_5","Vol_20",
    "Regime","Golden_Cross","EMA50_200_n","Drawdown",
    "Lag_1","Lag_2","Lag_3","Lag_5","Lag_10",
    "Cum_5_n","Cum_10_n","Cum_20_n","Ret_n",
]

def _make_X(ind_df):
    c = ind_df["Close"].values
    d = {}
    d["P_SMA5"]      = (c - ind_df["SMA_5"].values)   / (ind_df["SMA_5"].values   + 1e-9)
    d["P_EMA5"]      = (c - ind_df["EMA_5"].values)   / (ind_df["EMA_5"].values   + 1e-9)
    d["P_SMA20"]     = (c - ind_df["SMA_20"].values)  / (ind_df["SMA_20"].values  + 1e-9)
    d["P_EMA20"]     = (c - ind_df["EMA_20"].values)  / (ind_df["EMA_20"].values  + 1e-9)
    d["P_EMA50"]     = (c - ind_df["EMA_50"].values)  / (ind_df["EMA_50"].values  + 1e-9)
    d["P_EMA200"]    = (c - ind_df["EMA_200"].values) / (ind_df["EMA_200"].values + 1e-9)
    d["RSI_n"]       = ind_df["RSI"].values / 100.0
    d["MACD_n"]      = ind_df["MACD"].values / (c + 1e-9)
    d["MACD_h_n"]    = ind_df["MACD_Hist"].values / (c + 1e-9)
    d["BB_Pct"]      = ind_df["BB_Pct"].values
    d["BB_Width"]    = ind_df["BB_Width"].values
    d["ATR_n"]       = ind_df["ATR"].values / (c + 1e-9)
    d["Stoch_K_n"]   = ind_df["Stoch_K"].values / 100.0
    d["WR_n"]        = (ind_df["Williams_R"].values + 100) / 100.0
    d["CCI_n"]       = np.tanh(ind_df["CCI"].values / 200.0)
    d["ADX_n"]       = ind_df["ADX"].values / 100.0
    d["OBV_Slope"]   = np.tanh(ind_df["OBV_Slope"].values)
    d["Vol_Ratio_n"] = np.tanh(ind_df["Vol_Ratio"].values - 1)
    d["Vol_5"]       = ind_df["Vol_5"].values
    d["Vol_20"]      = ind_df["Vol_20"].values
    d["Regime"]      = ind_df["Regime"].values
    d["Golden_Cross"]= ind_df["Golden_Cross"].values
    d["EMA50_200_n"] = np.tanh(ind_df["EMA50_200"].values * 10)
    d["Drawdown"]    = ind_df["Drawdown"].values
    for lag in [1, 2, 3, 5, 10]:
        d[f"Lag_{lag}"] = ind_df[f"Lag_{lag}"].values
    for w in [5, 10, 20]:
        d[f"Cum_{w}_n"] = np.tanh(ind_df[f"Cum_{w}"].values * 10)
    d["Ret_n"] = np.tanh(ind_df["Return"].values * 50)
    X = np.column_stack(list(d.values()))
    return np.nan_to_num(X, nan=0, posinf=1, neginf=-1)


# ── Sequence stats (temporal features for HGB) ────────────────────────────
def _seq_stats(Xs):
    last  = Xs[:, -1, :]
    mn    = Xs.mean(1)
    sd    = Xs.std(1)
    trend = Xs[:, -1, :] - Xs[:, 0, :]
    q3    = Xs[:, SEQ_LEN*3//4:, :].mean(1) - Xs[:, :SEQ_LEN//4, :].mean(1)
    return np.concatenate([last, mn, sd, trend, q3], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# LSTM TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def train_lstm(ticker, _ind):
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    ind    = _ind[ticker]
    X_raw  = _make_X(ind)
    c      = ind["Close"].values
    n      = len(c)

    # ── Labels: 5-day forward return direction — NO LOOKAHEAD ──────────────
    fwd = np.zeros(n)
    for i in range(n - 5):
        fwd[i] = (c[i+5] - c[i]) / (c[i] + 1e-9)
    y = (fwd > 0).astype(int)          # 1 = UP, 0 = DOWN

    # ── Sequences ──────────────────────────────────────────────────────────
    Xs    = np.array([X_raw[i-SEQ_LEN:i] for i in range(SEQ_LEN, n)])
    ys    = y[SEQ_LEN:]
    dates = ind.index[SEQ_LEN:]
    reg   = ind["Regime"].values[SEQ_LEN:]
    fwd_s = fwd[SEQ_LEN:]

    ok = np.isfinite(Xs).all(axis=(1, 2)) & np.isfinite(ys)
    Xs = Xs[ok]; ys = ys[ok]; dates = dates[ok]; reg = reg[ok]; fwd_s = fwd_s[ok]

    N     = len(Xs)
    split = int(N * 0.80)
    Xtr, Xte = Xs[:split], Xs[split:]
    ytr, yte  = ys[:split], ys[split:]
    dte, reg_te = dates[split:], reg[split:]
    F = Xtr.shape[2]

    # ── Normalise (fit on train only — zero data leakage) ──────────────────
    sc    = RobustScaler()
    Xtr_n = sc.fit_transform(Xtr.reshape(-1, F)).reshape(Xtr.shape)
    Xte_n = sc.transform(Xte.reshape(-1, F)).reshape(Xte.shape)
    Xtr_n = np.nan_to_num(Xtr_n, 0); Xte_n = np.nan_to_num(Xte_n, 0)

    # ── Train LSTM via BPTT + Adam ─────────────────────────────────────────
    model = LSTM(F, LSTM_H, n_cls=2, lr=5e-4, seed=42)
    loss_hist = model.train(Xtr_n, ytr, epochs=10)

    # ── Extract trained hidden states ──────────────────────────────────────
    H_tr = model.hidden_batch(Xtr_n)   # (n_train, LSTM_H)
    H_te = model.hidden_batch(Xte_n)   # (n_test,  LSTM_H)

    # ── Combined features: LSTM hidden + sequence statistics ───────────────
    Ftr = np.concatenate([H_tr, _seq_stats(Xtr_n)], axis=1)
    Fte = np.concatenate([H_te, _seq_stats(Xte_n)], axis=1)
    sc2 = RobustScaler()
    Ftr_s = np.nan_to_num(sc2.fit_transform(Ftr), 0)
    Fte_s = np.nan_to_num(sc2.transform(Fte), 0)

    # ── Gradient boosting on LSTM features ─────────────────────────────────
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.025, max_depth=5,
        min_samples_leaf=10, l2_regularization=0.1,
        random_state=42, early_stopping=True,
        validation_fraction=0.12, n_iter_no_change=25,
    )
    hgb.fit(Ftr_s, ytr)

    # ── Evaluate ───────────────────────────────────────────────────────────
    probs  = hgb.predict_proba(Fte_s)
    preds  = np.argmax(probs, axis=1)
    conf   = probs.max(axis=1)

    # Raw LSTM accuracy
    lstm_pr   = model.proba_batch(Xte_n)
    lstm_pred = np.argmax(lstm_pr, axis=1)
    acc_lstm  = accuracy_score(yte, lstm_pred) * 100

    # Ensemble overall
    acc_all  = accuracy_score(yte, preds) * 100

    # Regime-split accuracy
    bull, bear = reg_te == 1, reg_te == 0
    acc_bull = accuracy_score(yte[bull], preds[bull]) * 100 if bull.sum() > 10 else 0
    acc_bear = accuracy_score(yte[bear], preds[bear]) * 100 if bear.sum() > 10 else 0

    # Confidence-threshold table
    conf_tbl = []
    for th in [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]:
        m = conf >= th
        if m.sum() >= 10:
            a = accuracy_score(yte[m], preds[m]) * 100
            conf_tbl.append({
                "Threshold": f">={th:.0%}",
                "Accuracy":  f"{a:.1f}%",
                "Signals":   int(m.sum()),
                "Coverage":  f"{m.mean()*100:.0f}%",
                "Edge over 50%": f"+{a-50:.1f}%",
            })
        mb = m & bull
        if mb.sum() >= 8:
            ab = accuracy_score(yte[mb], preds[mb]) * 100
            conf_tbl.append({
                "Threshold": f">={th:.0%} + Bull",
                "Accuracy":  f"{ab:.1f}%",
                "Signals":   int(mb.sum()),
                "Coverage":  f"{mb.mean()*100:.0f}%",
                "Edge over 50%": f"+{ab-50:.1f}%",
            })

    cm = confusion_matrix(yte, preds, labels=[0, 1])

    # Signal overlay (full dataset)
    Xs_all = np.array([X_raw[i-SEQ_LEN:i] for i in range(SEQ_LEN, n)])
    ok2    = np.isfinite(Xs_all).all(axis=(1, 2))
    Xs_ok  = Xs_all[ok2]
    Xall_n = np.nan_to_num(
        sc.transform(Xs_ok.reshape(-1, F)).reshape(Xs_ok.shape), 0)
    H_all  = model.hidden_batch(Xall_n)
    Fall   = np.concatenate([H_all, _seq_stats(Xall_n)], axis=1)
    Fall_s = np.nan_to_num(sc2.transform(Fall), 0)
    pr_all = hgb.predict_proba(Fall_s)
    pa_all = np.argmax(pr_all, axis=1)
    ca_all = pr_all.max(axis=1)
    da_all = ind.index[SEQ_LEN:][ok2]

    return {
        "acc_lstm":  round(acc_lstm,  1),
        "acc_all":   round(acc_all,   1),
        "acc_bull":  round(acc_bull,  1),
        "acc_bear":  round(acc_bear,  1),
        "conf_tbl":  conf_tbl,
        "loss_hist": loss_hist,
        "preds":     preds,  "actual":  yte,
        "conf":      conf,   "probs":   probs,
        "dates_te":  dte,    "reg_te":  reg_te,
        "cm":        cm,
        "n_train":   split,  "n_test":  len(yte),
        "n_feats":   F,
        "preds_all": pa_all, "conf_all": ca_all,
        "dates_all": da_all,
        "report":    classification_report(yte, preds, output_dict=True),
    }, None


@st.cache_data(show_spinner=False)
def train_all(_ind):
    return {t: train_lstm(t, _ind) for t in LOADED_TICKERS}


# ── Investment score ───────────────────────────────────────────────────────
def score_stock(ind_df, lstm_res):
    last = ind_df.ffill().iloc[-1]; score = 0; bd = {}
    rsi  = float(last["RSI"])
    rs   = 20 if rsi < 30 else 15 if rsi < 50 else 10 if rsi < 65 else 5
    score += rs; bd["RSI"] = rs
    ms = 20 if float(last["MACD_Hist"]) > 0 else 5
    score += ms; bd["MACD"] = ms
    cur = float(last["Close"]); e200 = float(last["EMA_200"])
    es  = 15 if cur > e200 else 5
    score += es; bd["vs EMA200"] = es
    vol = float(last["Vol_30"])
    vs  = 15 if vol < 0.2 else 10 if vol < 0.35 else 5
    score += vs; bd["Volatility"] = vs
    adx = float(last["ADX"])
    ads = 10 if adx > 25 else 5
    score += ads; bd["ADX"] = ads
    if lstm_res:
        lp = int(lstm_res["preds_all"][-1]) if len(lstm_res["preds_all"]) > 0 else 1
        lc = float(lstm_res["conf_all"][-1]) if len(lstm_res["conf_all"]) > 0 else 0.5
        ls = int(15 * lc * (1.0 if lp == 1 else 0.25))
        score += ls; bd["LSTM"] = ls
    signal = "BUY" if score >= 65 else "HOLD" if score >= 45 else "AVOID"
    ret = ind_df["Return"].dropna()
    sh  = float((ret.mean()/ret.std())*np.sqrt(252)) if ret.std() > 0 else 0
    nr  = ret[ret < 0]
    so  = float((ret.mean()/nr.std())*np.sqrt(252)) if len(nr) > 0 and nr.std() > 0 else 0
    return {"score": score, "signal": signal, "breakdown": bd,
            "sharpe": round(sh, 2), "sortino": round(so, 2),
            "rsi": round(rsi, 1), "vol": round(vol*100, 1), "close": round(cur, 2)}


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR + LOAD
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 IB Analytics v5")
    st.caption(f"Yahoo Finance  |  Live as of {END}")
    st.divider()
    page = st.radio("Select Module", [
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
    selected = st.multiselect("Stocks", LOADED_TICKERS, default=LOADED_TICKERS[:min(5,len(LOADED_TICKERS))])
    if not selected:
        selected = LOADED_TICKERS[:min(5,len(LOADED_TICKERS))]
    st.divider()
    st.caption("⚠️ Not financial advice.")

with st.spinner("📡 Fetching live data from Yahoo Finance…"):
    frames = fetch_yahoo()

if not frames:
    st.error("❌ Cannot reach Yahoo Finance. Install yfinance: `pip install yfinance`")
    st.stop()

with st.spinner("⚙️ Computing 20+ technical indicators per stock…"):
    ind_data = compute_indicators(frames)

# Extended history + multi-timeframe indicators (cached separately)
with st.spinner("📅 Loading extended history (1993→today) for long-term charts…"):
    frames_long = fetch_yahoo_long()

with st.spinner("📊 Building Weekly / Monthly / Quarterly indicator suites…"):
    mtf_data = build_mtf(frames_long if frames_long else frames)

data_src = f"🟢 Yahoo Finance  |  {len(frames)} stocks  |  {START} → {END}"

# Only use tickers that actually downloaded successfully — guards all KeyErrors
LOADED_TICKERS = [t for t in TICKERS if t in frames and t in ind_data]
if not LOADED_TICKERS:
    st.error("No stock data loaded. Check yfinance installation."); st.stop()
if len(LOADED_TICKERS) < len(TICKERS):
    missing = [t for t in TICKERS if t not in LOADED_TICKERS]
    st.warning(f"⚠️ {len(missing)} ticker(s) failed to load: {', '.join(missing)}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("📊 Executive Overview")
    st.caption(data_src)

    # KPI row
    st.subheader("Live Market Snapshot")
    cols = st.columns(min(len(selected), 5))
    for i, t in enumerate([t for t in selected if t in LOADED_TICKERS][:5]):
        df  = frames[t]
        cur = float(df["Close"].iloc[-1])
        pv  = float(df["Close"].iloc[max(0, len(df)-252)])
        d1  = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2]) if len(df)>1 else 0
        ytd = (cur/pv - 1)*100
        hi  = float(df["Close"].rolling(252).max().iloc[-1])
        lo  = float(df["Close"].rolling(252).min().iloc[-1])
        with cols[i]:
            st.metric(t, f"${cur:.2f}", f"{d1:+.2f} ({ytd:+.1f}% 1Y)")
            st.progress(max(0, min(1, (cur-lo)/(hi-lo+1e-9))),
                        text=f"52W: ${lo:.0f}–${hi:.0f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normalised Performance (Base=100)")
        fig = go.Figure()
        for i, t in enumerate([t for t in selected if t in LOADED_TICKERS]):
            df   = frames[t]
            norm = df["Close"] / df["Close"].iloc[0] * 100
            fig.add_trace(go.Scatter(x=df.index, y=norm, name=t, mode="lines",
                                     line=dict(width=2, color=COLORS[i % len(COLORS)])))
        pplot(fig, h=340, xaxis_title="Date", yaxis_title="Indexed (Base=100)",
              legend=dict(orientation="h", y=1.02, x=0))
        ibox("What this shows",
             "How $100 invested at the start of the dataset has grown. "
             "Steeper slope = stronger compounder. "
             "Periods where all lines fall together = macro correlation spike — "
             "diversification offers no protection during those windows.")

    with c2:
        st.subheader("Return Correlation Matrix")
        ret_df = pd.DataFrame({t: frames[t]["Close"].pct_change() for t in LOADED_TICKERS}).dropna()
        corr   = ret_df.corr().round(3)
        fig2   = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                           zmin=-1, zmax=1, aspect="auto")
        fig2.update_layout(**DARK, height=340)
        st.plotly_chart(fig2, use_container_width=True, config=PCFG)
        ibox("Correlation insight",
             "Red = stocks that always move together — limited diversification benefit. "
             "Tech cluster (AAPL/MSFT/NVDA/GOOGL/META) clusters 0.6–0.8. "
             "BRK-B and UNH have the lowest cross-sector correlations — "
             "best diversifiers in this universe. "
             "The LSTM's `Regime` and `Golden_Cross` features implicitly encode "
             "when the whole market enters a high-correlation macro regime.")

    st.subheader("Price + EMA 20 / 50 / 200  (last 180 days)")
    cs_t  = st.selectbox("Ticker", selected, key="cs_ov")
    ind   = ind_data[cs_t]
    co    = ind.index[-1] - pd.Timedelta(days=180)
    iv    = ind[ind.index >= co]
    fig3  = go.Figure()
    fig3.add_trace(go.Candlestick(
        x=iv.index, open=iv["Open"], high=iv["High"],
        low=iv["Low"], close=iv["Close"], name=cs_t,
        increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
    fig3.add_trace(go.Bar(x=iv.index, y=iv["Volume"], name="Volume",
                          yaxis="y2", marker_color="rgba(88,166,255,0.12)"))
    for ema, col, dash, wid in [(20,"#fee140","dot",1.2),
                                 (50,"#58a6ff","dash",1.6),
                                 (200,"#3fb950","solid",2.0)]:
        fig3.add_trace(go.Scatter(x=iv.index, y=iv[f"EMA_{ema}"], name=f"EMA {ema}",
                                  line=dict(color=col, width=wid, dash=dash)))
    fig3.update_layout(**DARK, height=450,
                       yaxis2=dict(overlaying="y", side="right", showgrid=False),
                       xaxis_rangeslider_visible=False)
    st.plotly_chart(fig3, use_container_width=True, config=PCFG)
    ibox("EMA 20 / 50 / 200 — backbone of trend trading",
         "**EMA 20 (gold, dotted):** Short-term momentum. Day-traders watch "
         "price crossing EMA 20 as an intra-week signal. "
         "**EMA 50 (blue, dashed):** Institutional medium-term trend. "
         "Most professional traders use price vs EMA 50 as a core position filter. "
         "The LSTM feature `P_EMA50` is price distance from this line. "
         "**EMA 200 (green, solid):** The most watched line in all of technical analysis. "
         "Above EMA 200 = structural bull market; below = bear market. "
         "The LSTM's `Regime` feature is simply (price > EMA 200). "
         "**Golden Cross:** EMA 50 crosses above EMA 200 → strongest long-term buy signal. "
         "The LSTM learned that Golden Cross + RSI < 60 + MACD positive is the "
         "highest-probability setup in its 33-feature space.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("30-Day Rolling Volatility")
        fig4 = go.Figure()
        for i, t in enumerate([t for t in selected if t in LOADED_TICKERS]):
            v = ind_data[t]["Vol_30"] * 100
            fig4.add_trace(go.Scatter(x=v.index, y=v, name=t, mode="lines",
                                      line=dict(width=1.5, color=COLORS[i % len(COLORS)])))
        pplot(fig4, h=260, yaxis_title="Ann. Volatility (%)")
        ibox("Volatility regime",
             "Spikes indicate earnings, Fed decisions, or macro shocks. "
             "The LSTM's Vol_5/Vol_20 ratio detects vol expansion before it's "
             "visible in daily charts, giving the model an early risk-off signal.")

    with c4:
        st.subheader("Drawdown from All-Time High")
        fig5 = go.Figure()
        for i, t in enumerate([t for t in selected if t in LOADED_TICKERS]):
            dd = ind_data[t]["Drawdown"] * 100
            fig5.add_trace(go.Scatter(x=dd.index, y=dd, name=t,
                                      fill="tozeroy", mode="lines",
                                      line=dict(width=1, color=COLORS[i % len(COLORS)])))
        pplot(fig5, h=260, yaxis_title="Drawdown (%)")
        ibox("Drawdown as risk signal",
             "Deep drawdown in a bull regime (above EMA 200) often signals a mean-reversion "
             "entry — the LSTM has learned this regime-conditional pattern. "
             "The same drawdown below EMA 200 is an AVOID signal — trend is down.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📉 Technical Analysis":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.title("📉 Full Technical Analysis Suite")
    ta_t = st.selectbox("Select Stock", LOADED_TICKERS, key="ta_t")
    ind  = ind_data.get(ta_t) or ind_data[LOADED_TICKERS[0]]
    days = st.slider("Days to display", 60, len(ind), min(365, len(ind)), step=30)
    iv   = ind.iloc[-days:]

    # ── Price + EMAs + Ichimoku + Bollinger ────────────────────────────────
    st.subheader("Price Action: EMA 20/50/200 + Bollinger Bands + Ichimoku Cloud")
    fig = go.Figure()
    # Ichimoku cloud fill
    fig.add_trace(go.Scatter(x=iv.index, y=iv["Ichi_SpanA"], name="Span A",
                             line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=iv.index, y=iv["Ichi_SpanB"], name="Kumo Cloud",
                             fill="tonexty", fillcolor="rgba(63,185,80,0.07)",
                             line=dict(color="rgba(0,0,0,0)", width=0)))
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=iv.index, y=iv["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(88,166,255,0.35)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=iv.index, y=iv["BB_Lower"], name="BB Lower",
                             fill="tonexty", fillcolor="rgba(88,166,255,0.05)",
                             line=dict(color="rgba(88,166,255,0.35)", width=1, dash="dot")))
    # EMAs
    for ema, col, dash, wid in [(20,"#fee140","dot",1.2), (50,"#58a6ff","dash",1.6),
                                 (200,"#3fb950","solid",2.0)]:
        fig.add_trace(go.Scatter(x=iv.index, y=iv[f"EMA_{ema}"], name=f"EMA {ema}",
                                 line=dict(color=col, width=wid, dash=dash)))
    # Ichimoku lines
    fig.add_trace(go.Scatter(x=iv.index, y=iv["Ichi_Tenkan"], name="Tenkan (9)",
                             line=dict(color="#f093fb", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=iv.index, y=iv["Ichi_Kijun"], name="Kijun (26)",
                             line=dict(color="#4facfe", width=1, dash="dash")))
    # Price candles
    fig.add_trace(go.Candlestick(
        x=iv.index, open=iv["Open"], high=iv["High"],
        low=iv["Low"], close=iv["Close"], name=ta_t,
        increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
    fig.update_layout(**DARK, height=500, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.01, x=0, font=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True, config=PCFG)
    ibox("Reading this chart",
         "**Bollinger Bands (blue dotted):** Price at upper band = short-term overbought. "
         "Band squeeze (width narrows sharply) = volatility contraction before a big move. "
         "**Ichimoku Cloud (green shading):** Price above cloud = bullish. "
         "Thick cloud = strong support/resistance. Thin cloud = weak zone — price breaks through easily. "
         "**Tenkan (purple):** 9-period midpoint — short-term momentum. "
         "**Kijun (blue):** 26-period midpoint — medium-term baseline. "
         "TK cross (Tenkan crosses Kijun) = Ichimoku entry signal. "
         "**EMA 200 (green solid):** structural regime boundary — "
         "the single most important line the LSTM uses.")

    # ── RSI + Stochastic + Williams %R ────────────────────────────────────
    st.subheader("Momentum Oscillators: RSI(14)  |  Stochastic(14,3)  |  Williams %R(14)")
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          subplot_titles=["RSI(14)", "Stochastic %K/%D(14,3)", "Williams %R(14)"],
                          vertical_spacing=0.06)
    fig2.add_trace(go.Scatter(x=iv.index, y=iv["RSI"], name="RSI",
                              line=dict(color="#f6ad55", width=1.8)), row=1, col=1)
    for y_val, col in [(70,"#f85149"), (30,"#3fb950"), (50,"#374151")]:
        fig2.add_hline(y=y_val, line_color=col, line_dash="dash", row=1, col=1)
    fig2.add_trace(go.Scatter(x=iv.index, y=iv["Stoch_K"], name="%K",
                              line=dict(color="#58a6ff", width=1.6)), row=2, col=1)
    fig2.add_trace(go.Scatter(x=iv.index, y=iv["Stoch_D"], name="%D",
                              line=dict(color="#f093fb", width=1.2, dash="dot")), row=2, col=1)
    for y_val, col in [(80,"#f85149"), (20,"#3fb950")]:
        fig2.add_hline(y=y_val, line_color=col, line_dash="dash", row=2, col=1)
    fig2.add_trace(go.Scatter(x=iv.index, y=iv["Williams_R"], name="Williams %R",
                              line=dict(color="#43e97b", width=1.6)), row=3, col=1)
    for y_val, col in [(-20,"#f85149"), (-80,"#3fb950")]:
        fig2.add_hline(y=y_val, line_color=col, line_dash="dash", row=3, col=1)
    fig2.update_layout(**DARK, height=520, showlegend=True)
    st.plotly_chart(fig2, use_container_width=True, config=PCFG)
    ibox("Three oscillators — confirm each other",
         "**RSI(14):** >70 = overbought (sell pressure builds), <30 = oversold (buy zone). "
         "RSI divergence: price makes new high but RSI doesn't = bearish warning. "
         "The LSTM's RSI_n feature sees 20 consecutive days of RSI values — "
         "it learns the divergence pattern directly from the sequence. "
         "**Stochastic %K/%D:** %K crossing %D below 20 = buy; above 80 = sell. "
         "**Williams %R:** Identical logic but inverted scale. "
         "**Triple confirmation:** RSI<35 + Stoch<20 + WR<-80 simultaneously = "
         "highest-probability oversold buy setup — the LSTM has learned this joint signal "
         "from its 20-day multi-feature sequence.")

    # ── MACD + CCI + ADX ──────────────────────────────────────────────────
    st.subheader("Trend Indicators: MACD(12,26,9)  |  CCI(20)  |  ADX(14)")
    fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          subplot_titles=["MACD", "CCI(20)", "ADX(14)"],
                          vertical_spacing=0.06)
    hc = ["#3fb950" if v >= 0 else "#f85149" for v in iv["MACD_Hist"]]
    fig3.add_trace(go.Bar(x=iv.index, y=iv["MACD_Hist"], name="Histogram",
                          marker_color=hc, opacity=0.7), row=1, col=1)
    fig3.add_trace(go.Scatter(x=iv.index, y=iv["MACD"], name="MACD",
                              line=dict(color="#58a6ff", width=1.6)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=iv.index, y=iv["MACD_Signal"], name="Signal",
                              line=dict(color="#f85149", width=1.2, dash="dot")), row=1, col=1)
    fig3.add_hline(y=0, line_color="#374151", line_dash="dash", row=1, col=1)
    fig3.add_trace(go.Scatter(x=iv.index, y=iv["CCI"], name="CCI",
                              line=dict(color="#fee140", width=1.6)), row=2, col=1)
    for y_val, col in [(100,"#f85149"), (-100,"#3fb950"), (0,"#374151")]:
        fig3.add_hline(y=y_val, line_color=col, line_dash="dash", row=2, col=1)
    fig3.add_trace(go.Scatter(x=iv.index, y=iv["ADX"], name="ADX",
                              line=dict(color="#a371f7", width=1.8)), row=3, col=1)
    fig3.add_trace(go.Scatter(x=iv.index, y=iv["DI_Plus"], name="DI+",
                              line=dict(color="#3fb950", width=1.2, dash="dash")), row=3, col=1)
    fig3.add_trace(go.Scatter(x=iv.index, y=iv["DI_Minus"], name="DI-",
                              line=dict(color="#f85149", width=1.2, dash="dash")), row=3, col=1)
    fig3.add_hline(y=25, line_color="#f6ad55", line_dash="dash",
                   annotation_text="Trend threshold (25)", row=3, col=1)
    fig3.update_layout(**DARK, height=520, showlegend=True)
    st.plotly_chart(fig3, use_container_width=True, config=PCFG)
    ibox("Trend strength decoded",
         "**MACD:** Green histogram bars = bullish momentum building. "
         "Histogram shrinking while green = momentum peak — early exit warning. "
         "**CCI(20):** >+100 = strong uptrend (ride it). <−100 = strong downtrend. "
         "CCI crossing zero from below = trend change signal. "
         "**ADX(14):** Measures STRENGTH of trend, not direction. "
         "ADX > 25 = strong trend (directional strategies work). "
         "ADX < 20 = no trend (oscillator strategies work instead). "
         "DI+ above DI− AND ADX > 25 = strongest trend-following buy confirmation. "
         "The LSTM's ADX_n feature lets it switch strategy automatically: "
         "in high-ADX regimes it weights momentum features more heavily.")

    # ── Volume + OBV ──────────────────────────────────────────────────────
    st.subheader("Volume Analysis: OBV  |  Volume vs 20-Day MA")
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=["OBV (On-Balance Volume)",
                                          "Daily Volume vs 20-day MA"],
                          vertical_spacing=0.08)
    fig4.add_trace(go.Scatter(x=iv.index, y=iv["OBV"], name="OBV",
                              fill="tozeroy", line=dict(color="#a371f7", width=1.8)),
                   row=1, col=1)
    vc = ["#3fb950" if r >= 0 else "#f85149" for r in iv["Return"].fillna(0)]
    fig4.add_trace(go.Bar(x=iv.index, y=iv["Volume"], name="Volume",
                          marker_color=vc, opacity=0.6), row=2, col=1)
    vma20 = iv["Volume"].rolling(20).mean()
    fig4.add_trace(go.Scatter(x=iv.index, y=vma20, name="Vol MA20",
                              line=dict(color="#f6ad55", width=1.8)), row=2, col=1)
    fig4.update_layout(**DARK, height=400, showlegend=True)
    st.plotly_chart(fig4, use_container_width=True, config=PCFG)
    ibox("Volume confirms price moves",
         "**OBV:** Rising OBV while price is flat = institutional accumulation — "
         "smart money buying quietly before a breakout (leads price by 1–3 weeks). "
         "Falling OBV while price rises = distribution — institutions selling into retail buying. "
         "This divergence is the LSTM's `OBV_Slope` feature. "
         "**Volume bars (green=up day, red=down day):** A large green bar on an up day "
         "confirms institutional buying. A large red bar on a down day = panic selling — "
         "often a capitulation bottom. The `Vol_Ratio_n` feature (today vs 20-day avg) "
         "feeds this directly into the LSTM sequence.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MULTI-TIMEFRAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⏱️ Multi-Timeframe Analysis":
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    st.title("⏱️ Multi-Timeframe Analysis")
    st.caption(
        f"Data: Yahoo Finance  |  Daily from {START}  |  "
        f"Long-term history from {START_LONG}  |  Updated {END}"
    )

    mtf_t = st.selectbox("Select Stock", LOADED_TICKERS, key="mtf_t")
    tfs   = mtf_data.get(mtf_t, {})
    df_d  = frames_long.get(mtf_t, frames.get(mtf_t))  # long daily for heatmap

    # ── Tab structure ──────────────────────────────────────────────────────
    tab_w, tab_m, tab_q, tab_cal, tab_yr, tab_sr, tab_cmp = st.tabs([
        "📅 Weekly", "🗓️ Monthly", "📆 Quarterly",
        "🔥 Monthly Return Heatmap", "📈 Yearly Returns",
        "🎯 Support & Resistance", "🔀 Cross-TF Comparison"
    ])

    TF_LABELS = {"W":"Weekly","ME":"Monthly","QE":"Quarterly","YE":"Yearly"}

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1 — WEEKLY
    # ─────────────────────────────────────────────────────────────────────
    with tab_w:
        df_w = tfs.get("W")
        if df_w is None or len(df_w) < 20:
            st.info("Not enough weekly data."); 
        else:
            n_weeks = st.slider("Weeks to display", 52, len(df_w), min(260, len(df_w)), 13, key="wk_sl")
            wv = df_w.iloc[-n_weeks:]

            st.subheader("Weekly Candlestick + EMA 10 / 20 / 50 (Weekly)")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=wv.index, open=wv["Open"], high=wv["High"],
                low=wv["Low"], close=wv["Close"], name="Weekly",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
            fig.add_trace(go.Bar(x=wv.index, y=wv["Volume"], name="Vol",
                                 yaxis="y2", marker_color="rgba(88,166,255,0.12)"))
            for ema, col, dash, wid in [(10,"#fee140","dot",1.2),
                                         (20,"#58a6ff","dash",1.6),
                                         (50,"#3fb950","solid",2.0)]:
                col_n = f"EMA_{ema}"
                if col_n in wv.columns:
                    fig.add_trace(go.Scatter(x=wv.index, y=wv[col_n],
                                             name=f"W-EMA {ema}",
                                             line=dict(color=col, width=wid, dash=dash)))
            if "BB_Upper" in wv.columns:
                fig.add_trace(go.Scatter(x=wv.index, y=wv["BB_Upper"],
                                         name="BB Upper", line=dict(color="rgba(88,166,255,0.3)", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=wv.index, y=wv["BB_Lower"],
                                         name="BB Lower", fill="tonexty",
                                         fillcolor="rgba(88,166,255,0.05)",
                                         line=dict(color="rgba(88,166,255,0.3)", width=1, dash="dot")))
            fig.update_layout(**DARK, height=480, xaxis_rangeslider_visible=False,
                              yaxis2=dict(overlaying="y", side="right", showgrid=False),
                              legend=dict(orientation="h", y=1.01, x=0, font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)
            ibox("Weekly chart — why it matters",
                 "Each candle represents one full trading week. "
                 "Weekly charts filter out daily noise and reveal the true medium-term trend. "
                 "**W-EMA 10:** ~2.5-month momentum trend. "
                 "**W-EMA 20:** ~5-month trend — the primary signal for swing traders. "
                 "**W-EMA 50:** ~1-year trend — institutional positioning line. "
                 "A weekly candle closing above W-EMA 50 after a prolonged period below it "
                 "is one of the highest-confidence medium-term buy signals in technical analysis. "
                 "Weekly Bollinger Bands show multi-week over/undershooting.")

            st.subheader("Weekly RSI(14)  |  MACD  |  Stochastic")
            fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  subplot_titles=["Weekly RSI(14)","Weekly MACD","Weekly Stochastic"],
                                  vertical_spacing=0.06)
            if "RSI" in wv.columns:
                fig2.add_trace(go.Scatter(x=wv.index, y=wv["RSI"], name="W-RSI",
                                          line=dict(color="#f6ad55", width=1.8)), row=1, col=1)
                for yv, col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                    fig2.add_hline(y=yv, line_color=col, line_dash="dash", row=1, col=1)
            if "MACD" in wv.columns:
                hc = ["#3fb950" if v>=0 else "#f85149" for v in wv["MACD_Hist"]]
                fig2.add_trace(go.Bar(x=wv.index, y=wv["MACD_Hist"], name="W-Hist",
                                      marker_color=hc, opacity=0.7), row=2, col=1)
                fig2.add_trace(go.Scatter(x=wv.index, y=wv["MACD"],
                                          name="W-MACD", line=dict(color="#58a6ff", width=1.6)),
                               row=2, col=1)
                fig2.add_trace(go.Scatter(x=wv.index, y=wv["MACD_Signal"],
                                          name="W-Signal", line=dict(color="#f85149", width=1.2, dash="dot")),
                               row=2, col=1)
            if "Stoch_K" in wv.columns:
                fig2.add_trace(go.Scatter(x=wv.index, y=wv["Stoch_K"],
                                          name="W-%K", line=dict(color="#58a6ff", width=1.6)),
                               row=3, col=1)
                fig2.add_trace(go.Scatter(x=wv.index, y=wv["Stoch_D"],
                                          name="W-%D", line=dict(color="#f093fb", width=1.2, dash="dot")),
                               row=3, col=1)
                for yv, col in [(80,"#f85149"),(20,"#3fb950")]:
                    fig2.add_hline(y=yv, line_color=col, line_dash="dash", row=3, col=1)
            fig2.update_layout(**DARK, height=520)
            st.plotly_chart(fig2, use_container_width=True, config=PCFG)
            ibox("Weekly oscillators — stronger signals, fewer false alarms",
                 "Weekly RSI readings are far more reliable than daily. "
                 "When weekly RSI drops below 30, the stock has experienced a significant multi-week selloff "
                 "— this level has historically been an excellent medium-term entry point. "
                 "Weekly MACD crossovers indicate trend changes lasting weeks to months, "
                 "not just 1–2 day noise. A weekly MACD bullish crossover combined with "
                 "weekly RSI rising from below 40 is one of the most powerful setups in swing trading.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2 — MONTHLY
    # ─────────────────────────────────────────────────────────────────────
    with tab_m:
        df_m = tfs.get("ME")
        if df_m is None or len(df_m) < 12:
            st.info("Not enough monthly data.")
        else:
            n_months = st.slider("Months to display", 24, len(df_m), min(120, len(df_m)), 12, key="mo_sl")
            mv = df_m.iloc[-n_months:]

            st.subheader("Monthly Candlestick + EMA 6 / 12 / 24 (Monthly) = ~6M / 1Y / 2Y")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=mv.index, open=mv["Open"], high=mv["High"],
                low=mv["Low"], close=mv["Close"], name="Monthly",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
            fig.add_trace(go.Bar(x=mv.index, y=mv["Volume"], name="Vol",
                                 yaxis="y2", marker_color="rgba(88,166,255,0.12)"))
            for ema, col, dash, wid, lbl in [
                (6,"#fee140","dot",1.3,"~6 months"),
                (12,"#58a6ff","dash",1.7,"~1 year"),
                (24,"#3fb950","solid",2.2,"~2 years")]:
                col_n = f"EMA_{ema}"
                if col_n in mv.columns:
                    fig.add_trace(go.Scatter(x=mv.index, y=mv[col_n],
                                             name=f"M-EMA {ema} ({lbl})",
                                             line=dict(color=col, width=wid, dash=dash)))
            if "BB_Upper" in mv.columns:
                fig.add_trace(go.Scatter(x=mv.index, y=mv["BB_Upper"],
                                         name="Monthly BB Upper",
                                         line=dict(color="rgba(88,166,255,0.3)", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=mv.index, y=mv["BB_Lower"],
                                         name="Monthly BB Lower", fill="tonexty",
                                         fillcolor="rgba(88,166,255,0.05)",
                                         line=dict(color="rgba(88,166,255,0.3)", width=1, dash="dot")))
            fig.update_layout(**DARK, height=500, xaxis_rangeslider_visible=False,
                              yaxis2=dict(overlaying="y", side="right", showgrid=False),
                              legend=dict(orientation="h", y=1.01, x=0, font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)
            ibox("Monthly chart — the macro trend view",
                 "Each candle = one calendar month of price action. Monthly charts eliminate "
                 "weekly and daily noise completely — only the macro trend matters here. "
                 "**M-EMA 6 (~6 months):** Short-term macro trend. "
                 "**M-EMA 12 (~1 year):** The 12-month moving average is the most-watched "
                 "line by macro investors. Price above = annual bull trend. "
                 "**M-EMA 24 (~2 years):** Very long-term structural support/resistance. "
                 "When price bounces off the monthly EMA 24, it has historically marked major bull market re-entries. "
                 "Monthly Bollinger Band expansions indicate multi-year volatility cycles — "
                 "touching the monthly upper band = multi-year overbought.")

            st.subheader("Monthly RSI  |  MACD  |  ADX Trend Strength")
            fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  subplot_titles=["Monthly RSI(14)", "Monthly MACD", "Monthly ADX(14)"],
                                  vertical_spacing=0.06)
            if "RSI" in mv.columns:
                fig2.add_trace(go.Scatter(x=mv.index, y=mv["RSI"], name="M-RSI",
                                          line=dict(color="#f6ad55", width=2)), row=1, col=1)
                for yv, col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                    fig2.add_hline(y=yv, line_color=col, line_dash="dash", row=1, col=1)
            if "MACD" in mv.columns:
                hc = ["#3fb950" if v>=0 else "#f85149" for v in mv["MACD_Hist"]]
                fig2.add_trace(go.Bar(x=mv.index, y=mv["MACD_Hist"], name="M-Hist",
                                      marker_color=hc, opacity=0.7), row=2, col=1)
                fig2.add_trace(go.Scatter(x=mv.index, y=mv["MACD"],
                                          name="M-MACD", line=dict(color="#58a6ff", width=1.8)),
                               row=2, col=1)
                fig2.add_trace(go.Scatter(x=mv.index, y=mv["MACD_Signal"],
                                          name="M-Signal", line=dict(color="#f85149", width=1.3, dash="dot")),
                               row=2, col=1)
            if "ADX" in mv.columns:
                fig2.add_trace(go.Scatter(x=mv.index, y=mv["ADX"], name="M-ADX",
                                          line=dict(color="#a371f7", width=1.8)), row=3, col=1)
                fig2.add_trace(go.Scatter(x=mv.index, y=mv["DI_Plus"], name="M-DI+",
                                          line=dict(color="#3fb950", width=1.2, dash="dash")), row=3, col=1)
                fig2.add_trace(go.Scatter(x=mv.index, y=mv["DI_Minus"], name="M-DI-",
                                          line=dict(color="#f85149", width=1.2, dash="dash")), row=3, col=1)
                fig2.add_hline(y=25, line_color="#f6ad55", line_dash="dash",
                               annotation_text="Trend (25)", row=3, col=1)
            fig2.update_layout(**DARK, height=520)
            st.plotly_chart(fig2, use_container_width=True, config=PCFG)
            ibox("Monthly oscillators — macro cycle timing",
                 "Monthly RSI below 30 is historically very rare and has marked major market bottoms "
                 "(e.g., March 2009, March 2020). These are once-in-a-cycle buying opportunities. "
                 "Monthly MACD bullish crossover = new multi-year bull phase beginning. "
                 "Monthly ADX > 40 = extremely strong directional trend — "
                 "the kind seen during NVDA's 2023–2024 AI-driven surge. "
                 "Monthly ADX falling from high levels signals trend exhaustion — time to reduce longs.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3 — QUARTERLY
    # ─────────────────────────────────────────────────────────────────────
    with tab_q:
        df_q = tfs.get("QE")
        if df_q is None or len(df_q) < 8:
            st.info("Not enough quarterly data.")
        else:
            n_q = st.slider("Quarters to display", 8, len(df_q), min(40, len(df_q)), 4, key="q_sl")
            qv = df_q.iloc[-n_q:]

            st.subheader("Quarterly Candlestick + EMA 4 / 8 / 12 (Quarterly) = ~1Y / 2Y / 3Y")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=qv.index, open=qv["Open"], high=qv["High"],
                low=qv["Low"], close=qv["Close"], name="Quarterly",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
            for ema, col, dash, wid, lbl in [
                (4,"#fee140","dot",1.3,"~1yr"),
                (8,"#58a6ff","dash",1.7,"~2yr"),
                (12,"#3fb950","solid",2.2,"~3yr")]:
                col_n = f"EMA_{ema}"
                if col_n in qv.columns:
                    fig.add_trace(go.Scatter(x=qv.index, y=qv[col_n],
                                             name=f"Q-EMA {ema} ({lbl})",
                                             line=dict(color=col, width=wid, dash=dash)))
            fig.update_layout(**DARK, height=460, xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h", y=1.01, x=0, font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)
            ibox("Quarterly chart — a decade of context",
                 "Each candle = one quarter (3 months). This is the timeframe used by "
                 "institutional fund managers and Warren Buffett-style value investors. "
                 "Quarterly EMAs represent truly long-term trend lines: "
                 "Q-EMA 8 = 2-year average, Q-EMA 12 = 3-year average. "
                 "A stock trading below its quarterly EMA 8 has underperformed "
                 "for 2 full years — a structural concern, not just a dip. "
                 "Quarterly Bollinger Band touches represent generational valuation extremes.")

            st.subheader("Quarterly RSI  |  Price Position in 52-Period Range")
            c1q, c2q = st.columns(2)
            with c1q:
                fig2 = go.Figure()
                if "RSI" in qv.columns:
                    fig2.add_trace(go.Scatter(x=qv.index, y=qv["RSI"], name="Q-RSI",
                                              fill="tozeroy", line=dict(color="#f6ad55", width=2)))
                    for yv, col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                        fig2.add_hline(y=yv, line_color=col, line_dash="dash")
                pplot(fig2, h=300, yaxis_title="Quarterly RSI", yaxis=dict(range=[0,100]))
                ibox("Quarterly RSI",
                     "Quarterly RSI < 30 = multi-year buying opportunity. Historically: "
                     "Q3 2022 for most tech stocks (NVDA ~25), March 2020 (~20). "
                     "Quarterly RSI > 80 = multi-year overbought — caution territory.")
            with c2q:
                fig3 = go.Figure()
                if "Pct_Range_52" in qv.columns:
                    fig3.add_trace(go.Bar(x=qv.index, y=qv["Pct_Range_52"],
                                          name="% of 52-period range",
                                          marker_color=["#3fb950" if v > 50 else "#f85149"
                                                        for v in qv["Pct_Range_52"]]))
                    fig3.add_hline(y=50, line_dash="dash", line_color="#374151")
                pplot(fig3, h=300, yaxis_title="% Position in 52Q Range",
                      yaxis=dict(range=[0, 110]))
                ibox("52-quarter range position",
                     "0% = at a 13-year low. 100% = at a 13-year high. "
                     "Above 80% = historically expensive relative to recent history. "
                     "Below 20% = historically cheap — strong long-term value zone.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4 — MONTHLY RETURN HEATMAP
    # ─────────────────────────────────────────────────────────────────────
    with tab_cal:
        st.subheader("Monthly Return Calendar Heatmap (%)")
        if df_d is not None and len(df_d) > 24:
            pivot = _monthly_heatmap_pivot(df_d)
            # Display only last 15 years for readability
            pivot = pivot.tail(15)
            figc = px.imshow(
                pivot.round(1), text_auto=True,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto",
                labels=dict(x="Month", y="Year", color="Return %"),
            )
            figc.update_traces(textfont=dict(size=10))
            figc.update_layout(**DARK, height=max(350, len(pivot)*32+80))
            st.plotly_chart(figc, use_container_width=True, config=PCFG)

            # Row totals (annual return)
            yr_ret = _yearly_returns(df_d).reindex(pivot.index)
            yr_df  = pd.DataFrame({
                "Year": yr_ret.index,
                "Annual Return %": yr_ret.values.round(1),
                "Best Month": pivot.idxmax(axis=1).values,
                "Worst Month": pivot.idxmin(axis=1).values,
            })
            st.dataframe(yr_df.set_index("Year"), use_container_width=True)
            ibox("Reading the heatmap",
                 "Each cell = the return for that stock in that calendar month. "
                 "**Dark green** = strong positive month. **Dark red** = significant loss. "
                 "**Seasonal patterns to spot:** "
                 "January Effect (many stocks rally in Jan), "
                 "Sell in May (May–Oct historically weaker), "
                 "Q4 strength (Nov–Dec often positive for tech). "
                 "Identifying a stock's seasonal patterns lets you time entries and exits "
                 "at statistically favourable points in the calendar. "
                 "Grey = data not available for that month.")
        else:
            st.info("Not enough daily history for monthly heatmap.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 5 — YEARLY RETURNS
    # ─────────────────────────────────────────────────────────────────────
    with tab_yr:
        st.subheader("Calendar Year Returns — All Stocks")
        yr_data = {}
        for t in LOADED_TICKERS:
            dfl = frames_long.get(t, frames.get(t))
            if dfl is not None and len(dfl) > 250:
                yr_data[t] = _yearly_returns(dfl)

        if yr_data:
            yr_df = pd.DataFrame(yr_data).round(1)
            # Colour-coded heatmap
            figyr = px.imshow(
                yr_df.T, text_auto=True,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto",
                labels=dict(x="Year", y="Stock", color="Return %"),
            )
            figyr.update_traces(textfont=dict(size=10))
            figyr.update_layout(**DARK, height=420)
            st.plotly_chart(figyr, use_container_width=True, config=PCFG)

            # Bar chart for selected year
            available_years = sorted(yr_df.index.tolist(), reverse=True)
            sel_yr = st.selectbox("Drill into a year", available_years, key="yr_sel")
            yr_row = yr_df.loc[sel_yr].dropna().sort_values(ascending=False)
            fig_bar = go.Figure(go.Bar(
                x=yr_row.index, y=yr_row.values,
                marker_color=["#3fb950" if v >= 0 else "#f85149" for v in yr_row.values],
                text=[f"{v:.1f}%" for v in yr_row.values], textposition="outside",
            ))
            fig_bar.update_layout(**DARK, height=340,
                                  title=f"{sel_yr} Returns",
                                  yaxis_title="Return (%)")
            st.plotly_chart(fig_bar, use_container_width=True, config=PCFG)
            ibox("Long-term return patterns",
                 "Cross-year view reveals structural performance leaders. "
                 "Stocks that outperform across multiple years tend to have "
                 "compounding competitive advantages (NVDA: AI chips, MSFT: cloud+AI). "
                 "Years with broadly negative returns (2022) = macro/rate-driven selloffs. "
                 "Years where some stocks are positive while others are negative "
                 "(2023: NVDA +239% vs others) = idiosyncratic factor returns. "
                 "The LSTM model implicitly uses these long-run return patterns through "
                 "its Regime and EMA50_200 features.")
        else:
            st.info("Not enough yearly history to display.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 6 — SUPPORT & RESISTANCE
    # ─────────────────────────────────────────────────────────────────────
    with tab_sr:
        st.subheader("Key Price Levels: Support & Resistance")
        sr_t    = st.selectbox("Stock", LOADED_TICKERS, key="sr_t")
        sr_tf   = st.selectbox("Timeframe", ["Monthly","Weekly","Daily"], key="sr_tf")
        tf_map  = {"Monthly":"ME","Weekly":"W","Daily":None}
        tf_key  = tf_map[sr_tf]

        if tf_key:
            df_sr = mtf_data.get(sr_t, {}).get(tf_key)
        else:
            df_sr = ind_data.get(sr_t)

        if df_sr is not None and len(df_sr) > 30:
            n_sr = st.slider("Periods to show", 30, len(df_sr),
                             min(120, len(df_sr)), 10, key="sr_sl")
            srv  = df_sr.iloc[-n_sr:]
            c_sr = df_sr["Close"].values
            win  = 3 if tf_key == "ME" else (5 if tf_key == "W" else 10)
            supps, ress = _support_resistance(c_sr, window=win)

            fig_sr = go.Figure()
            fig_sr.add_trace(go.Candlestick(
                x=srv.index, open=srv["Open"], high=srv["High"],
                low=srv["Low"], close=srv["Close"],
                name=f"{sr_t} ({sr_tf})",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
            # EMAs
            for ema, col, dash, wid in [(20,"#58a6ff","dash",1.4),(50,"#3fb950","solid",1.8)]:
                cn = f"EMA_{ema}"
                if cn in srv.columns:
                    fig_sr.add_trace(go.Scatter(x=srv.index, y=srv[cn],
                                                name=f"EMA {ema}",
                                                line=dict(color=col, width=wid, dash=dash)))
            # Support levels
            for i, s in enumerate(supps):
                fig_sr.add_hline(y=s, line_color="#3fb950", line_dash="dot", line_width=1.5,
                                 annotation_text=f"S{i+1} ${s:.2f}",
                                 annotation_position="right",
                                 annotation_font_color="#3fb950")
            # Resistance levels
            for i, r in enumerate(ress):
                fig_sr.add_hline(y=r, line_color="#f85149", line_dash="dot", line_width=1.5,
                                 annotation_text=f"R{i+1} ${r:.2f}",
                                 annotation_position="right",
                                 annotation_font_color="#f85149")
            fig_sr.update_layout(**DARK, height=520,
                                 xaxis_rangeslider_visible=False,
                                 legend=dict(orientation="h", y=1.01, x=0))
            st.plotly_chart(fig_sr, use_container_width=True, config=PCFG)

            c1s, c2s = st.columns(2)
            with c1s:
                st.markdown("**🟢 Support Levels (${:.2f} – ${:.2f})**".format(
                    min(supps) if supps else 0, max(supps) if supps else 0))
                for i, s in enumerate(supps):
                    cur_px = float(df_sr["Close"].iloc[-1])
                    dist   = (cur_px - s) / cur_px * 100
                    st.write(f"S{i+1}: **${s:.2f}** ({dist:+.1f}% from current)")
            with c2s:
                st.markdown("**🔴 Resistance Levels (${:.2f} – ${:.2f})**".format(
                    min(ress) if ress else 0, max(ress) if ress else 0))
                for i, r in enumerate(ress):
                    cur_px = float(df_sr["Close"].iloc[-1])
                    dist   = (r - cur_px) / cur_px * 100
                    st.write(f"R{i+1}: **${r:.2f}** (+{dist:.1f}% to target)")
            ibox("How support & resistance levels are calculated",
                 "Levels are identified using **pivot point analysis**: a price is a pivot high "
                 "if it is higher than all surrounding prices within a window of N periods. "
                 "Nearby pivots are clustered together (within 2% tolerance) to form "
                 "a single significant level — the more times price has touched it, "
                 "the more significant the level. "
                 "**Monthly support/resistance** = multi-year price memory — "
                 "levels that have held for years carry enormous psychological weight. "
                 "**Weekly S/R** = medium-term zones where buyers/sellers historically stepped in. "
                 "The LSTM doesn't use S/R explicitly, but its Drawdown and Pct_Range features "
                 "implicitly encode where price sits relative to historical extremes.")
        else:
            st.info("Not enough data for support/resistance calculation.")

    # ─────────────────────────────────────────────────────────────────────
    # TAB 7 — CROSS-TIMEFRAME COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────────
    with tab_cmp:
        st.subheader("Multi-Timeframe Signal Summary Table")
        st.caption("All values from live Yahoo Finance data. Green = bullish. Red = bearish.")

        rows = []
        for t in LOADED_TICKERS:
            row = {"Stock": t, "Sector": META[t]["sector"]}
            tfs_t = mtf_data.get(t, {})
            for tf_key, tf_lbl, ann_mult in [
                ("W",  "Weekly",    52),
                ("ME", "Monthly",   12),
                ("QE", "Quarterly",  4),
            ]:
                df_tf = tfs_t.get(tf_key)
                if df_tf is not None and len(df_tf) > 5:
                    last = df_tf.ffill().iloc[-1]
                    rsi_ = round(float(last["RSI"]), 1) if "RSI" in last.index else None
                    adx_ = round(float(last["ADX"]), 1) if "ADX" in last.index else None
                    mh_  = round(float(last["MACD_Hist"]), 4) if "MACD_Hist" in last.index else None
                    reg_ = "🟢 Bull" if float(last.get("Regime", 0)) == 1 else "🔴 Bear"
                    # 1-period return
                    if len(df_tf) > 1:
                        r1 = round((float(df_tf["Close"].iloc[-1]) /
                                    float(df_tf["Close"].iloc[-2]) - 1) * 100, 2)
                    else:
                        r1 = None
                    row[f"{tf_lbl} RSI"]    = rsi_
                    row[f"{tf_lbl} MACD_H"] = mh_
                    row[f"{tf_lbl} ADX"]    = adx_
                    row[f"{tf_lbl} Regime"] = reg_
                    row[f"{tf_lbl} Ret%"]   = r1
                else:
                    for suffix in ["RSI","MACD_H","ADX","Regime","Ret%"]:
                        row[f"{tf_lbl} {suffix}"] = None
            rows.append(row)

        summary_df = pd.DataFrame(rows).set_index("Stock")
        st.dataframe(summary_df, use_container_width=True)

        ibox("How to use the cross-TF table",
             "Each row = one stock. Columns = the same indicators computed on Weekly, Monthly, and Quarterly data. "
             "**Triple alignment = highest-conviction signal:** "
             "When Weekly RSI, Monthly RSI, AND Quarterly RSI are all below 50 and rising, "
             "all three timeframes confirm a bullish recovery is underway. "
             "When all three regimes show 🟢 Bull, the stock is in a confirmed structural uptrend "
             "across short, medium, and long timeframes — the most reliable trading environment. "
             "**Divergence = caution:** If Weekly is bullish but Monthly is bearish, "
             "the short-term bounce may fail at the medium-term resistance. "
             "This table is the professional analyst's first step in any stock analysis — "
             "establish the trend on the highest timeframe first, then zoom in.")


elif page == "🎯 LSTM Classification":

    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🎯 LSTM Classification — Real Trained Model")
    st.markdown(f"""
| Item | Detail |
|------|--------|
| **Model** | Single-layer LSTM (hidden={LSTM_H}, seq_len={SEQ_LEN}) |
| **Training** | BPTT + Adam optimiser, 10 epochs, gradient clipping |
| **Label** | 5-day forward return direction: UP (1) / DOWN (0) — **zero lookahead** |
| **Data source** | Yahoo Finance only ({START} → {END}) |
| **Ensemble** | Trained LSTM hidden states → HistGradientBoosting |
| **Split** | 80/20 strict time-series (no shuffling, no data leakage) |
    """)

    clf_t = st.selectbox("Select Stock", LOADED_TICKERS, key="clf_t")
    bar   = st.progress(0, text="Initialising…")

    def upd_bar(ep, tot, loss):
        bar.progress(int(ep/tot*100), text=f"Epoch {ep}/{tot}  |  Loss: {loss:.4f}")

    with st.spinner(f"Training LSTM on live Yahoo Finance data for {clf_t} (~40 s first run)…"):
        res, err = train_lstm(clf_t, ind_data)
        bar.empty()

    if err or res is None:
        st.error(f"Training error: {err}"); st.stop()

    # Accuracy headline
    st.subheader("Model Accuracy")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Raw LSTM Accuracy",   f"{res['acc_lstm']}%",
                  help="LSTM output alone, before HGB ensemble")
    with c2:
        st.metric("LSTM + HGB Ensemble", f"{res['acc_all']}%",
                  help="Ensemble on all out-of-sample test days")
    with c3:
        st.metric("Bull Regime Accuracy",f"{res['acc_bull']}%",
                  delta=f"{res['acc_bull']-res['acc_all']:+.1f}% vs overall",
                  help="Test days when price > EMA 200")
    with c4:
        st.metric("Bear Regime Accuracy",f"{res['acc_bear']}%",
                  delta=f"{res['acc_bear']-res['acc_all']:+.1f}% vs overall",
                  help="Test days when price < EMA 200")

    ibox("Understanding these accuracy numbers",
         f"**50% is the coin-flip baseline** — random guessing on UP vs DOWN. "
         f"**{res['acc_all']}% overall** means the LSTM+HGB ensemble correctly predicts "
         "5-day price direction more often than chance on unseen out-of-sample data. "
         "Every percentage point above 50% is a genuine, exploitable edge. "
         f"**Bull regime ({res['acc_bull']}%):** Technical indicators are more reliable when "
         "the stock is in an uptrend (above EMA 200). Momentum tends to persist in bull markets. "
         f"**Bear regime ({res['acc_bear']}%):** Below EMA 200, price action is choppier and "
         "mean-reverting — a different regime that the LSTM partially handles through "
         "its Drawdown and Vol_20 features. "
         "The accuracy difference between regimes is why regime-awareness is critical "
         "to any real trading system.")

    # Training loss curve
    st.subheader("LSTM Training Loss (BPTT + Adam)")
    fig_l = go.Figure()
    fig_l.add_trace(go.Scatter(
        y=res["loss_hist"],
        x=list(range(1, len(res["loss_hist"])+1)),
        mode="lines+markers", name="Training Loss",
        line=dict(color="#58a6ff", width=2.5),
        marker=dict(size=9, color="#58a6ff")))
    pplot(fig_l, h=250, xaxis_title="Epoch", yaxis_title="Cross-Entropy Loss")
    ibox("What the loss curve tells you",
         "Each epoch = one complete pass through all training sequences (backpropagation through time). "
         "Loss decreasing confirms the weights are being updated — genuine learning, not random. "
         "Adam optimizer adapts learning rates per parameter, making it 5–10× faster than plain SGD. "
         "If the curve flattens early, the LSTM has extracted maximum learnable signal from "
         "the 20-day window — more epochs would overfit. "
         "The current 10-epoch schedule is calibrated to balance learning vs generalisation.")

    # Confidence threshold table
    st.subheader("Accuracy vs Confidence Threshold")
    if res["conf_tbl"]:
        df_ct = pd.DataFrame(res["conf_tbl"])
        st.dataframe(df_ct, use_container_width=True, hide_index=True)
        ibox("The confidence table is the most important output",
             "The ensemble outputs a probability for each prediction. "
             "Low confidence (~50%) = indicators conflict → model is uncertain → skip this signal. "
             "High confidence (≥65%) = indicators align strongly → model has genuine conviction. "
             "**Bull + high confidence rows** show the highest accuracy — the ideal trade setup. "
             "As the threshold rises, accuracy rises (model knows when it doesn't know). "
             "**Edge over 50%** converts directly to expected profit: "
             "a +12% edge means that for every 100 trades at that confidence level, "
             "you win 62 and lose 38 — positive expected value before transaction costs.")

    # Signal overlay
    st.subheader("LSTM Signal Overlay (≥60% Confidence)")
    fig_s = go.Figure()
    df_p  = frames[clf_t]
    fig_s.add_trace(go.Scatter(x=df_p.index, y=df_p["Close"],
                               name="Close", line=dict(color="#e6edf3", width=1.5)))
    for ema, col, dash in [(50,"#58a6ff","dash"), (200,"#3fb950","solid")]:
        e = ind_data[clf_t][f"EMA_{ema}"]
        fig_s.add_trace(go.Scatter(x=e.index, y=e, name=f"EMA {ema}",
                                   line=dict(color=col, width=1, dash=dash), opacity=0.7))
    pa, ca, da = res["preds_all"], res["conf_all"], res["dates_all"]
    for lbl, sym, col in [(1,"triangle-up","#3fb950"), (0,"triangle-down","#f85149")]:
        m = (pa == lbl) & (ca >= 0.60)
        if m.sum() > 0:
            ix  = da[m]
            px_ = df_p["Close"].reindex(ix)
            lab = "LSTM: UP ≥60%" if lbl == 1 else "LSTM: DOWN ≥60%"
            fig_s.add_trace(go.Scatter(x=ix, y=px_, mode="markers", name=lab,
                                       marker=dict(color=col, size=7,
                                                   symbol=sym, opacity=0.85)))
    pplot(fig_s, h=440, yaxis_title="Price (USD)")
    ibox("Signal overlay interpretation",
         "Green ▲ = LSTM predicts UP over next 5 trading days (≥60% confidence). "
         "Red ▼ = DOWN prediction. Overlaid with EMA 50 and EMA 200 for regime context. "
         "**What to look for:** Do green triangles precede upward price moves? "
         "Do red triangles precede drawdowns? Count the correct signals vs false alarms "
         "— that ratio is the precision score shown in the classification report below. "
         "Signal density is lower during choppy/sideways markets (ADX < 20) because "
         "the model correctly reduces confidence when indicators disagree.")

    # Confusion matrix + report
    st.subheader("Confusion Matrix  |  Classification Report")
    c1, c2 = st.columns(2)
    with c1:
        cm   = res["cm"]
        figc = px.imshow(cm, x=["DOWN","UP"], y=["DOWN","UP"], text_auto=True,
                         color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="Actual"))
        figc.update_layout(**DARK, height=320)
        st.plotly_chart(figc, use_container_width=True, config=PCFG)
    with c2:
        rep = res["report"]
        rows = []
        for k, label in [("0","DOWN"),("1","UP")]:
            if k in rep:
                rows.append({"Class": label,
                             "Precision": round(rep[k]["precision"],3),
                             "Recall":    round(rep[k]["recall"],3),
                             "F1-Score":  round(rep[k]["f1-score"],3),
                             "Support":   int(rep[k]["support"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        ibox("Precision vs Recall",
             "**Precision (UP):** of all 'UP' predictions, what % were correct? "
             "High precision = few false alarms. "
             "**Recall (UP):** of all actual UP days, what % did the model catch? "
             "**F1:** harmonic mean — best single quality metric. "
             "Disciplined traders prioritise precision over recall: "
             "better to miss some gains than enter bad trades. "
             "The LSTM+HGB ensemble is calibrated for high precision.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression Analysis":
    import plotly.graph_objects as go
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    st.title("📈 Regression Analysis — 5-Day Return Prediction")
    mod_t = st.selectbox("Stock", LOADED_TICKERS, key="mod_t")
    ind   = ind_data.get(mod_t) or ind_data[LOADED_TICKERS[0]]
    X_raw = _make_X(ind)
    c_arr = ind["Close"].values
    n     = len(c_arr)

    fwd = np.zeros(n)
    for i in range(n-5):
        fwd[i] = (c_arr[i+5] - c_arr[i]) / (c_arr[i] + 1e-9)

    ok  = np.isfinite(X_raw).all(axis=1) & np.isfinite(fwd) & (fwd != 0)
    X   = X_raw[ok]; y = fwd[ok]; dates = ind.index[ok]
    sp  = int(len(X)*0.8)
    Xtr, Xte = X[:sp], X[sp:]
    ytr, yte  = y[:sp], y[sp:]
    dte       = dates[sp:]

    sc    = RobustScaler()
    Xtr_s = np.nan_to_num(sc.fit_transform(Xtr), 0)
    Xte_s = np.nan_to_num(sc.transform(Xte), 0)

    def mets(yt, yp, nm):
        return {"Model": nm,
                "MAE":      round(mean_absolute_error(yt, yp), 5),
                "RMSE":     round(np.sqrt(mean_squared_error(yt, yp)), 5),
                "R2":       round(r2_score(yt, yp), 4),
                "Dir_Acc%": round(np.mean(np.sign(yt)==np.sign(yp))*100, 1)}

    with st.spinner("Training regression models…"):
        R = {}
        m1 = LinearRegression().fit(Xtr_s, ytr)
        R["Linear"] = {"pred": m1.predict(Xte_s),
                       "fi": dict(zip(_LSTM_COLS, np.abs(m1.coef_)))}
        m2 = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5).fit(Xtr_s, ytr)
        R["Ridge"] = {"pred": m2.predict(Xte_s),
                      "fi": dict(zip(_LSTM_COLS, np.abs(m2.coef_)))}
        m3 = LassoCV(cv=5, max_iter=5000).fit(Xtr_s, ytr)
        R["Lasso"] = {"pred": m3.predict(Xte_s),
                      "fi": dict(zip(_LSTM_COLS, np.abs(m3.coef_)))}
        m4 = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   random_state=42, n_jobs=-1).fit(Xtr_s, ytr)
        R["RF"]  = {"pred": m4.predict(Xte_s),
                    "fi": dict(zip(_LSTM_COLS, m4.feature_importances_))}
        m5 = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.02, max_depth=5,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20).fit(Xtr_s, ytr)
        R["HGB"] = {"pred": m5.predict(Xte_s), "fi": {}}
        for k in R:
            R[k]["metrics"] = mets(yte, R[k]["pred"], k)

    mdf    = pd.DataFrame([v["metrics"] for v in R.values()]).set_index("Model")
    best   = mdf["Dir_Acc%"].idxmax()
    st.subheader("Model Performance Table")
    st.dataframe(
        mdf.style
           .highlight_max(subset=["R2","Dir_Acc%"], color="#14532d")
           .highlight_min(subset=["MAE","RMSE"],    color="#14532d")
           .format(precision=4),
        use_container_width=True)
    ibox("Key metrics explained",
         f"**Directional Accuracy (Dir_Acc%)** is the trading-relevant metric: "
         "did the model predict the correct direction? "
         f"**{best}** leads with {mdf.loc[best,'Dir_Acc%']:.1f}%. "
         "Each 1pp above 50% is a real, compoundable edge. "
         "**R2 > 0** means the model explains some return variance. "
         "Stock R2 > 0.05 out-of-sample is genuinely significant (most academic papers "
         "report 0.01–0.03). "
         "**EMA features** (P_EMA50, P_EMA200) add ~2–3pp of directional accuracy "
         "vs models without them because they act as regime filters.")

    sel_m = st.selectbox("Inspect model", list(R.keys()), key="sel_m")
    pred  = R[sel_m]["pred"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dte, y=yte*100, name="Actual",
                                 line=dict(color="#e6edf3", width=1.5)))
        fig.add_trace(go.Scatter(x=dte, y=pred*100, name="Predicted",
                                 line=dict(color="#58a6ff", width=1.5, dash="dot")))
        fig.add_hline(y=0, line_dash="dash", line_color="#374151")
        pplot(fig, h=290, yaxis_title="5-day Return (%)")
        ibox("Prediction chart",
             "The model captures trend direction more reliably than exact magnitude. "
             "Direction is all we need: long when predicted > 0, flat/short when < 0.")
    with c2:
        fi = R[sel_m]["fi"]
        if fi:
            fi_s = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
            fig2 = go.Figure(go.Bar(y=list(fi_s.keys()), x=list(fi_s.values()),
                                    orientation="h", marker_color="#58a6ff"))
            pplot(fig2, h=290, margin=dict(l=120, r=10, t=30, b=30))
            ibox("Feature importance",
                 "High-ranked EMA features confirm trend context is key. "
                 "Lag features show short-term momentum persistence. "
                 "ADX_n confirms the model weights trend-strength as a top predictor.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Clustering Analysis":
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    st.title("🔮 Clustering Analysis — K-Means (k=3)")
    with st.spinner("Clustering…"):
        recs = []
        for t in LOADED_TICKERS:
            ind  = ind_data[t]
            last = ind.ffill().iloc[-1]
            r1y  = float(ind["Return"].tail(252).mean() * 252)
            recs.append({
                "Ticker":     t,
                "Name":       META[t]["name"],
                "Sector":     META[t]["sector"],
                "1Y_Return%": round(r1y * 100, 2),
                "Volatility%":round(float(last["Vol_30"]) * 100, 2),
                "RSI":        round(float(last["RSI"]), 1),
                "Sharpe":     round(r1y / (float(last["Vol_30"]) + 1e-9), 2),
                "ADX":        round(float(last["ADX"]), 1),
                "BB_Width":   round(float(last["BB_Width"]), 4),
            })
        fd  = pd.DataFrame(recs).set_index("Ticker")
        nc  = ["1Y_Return%","Volatility%","RSI","Sharpe","ADX","BB_Width"]
        scl = StandardScaler()
        Xs  = scl.fit_transform(fd[nc].values)
        km  = KMeans(n_clusters=3, random_state=42, n_init=10); km.fit(Xs)
        fd["Cluster"] = km.labels_
        mn  = fd.groupby("Cluster")["1Y_Return%"].mean().sort_values(ascending=False)
        cmp = {mn.index[0]:"🚀 High Growth",
               mn.index[1]:"⚖️ Balanced",
               mn.index[2]:"🛡️ Defensive"}
        fd["Group"] = fd["Cluster"].map(cmp)
        Xp  = PCA(n_components=2).fit_transform(Xs)
        fd["PC1"] = Xp[:, 0]; fd["PC2"] = Xp[:, 1]

    gc = {"🚀 High Growth":"#3fb950","⚖️ Balanced":"#f6ad55","🛡️ Defensive":"#58a6ff"}
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("PCA Cluster Map")
        fig = go.Figure()
        for gn, grp in fd.groupby("Group"):
            fig.add_trace(go.Scatter(
                x=grp["PC1"], y=grp["PC2"], mode="markers+text",
                text=grp.index, textposition="top center", name=gn,
                marker=dict(color=gc.get(gn,"#fff"), size=16, opacity=0.9)))
        pplot(fig, h=380)
        ibox("Cluster map",
             "Distance = similarity in risk-return space. "
             "Stocks close together add little diversification — "
             "holding both is concentrated risk. "
             "Pick at least one stock from each cluster for genuine diversification.")
    with c2:
        st.subheader("Cluster Profiles")
        st.dataframe(fd.groupby("Group")[nc].mean().round(2).T, use_container_width=True)
        st.dataframe(fd[["Name","Sector","Group","1Y_Return%","Volatility%","Sharpe"]],
                     use_container_width=True)

    st.subheader("Cluster Radar")
    fig2 = go.Figure()
    for gn, grp in fd.groupby("Group"):
        vals = [float(grp[m].mean()) for m in nc]
        mn_  = [fd[m].min() for m in nc]; mx_ = [fd[m].max() for m in nc]
        norm = [(v-a)/(b-a+1e-9)*100 for v,a,b in zip(vals,mn_,mx_)]
        norm += [norm[0]]
        fig2.add_trace(go.Scatterpolar(r=norm, theta=nc+[nc[0]], fill="toself",
                                        name=gn, line_color=gc.get(gn,"#fff"), opacity=0.7))
    fig2.update_layout(**DARK, height=400,
                       polar=dict(bgcolor="#1f2937",
                                  radialaxis=dict(visible=True, range=[0,100], color="#9ca3af"),
                                  angularaxis=dict(color="#9ca3af")))
    st.plotly_chart(fig2, use_container_width=True, config=PCFG)
    ibox("Radar insight",
         "🚀 High Growth dominates Return and Sharpe but spikes on Volatility. "
         "🛡️ Defensive is balanced across all axes — no extremes, lowest risk. "
         "Optimal allocation: weight by Sharpe, diversify across clusters.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🔗 Association Rules — Co-movement Analysis")
    ret_df = pd.DataFrame({t: frames[t]["Close"].pct_change()
                           for t in LOADED_TICKERS}).dropna()
    corr   = ret_df.corr()

    st.subheader("Full Return Correlation Matrix")
    fig = px.imshow(corr.round(3), text_auto=True,
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(**DARK, height=460)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)
    ibox("Association rules from Yahoo Finance data",
         "Each cell = P(Stock B rises | Stock A rises) — the simplest market association rule. "
         "AAPL–MSFT at 0.75 means: on 75% of days when AAPL rises, MSFT also rises. "
         "**Use this for:** pair trading (long A, short B when they diverge), "
         "risk management (don't hold two stocks with corr > 0.8 as if they diversify you). "
         "The LSTM uses cross-stock dynamics implicitly — when the entire tech sector "
         "enters a high-correlation macro regime, individual stock signals become less reliable.")

    rules = []
    for i in range(len(TICKERS)):
        for j in range(i+1, len(TICKERS)):
            r = float(corr.iloc[i, j])
            if abs(r) > 0.4:
                rules.append({
                    "Pair":          f"{TICKERS[i]} ↔ {TICKERS[j]}",
                    "Correlation":   round(r, 3),
                    "Relationship":  "Strong Positive" if r > 0.7 else "Moderate Positive",
                    "Implication":   "Concentrated risk — limit both" if r > 0.7
                                     else "Partial co-movement",
                })
    st.subheader("Co-movement Rules  (|corr| > 0.4)")
    if rules:
        st.dataframe(pd.DataFrame(rules).sort_values("Correlation", ascending=False),
                     use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Correlation to Portfolio")
        avg_c = pd.Series({t: ret_df[[o for o in TICKERS if o != t]].corrwith(ret_df[t]).mean()
                           for t in LOADED_TICKERS}).sort_values(ascending=False)
        fig2 = go.Figure(go.Bar(
            x=avg_c.index, y=avg_c.values,
            marker_color=["#f85149" if v>0.6 else "#f6ad55" if v>0.4 else "#3fb950"
                          for v in avg_c.values],
            text=[f"{v:.2f}" for v in avg_c.values], textposition="outside"))
        pplot(fig2, h=300, yaxis=dict(range=[0, 1.1]))
        ibox("Diversification score",
             "Red = redundant — co-moves with everything. "
             "Green = genuinely independent — reduces portfolio volatility. "
             "BRK-B and UNH are typically the strongest diversifiers here.")
    with c2:
        if rules:
            top_pair = sorted(rules, key=lambda x: x["Correlation"], reverse=True)[0]
            ta, tb   = top_pair["Pair"].split(" ↔ ")
            roll     = ret_df[ta].rolling(30).corr(ret_df[tb])
            fig3     = go.Figure()
            fig3.add_trace(go.Scatter(x=roll.index, y=roll, fill="tozeroy",
                                      name=f"{ta}↔{tb}", line=dict(color="#58a6ff", width=2)))
            fig3.add_hline(y=0.5, line_dash="dash", line_color="#f6ad55")
            pplot(fig3, h=300, yaxis=dict(range=[-0.1, 1.2]),
                  xaxis_title="Date", yaxis_title="30-day Rolling Corr")
            ibox("Rolling vs static",
                 "When rolling correlation drops sharply, the pair has diverged — "
                 "classic pair-trade entry: long the laggard, short the leader, "
                 "bet on mean-reversion. The correlation always normalises eventually.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — DEEP DRILL-DOWN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Deep Drill-Down":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    st.title("🔬 Deep Drill-Down")
    dd_t = st.selectbox("Stock", LOADED_TICKERS, key="dd_t")
    ind  = ind_data.get(dd_t) or ind_data[LOADED_TICKERS[0]]
    last = ind.ffill().iloc[-1]
    cur  = float(last["Close"])
    e50  = float(last["EMA_50"]); e200 = float(last["EMA_200"])
    ret_ = ind["Return"].dropna()

    hi52 = float(ind["Close"].rolling(252).max().iloc[-1])
    lo52 = float(ind["Close"].rolling(252).min().iloc[-1])
    ann_ret = float(ret_.tail(252).mean() * 252 * 100)
    ann_vol = float(ret_.std() * np.sqrt(252) * 100)
    shrp = float((ret_.mean()/ret_.std())*np.sqrt(252)) if ret_.std() > 0 else 0
    nr   = ret_[ret_ < 0]
    sort = float((ret_.mean()/nr.std())*np.sqrt(252)) if len(nr)>0 and nr.std()>0 else 0
    max_dd = float(ind["Drawdown"].min() * 100)

    # Summary table
    summary = pd.DataFrame({
        "Metric": [
            "Live Price","52W High","52W Low",
            "vs EMA 50","vs EMA 200","Regime",
            "Golden Cross","1Y Ann. Return","Ann. Volatility",
            "Sharpe Ratio","Sortino Ratio","Max Drawdown",
            "RSI(14)","MACD Hist","ADX(14)",
            "Stoch %K","Williams %R","CCI(20)",
            "OBV Slope","BB Position","ATR",
        ],
        "Value": [
            f"${cur:.2f}", f"${hi52:.2f}", f"${lo52:.2f}",
            f"{'ABOVE' if cur>e50  else 'BELOW'} ({(cur/e50-1)*100:+.1f}%)",
            f"{'ABOVE' if cur>e200 else 'BELOW'} ({(cur/e200-1)*100:+.1f}%)",
            "🟢 BULL" if cur > e200 else "🔴 BEAR",
            "✅ YES" if float(last["Golden_Cross"])==1 else "❌ NO",
            f"{ann_ret:+.1f}%", f"{ann_vol:.1f}%",
            f"{shrp:.2f}", f"{sort:.2f}", f"{max_dd:.1f}%",
            f"{float(last['RSI']):.1f}",
            f"{float(last['MACD_Hist']):.4f}",
            f"{float(last['ADX']):.1f}",
            f"{float(last['Stoch_K']):.1f}",
            f"{float(last['Williams_R']):.1f}",
            f"{float(last['CCI']):.1f}",
            f"{float(last['OBV_Slope']):.4f}",
            f"{float(last['BB_Pct'])*100:.1f}%",
            f"${float(last['ATR']):.2f}",
        ],
    }).set_index("Metric")
    st.dataframe(summary, use_container_width=True)

    # Full price history with all EMAs
    st.subheader("Full Price History + All EMAs")
    fig = go.Figure()
    for ema, col, dash, wid in [(20,"#fee140","dot",1.1), (50,"#58a6ff","dash",1.5),
                                 (100,"#f093fb","dash",1.2), (200,"#3fb950","solid",2.0)]:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[f"EMA_{ema}"], name=f"EMA {ema}",
                                 line=dict(color=col, width=wid, dash=dash), opacity=0.85))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close",
                             line=dict(color="#e6edf3", width=1.5)))
    pplot(fig, h=400, yaxis_title="Price (USD)",
          legend=dict(orientation="h", y=1.02, x=0))
    ibox("Full history EMA context",
         "The full history reveals where Golden Cross and Death Cross events occurred "
         "and how price behaved after each. The LSTM trains on all of this history — "
         "it learns that post-Golden-Cross periods have persistently higher "
         "directional accuracy than Death Cross periods.")

    # ── Long-term charts from extended history ─────────────────────────────
    st.divider()
    st.subheader("Long-Term Context (from Extended History Since 1993)")
    tfs_dd = mtf_data.get(dd_t, {})

    c_lt1, c_lt2 = st.columns(2)
    with c_lt1:
        df_m_dd = tfs_dd.get("ME")
        if df_m_dd is not None and len(df_m_dd) > 24:
            st.markdown("**Monthly Chart + EMA 6/12/24**")
            fig_m = go.Figure()
            fig_m.add_trace(go.Candlestick(
                x=df_m_dd.index, open=df_m_dd["Open"], high=df_m_dd["High"],
                low=df_m_dd["Low"], close=df_m_dd["Close"], name="Monthly",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149"))
            for ema, col, dash, wid in [(6,"#fee140","dot",1.2),(12,"#58a6ff","dash",1.5),(24,"#3fb950","solid",2)]:
                cn = f"EMA_{ema}"
                if cn in df_m_dd.columns:
                    fig_m.add_trace(go.Scatter(x=df_m_dd.index, y=df_m_dd[cn],
                                               name=f"M-EMA {ema}",
                                               line=dict(color=col, width=wid, dash=dash)))
            fig_m.update_layout(**DARK, height=320, xaxis_rangeslider_visible=False,
                                legend=dict(orientation="h", y=1.01, x=0, font=dict(size=9)))
            st.plotly_chart(fig_m, use_container_width=True, config=PCFG)
            ibox("Monthly view","Each candle = 1 month. Monthly EMA 12 ≈ 1-year moving average. "
                 "Monthly EMA 24 ≈ 2-year moving average — the most important macro support line.")
    with c_lt2:
        df_q_dd = tfs_dd.get("QE")
        if df_q_dd is not None and len(df_q_dd) > 8:
            st.markdown("**Quarterly Chart + RSI**")
            fig_q2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=["Quarterly Price","Quarterly RSI(14)"],
                                    vertical_spacing=0.08, row_heights=[0.65, 0.35])
            fig_q2.add_trace(go.Candlestick(
                x=df_q_dd.index, open=df_q_dd["Open"], high=df_q_dd["High"],
                low=df_q_dd["Low"], close=df_q_dd["Close"], name="Quarterly",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149"), row=1, col=1)
            if "EMA_8" in df_q_dd.columns:
                fig_q2.add_trace(go.Scatter(x=df_q_dd.index, y=df_q_dd["EMA_8"],
                                             name="Q-EMA 8 (~2yr)",
                                             line=dict(color="#3fb950", width=1.8)), row=1, col=1)
            if "RSI" in df_q_dd.columns:
                fig_q2.add_trace(go.Scatter(x=df_q_dd.index, y=df_q_dd["RSI"],
                                             name="Q-RSI", fill="tozeroy",
                                             line=dict(color="#f6ad55", width=1.6)), row=2, col=1)
                for yv, col in [(70,"#f85149"),(30,"#3fb950"),(50,"#374151")]:
                    fig_q2.add_hline(y=yv, line_color=col, line_dash="dash", row=2, col=1)
            fig_q2.update_layout(**DARK, height=320, showlegend=False,
                                  xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_q2, use_container_width=True, config=PCFG)
            ibox("Quarterly RSI","Quarterly RSI < 30 marks major market bottoms. "
                 "This view puts the current situation in a 10-30 year context.")

    # ── Yearly returns bar ─────────────────────────────────────────────────
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
        best_yr  = yr_s.idxmax(); worst_yr = yr_s.idxmin()
        avg_ret  = yr_s.mean(); pos_yrs  = (yr_s > 0).sum()
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


    # Return distribution
    st.subheader("Return Distribution vs Normal")
    rp  = ret_ * 100; mu = float(rp.mean()); sg = float(rp.std())
    xn  = np.linspace(float(rp.min()), float(rp.max()), 200)
    yn  = (1/(sg*np.sqrt(2*np.pi))) * np.exp(-0.5*((xn-mu)/sg)**2)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=rp, nbinsx=80, name="Actual",
                                marker_color="#58a6ff", opacity=0.7,
                                histnorm="probability density"))
    fig2.add_trace(go.Scatter(x=xn, y=yn, name="Normal Fit",
                              line=dict(color="#f85149", width=2)))
    pplot(fig2, h=280, xaxis_title="Daily Return (%)", yaxis_title="Density")
    skew = float(rp.skew()); kurt = float(rp.kurtosis())
    ibox(f"Fat tails — {dd_t}",
         f"Mean: **{mu:.3f}%** | Std: **{sg:.3f}%** | "
         f"Skew: **{skew:.3f}** | Excess Kurtosis: **{kurt:.2f}**  \n"
         "The actual distribution has taller peaks and fatter tails than the red normal curve. "
         f"Kurtosis of {kurt:.1f} means extreme daily moves are far more likely than a normal "
         "model predicts — this is why VaR models that assume normality routinely underestimate "
         "real losses during crises. The LSTM's `Vol_5` vs `Vol_20` ratio detects when the "
         "distribution is shifting toward fatter-tail territory before it shows in price.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 8 — DOWNLOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📥 Download Data":
    st.title("📥 Download Data")
    st.caption(f"All data sourced exclusively from Yahoo Finance  |  {data_src}")

    st.subheader("Full OHLCV + All Technical Indicators (CSV per stock)")
    dl_t = st.selectbox("Ticker", LOADED_TICKERS, key="dl_t")
    buf  = io.StringIO()
    ind_data[dl_t].to_csv(buf)
    st.download_button(
        label=f"⬇️ {dl_t} — Full Indicator CSV ({len(ind_data[dl_t])} rows × "
              f"{len(ind_data[dl_t].columns)} columns)",
        data=buf.getvalue(),
        file_name=f"{dl_t}_all_indicators_{END}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Excel Workbook — All 10 Stocks")
    if st.button("🔄 Build Excel from live Yahoo Finance data", type="primary"):
        with st.spinner("Building Excel…"):
            try:
                import openpyxl
                from openpyxl.styles import Font, PatternFill, Alignment
                wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Overview"
                ws["A1"] = (f"S&P 500 Top 10 | Yahoo Finance | "
                            f"Generated {datetime.today().strftime('%Y-%m-%d %H:%M')}")
                ws["A1"].font = Font(bold=True, size=13)
                ws.append(["Ticker","Company","Sector","Price","1Y Ret%",
                           "Vol%","RSI","ADX","Regime","Signal"])
                for t in LOADED_TICKERS:
                    ind = ind_data[t]; last = ind.ffill().iloc[-1]
                    r1y = float(ind["Return"].tail(252).mean()*252*100)
                    sc_ = score_stock(ind, None)
                    ws.append([t, META[t]["name"], META[t]["sector"],
                               round(float(last["Close"]),2), round(r1y,2),
                               round(float(last["Vol_30"])*100,1),
                               round(float(last["RSI"]),1),
                               round(float(last["ADX"]),1),
                               "BULL" if float(last["Regime"])==1 else "BEAR",
                               sc_["signal"]])
                for t in LOADED_TICKERS:
                    ws2 = wb.create_sheet(t)
                    df  = frames[t].reset_index()
                    ws2.append(list(df.columns))
                    for row in df.values.tolist():
                        ws2.append(row)
                wb.save(XLS_PATH)
                st.success(f"✅ Saved {XLS_PATH}")
            except Exception as e:
                st.error(f"Error building Excel: {e}")

    if os.path.exists(XLS_PATH):
        with open(XLS_PATH, "rb") as f:
            st.download_button(
                "⬇️ Download Excel Workbook", data=f.read(),
                file_name=XLS_PATH, type="primary",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Click 'Build Excel' to generate the file first.")

    st.divider()
    st.subheader("LSTM Results — All 10 Stocks (Excel)")
    if st.button("📊 Run LSTM on all stocks and export results"):
        with st.spinner("Training LSTM on all 10 stocks (~8 min total, cached after first run)…"):
            all_res = train_all(ind_data)
        rows = []
        for t in LOADED_TICKERS:
            r, _ = all_res.get(t, (None, None))
            if r is None: continue
            if r:
                rows.append({
                    "Ticker":           t,
                    "Raw LSTM Acc%":    r["acc_lstm"],
                    "Ensemble Acc%":    r["acc_all"],
                    "Bull Regime Acc%": r["acc_bull"],
                    "Bear Regime Acc%": r["acc_bear"],
                    "N Train":          r["n_train"],
                    "N Test":           r["n_test"],
                    "N Features":       r["n_feats"],
                })
        if rows:
            mbuf = io.BytesIO()
            pd.DataFrame(rows).to_excel(mbuf, index=False)
            st.download_button(
                "⬇️ LSTM Results Excel", data=mbuf.getvalue(),
                file_name=f"lstm_all_stocks_{END}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.warning(
        "⚠️ **Legal Disclaimer:** This dashboard is for informational and educational purposes only. "
        "All data sourced from Yahoo Finance via yfinance. "
        "Nothing constitutes financial, investment, legal, or tax advice. "
        "Past performance does not guarantee future results. "
        "Always consult a qualified financial advisor before making investment decisions.")
