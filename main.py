"""
Kwality Construction Company — Forensic Financial Dashboard
Run:  pip install -r requirements.txt && streamlit run main.py
"""

import os, pathlib

# ── Auto-create .streamlit/config.toml next to this script ──
_dir = pathlib.Path(__file__).parent / ".streamlit"
_dir.mkdir(exist_ok=True)
(_dir / "config.toml").write_text(
    "[theme]\n"
    'primaryColor = "#1e40af"\n'
    'backgroundColor = "#f8fafc"\n'
    'secondaryBackgroundColor = "#ffffff"\n'
    'textColor = "#0f172a"\n'
    'font = "sans serif"\n'
    "[server]\nheadless = true\nport = 8501\n"
    "[browser]\ngatherUsageStats = false\n"
)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from bank_data import (
    SUMMARY, MONTHS, MONTHLY_CREDITS, MONTHLY_DEBITS, MONTHLY_CLOSING,
    MONTHLY_NET, EXPENSE_CATEGORIES, REVENUE_SOURCES, SUSPICIOUS,
    RECOMMENDATIONS, PROFIT_BRIDGE,
)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kwality Construction — Financial Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }
    .section-head {
        font-size: 1.3rem; font-weight: 800;
        margin-top: 0.6rem; margin-bottom: 0.2rem;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border-left: 4px solid #3b82f6; border-radius: 0.5rem;
        padding: 1rem 1.2rem; margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem; line-height: 1.7;
    }
    .alert-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
        border-left: 4px solid #ef4444; border-radius: 0.5rem;
        padding: 1rem 1.2rem; margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem; line-height: 1.7;
    }
    .green-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border-left: 4px solid #10b981; border-radius: 0.5rem;
        padding: 1rem 1.2rem; margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem; line-height: 1.7;
    }
    div[data-testid="stExpander"] details summary p { font-weight: 700; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────
def fmt(n):
    a = abs(n)
    sign = "-" if n < 0 else ""
    if a >= 1e7:  return f"{sign}₹{a/1e7:.2f} Cr"
    if a >= 1e5:  return f"{sign}₹{a/1e5:.1f} L"
    if a >= 1e3:  return f"{sign}₹{a/1e3:.0f}K"
    return f"{sign}₹{a:,.0f}"

def fmt_full(n):
    return f"₹{abs(n):,.0f}"

PLOTLY = dict(
    font=dict(family="DM Sans, sans-serif", size=12),
    margin=dict(l=40, r=20, t=45, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="white", font_size=12),
)


# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("### 🏗️ Kwality Construction Company")
    st.caption(f"{SUMMARY['period']}  ·  HDFC Bank A/c {SUMMARY['account']}  ·  Mussoorie")
with h2:
    st.markdown(
        "<div style='text-align:right;margin-top:12px'>"
        "<span style='background:#fef2f2;color:#dc2626;padding:6px 16px;"
        "border-radius:20px;font-weight:800;font-size:0.85rem'>"
        f"⚠ NET LOSS: {fmt(SUMMARY['net_change'])}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ─── KPI ROW ─────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Inflows",  fmt(SUMMARY["total_credits"]),  f"{SUMMARY['credit_count']} txns")
k2.metric("Total Outflows", fmt(SUMMARY["total_debits"]),   f"{SUMMARY['debit_count']} txns")
k3.metric("Net Cash Change", fmt(SUMMARY["net_change"]),    "Loss ↓", delta_color="inverse")
k4.metric("Opening Balance", fmt(SUMMARY["opening_balance"]))
k5.metric("Closing Balance", fmt(SUMMARY["closing_balance"]),
          f"{((SUMMARY['closing_balance']/SUMMARY['opening_balance'])-1)*100:.0f}%",
          delta_color="inverse")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Expense Deep Dive",
    "📈 Cash Flow & Revenue",
    "🔴 Suspicious Transactions",
    "🚀 Loss → Profit Roadmap",
])


# ═══════════════════════════════════════════════════════════════════
#  TAB 1 — EXPENSE DEEP DIVE
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-head">Where Every Rupee Goes</p>', unsafe_allow_html=True)
    st.caption("Total categorized: ₹8.73 Cr of ₹10.11 Cr debits  ·  Click categories below for vendor detail")

    # Donut + Bar
    c_donut, c_bar = st.columns([2, 3])

    with c_donut:
        fig = go.Figure(go.Pie(
            labels=[c["name"] for c in EXPENSE_CATEGORIES],
            values=[c["amount"] for c in EXPENSE_CATEGORIES],
            hole=0.55,
            marker_colors=[c["color"] for c in EXPENSE_CATEGORIES],
            textinfo="label+percent", textposition="outside",
            textfont_size=10,
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY, height=430, showlegend=False,
            annotations=[dict(text="<b>₹8.73Cr</b><br>Total Expenses",
                              x=0.5, y=0.5, font_size=13, showarrow=False,
                              font_family="JetBrains Mono")])
        st.plotly_chart(fig, use_container_width=True)

    with c_bar:
        cats = sorted(EXPENSE_CATEGORIES, key=lambda c: c["amount"])
        fig = go.Figure(go.Bar(
            y=[c["name"] for c in cats],
            x=[c["amount"] for c in cats],
            orientation="h",
            marker_color=[c["color"] for c in cats],
            text=[fmt(c["amount"]) for c in cats],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
            customdata=[c["pct"] for c in cats],
            hovertemplate="<b>%{y}</b><br>₹%{x:,.0f} (%{customdata:.1f}%)<extra></extra>",
        ))
        fig.update_layout(**PLOTLY, height=430,
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Amount (₹)"),
            yaxis_tickfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="alert-box">'
        '<strong>🔍 Key Finding:</strong> Materials (31%) + Sub-contractors (21%) + Labor (13%) '
        '= <strong>65% of outflows</strong>.  Cash withdrawals (12%) + Suspicious (6%) = '
        '<strong>18% with NO audit trail</strong> — ₹1.58 Cr with inadequate documentation.'
        '</div>', unsafe_allow_html=True)

    # ── Vendor Drill-Down ──
    st.markdown('<p class="section-head">Vendor-Level Drill Down</p>', unsafe_allow_html=True)

    for cat in EXPENSE_CATEGORIES:
        with st.expander(f"{cat['icon']}  **{cat['name']}** — {fmt(cat['amount'])} ({cat['pct']}%)"):
            ct, cv = st.columns([3, 2])
            with ct:
                df = pd.DataFrame(cat["vendors"], columns=["Vendor", "Amount (₹)"])
                df["Share"] = (df["Amount (₹)"] / cat["amount"] * 100).round(1).astype(str) + "%"
                df.index += 1
                df["Amount (₹)"] = df["Amount (₹)"].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(df, use_container_width=True,
                             height=min(len(cat["vendors"]) * 36 + 42, 420))
            with cv:
                top = cat["vendors"][:8]
                fig = go.Figure(go.Bar(
                    x=[v[1] for v in top],
                    y=[v[0][:28] for v in top],
                    orientation="h",
                    marker_color=cat["color"], marker_opacity=0.8,
                    text=[fmt(v[1]) for v in top],
                    textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono"),
                ))
                fig.update_layout(**PLOTLY, height=300,
                    xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                    yaxis=dict(autorange="reversed", tickfont_size=9))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f'<div class="insight-box">'
                f'📊 <strong>CA Insight:</strong> {cat["insight"]}<br><br>'
                f'📏 <strong>Benchmark:</strong> {cat["benchmark"]}'
                f'</div>', unsafe_allow_html=True)

    # ── Top 15 Payees ──
    st.markdown('<p class="section-head">Top 15 Payees (All Categories)</p>', unsafe_allow_html=True)
    rows = []
    for cat in EXPENSE_CATEGORIES:
        for vn, va in cat["vendors"]:
            rows.append(dict(Vendor=vn, Amount=va, Category=cat["name"]))
    dfa = pd.DataFrame(rows).sort_values("Amount", ascending=False).head(15)

    fig = go.Figure(go.Bar(
        x=dfa["Amount"], y=dfa["Vendor"].str[:32],
        orientation="h", marker_color="#1e40af",
        text=dfa["Amount"].apply(fmt), textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
        customdata=dfa["Category"],
        hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY, height=500,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Total Paid (₹)"),
        yaxis=dict(autorange="reversed", tickfont_size=10))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  TAB 2 — CASH FLOW & REVENUE
# ═══════════════════════════════════════════════════════════════════
with tab2:

    # ── Monthly Inflow vs Outflow ──
    st.markdown('<p class="section-head">Monthly Inflows vs Outflows</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Inflows",  x=MONTHS, y=MONTHLY_CREDITS,
        marker_color="#10b981", marker_opacity=0.85,
        text=[fmt(v) for v in MONTHLY_CREDITS], textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono")))
    fig.add_trace(go.Bar(name="Outflows", x=MONTHS, y=MONTHLY_DEBITS,
        marker_color="#ef4444", marker_opacity=0.85,
        text=[fmt(v) for v in MONTHLY_DEBITS], textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono")))
    fig.update_layout(**PLOTLY, height=430, barmode="group",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Amount (₹)"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Net Surplus / Deficit ──
    st.markdown('<p class="section-head">Monthly Net Surplus / Deficit</p>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=MONTHS, y=MONTHLY_NET,
        marker_color=["#10b981" if v >= 0 else "#ef4444" for v in MONTHLY_NET],
        text=[fmt(v) for v in MONTHLY_NET], textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
    ))
    fig.update_layout(**PLOTLY, height=320,
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                   zeroline=True, zerolinecolor="#475569", zerolinewidth=1.5))
    st.plotly_chart(fig, use_container_width=True)

    loss_months = sum(1 for v in MONTHLY_NET if v < 0)
    st.markdown(
        f'<div class="alert-box">'
        f'⚠️ <strong>{loss_months} of 12 months</strong> had negative net cash flow. '
        f'Worst: <strong>Mar-25 (−₹1.22 Cr)</strong> — massive year-end vendor settlements. '
        f'Only Sep-24 and Oct-24 showed significant surplus.'
        f'</div>', unsafe_allow_html=True)

    # ── Balance Trend ──
    st.markdown('<p class="section-head">Account Balance Trend</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=MONTHS, y=MONTHLY_CLOSING, mode="lines+markers+text",
        line=dict(color="#6366f1", width=3),
        marker=dict(size=8,
            color=["#ef4444" if v < 1e5 else "#f59e0b" if v < 5e5 else "#6366f1"
                   for v in MONTHLY_CLOSING]),
        text=[fmt(v) for v in MONTHLY_CLOSING],
        textposition="top center",
        textfont=dict(size=9, family="JetBrains Mono"),
    ))
    fig.add_hline(y=5e5, line_dash="dash", line_color="#f59e0b", opacity=.5,
                  annotation_text="₹5L Safe Minimum", annotation_position="top right")
    fig.add_hline(y=5e4, line_dash="dash", line_color="#ef4444", opacity=.5,
                  annotation_text="₹50K Danger", annotation_position="bottom right")
    fig.update_layout(**PLOTLY, height=360,
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Closing Balance (₹)"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Revenue Sources ──
    st.markdown('<p class="section-head">Revenue Sources — Who Pays You</p>', unsafe_allow_html=True)
    cr, cb = st.columns([2, 3])
    with cr:
        fig = go.Figure(go.Pie(
            labels=[r[0] for r in REVENUE_SOURCES],
            values=[r[1] for r in REVENUE_SOURCES],
            hole=0.5, marker_colors=[r[3] for r in REVENUE_SOURCES],
            textinfo="label+percent", textposition="outside", textfont_size=10,
        ))
        fig.update_layout(**PLOTLY, height=400, showlegend=False,
            annotations=[dict(text="<b>₹9.97Cr</b><br>Revenue",
                              x=0.5, y=0.5, font_size=13, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        dfr = pd.DataFrame(REVENUE_SOURCES, columns=["Source","Amount","Pct","Color"])
        fig = go.Figure(go.Bar(
            y=dfr["Source"], x=dfr["Amount"], orientation="h",
            marker_color=dfr["Color"],
            text=dfr["Amount"].apply(fmt), textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
        ))
        fig.update_layout(**PLOTLY, height=400,
            yaxis=dict(autorange="reversed", tickfont_size=10),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="alert-box">'
        '🎯 <strong>Concentration Risk:</strong> Wynberg Allen (32.5%) + Woodstock (25.9%) = '
        '<strong>58.4% from just 2 clients</strong>. Loss of either = insolvency within 2 months. '
        'Government payments (20%) arrive in large irregular lumps causing severe cash flow swings.'
        '</div>', unsafe_allow_html=True)

    # ── Transaction Imbalance ──
    st.markdown('<p class="section-head">Transaction Volume Imbalance</p>', unsafe_allow_html=True)
    cv1, cv2 = st.columns([1, 1])
    with cv1:
        fig = go.Figure(go.Pie(
            labels=["Debits (1,130)", "Credits (82)"],
            values=[SUMMARY["debit_count"], SUMMARY["credit_count"]],
            hole=0.6, marker_colors=["#ef4444", "#10b981"],
            textinfo="label+value", textfont_size=13,
        ))
        fig.update_layout(**PLOTLY, height=300, showlegend=False,
            annotations=[dict(text="<b>1 : 14</b><br>Ratio",
                              x=0.5, y=0.5, font_size=16, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
    with cv2:
        st.markdown("")
        st.metric("Avg Credit Size", fmt(SUMMARY["total_credits"] / SUMMARY["credit_count"]))
        st.metric("Avg Debit Size",  fmt(SUMMARY["total_debits"]  / SUMMARY["debit_count"]))
        st.markdown(
            '<div class="insight-box">'
            'Money arrives via <strong>82 large payments</strong> (avg ₹12.2L) from ~10 sources. '
            'It leaves as <strong>1,130 small payments</strong> (avg ₹89K) to 200+ parties. '
            'This extreme dispersion makes oversight extremely difficult.'
            '</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  TAB 3 — SUSPICIOUS TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-head">🔴 Flagged Transactions & Entities</p>', unsafe_allow_html=True)
    st.caption("Sorted by severity · entities requiring immediate investigation")

    sc1, sc2, sc3 = st.columns(3)
    crit  = [s for s in SUSPICIOUS if s["severity"] == "CRITICAL"]
    high  = [s for s in SUSPICIOUS if s["severity"] == "HIGH"]
    med   = [s for s in SUSPICIOUS if s["severity"] == "MEDIUM"]
    sc1.metric("🔴 CRITICAL", f"{len(crit)} findings",
               fmt(sum(s["amount"] for s in crit)))
    sc2.metric("🟠 HIGH", f"{len(high)} findings",
               fmt(sum(s["amount"] for s in high if s["amount"] > 0)))
    sc3.metric("🟡 MEDIUM", f"{len(med)} findings",
               fmt(sum(s["amount"] for s in med)))

    # ── Suspicious amounts chart ──
    sw = [s for s in SUSPICIOUS if s["amount"] > 0]
    fig = go.Figure(go.Bar(
        x=[s["amount"] for s in sw],
        y=[s["entity"][:38] for s in sw],
        orientation="h",
        marker_color=[
            "#dc2626" if s["severity"] == "CRITICAL"
            else "#ea580c" if s["severity"] == "HIGH"
            else "#eab308" for s in sw],
        text=[fmt(s["amount"]) for s in sw],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
        customdata=[s["risk"] for s in sw],
        hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<br>Risk: %{customdata}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY, height=380,
        title="Suspicious Outflows by Entity",
        yaxis=dict(autorange="reversed", tickfont_size=10),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Amount (₹)"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Detail expanders ──
    for s in SUSPICIOUS:
        sev_c = "#dc2626" if s["severity"]=="CRITICAL" else "#ea580c" if s["severity"]=="HIGH" else "#eab308"
        sev_bg = "#fef2f2" if s["severity"]=="CRITICAL" else "#fff7ed" if s["severity"]=="HIGH" else "#fefce8"
        icon = "🔴" if s["severity"]=="CRITICAL" else "🟠" if s["severity"]=="HIGH" else "🟡"

        with st.expander(
            f"{icon}  **{s['entity']}** — "
            f"{fmt(s['amount']) if s['amount'] > 0 else 'Risk Flag'}  "
            f"({s['severity']})"
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Amount",  fmt(s["amount"]) if s["amount"] > 0 else "N/A")
            m2.metric("Txns",    s["count"])
            m3.metric("Avg/Txn", fmt(s["avg"]) if s["avg"] > 0 else "N/A")
            m4.metric("Risk",    s["severity"])

            st.markdown(
                f'<div style="background:{sev_bg};border-left:4px solid {sev_c};'
                f'border-radius:.5rem;padding:1rem 1.2rem;margin:.5rem 0;'
                f'font-size:.9rem;line-height:1.7">'
                f'<strong style="color:{sev_c}">Risk: {s["risk"]}</strong><br>'
                f'{s["detail"]}</div>', unsafe_allow_html=True)

            # Special chart for Noorkhan
            if s["entity"] == "Noorkhan S/O Gulam Rasool":
                st.markdown("#### 📉 Noorkhan — Monthly Payment Pattern")
                nk_amt = [195000,165000,235000,177000,107000,145000,
                          159000,74500,107000,140000,145000,190850]
                nk_cnt = [12,10,14,11,8,9,10,5,7,8,8,9]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=MONTHS, y=nk_amt, name="₹ Paid",
                    marker_color="#dc2626", marker_opacity=.7,
                    text=[fmt(v) for v in nk_amt], textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono")))
                fig.add_trace(go.Scatter(x=MONTHS, y=nk_cnt, name="Txn Count",
                    yaxis="y2", mode="lines+markers",
                    line=dict(color="#6366f1", width=2.5), marker_size=7))
                fig.update_layout(**PLOTLY, height=320,
                    yaxis=dict(title="Amount (₹)", showgrid=True, gridcolor="#f1f5f9"),
                    yaxis2=dict(title="# Transactions", overlaying="y", side="right"),
                    legend=dict(orientation="h", y=1.08, x=.5, xanchor="center"))
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  TAB 4 — LOSS → PROFIT ROADMAP
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-head">🎯 The Profit Bridge</p>', unsafe_allow_html=True)
    st.caption("Turning a ₹13.6L loss into ₹94.8L profit — step by step")

    # ── Waterfall ──
    labels = ["Current Loss", "Stop Leakages", "Cost Reduction",
              "Revenue Growth", "Projected Profit"]
    vals = [PROFIT_BRIDGE["current_loss"], PROFIT_BRIDGE["stop_leakages"],
            PROFIT_BRIDGE["cost_reduction"], PROFIT_BRIDGE["revenue_growth"],
            PROFIT_BRIDGE["projected_profit"]]

    fig = go.Figure(go.Waterfall(
        x=labels, y=vals,
        measure=["absolute", "relative", "relative", "relative", "total"],
        connector_line=dict(color="#94a3b8", width=1),
        increasing_marker_color="#10b981",
        decreasing_marker_color="#ef4444",
        totals_marker_color="#2563eb",
        text=[fmt(v) for v in vals],
        textposition="outside",
        textfont=dict(size=13, family="JetBrains Mono"),
    ))
    fig.update_layout(**PLOTLY, height=420,
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Amount (₹)",
                   zeroline=True, zerolinecolor="#0f172a", zerolinewidth=1.5),
        xaxis_tickfont_size=11)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="green-box">'
        '<strong>✅ Bottom Line:</strong> Your company isn\'t fundamentally broken — it\'s '
        '<strong>leaking money</strong>. With ₹10Cr turnover, even 10% net margin = ₹1Cr profit. '
        'The ₹13.6L loss is caused by ₹52.6L suspicious outflows, ₹1.05Cr untraceable cash, '
        'and ₹72L poorly-managed GST. <strong>Fix these three and you\'re profitable by Q2.</strong>'
        '</div>', unsafe_allow_html=True)

    # ── Savings Breakdown ──
    st.markdown('<p class="section-head">Savings Potential Breakdown</p>', unsafe_allow_html=True)

    savings = [
        ("Separate Personal Expenses", 30_60_000, "#fb923c"),
        ("Vendor Consolidation",       25_00_000, "#f59e0b"),
        ("Recover Rajender Kumar ₹22L",22_00_000, "#f87171"),
        ("Stop Noorkhan (₹18.4L)",     18_40_000, "#dc2626"),
        ("Reduce Sub-contracting",     17_50_000, "#8b5cf6"),
        ("Eliminate Cash Drain",       14_00_000, "#ef4444"),
        ("GST Input Credit Recovery",  12_50_000, "#10b981"),
        ("Loan Refinancing",            4_00_000, "#6366f1"),
        ("Fix GST Compliance",          2_50_000, "#14b8a6"),
        ("Fuel Savings",                2_50_000, "#06b6d4"),
    ]

    sd, sl = st.columns([2, 3])
    with sd:
        fig = go.Figure(go.Pie(
            labels=[s[0] for s in savings],
            values=[s[1] for s in savings],
            marker_colors=[s[2] for s in savings],
            hole=0.55, textinfo="percent", textposition="inside", textfont_size=9,
        ))
        tot = sum(s[1] for s in savings)
        fig.update_layout(**PLOTLY, height=400, showlegend=False,
            annotations=[dict(text=f"<b>{fmt(tot)}</b><br>Total Potential",
                              x=.5, y=.5, font_size=12, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with sl:
        for label, amt, color in savings:
            pct = amt / tot * 100
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
                f'<div style="width:12px;height:12px;border-radius:3px;background:{color};flex-shrink:0"></div>'
                f'<div style="flex:1;font-size:0.88rem">{label}</div>'
                f'<div style="font-family:JetBrains Mono;font-weight:700;font-size:0.88rem">{fmt(amt)}</div>'
                f'<div style="color:#94a3b8;font-size:0.78rem;width:40px;text-align:right">{pct:.0f}%</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Detailed Recommendations ──
    st.markdown('<p class="section-head">Detailed Action Plan — 17 Recommendations</p>',
                unsafe_allow_html=True)

    for section in RECOMMENDATIONS:
        st.markdown(
            f'<div style="background:{section["color"]}0d;'
            f'border:1px solid {section["color"]}33;border-radius:.5rem;'
            f'padding:.7rem 1rem;margin-top:1rem;margin-bottom:.4rem">'
            f'<strong style="color:{section["color"]};font-size:1.05rem">'
            f'{section["section"]}</strong>'
            f'<span style="float:right;background:{section["color"]}1a;'
            f'color:{section["color"]};padding:2px 10px;border-radius:12px;'
            f'font-size:.8rem;font-weight:700">Potential: {section["potential"]}</span>'
            f'</div>', unsafe_allow_html=True)

        for action, saving, timeline, detail in section["items"]:
            with st.expander(f"**{action}** — 💰 {saving}  ·  ⏱ {timeline}"):
                st.markdown(f'<div class="insight-box">{detail}</div>',
                            unsafe_allow_html=True)
                r1, r2 = st.columns(2)
                r1.metric("💰 Expected Savings", saving)
                r2.metric("⏱ Implementation", timeline)

    # ── 90-Day Plan ──
    st.markdown('<p class="section-head">📋 90-Day Execution Timeline</p>', unsafe_allow_html=True)

    phases = [
        ("Week 1", "#dc2626",
         "Freeze Noorkhan payments · Investigate Rajender Kumar ₹22L · "
         "Set ₹5L min balance rule · Demand invoices from all laborers"),
        ("Week 2-3", "#ea580c",
         "Stop cash withdrawals >₹50K · Setup UPI/NEFT for all vendors · "
         "Separate Ashish Sharma account · Apply for OD facility"),
        ("Month 2", "#d97706",
         "Consolidate to 4 material vendors with rate contracts · File pending GST · "
         "Get GST invoices from laborers · Start monthly filing"),
        ("Month 3", "#059669",
         "Reduce sub-contracting (2 categories in-house) · Setup bill discounting · "
         "Bid for 2 new PWD tenders · Refinance high-EMI loan"),
    ]

    fig = go.Figure()
    widths = [1, 2, 4, 4]
    starts = [0, 1, 3, 7]
    for i, (phase, color, _) in enumerate(phases):
        fig.add_trace(go.Bar(
            x=[widths[i]], y=[0], base=starts[i],
            orientation="h", marker_color=color, marker_opacity=.8,
            name=phase, text=[phase], textposition="inside",
            textfont=dict(size=11, color="white", family="DM Sans"),
            hoverinfo="name",
        ))
    fig.update_layout(**PLOTLY, height=100, showlegend=False,
        xaxis=dict(title="Weeks →", range=[0, 12], showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(visible=False), bargap=0)
    st.plotly_chart(fig, use_container_width=True)

    for phase, color, tasks in phases:
        st.markdown(
            f'<div style="display:flex;gap:12px;margin-bottom:8px;align-items:flex-start">'
            f'<div style="background:{color};color:white;font-size:.75rem;font-weight:800;'
            f'padding:4px 10px;border-radius:4px;flex-shrink:0;min-width:72px;text-align:center">'
            f'{phase}</div>'
            f'<div style="font-size:.88rem;line-height:1.6;color:#334155">{tasks}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("")
    st.success(
        "🎯 **Your company turns ₹10 Cr in revenue. Even a 10% net margin = ₹1 Cr profit. "
        "The path from loss to profit is clear — it requires discipline, not miracles.**"
    )


# ═══════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.caption("📊 Forensic Financial Analysis  ·  Kwality Construction Company  ·  FY 2024-25  ·  Confidential")
