"""
ECO 317 - Intermediate Macroeconomic Theory
AI-Assisted Macroeconomic Modeling Dashboard
Professor Jonathan Wolff | Spring 2026

Run with:  python app.py
Then open: http://127.0.0.1:8050  in your browser

Three models solved via Value Function Iteration (VFI):
  Model 1: Stochastic Consumption-Savings (CES/CRRA preferences, Markov income shock)
  Model 2: Stochastic Robinson Crusoe Economy (capital accumulation, TFP shock)
  Model 3: Endogenous Labor Supply (consumption-leisure trade-off, Markov wage shock)
"""

import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser, threading, time

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
NAV    = "#0b1120"
NAV2   = "#0d1a2e"
BORDER = "#1e3a5f"
ACCENT = "#4a90d9"
TEXT   = "#e8e0d0"
SUBTEXT= "#7fa8c8"
WHITE  = "#ffffff"

C0 = "#1565c0"
C1 = "#c62828"
C2 = "#2e7d32"
C3 = "#6a1b9a"
C4 = "#e65100"

GRID_CLR = "#e8e8e8"
FONT_FAM = "EB Garamond, Georgia, serif"

# ─────────────────────────────────────────────
# PLOTLY LAYOUT HELPER
# ─────────────────────────────────────────────
def base_layout(title="", xtitle="", ytitle="", height=350):
    return dict(
        title=dict(text=title, font=dict(family=FONT_FAM, size=13, color="#111")),
        xaxis=dict(title=xtitle, gridcolor=GRID_CLR, showgrid=True,
                   zeroline=False, tickfont=dict(family=FONT_FAM, size=10)),
        yaxis=dict(title=ytitle, gridcolor=GRID_CLR, showgrid=True,
                   zeroline=False, tickfont=dict(family=FONT_FAM, size=10)),
        font=dict(family=FONT_FAM, color="#222"),
        height=height,
        margin=dict(l=55, r=20, t=44, b=50),
        legend=dict(font=dict(family=FONT_FAM, size=10),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1),
    )

# ─────────────────────────────────────────────
# MARKOV UTILITIES
# ─────────────────────────────────────────────
def make_markov(p_hh, p_ll):
    return np.array([[p_ll, 1-p_ll], [1-p_hh, p_hh]])

def stationary_dist(P):
    pi_h = (1 - P[0,0]) / (2 - P[0,0] - P[1,1])
    return np.array([1-pi_h, pi_h])

def simulate_markov(P, T, seed=42):
    rng = np.random.default_rng(seed)
    states = np.zeros(T, dtype=int)
    pi = stationary_dist(P)
    states[0] = rng.choice(2, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(2, p=P[states[t-1]])
    return states

# ─────────────────────────────────────────────
# MODEL 1 — STOCHASTIC CONSUMPTION-SAVINGS
# Bellman: V(a,s) = max_{a'>=0} [U(c) + beta*E[V(a',s')]]
# Budget:  c = (1+r)*a + y(s) - a',  c > 0
# Utility: U(c) = (c^{1-sigma}-1)/(1-sigma)  [CRRA]
# ─────────────────────────────────────────────
def solve_model1(beta, sigma, r, y_low, y_high, p_hh, p_ll,
                 n_grid=400, tol=1e-6, max_iter=2000):
    P = make_markov(p_hh, p_ll)
    y = np.array([y_low, y_high])
    a_grid = np.linspace(0.0, y_high * 30, n_grid)

    def util(c):
        c = np.maximum(c, 1e-10)
        return np.log(c) if abs(sigma-1.0) < 1e-9 else (c**(1-sigma)-1)/(1-sigma)

    V = np.zeros((n_grid, 2))
    pol_a = np.zeros((n_grid, 2))
    pol_c = np.zeros((n_grid, 2))

    for _ in range(max_iter):
        V_old = V.copy()
        for s in range(2):
            coh   = (1+r)*a_grid + y[s]
            EV    = P[s,0]*V_old[:,0] + P[s,1]*V_old[:,1]
            c_mat = coh[:,None] - a_grid[None,:]
            feas  = c_mat > 1e-10
            obj   = np.where(feas, util(np.maximum(c_mat,1e-10)), -1e15) + beta*EV[None,:]
            obj[~feas] = -1e15
            j = np.argmax(obj, axis=1)
            V[:,s]     = obj[np.arange(n_grid), j]
            pol_a[:,s] = a_grid[j]
            pol_c[:,s] = coh - a_grid[j]
        if np.max(np.abs(V-V_old)) < tol:
            break
    return a_grid, pol_a, pol_c, P

def simulate_model1(a_grid, pol_a, pol_c, P, y_low, y_high, T=200, seed=42):
    states = simulate_markov(P, T, seed)
    a = np.zeros(T); c = np.zeros(T)
    a[0] = a_grid[len(a_grid)//4]
    for t in range(T):
        s = states[t]
        c[t] = max(np.interp(a[t], a_grid, pol_c[:,s]), 1e-10)
        if t+1 < T:
            a[t+1] = np.interp(a[t], a_grid, pol_a[:,s])
    y_path = np.where(states==0, y_low, y_high)
    return c, a, y_path

def forecast_model1(a0, shocks, a_grid, pol_a, pol_c):
    a = a0; H = len(shocks)
    c_fc = np.zeros(H); a_fc = np.zeros(H)
    for t in range(H):
        s = shocks[t]
        c_fc[t] = np.interp(a, a_grid, pol_c[:,s])
        a_fc[t] = np.interp(a, a_grid, pol_a[:,s])
        a = a_fc[t]
    return c_fc, a_fc

# ─────────────────────────────────────────────
# MODEL 2 — STOCHASTIC ROBINSON CRUSOE
# Production: Y = z*K^alpha
# Budget:     C = z*K^alpha + (1-delta)*K - K',  C > 0
# Bellman:    V(K,z) = max_{K'>=0} [U(C) + beta*E[V(K',z')]]
# ─────────────────────────────────────────────
def solve_model2(beta, sigma, alpha, delta, z_low, z_high, p_hh, p_ll,
                 n_grid=400, tol=1e-6, max_iter=2000):
    P = make_markov(p_hh, p_ll)
    z = np.array([z_low, z_high])
    z_m = stationary_dist(P) @ z
    K_ss = (alpha*z_m / (1/beta - 1 + delta))**(1/(1-alpha))
    k_grid = np.linspace(0.01, 4.0*K_ss, n_grid)

    def util(c):
        c = np.maximum(c, 1e-10)
        return np.log(c) if abs(sigma-1.0) < 1e-9 else (c**(1-sigma)-1)/(1-sigma)

    V = np.zeros((n_grid, 2))
    pol_k = np.zeros((n_grid, 2))
    pol_c = np.zeros((n_grid, 2))

    for _ in range(max_iter):
        V_old = V.copy()
        for s in range(2):
            res   = z[s]*k_grid**alpha + (1-delta)*k_grid
            EV    = P[s,0]*V_old[:,0] + P[s,1]*V_old[:,1]
            c_mat = res[:,None] - k_grid[None,:]
            feas  = c_mat > 1e-10
            obj   = np.where(feas, util(np.maximum(c_mat,1e-10)), -1e15) + beta*EV[None,:]
            obj[~feas] = -1e15
            j = np.argmax(obj, axis=1)
            V[:,s]     = obj[np.arange(n_grid), j]
            pol_k[:,s] = k_grid[j]
            pol_c[:,s] = res - k_grid[j]
        if np.max(np.abs(V-V_old)) < tol:
            break
    return k_grid, K_ss, pol_k, pol_c, P, z, alpha, delta

def simulate_model2(k_grid, pol_k, pol_c, P, z_vals, alpha, delta, T=200, seed=42):
    states = simulate_markov(P, T, seed)
    k = np.zeros(T); c = np.zeros(T); y = np.zeros(T)
    k[0] = k_grid[len(k_grid)//3]
    for t in range(T):
        s = states[t]
        y[t] = z_vals[s] * k[t]**alpha
        c[t] = max(np.interp(k[t], k_grid, pol_c[:,s]), 1e-10)
        if t+1 < T:
            k[t+1] = np.interp(k[t], k_grid, pol_k[:,s])
    return c, k, y

def forecast_model2(k0, shocks, k_grid, pol_k, pol_c, z_vals, alpha):
    k = k0; H = len(shocks)
    c_fc = np.zeros(H); k_fc = np.zeros(H); y_fc = np.zeros(H)
    for t in range(H):
        s = shocks[t]
        y_fc[t] = z_vals[s] * k**alpha
        c_fc[t] = np.interp(k, k_grid, pol_c[:,s])
        k_fc[t] = np.interp(k, k_grid, pol_k[:,s])
        k = k_fc[t]
    return c_fc, k_fc, y_fc

# ─────────────────────────────────────────────
# MODEL 3 — ENDOGENOUS LABOR SUPPLY
# U(C,L) = C^{1-s}/(1-s) + phi*(1-L)^{1-s}/(1-s)
# Budget:  C = (1+r)*a + w*L - a'
# FOC:     L* = (R - A)/(w + R),  R = (w*phi)^{1/sigma}
# ─────────────────────────────────────────────
def solve_model3(beta, sigma, phi, r, w_low, w_high, p_hh, p_ll,
                 n_grid=300, tol=1e-6, max_iter=2000):
    P = make_markov(p_hh, p_ll)
    w = np.array([w_low, w_high])
    a_grid = np.linspace(0.0, max(w_high*25, 5.0), n_grid)

    def util_jt(c, l):
        c    = np.maximum(c, 1e-10)
        leis = np.maximum(1-l, 1e-10)
        if abs(sigma-1.0) < 1e-9:
            return np.log(c) + phi*np.log(leis)
        return (c**(1-sigma)-1)/(1-sigma) + phi*(leis**(1-sigma)-1)/(1-sigma)

    def opt_L(A, w_s):
        R = (w_s*phi)**(1.0/sigma)
        return np.clip((R - A)/(w_s + R), 0.0, 1.0)

    V = np.zeros((n_grid, 2))
    pol_a = np.zeros((n_grid, 2))
    pol_c = np.zeros((n_grid, 2))
    pol_l = np.zeros((n_grid, 2))

    for _ in range(max_iter):
        V_old = V.copy()
        for s in range(2):
            w_s = w[s]
            EV  = P[s,0]*V_old[:,0] + P[s,1]*V_old[:,1]
            A   = (1+r)*a_grid[:,None] - a_grid[None,:]
            L   = opt_L(A, w_s)
            C   = A + w_s*L
            feas = (C > 1e-10) & (L >= 0) & (L <= 1.0)
            obj  = np.where(feas, util_jt(C,L), -1e15) + beta*EV[None,:]
            obj[~feas] = -1e15
            j = np.argmax(obj, axis=1)
            V[:,s]     = obj[np.arange(n_grid), j]
            pol_a[:,s] = a_grid[j]
            A_opt = (1+r)*a_grid - a_grid[j]
            L_opt = opt_L(A_opt, w_s)
            pol_l[:,s] = L_opt
            pol_c[:,s] = np.maximum(A_opt + w_s*L_opt, 1e-10)
        if np.max(np.abs(V-V_old)) < tol:
            break
    return a_grid, pol_a, pol_c, pol_l, P, w

def simulate_model3(a_grid, pol_a, pol_c, pol_l, P, w_vals, T=200, seed=42):
    states = simulate_markov(P, T, seed)
    a = np.zeros(T); c = np.zeros(T); l = np.zeros(T); y = np.zeros(T)
    a[0] = a_grid[len(a_grid)//4]
    for t in range(T):
        s = states[t]
        c[t] = max(np.interp(a[t], a_grid, pol_c[:,s]), 1e-10)
        l[t] = np.clip(np.interp(a[t], a_grid, pol_l[:,s]), 0, 1)
        y[t] = w_vals[s]*l[t]
        if t+1 < T:
            a[t+1] = np.interp(a[t], a_grid, pol_a[:,s])
    return c, a, l, y

def forecast_model3(a0, shocks, a_grid, pol_a, pol_c, pol_l):
    a = a0; H = len(shocks)
    c_fc = np.zeros(H); a_fc = np.zeros(H); l_fc = np.zeros(H)
    for t in range(H):
        s = shocks[t]
        c_fc[t] = np.interp(a, a_grid, pol_c[:,s])
        l_fc[t] = np.clip(np.interp(a, a_grid, pol_l[:,s]), 0, 1)
        a_fc[t] = np.interp(a, a_grid, pol_a[:,s])
        a = a_fc[t]
    return c_fc, a_fc, l_fc

# ─────────────────────────────────────────────
# MOMENTS
# ─────────────────────────────────────────────
def compute_moments(series_dict, output_key):
    out = series_dict.get(output_key)
    res = {}
    for name, x in series_dict.items():
        x  = np.asarray(x, dtype=float)
        sd = float(np.std(x))
        ac1 = float(np.corrcoef(x[:-1], x[1:])[0,1]) if sd > 1e-10 else 0.0
        if out is not None and name != output_key and sd > 1e-10 and np.std(out) > 1e-10:
            cy = float(pearsonr(x, out)[0])
        else:
            cy = 1.0 if name == output_key else float("nan")
        res[name] = {"Mean": float(np.mean(x)), "Std Dev": sd,
                     "Variance": float(np.var(x)), "AR(1)": ac1, "Corr w/ Output": cy}
    return res

def shock_path(label, horizon):
    MAP = {"Low–Low–Low":[0,0,0],"Low–High–Low":[0,1,0],
           "High–High–High":[1,1,1],"High–Low–High":[1,0,1],
           "Alternating L-H":[0,1,0]}
    b = MAP.get(label, [0,0,0])
    return np.array((b * ((horizon//3)+2))[:horizon], dtype=int)

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def sld(id_, label, mn, mx, val, step):
    return html.Div([
        html.Div([
            html.Span(label, style={"color": SUBTEXT, "fontSize":"0.76rem",
                                    "letterSpacing":"0.07em","textTransform":"uppercase"}),
            html.Span(id=f"{id_}-val", style={"color":TEXT,"fontSize":"0.88rem",
                                               "marginLeft":"8px","fontWeight":"600"}),
        ], style={"display":"flex","alignItems":"center","marginBottom":"3px"}),
        dcc.Slider(id=id_, min=mn, max=mx, value=val, step=step,
                   marks=None, tooltip={"placement":"bottom","always_visible":False}),
    ], style={"marginBottom":"12px"})

def sec(text):
    return html.Div(text, style={
        "color":SUBTEXT,"fontSize":"0.69rem","letterSpacing":"0.12em",
        "textTransform":"uppercase","background":"#162840","padding":"2px 9px",
        "borderRadius":"3px","display":"inline-block","marginBottom":"8px","marginTop":"4px"})

def card(lbl, val="—"):
    """Render a metric card with a direct value (no callback needed)."""
    return html.Div([
        html.Div(lbl, style={"color":SUBTEXT,"fontSize":"0.67rem","letterSpacing":"0.09em",
                              "textTransform":"uppercase","marginBottom":"3px"}),
        html.Div(str(val),
                 style={"color":TEXT,"fontSize":"1.25rem","fontWeight":"600"}),
    ], style={"background":"#0f2040","border":f"1px solid {BORDER}","borderRadius":"6px",
               "padding":"11px 15px","flex":"1","minWidth":"115px"})

def chart_wrap(fig_id):
    return html.Div(dcc.Graph(id=fig_id, config={"displayModeBar":False}),
                    style={"background":WHITE,"borderRadius":"8px","padding":"6px",
                           "marginBottom":"12px","boxShadow":"0 2px 8px rgba(0,0,0,0.3)"})

def row(*cols):
    return html.Div(list(cols), style={"display":"flex","gap":"14px","flexWrap":"wrap"})

def col(*ch, flex="1", minw="280px"):
    return html.Div(list(ch), style={"flex":flex,"minWidth":minw})

def analysis_box(txt):
    return html.Div(txt, style={
        "background":"linear-gradient(135deg,#0f2040 0%,#0d1a2e 100%)",
        "borderLeft":f"3px solid {ACCENT}","borderRadius":"0 6px 6px 0",
        "padding":"13px 18px","marginTop":"8px","marginBottom":"16px",
        "fontStyle":"italic","fontSize":"0.95rem","lineHeight":"1.8","color":"#c8d8e8"})

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
SIDEBAR = html.Div([
    html.Div([
        html.Span("⚙ ", style={"fontSize":"1.3rem"}),
        html.Span("Parameters", style={"fontSize":"1.05rem","fontWeight":"700",
                                        "color":TEXT,"letterSpacing":"0.04em"}),
    ], style={"display":"flex","alignItems":"center","marginBottom":"16px",
               "borderBottom":f"1px solid {BORDER}","paddingBottom":"10px"}),

    sec("Preferences"),
    sld("beta",  "Discount factor β", 0.85, 0.99, 0.96, 0.01),
    sld("sigma", "Risk aversion σ",   0.5,  5.0,  2.0,  0.1),

    sec("Technology"),
    sld("r-rate", "Interest rate r",   0.01, 0.10, 0.04, 0.005),
    sld("alpha",  "Capital share α",   0.20, 0.45, 0.33, 0.01),
    sld("delta",  "Depreciation δ",    0.02, 0.15, 0.10, 0.01),
    sld("phi",    "Leisure weight φ",  0.3,  3.0,  1.0,  0.1),

    sec("Income / Wage / TFP Shocks"),
    sld("y-low",  "Low state",  0.5, 1.5, 0.80, 0.05),
    sld("y-high", "High state", 0.5, 2.5, 1.20, 0.05),

    sec("Markov Transition"),
    sld("p-hh", "P(High→High)", 0.50, 0.99, 0.80, 0.01),
    sld("p-ll", "P(Low→Low)",   0.50, 0.99, 0.80, 0.01),

    sec("Simulation"),
    sld("T-sim",   "Periods T",        100, 500, 200, 50),
    sld("horizon", "Forecast horizon", 5,   20,  10,  1),

    html.Div([
        html.Div("Shock path", style={"color":SUBTEXT,"fontSize":"0.76rem",
                                       "letterSpacing":"0.07em","textTransform":"uppercase",
                                       "marginBottom":"5px"}),
        dcc.Dropdown(id="shock-sel",
            options=[{"label":s,"value":s} for s in
                     ["Low–Low–Low","Low–High–Low","High–High–High",
                      "High–Low–High","Alternating L-H"]],
            value="Low–High–Low", clearable=False,
            style={"fontSize":"0.88rem"}),
    ], style={"marginBottom":"16px"}),

    html.Div("ECO 317 · Spring 2026 · Prof. Wolff",
             style={"color":"#3a6080","fontSize":"0.7rem","marginTop":"18px",
                    "borderTop":f"1px solid {BORDER}","paddingTop":"10px"}),
], style={
    "width":"270px","minWidth":"270px",
    "background":f"linear-gradient(180deg,{NAV2} 0%,{NAV} 100%)",
    "borderRight":f"1px solid {BORDER}","padding":"18px 16px",
    "overflowY":"auto","height":"100vh","position":"fixed",
    "top":"0","left":"0","fontFamily":FONT_FAM,
})

# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────
app = dash.Dash(__name__, title="ECO 317 · Macro Dashboard")
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    SIDEBAR,
    html.Div([
        # Header
        html.Div([
            html.Span("📊 ", style={"fontSize":"1.7rem"}),
            html.Div([
                html.H1("Macroeconomic Modeling Dashboard",
                        style={"margin":"0","fontSize":"1.5rem","fontWeight":"700",
                               "color":TEXT,"letterSpacing":"0.02em"}),
                html.Div("ECO 317 · Intermediate Macroeconomic Theory · Prof. Wolff · Spring 2026",
                         style={"color":SUBTEXT,"fontSize":"0.86rem","fontStyle":"italic","marginTop":"2px"}),
            ]),
        ], style={"display":"flex","alignItems":"center","padding":"16px 26px 12px",
                   "borderBottom":f"1px solid {BORDER}","background":NAV2}),

        # Tabs
        dcc.Tabs(id="tabs", value="m1", children=[
            dcc.Tab(label="  Model 1 · Consumption-Savings  ", value="m1"),
            dcc.Tab(label="  Model 2 · Robinson Crusoe  ",     value="m2"),
            dcc.Tab(label="  Model 3 · Endogenous Labor  ",    value="m3"),
        ], style={"fontFamily":FONT_FAM},
           colors={"border":BORDER,"primary":ACCENT,"background":NAV2}),

        html.Div(id="tab-body", style={"padding":"18px 26px"}),
    ], style={"marginLeft":"270px","display":"flex","flexDirection":"column",
               "minHeight":"100vh","background":NAV,"fontFamily":FONT_FAM}),
], style={"display":"flex","fontFamily":FONT_FAM,"background":NAV,"color":TEXT})

# ─────────────────────────────────────────────
# SLIDER DISPLAY CALLBACKS  (single callback, no closure bug)
# ─────────────────────────────────────────────
@app.callback(
    [Output("beta-val","children"),   Output("sigma-val","children"),
     Output("r-rate-val","children"), Output("alpha-val","children"),
     Output("delta-val","children"),  Output("phi-val","children"),
     Output("y-low-val","children"),  Output("y-high-val","children"),
     Output("p-hh-val","children"),   Output("p-ll-val","children"),
     Output("T-sim-val","children"),  Output("horizon-val","children")],
    [Input("beta","value"),   Input("sigma","value"),
     Input("r-rate","value"), Input("alpha","value"),
     Input("delta","value"),  Input("phi","value"),
     Input("y-low","value"),  Input("y-high","value"),
     Input("p-hh","value"),   Input("p-ll","value"),
     Input("T-sim","value"),  Input("horizon","value")],
)
def update_slider_labels(beta,sigma,r,alpha,delta,phi,ylo,yhi,phh,pll,T,H):
    return [f"= {v}" for v in (beta,sigma,r,alpha,delta,phi,ylo,yhi,phh,pll,T,H)]

# ─────────────────────────────────────────────
# MAIN RENDER CALLBACK
# ─────────────────────────────────────────────
@app.callback(
    Output("tab-body","children"),
    [Input("tabs","value"),
     Input("beta","value"),  Input("sigma","value"),
     Input("r-rate","value"),Input("alpha","value"),  Input("delta","value"),
     Input("phi","value"),   Input("y-low","value"),  Input("y-high","value"),
     Input("p-hh","value"),  Input("p-ll","value"),
     Input("T-sim","value"), Input("horizon","value"),
     Input("shock-sel","value")],
)
def render(tab, beta, sigma, r, alpha, delta, phi,
           y_low, y_high, p_hh, p_ll, T, horizon, shock_lbl):

    shocks = shock_path(shock_lbl, horizon)
    cbox   = {"background":WHITE,"borderRadius":"8px","padding":"6px",
               "marginBottom":"12px","boxShadow":"0 2px 8px rgba(0,0,0,0.28)"}

    # ════════════════════════════════════
    # MODEL 1
    # ════════════════════════════════════
    if tab == "m1":
        a_grid, pol_a, pol_c, P = solve_model1(
            beta, sigma, r, y_low, y_high, p_hh, p_ll)
        c1, a1, y1 = simulate_model1(
            a_grid, pol_a, pol_c, P, y_low, y_high, T=T)
        c1_fc, a1_fc = forecast_model1(a1[-1], shocks, a_grid, pol_a, pol_c)
        m = compute_moments({"Consumption":c1,"Savings":a1,"Income":y1}, "Income")
        vr = m["Consumption"]["Variance"]/m["Income"]["Variance"] \
             if m["Income"]["Variance"] > 0 else float("nan")
        t_ax  = np.arange(T)
        fc_ax = np.arange(1, horizon+1)
        a_sh  = a_grid[:200]

        # Policy functions
        fig_pf = make_subplots(rows=1, cols=2,
            subplot_titles=["Consumption Policy  C*(a, s)",
                            "Savings Policy  a'(a, s)"])
        for s_idx, (nm, col_) in enumerate(zip(["Low income","High income"],[C1,C0])):
            fig_pf.add_trace(go.Scatter(x=a_sh, y=pol_c[:200,s_idx], name=nm,
                line=dict(color=col_,width=2)), row=1, col=1)
            fig_pf.add_trace(go.Scatter(x=a_sh, y=pol_a[:200,s_idx],
                line=dict(color=col_,width=2), showlegend=False), row=1, col=2)
        fig_pf.add_trace(go.Scatter(x=a_sh, y=a_sh, name="45° line",
            line=dict(color="#aaa",width=1,dash="dash"),showlegend=False), row=1, col=2)
        fig_pf.update_layout(**base_layout(height=360))
        fig_pf.update_xaxes(gridcolor=GRID_CLR, title_text="Asset holdings  a")
        fig_pf.update_yaxes(gridcolor=GRID_CLR)

        # Simulation
        fig_sim = make_subplots(rows=2,cols=1,shared_xaxes=True,
            subplot_titles=["Consumption  C_t","Asset Holdings  a_t"],
            vertical_spacing=0.1)
        fig_sim.add_trace(go.Scatter(x=t_ax,y=c1,line=dict(color=C0,width=1.4),name="C_t"),row=1,col=1)
        fig_sim.add_trace(go.Scatter(x=t_ax,y=a1,line=dict(color=C2,width=1.4),
            fill="tozeroy",fillcolor="rgba(46,125,50,0.12)",name="a_t"),row=2,col=1)
        fig_sim.update_layout(**base_layout(height=400))
        fig_sim.update_xaxes(gridcolor=GRID_CLR); fig_sim.update_yaxes(gridcolor=GRID_CLR)

        # Forecast
        sc = [C1 if s==0 else C0 for s in shocks]
        fig_fc = make_subplots(rows=2,cols=1,shared_xaxes=True,
            subplot_titles=[f"Forecast Consumption  ({shock_lbl})","Forecast Asset Holdings"],
            vertical_spacing=0.1)
        fig_fc.add_trace(go.Bar(x=fc_ax,y=c1_fc,marker_color=sc,name="C forecast"),row=1,col=1)
        fig_fc.add_trace(go.Bar(x=fc_ax,y=a1_fc,marker_color=C2,name="a forecast"),row=2,col=1)
        fig_fc.update_layout(**base_layout(height=400))
        fig_fc.update_xaxes(gridcolor=GRID_CLR); fig_fc.update_yaxes(gridcolor=GRID_CLR)

        # MPC vs wealth
        mpc_lo = np.gradient(pol_c[:200,0], a_sh)
        mpc_hi = np.gradient(pol_c[:200,1], a_sh)
        fig_mpc = go.Figure([
            go.Scatter(x=a_sh,y=mpc_lo,name="MPC (Low income)",line=dict(color=C1,width=2)),
            go.Scatter(x=a_sh,y=mpc_hi,name="MPC (High income)",line=dict(color=C0,width=2)),
        ])
        fig_mpc.add_hline(y=r/(1+r),line_dash="dash",line_color="#888",
            annotation_text=f"PIH benchmark r/(1+r)={r/(1+r):.3f}")
        fig_mpc.update_layout(**base_layout("Marginal Propensity to Consume vs. Wealth",
            "Asset Holdings  a","MPC",320))
        fig_mpc.update_yaxes(range=[0,0.35], gridcolor=GRID_CLR)
        fig_mpc.update_xaxes(gridcolor=GRID_CLR)

        summ = (f"Simulated mean consumption: {m['Consumption']['Mean']:.3f} | "
                f"Std Dev: {m['Consumption']['Std Dev']:.3f} | "
                f"AR(1): {m['Consumption']['AR(1)']:.3f} | "
                f"Var(C)/Var(Y): {vr:.3f} — "
                + ("PIH consumption-smoothing holds (ratio < 1)." if vr < 1
                   else "Amplification present (ratio > 1).")
                + f" Higher β = {beta:.2f} implies "
                + ("strong" if beta > 0.95 else "moderate")
                + " precautionary savings motive.")

        return html.Div([
            html.H2("Model 1 — Stochastic Consumption-Savings",
                    style={"color":TEXT,"marginBottom":"2px","marginTop":"0"}),
            sec("VFI · CRRA Utility · Markov Income Shock · Budget: c = (1+r)a + y(s) − a'"),
            html.H3("Simulated Moments", style={"color":TEXT,"marginTop":"16px","marginBottom":"8px"}),
            row(card("Mean (C)",     f"{m['Consumption']['Mean']:.3f}"),
                card("Std Dev (C)",  f"{m['Consumption']['Std Dev']:.3f}"),
                card("AR(1) of C",  f"{m['Consumption']['AR(1)']:.3f}"),
                card("Corr(C,Y)",   f"{m['Consumption']['Corr w/ Output']:.3f}"),
                card("Var(C)/Var(Y)", f"{vr:.3f}")),
            html.H3("Policy Functions", style={"color":TEXT,"marginTop":"20px"}),
            html.Div(dcc.Graph(figure=fig_pf,config={"displayModeBar":False}),style=cbox),
            html.H3("Simulation & Forecast", style={"color":TEXT}),
            row(col(html.Div(dcc.Graph(figure=fig_sim,config={"displayModeBar":False}),style=cbox)),
                col(html.Div(dcc.Graph(figure=fig_fc,config={"displayModeBar":False}),style=cbox))),
            html.H3("Static Intuition: MPC vs. Wealth", style={"color":TEXT}),
            html.Div(dcc.Graph(figure=fig_mpc,config={"displayModeBar":False}),style=cbox),
            analysis_box(summ),
        ])

    # ════════════════════════════════════
    # MODEL 2
    # ════════════════════════════════════
    elif tab == "m2":
        z_lo, z_hi = y_low, y_high
        k_grid, K_ss, pol_k, pol_c2, P2, z_vals, a2, d2 = solve_model2(
            beta, sigma, alpha, delta, z_lo, z_hi, p_hh, p_ll)
        c2, k2, y2 = simulate_model2(k_grid, pol_k, pol_c2, P2, z_vals, a2, d2, T=T)
        inv2 = np.diff(k2,prepend=k2[0]) + d2*k2
        c2_fc, k2_fc, y2_fc = forecast_model2(k2[-1],shocks,k_grid,pol_k,pol_c2,z_vals,a2)
        m2 = compute_moments({"Consumption":c2,"Capital":k2,"Output":y2,"Investment":inv2},"Output")
        t_ax  = np.arange(T)
        fc_ax = np.arange(1,horizon+1)
        k_sh  = k_grid[:200]

        # Policy functions
        fig_pf2 = make_subplots(rows=1,cols=2,
            subplot_titles=["Consumption Policy  C*(K, z)","Capital Policy  K'(K, z)"])
        for s_idx,(nm,col_) in enumerate(zip([f"Low TFP z={z_lo:.2f}",
                                               f"High TFP z={z_hi:.2f}"],[C1,C0])):
            fig_pf2.add_trace(go.Scatter(x=k_sh,y=pol_c2[:200,s_idx],name=nm,
                line=dict(color=col_,width=2)),row=1,col=1)
            fig_pf2.add_trace(go.Scatter(x=k_sh,y=pol_k[:200,s_idx],
                line=dict(color=col_,width=2),showlegend=False),row=1,col=2)
        fig_pf2.add_trace(go.Scatter(x=k_sh,y=k_sh,name="45°",
            line=dict(color="#aaa",width=1,dash="dash"),showlegend=False),row=1,col=2)
        fig_pf2.add_vline(x=K_ss,line_dash="dot",line_color="#555",
            annotation_text=f"K*={K_ss:.2f}")
        fig_pf2.update_layout(**base_layout(height=360))
        fig_pf2.update_xaxes(gridcolor=GRID_CLR,title_text="Capital stock  K")
        fig_pf2.update_yaxes(gridcolor=GRID_CLR)

        # Simulation
        fig_sim2 = make_subplots(rows=3,cols=1,shared_xaxes=True,
            subplot_titles=["Output  Y_t","Consumption  C_t","Capital  K_t"],
            vertical_spacing=0.07)
        fig_sim2.add_trace(go.Scatter(x=t_ax,y=y2,line=dict(color=C0,width=1.4),name="Y_t"),row=1,col=1)
        fig_sim2.add_trace(go.Scatter(x=t_ax,y=c2,line=dict(color=C2,width=1.4),
            fill="tozeroy",fillcolor="rgba(46,125,50,0.1)",name="C_t"),row=2,col=1)
        fig_sim2.add_trace(go.Scatter(x=t_ax,y=k2,line=dict(color=C4,width=1.4),name="K_t"),row=3,col=1)
        fig_sim2.update_layout(**base_layout(height=490))
        fig_sim2.update_xaxes(gridcolor=GRID_CLR); fig_sim2.update_yaxes(gridcolor=GRID_CLR)

        # Forecast
        sc2 = [C1 if s==0 else C0 for s in shocks]
        fig_fc2 = make_subplots(rows=3,cols=1,shared_xaxes=True,
            subplot_titles=["Forecast Output","Forecast Consumption","Forecast Capital"],
            vertical_spacing=0.07)
        fig_fc2.add_trace(go.Bar(x=fc_ax,y=y2_fc,marker_color=sc2),row=1,col=1)
        fig_fc2.add_trace(go.Bar(x=fc_ax,y=c2_fc,marker_color=C2),row=2,col=1)
        fig_fc2.add_trace(go.Bar(x=fc_ax,y=k2_fc,marker_color=C4),row=3,col=1)
        fig_fc2.update_layout(**base_layout(height=490))
        fig_fc2.update_xaxes(gridcolor=GRID_CLR); fig_fc2.update_yaxes(gridcolor=GRID_CLR)

        # Steady-state capital vs patience
        b_rng  = np.linspace(0.80,0.99,80)
        z_mean = stationary_dist(P2) @ z_vals
        kss_b  = (alpha*z_mean/(1/b_rng - 1 + delta))**(1/(1-alpha))
        fig_ss = go.Figure([
            go.Scatter(x=b_rng,y=kss_b,line=dict(color=C0,width=2.5),name="K*(β)")
        ])
        fig_ss.add_vline(x=beta,line_dash="dash",line_color=C1,
            annotation_text=f"β={beta:.2f} → K*={K_ss:.2f}")
        fig_ss.update_layout(**base_layout("Steady-State Capital vs. Patience","β","K*",310))
        fig_ss.update_xaxes(gridcolor=GRID_CLR); fig_ss.update_yaxes(gridcolor=GRID_CLR)

        summ2 = (f"Steady-state K* = {K_ss:.3f}. Mean output: {m2['Output']['Mean']:.3f}, "
                 f"AR(1) = {m2['Output']['AR(1)']:.3f}. "
                 f"Corr(C,Y) = {m2['Consumption']['Corr w/ Output']:.3f} — "
                 + ("strong RBC co-movement." if m2['Consumption']['Corr w/ Output'] > 0.5
                    else "moderate co-movement.") +
                 f" Investment std ({m2['Investment']['Std Dev']:.3f}) "
                 + ("exceeds" if m2['Investment']['Std Dev'] > m2['Consumption']['Std Dev'] else "is below")
                 + f" consumption std ({m2['Consumption']['Std Dev']:.3f}), "
                 + ("consistent with standard RBC mechanics." if m2['Investment']['Std Dev'] > m2['Consumption']['Std Dev']
                    else "consider widening the TFP shock range."))

        return html.Div([
            html.H2("Model 2 — Stochastic Robinson Crusoe Economy",
                    style={"color":TEXT,"marginBottom":"2px","marginTop":"0"}),
            sec("VFI · Capital Accumulation · TFP Shock · Y = z·K^α"),
            html.H3("Simulated Moments",style={"color":TEXT,"marginTop":"16px","marginBottom":"8px"}),
            row(card("Mean Output",  f"{m2['Output']['Mean']:.3f}"),
                card("Std Dev (Y)",  f"{m2['Output']['Std Dev']:.3f}"),
                card("AR(1) of Y",  f"{m2['Output']['AR(1)']:.3f}"),
                card("Corr(C,Y)",   f"{m2['Consumption']['Corr w/ Output']:.3f}"),
                card("SS Capital K*", f"{K_ss:.3f}")),
            html.H3("Policy Functions",style={"color":TEXT,"marginTop":"20px"}),
            html.Div(dcc.Graph(figure=fig_pf2,config={"displayModeBar":False}),style=cbox),
            html.H3("Simulation & Forecast",style={"color":TEXT}),
            row(col(html.Div(dcc.Graph(figure=fig_sim2,config={"displayModeBar":False}),style=cbox)),
                col(html.Div(dcc.Graph(figure=fig_fc2,config={"displayModeBar":False}),style=cbox))),
            html.H3("Static Intuition: K* vs. Patience (β)",style={"color":TEXT}),
            html.Div(dcc.Graph(figure=fig_ss,config={"displayModeBar":False}),style=cbox),
            analysis_box(summ2),
        ])

    # ════════════════════════════════════
    # MODEL 3
    # ════════════════════════════════════
    else:
        a_grid3,pol_a3,pol_c3,pol_l3,P3,w3 = solve_model3(
            beta,sigma,phi,r,y_low,y_high,p_hh,p_ll)
        c3,a3,l3,y3 = simulate_model3(a_grid3,pol_a3,pol_c3,pol_l3,P3,w3,T=T)
        c3_fc,a3_fc,l3_fc = forecast_model3(a3[-1],shocks,a_grid3,pol_a3,pol_c3,pol_l3)
        m3 = compute_moments({"Consumption":c3,"Savings":a3,"Labor":l3,"Income":y3},"Income")
        t_ax  = np.arange(T)
        fc_ax = np.arange(1,horizon+1)
        a_sh3 = a_grid3[:150]

        # Policy functions
        fig_pf3 = make_subplots(rows=1,cols=2,
            subplot_titles=["Consumption Policy  C*(a, w)","Labor Policy  L*(a, w)"])
        for s_idx,(nm,col_) in enumerate(zip([f"Low w={y_low:.2f}",
                                               f"High w={y_high:.2f}"],[C1,C0])):
            fig_pf3.add_trace(go.Scatter(x=a_sh3,y=pol_c3[:150,s_idx],name=nm,
                line=dict(color=col_,width=2)),row=1,col=1)
            fig_pf3.add_trace(go.Scatter(x=a_sh3,y=pol_l3[:150,s_idx],
                line=dict(color=col_,width=2),showlegend=False),row=1,col=2)
        fig_pf3.update_layout(**base_layout(height=360))
        fig_pf3.update_xaxes(gridcolor=GRID_CLR,title_text="Asset holdings  a")
        fig_pf3.update_yaxes(row=1,col=2,range=[0,1.05],gridcolor=GRID_CLR)
        fig_pf3.update_yaxes(row=1,col=1,gridcolor=GRID_CLR)

        # Simulation
        fig_sim3 = make_subplots(rows=3,cols=1,shared_xaxes=True,
            subplot_titles=["Consumption  C_t","Labor Supply  L_t","Leisure  (1−L_t)"],
            vertical_spacing=0.07)
        fig_sim3.add_trace(go.Scatter(x=t_ax,y=c3,line=dict(color=C0,width=1.4),name="C_t"),row=1,col=1)
        fig_sim3.add_trace(go.Scatter(x=t_ax,y=l3,line=dict(color=C3,width=1.4),
            fill="tozeroy",fillcolor="rgba(106,27,154,0.1)",name="L_t"),row=2,col=1)
        fig_sim3.add_trace(go.Scatter(x=t_ax,y=1-l3,line=dict(color=C2,width=1.4),
            fill="tozeroy",fillcolor="rgba(46,125,50,0.1)",name="1-L_t"),row=3,col=1)
        fig_sim3.update_layout(**base_layout(height=490))
        fig_sim3.update_yaxes(row=2,col=1,range=[0,1.05])
        fig_sim3.update_yaxes(row=3,col=1,range=[0,1.05])
        fig_sim3.update_xaxes(gridcolor=GRID_CLR); fig_sim3.update_yaxes(gridcolor=GRID_CLR)

        # Forecast
        sc3 = [C1 if s==0 else C0 for s in shocks]
        fig_fc3 = make_subplots(rows=3,cols=1,shared_xaxes=True,
            subplot_titles=["Forecast Consumption","Forecast Labor","Forecast Leisure"],
            vertical_spacing=0.07)
        fig_fc3.add_trace(go.Bar(x=fc_ax,y=c3_fc,marker_color=sc3),row=1,col=1)
        fig_fc3.add_trace(go.Bar(x=fc_ax,y=l3_fc,marker_color=C3),row=2,col=1)
        fig_fc3.add_trace(go.Bar(x=fc_ax,y=1-l3_fc,marker_color=C2),row=3,col=1)
        fig_fc3.update_layout(**base_layout(height=490))
        fig_fc3.update_xaxes(gridcolor=GRID_CLR); fig_fc3.update_yaxes(gridcolor=GRID_CLR)

        # Intra-temporal labor supply curve (analytical)
        w_rng = np.linspace(0.3,3.0,200)
        A_app = r * a_grid3[len(a_grid3)//3]
        R_rng = (w_rng*phi)**(1/sigma)
        L_crv = np.clip((R_rng - A_app)/(w_rng + R_rng), 0, 1)
        fig_ls = go.Figure([
            go.Scatter(x=w_rng,y=L_crv,line=dict(color=C0,width=2.5),name="L*(w)")
        ])
        fig_ls.add_vline(x=y_low, line_dash="dash",line_color=C1,
            annotation_text=f"w_low={y_low:.2f}")
        fig_ls.add_vline(x=y_high,line_dash="dash",line_color=C0,
            annotation_text=f"w_high={y_high:.2f}")
        fig_ls.update_layout(**base_layout(
            f"Intra-temporal Labor Supply Curve  (φ={phi:.1f}, σ={sigma:.1f})",
            "Real Wage  w","Optimal Labor  L*(w)",310))
        fig_ls.update_yaxes(range=[0,1.05],gridcolor=GRID_CLR)
        fig_ls.update_xaxes(gridcolor=GRID_CLR)

        summ3 = (f"Mean labor: {m3['Labor']['Mean']:.3f} (leisure: {1-m3['Labor']['Mean']:.3f}). "
                 f"Labor std: {m3['Labor']['Std Dev']:.3f}. "
                 f"Corr(L,Y) = {m3['Labor']['Corr w/ Output']:.3f} — "
                 + ("substitution effect dominates; higher wages → more work." if m3['Labor']['Corr w/ Output'] > 0.3
                    else "income effect partially offsets substitution effect.") +
                 f" Forecast mean labor: {np.mean(l3_fc):.3f}, "
                 f"consumption: {np.mean(c3_fc):.3f} under '{shock_lbl}' path.")

        return html.Div([
            html.H2("Model 3 — Endogenous Labor Supply",
                    style={"color":TEXT,"marginBottom":"2px","marginTop":"0"}),
            sec("VFI · Consumption-Leisure Trade-off · Markov Wage Shock · T=1 time endowment"),
            html.H3("Simulated Moments",style={"color":TEXT,"marginTop":"16px","marginBottom":"8px"}),
            row(card("Mean Labor L", f"{m3['Labor']['Mean']:.3f}"),
                card("Std Dev (L)",  f"{m3['Labor']['Std Dev']:.3f}"),
                card("AR(1) of C",  f"{m3['Consumption']['AR(1)']:.3f}"),
                card("Corr(C,Y)",   f"{m3['Consumption']['Corr w/ Output']:.3f}"),
                card("Corr(L,Y)",   f"{m3['Labor']['Corr w/ Output']:.3f}")),
            html.H3("Policy Functions",style={"color":TEXT,"marginTop":"20px"}),
            html.Div(dcc.Graph(figure=fig_pf3,config={"displayModeBar":False}),style=cbox),
            html.H3("Simulation & Forecast",style={"color":TEXT}),
            row(col(html.Div(dcc.Graph(figure=fig_sim3,config={"displayModeBar":False}),style=cbox)),
                col(html.Div(dcc.Graph(figure=fig_fc3, config={"displayModeBar":False}),style=cbox))),
            html.H3("Static Intuition: Intra-temporal Labor Supply Curve",style={"color":TEXT}),
            html.Div(dcc.Graph(figure=fig_ls,config={"displayModeBar":False}),style=cbox),
            analysis_box(summ3),
        ])

# ─────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    def _open():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:8050")
    threading.Thread(target=_open, daemon=True).start()
    print("\n  ECO 317 · Macroeconomic Modeling Dashboard")
    print("  ─────────────────────────────────────────────")
    print("  Running at  http://127.0.0.1:8050")
    print("  Press  Ctrl+C  to stop\n")
    app.run(debug=False, port=8050)