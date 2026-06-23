"""
config/theme.py
Shared theme system for the Capstone Macroeconomic Engine.

Provides dark/navy and light themes, Plotly layout helpers, and global CSS
injection (Garamond font, dark background, white charting boxes).
"""

import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------
THEMES = {
    "Dark": dict(
        bg="#0e1a2b",
        sidebar_bg="#0a1420",
        text="#e8e8e8",
        heading="#ffffff",
        card_bg="#162236",
        card_text="#e8e8e8",
        divider="#2a3f5c",
        caption="#8899aa",
        plot_paper="#162338",
        plot_bg="#0e1a2b",
        plot_grid="rgba(255,255,255,0.1)",
        plot_text="#c9d6e3",
        line1="#5b9bd5",
        line2="#ed7d31",
        line3="#70ad47",
        line4="#ffc000",
        line5="#9b59b6",
        plotly_template="plotly_dark",
    ),
    "Light": dict(
        bg="#fafafa",
        sidebar_bg="#f0f2f5",
        text="#1a1a1a",
        heading="#111111",
        card_bg="#ffffff",
        card_text="#1a1a1a",
        divider="#cccccc",
        caption="#666666",
        plot_paper="#ffffff",
        plot_bg="#ffffff",
        plot_grid="rgba(0,0,0,0.1)",
        plot_text="#333333",
        line1="#2b5c8a",
        line2="#d35400",
        line3="#27ae60",
        line4="#8e44ad",
        line5="#e74c3c",
        plotly_template="plotly_white",
    ),
}

FONT = "EB Garamond, Garamond, Georgia, serif"
PLOTLY_CFG = {"displayModeBar": True, "scrollZoom": True}


def get_theme() -> dict:
    """Return the currently selected theme dict from session state."""
    return THEMES.get(st.session_state.get("theme_choice", "Dark"), THEMES["Dark"])


def colors(th: dict | None = None) -> list[str]:
    """Return the 5 theme line colors."""
    th = th or get_theme()
    return [th["line1"], th["line2"], th["line3"], th["line4"], th["line5"]]


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def inject_css(th: dict | None = None):
    """Inject the global dark/navy CSS with Garamond font."""
    th = th or get_theme()
    dark = th["bg"] == "#0e1a2b"

    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');
    html, body, [class*="css"] {{
        font-family: "EB Garamond", Garamond, "Times New Roman", serif;
    }}
    .stApp {{
        background: {th['bg']};
        color: {th['text']};
    }}
    [data-testid="stSidebar"] {{
        background: {th['sidebar_bg']};
    }}
    [data-testid="stSidebarNav"] {{
        display: none;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {{
        color: {'#d0d8e8' if dark else '#1a1a1a'} !important;
    }}
    h1, h2, h3 {{
        font-family: "EB Garamond", Garamond, serif;
        color: {th['heading']};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: "EB Garamond", Garamond, serif;
        font-size: 1.1rem;
        color: {'#c0c8d8' if dark else '#444'};
    }}
    .stTabs [aria-selected="true"] {{
        color: {'#ffffff' if dark else '#111'};
        border-bottom-color: {'#4a90d9' if dark else '#2f80ed'};
    }}
    [data-testid="stMetric"] {{
        background: {th['card_bg']};
        color: {th['text']};
        border: 1px solid {'#2a3a52' if dark else '#ddd'};
        border-radius: 8px;
        padding: 10px 14px;
    }}
    [data-testid="stMetricLabel"] {{ color: {'#a0b0c8' if dark else '#555'} !important; }}
    [data-testid="stMetricValue"] {{ color: {'#ffffff' if dark else '#111'} !important; }}
    .stMarkdown, .stMarkdown p, .stMarkdown li {{ color: {th['text']} !important; }}
    .katex, .katex * {{ color: {th['text']} !important; }}
    .stSelectbox label, .stNumberInput label, .stSlider label {{
        color: {th['text']} !important;
    }}
    </style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly layout helpers
# ---------------------------------------------------------------------------
def plotly_layout(th: dict | None = None, **kw) -> dict:
    """
    Return a Plotly layout dict that matches the current theme.
    Pass any keyword overrides (title, xaxis_title, yaxis_title, height, etc.)
    """
    th = th or get_theme()
    plot_font = dict(family=FONT, color=th["plot_text"])

    base = dict(
        template=th["plotly_template"],
        paper_bgcolor=th["plot_paper"],
        plot_bgcolor=th["plot_bg"],
        font=plot_font,
        legend=dict(font=plot_font),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    base.update(kw)

    # Force title font
    title_text = base.get("title", "")
    if isinstance(title_text, dict):
        title_text = title_text.get("text", "")
    base["title"] = dict(text=title_text or "", font=plot_font)

    # Force axis fonts
    for axis_key in ("xaxis", "yaxis"):
        axis = base.get(axis_key) or {}
        title_kw = base.pop(f"{axis_key}_title", None)
        if title_kw is not None:
            axis["title"] = dict(text=title_kw, font=plot_font)
        else:
            existing = axis.get("title")
            cur = existing.get("text", "") if isinstance(existing, dict) else (existing or "")
            axis["title"] = dict(text=cur, font=plot_font)
        axis["tickfont"] = plot_font
        base[axis_key] = axis

    return base


def style_axes(fig: go.Figure, th: dict | None = None) -> go.Figure:
    """Apply theme grid + font styling to all axes on a figure."""
    th = th or get_theme()
    plot_font = dict(family=FONT, color=th["plot_text"])
    fig.update_xaxes(gridcolor=th["plot_grid"], title_font=plot_font, tickfont=plot_font)
    fig.update_yaxes(gridcolor=th["plot_grid"], title_font=plot_font, tickfont=plot_font)
    fig.update_layout(title_font=plot_font)
    return fig


def download_btns(fig, data_dict, label, key_prefix):
    """Add CSV + PNG download buttons for a chart."""
    import pandas as pd
    c1, c2, _ = st.columns([1, 1, 4])
    csv_bytes = pd.DataFrame(data_dict).to_csv(index=False).encode()
    c1.download_button(f"CSV: {label}", csv_bytes, f"{label}.csv",
                       "text/csv", key=f"csv_{key_prefix}")
    try:
        h = fig.layout.height or 700
        png = fig.to_image(format="png", width=1200, height=h, scale=2)
        c2.download_button(f"PNG: {label}", png, f"{label}.png",
                           "image/png", key=f"png_{key_prefix}")
    except Exception:
        c2.caption("Install *kaleido* for PNG export")
