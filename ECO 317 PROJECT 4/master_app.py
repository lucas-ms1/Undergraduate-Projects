"""
master_app.py
=============
ECO 317 Project 4 -- The Capstone Macroeconomic Engine
------------------------------------------------------
Launch with:  streamlit run master_app.py

Unifies four macroeconomic modules into a single, professional Streamlit
dashboard with sidebar navigation, consistent dark/navy theming, and
Garamond typography throughout.

Modules
-------
I.   Empirical Data Suite        (from Assignment 1)
II.  Long-Run Growth: Solow      (NEW)
III. Micro-Founded VFI Models    (from Assignment 2)
IV.  DSGE & Fiscal Policy        (from Assignment 3)
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="ECO 317 Capstone Macro Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme system ─────────────────────────────────────────────────────────────
from config.theme import inject_css, get_theme

# Sidebar: theme toggle
with st.sidebar:
    st.title("Macro Engine")
    st.caption("ECO 317 · Spring 2026")
    st.markdown("---")
    theme_choice = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="theme_choice")
    st.markdown("---")

# Inject CSS (dark/navy bg, Garamond font, etc.)
inject_css()

# ── Module navigation ────────────────────────────────────────────────────────
MODULE_LABELS = {
    "I.  Empirical Data Suite":       "empirical",
    "II. Long-Run Growth (Solow)":    "solow",
    "III. VFI Dynamic Models":        "vfi",
    "IV. DSGE & Fiscal Policy":       "dsge",
}

with st.sidebar:
    selected = st.radio(
        "Select Module",
        list(MODULE_LABELS.keys()),
        index=1,  # Default to Solow (the new module)
        key="module_nav",
    )
    module_key = MODULE_LABELS[selected]
    st.markdown("---")
    st.caption(
        "Lucas Sneller, Rida, Lindsey,\n"
        "Aidan, Ben · Miami University"
    )

# ── Render selected module ───────────────────────────────────────────────────
if module_key == "empirical":
    from pages.empirical import render
    render()

elif module_key == "solow":
    from pages.solow import render
    render()

elif module_key == "vfi":
    from pages.vfi_models import render
    render()

elif module_key == "dsge":
    from pages.dsge_fiscal import render
    render()

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "ECO 317 · Intermediate Macroeconomic Theory · Spring 2026 · "
    "The Capstone Macroeconomic Engine · Professor Jonathan Wolff · Miami University"
)
