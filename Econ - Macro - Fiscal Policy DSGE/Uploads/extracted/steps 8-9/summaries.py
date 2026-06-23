"""
utils/summaries.py
Step 9 – Dynamic commentary generator
---------------------------------------
Deterministic f-string generators that take the moments dictionary (from
moments.py) and the current slider values and return plain-English paragraphs
for Tab 1 of the Streamlit app.

Each structural parameter (habit, utilization, price stickiness, wage
stickiness, debt feedback) has at least one dedicated sentence.

Design
------
• Pure Python – no Streamlit dependency.  All formatting lives here; app.py
  simply calls generate_tab1_commentary(...) and renders the result.
• Deterministic: same moments + same params → same text.  No LLM calls.
• Gracefully handles missing keys (moments may have variable names depending
  on the model's observable set).
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(value: float, decimals: int = 1) -> str:
    """Format a variance ratio or fraction as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def _ratio(num: float, den: float) -> Optional[float]:
    if den == 0 or np.isnan(den) or np.isnan(num):
        return None
    return num / den


def _get(d: Dict[str, float], key: str) -> float:
    return d.get(key, float("nan"))


def _fmt(x: float, decimals: int = 4) -> str:
    if np.isnan(x):
        return "N/A"
    return f"{x:.{decimals}f}"


# ---------------------------------------------------------------------------
# Per-variable sentence builders
# ---------------------------------------------------------------------------

def _output_sentence(m: Dict[str, float]) -> str:
    ac1_y = _get(m, "ac1_y")
    return (
        f"Output (ŷ) has a variance of {_fmt(_get(m, 'var_y'))} and a "
        f"first-order autocorrelation of {_fmt(ac1_y, 3)}, "
        f"{'indicating substantial persistence' if ac1_y > 0.7 else 'suggesting moderate persistence' if ac1_y > 0.4 else 'indicating relatively low persistence'} "
        f"in the business cycle."
    )


def _consumption_sentence(m: Dict[str, float], h: float) -> str:
    ratio = _ratio(_get(m, "var_c"), _get(m, "var_y"))
    ratio_str = _pct(ratio) if ratio is not None else "N/A"
    smoothness = (
        "well below" if (ratio or 1) < 0.8
        else "slightly below" if (ratio or 1) < 1.0
        else "above"
    )
    habit_effect = (
        "amplified habit-formation dampens the consumption response"
        if h >= 0.6
        else "moderate habit persistence provides some consumption smoothing"
        if h >= 0.3
        else "low habit persistence, so consumption tracks income relatively closely"
    )
    return (
        f"With h = {h:.2f}, consumption volatility is {ratio_str} of output volatility "
        f"({smoothness} the stylized benchmark of ~50–80%), consistent with "
        f"{habit_effect}."
    )


def _investment_sentence(m: Dict[str, float]) -> str:
    ratio = _ratio(_get(m, "var_i"), _get(m, "var_y"))
    ratio_str = _pct(ratio) if ratio is not None else "N/A"
    sign = "exceeds" if (ratio or 0) > 1 else "falls short of"
    return (
        f"Investment volatility is {ratio_str} of output volatility, which "
        f"{sign} the empirical benchmark (investment is typically 3–4× more "
        f"volatile than output)."
    )


def _hours_sentence(m: Dict[str, float], utilization: float) -> str:
    corr = _get(m, "corr_l")
    utilization_note = (
        f"With capital utilization flexibility set to {utilization:.2f}, the model "
        f"{'allows firms to adjust the intensity of capital use, muting hours fluctuations'
           if utilization > 0.5
           else 'restricts capital utilization adjustment, pushing more of the margin onto hours'}."
    )
    return (
        f"Hours (l̂) have a correlation with output of {_fmt(corr, 3)}, "
        f"{'strongly procyclical as expected' if corr > 0.7 else 'moderately procyclical'}. "
        + utilization_note
    )


def _inflation_sentence(m: Dict[str, float], price_stickiness: float, wage_stickiness: float) -> str:
    ac1_pi = _get(m, "ac1_pi")
    corr_pi = _get(m, "corr_pi")
    stick_desc = (
        "high price stickiness (Calvo parameter near 1)" if price_stickiness > 0.8
        else "moderate price stickiness" if price_stickiness > 0.5
        else "low price stickiness (frequent re-pricing)"
    )
    wage_desc = (
        "rigid wages further dampen cost pass-through" if wage_stickiness > 0.7
        else "moderate wage stickiness"
    )
    return (
        f"Inflation (π̂) exhibits a first-order autocorrelation of {_fmt(ac1_pi, 3)} "
        f"and a correlation with output of {_fmt(corr_pi, 3)}. "
        f"The model features {stick_desc} and {wage_desc}, "
        f"{'producing the sluggish inflation dynamics typical of NK models'
           if price_stickiness > 0.6
           else 'allowing relatively rapid price adjustment'}."
    )


def _debt_feedback_sentence(debt_feedback: float) -> str:
    if debt_feedback > 0.5:
        desc = (
            f"The debt feedback coefficient of {debt_feedback:.2f} is large, "
            f"meaning fiscal consolidation is aggressive: higher debt quickly "
            f"triggers spending cuts or tax rises, stabilising debt ratios but "
            f"potentially amplifying short-run output fluctuations."
        )
    elif debt_feedback > 0.1:
        desc = (
            f"With a debt feedback coefficient of {debt_feedback:.2f}, fiscal "
            f"policy responds gradually to debt deviations, providing mild "
            f"stabilisation without strong procyclical pressure."
        )
    else:
        desc = (
            f"The near-zero debt feedback coefficient ({debt_feedback:.2f}) implies "
            f"almost no automatic fiscal response to debt accumulation, leaving "
            f"the debt path largely open-loop in the short run."
        )
    return desc


# ---------------------------------------------------------------------------
# Empirical comparison sentence
# ---------------------------------------------------------------------------

def _empirical_comparison_sentence(
    model_moments: Dict[str, float],
    empirical_moments: Optional[Dict[str, float]],
) -> str:
    if not empirical_moments:
        return ""

    hits = 0
    total = 0
    for key in ["var_y", "var_c", "var_i", "var_l", "var_pi",
                "ac1_y", "ac1_c", "ac1_i", "corr_c", "corr_i"]:
        m_val = model_moments.get(key, float("nan"))
        e_val = empirical_moments.get(key, float("nan"))
        if np.isnan(m_val) or np.isnan(e_val) or e_val == 0:
            continue
        ratio = m_val / e_val
        total += 1
        if 0.5 <= ratio <= 2.0:   # within 2× of empirical = "reproduces"
            hits += 1

    if total == 0:
        return ""

    frac = f"{hits}/{total}"
    quality = (
        "strong" if hits / total >= 0.7
        else "partial" if hits / total >= 0.4
        else "limited"
    )

    # Highlight consumption specifically
    m_var_c = model_moments.get("var_c", float("nan"))
    e_var_c = empirical_moments.get("var_c", float("nan"))
    if not (np.isnan(m_var_c) or np.isnan(e_var_c) or e_var_c == 0):
        c_ratio_pct = _pct(m_var_c / e_var_c)
        c_note = (
            f" In particular, the model reproduces {c_ratio_pct} of empirical "
            f"consumption volatility."
        )
    else:
        c_note = ""

    return (
        f"Overall, the model achieves {quality} empirical moment matching, "
        f"reproducing {frac} target moments within a factor of two of their "
        f"empirical counterparts.{c_note}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tab1_commentary(
    moments: Dict[str, float],
    h: float,
    utilization: float,
    price_stickiness: float,
    wage_stickiness: float,
    debt_feedback: float,
    empirical_moments: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate a two-paragraph plain-English commentary for Tab 1.

    Parameters
    ----------
    moments           : model moment dict from moments.py (raw or HP-filtered)
    h                 : habit persistence parameter
    utilization       : capital utilization flexibility [0, 1]
    price_stickiness  : Calvo price-stickiness parameter [0, 1)
    wage_stickiness   : Calvo wage-stickiness parameter [0, 1)
    debt_feedback     : fiscal debt-feedback coefficient
    empirical_moments : optional empirical moment dict from empirical.py

    Returns
    -------
    Multi-sentence string suitable for st.markdown() or st.write().
    """
    lines = [
        "**Model Dynamics – Moment Interpretation**\n",
        _output_sentence(moments),
        _consumption_sentence(moments, h),
        _investment_sentence(moments),
        _hours_sentence(moments, utilization),
        _inflation_sentence(moments, price_stickiness, wage_stickiness),
        "\n**Fiscal Channel**\n",
        _debt_feedback_sentence(debt_feedback),
    ]

    if empirical_moments:
        lines.append("\n**Empirical Fit**\n")
        lines.append(_empirical_comparison_sentence(moments, empirical_moments))

    return "  \n".join(lines)
