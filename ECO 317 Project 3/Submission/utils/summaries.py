"""
utils/summaries.py
Step 9 - Dynamic commentary generator with LaTeX math
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np


def _pct(value, decimals=1):
    return f"{value * 100:.{decimals}f}%"

def _ratio(num, den):
    if den == 0 or np.isnan(den) or np.isnan(num):
        return None
    return num / den

def _get(d, key):
    return d.get(key, float("nan"))

def _fmt(x, decimals=4):
    if np.isnan(x):
        return "N/A"
    return f"{x:.{decimals}f}"


def _output_sentence(m):
    ac1_y = _get(m, "ac1_y")
    persist = ('indicating substantial persistence' if ac1_y > 0.7
               else 'suggesting moderate persistence' if ac1_y > 0.4
               else 'indicating relatively low persistence')
    var_y = _fmt(_get(m, 'var_y'))
    ac1_s = _fmt(ac1_y, 3)
    return (
        "Output $\\hat{y}_t$ has a variance of " + var_y + " and a "
        "first-order autocorrelation of " + ac1_s + ", "
        "" + persist + " in the business cycle."
    )


def _consumption_sentence(m, h):
    ratio = _ratio(_get(m, "var_c"), _get(m, "var_y"))
    ratio_str = _pct(ratio) if ratio is not None else "N/A"
    smoothness = (
        "well below" if (ratio or 1) < 0.8
        else "slightly below" if (ratio or 1) < 1.0
        else "above"
    )
    habit_effect = (
        "amplified habit formation dampens the consumption response"
        if h >= 0.6
        else "moderate habit persistence provides some consumption smoothing"
        if h >= 0.3
        else "low habit persistence, so consumption tracks income relatively closely"
    )
    return (
        "With $h = " + f"{h:.2f}" + "$, "
        "$\\mathrm{Var}(\\hat{c}) / \\mathrm{Var}(\\hat{y})$ = " + ratio_str + " "
        "(" + smoothness + " the stylised benchmark of ~50--80%), consistent with "
        "" + habit_effect + "."
    )


def _investment_sentence(m):
    ratio = _ratio(_get(m, "var_i"), _get(m, "var_y"))
    ratio_str = _pct(ratio) if ratio is not None else "N/A"
    sign = "exceeds" if (ratio or 0) > 1 else "falls short of"
    return (
        "Investment volatility is "
        "$\\mathrm{Var}(\\hat{\\imath}) / \\mathrm{Var}(\\hat{y})$ = " + ratio_str + ", which "
        "" + sign + " the empirical benchmark (investment is typically $3$--$4\\times$ more "
        "volatile than output)."
    )


def _hours_sentence(m, utilisation):
    corr = _get(m, "corr_l")
    adj = ("allows firms to adjust intensity of capital use, muting hours fluctuations"
           if utilisation > 0.5
           else "restricts capital utilisation adjustment, pushing more of the margin onto hours")
    corr_s = _fmt(corr, 3)
    cyc = 'strongly procyclical as expected' if corr > 0.7 else 'moderately procyclical'
    return (
        "Hours $\\hat{\\ell}_t$ have a correlation with output of " + corr_s + ", "
        "" + cyc + ". "
        "With capital utilisation flexibility $\\psi = " + f"{utilisation:.2f}" + "$, the model " + adj + "."
    )


def _inflation_sentence(m, price_stickiness, wage_stickiness):
    ac1_pi = _get(m, "ac1_pi")
    corr_pi = _get(m, "corr_pi")
    stick_desc = (
        "high price stickiness ($\\theta_p$ near 1)" if price_stickiness > 0.8
        else "moderate price stickiness" if price_stickiness > 0.5
        else "low price stickiness (frequent re-pricing)"
    )
    wage_desc = (
        "rigid wages ($\\theta_w > 0.7$) further dampen cost pass-through" if wage_stickiness > 0.7
        else "moderate wage stickiness"
    )
    dynamics = ("producing the sluggish inflation dynamics typical of NK models"
                if price_stickiness > 0.6
                else "allowing relatively rapid price adjustment")
    return (
        "Inflation $\\hat{\\pi}_t$ exhibits a first-order autocorrelation of " + _fmt(ac1_pi, 3) + " "
        "and a correlation with output of " + _fmt(corr_pi, 3) + ". "
        "The model features " + stick_desc + " and " + wage_desc + ", " + dynamics + "."
    )


def _debt_feedback_sentence(debt_feedback):
    phi_str = f"{debt_feedback:.2f}"
    if debt_feedback > 0.5:
        return (
            "The debt feedback coefficient $\\phi_b = " + phi_str + "$ is large, "
            "meaning fiscal consolidation is aggressive: higher "
            "$\\hat{B}_t$ quickly triggers spending cuts or tax rises, stabilising debt ratios but "
            "potentially amplifying short-run output fluctuations."
        )
    elif debt_feedback > 0.1:
        return (
            "With $\\phi_b = " + phi_str + "$, fiscal "
            "policy responds gradually to $\\hat{B}_t$ deviations, providing mild "
            "stabilisation without strong procyclical pressure."
        )
    else:
        return (
            "The near-zero debt feedback coefficient $\\phi_b = " + phi_str + "$ implies "
            "almost no automatic fiscal response to $\\hat{B}_t$ accumulation, leaving "
            "the debt path largely open-loop in the short run."
        )


def _empirical_comparison_sentence(model_moments, empirical_moments):
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
        if 0.5 <= ratio <= 2.0:
            hits += 1
    if total == 0:
        return ""
    frac = str(hits) + "/" + str(total)
    quality = (
        "strong" if hits / total >= 0.7
        else "partial" if hits / total >= 0.4
        else "limited"
    )
    c_note = ""
    m_var_c = model_moments.get("var_c", float("nan"))
    e_var_c = empirical_moments.get("var_c", float("nan"))
    if not (np.isnan(m_var_c) or np.isnan(e_var_c) or e_var_c == 0):
        c_ratio_pct = _pct(m_var_c / e_var_c)
        c_note = (" In particular, the model reproduces " + c_ratio_pct +
                  " of empirical $\\mathrm{Var}(\\hat{c})$.")
    return (
        "Overall, the model achieves " + quality + " empirical moment matching, "
        "reproducing " + frac + " target moments within a factor of two of their "
        "empirical counterparts." + c_note
    )


def _econometrics_sentence(econometrics):
    if not econometrics:
        return ""
    lines = []
    taylor = econometrics.get("taylor_rule")
    if taylor is not None:
        coefs = taylor.as_dict()
        phi_pi = coefs.get("phi_pi_hat", float("nan"))
        if not np.isnan(phi_pi):
            closeness = "near" if abs(phi_pi - 1.50) <= 0.25 else "away from"
            lines.append(
                "The Taylor-rule regression reports structural coefficients; "
                "phi_pi_hat = " + _fmt(phi_pi, 3) + " is " + closeness +
                " the calibration target phi_pi = 1.50."
            )
    smooth = econometrics.get("consumption_smoothness")
    if smooth is not None:
        coefs = smooth.as_dict()
        c_lag = coefs.get("c_lag1", float("nan"))
        if not np.isnan(c_lag):
            lines.append(
                "The consumption-smoothness regression has c_lag1 = " +
                _fmt(c_lag, 3) + ", which is direct evidence on habit persistence "
                "and rule-of-thumb household propagation."
            )
    ratios = econometrics.get("variance_ratios")
    if ratios:
        lines.append(
            "The variance-ratio estimates are model-fit diagnostics: values far "
            "from empirical ratios point to volatility gaps rather than policy effects."
        )
    return " ".join(lines)


def generate_tab1_commentary(moments, h, utilisation, price_stickiness,
                              wage_stickiness, debt_feedback,
                              empirical_moments=None, econometrics=None):
    """Generate multi-paragraph plain-English commentary for Tab 1."""
    lines = [
        "**Model Dynamics - Moment Interpretation**\n",
        _output_sentence(moments),
        _consumption_sentence(moments, h),
        _investment_sentence(moments),
        _hours_sentence(moments, utilisation),
        _inflation_sentence(moments, price_stickiness, wage_stickiness),
        "\n**Fiscal Channel**\n",
        _debt_feedback_sentence(debt_feedback),
    ]
    if empirical_moments:
        lines.append("\n**Empirical Fit**\n")
        lines.append(_empirical_comparison_sentence(moments, empirical_moments))
    econ_text = _econometrics_sentence(econometrics)
    if econ_text:
        lines.append("\n**Econometric Diagnostics**\n")
        lines.append(econ_text)
    return "  \n".join(lines)


def generate_tab2_briefing(shock_name, financing_rule, impact_mult,
                            cumulative_mult, drag_horizon):
    """Generate automated policy briefing for Tab 2."""
    if drag_horizon is not None:
        drag_text = ("Fiscal drag begins at $t^* = " + str(drag_horizon) + "$, when the financing "
                     "instrument's distortionary effects push $\\hat{y}_t$ below steady state.")
    else:
        drag_text = ("$\\hat{y}_t$ does not turn negative within the 40-quarter window, "
                     "suggesting the financing rule is relatively non-distortionary.")

    if impact_mult > 1.0:
        mult_comparison = ("The impact multiplier exceeds unity ($\\mathrm{IM} > 1$), indicating strong "
                           "short-run demand effects.")
    elif impact_mult > 0.5:
        mult_comparison = ("The impact multiplier is moderate, consistent with standard "
                           "New Keynesian predictions.")
    else:
        mult_comparison = ("The impact multiplier is relatively low, likely reflecting "
                           "significant crowding-out or distortionary financing effects.")

    im_s = f"{impact_mult:.3f}"
    cm_s = f"{cumulative_mult:.3f}"
    header = "**Policy Briefing: " + shock_name + " under " + financing_rule + "**"
    body = ("The impact multiplier is $\\mathrm{IM} = " + im_s + "$ and the cumulative "
            "(discounted) multiplier is $\\mathrm{CM} = " + cm_s + "$. " + mult_comparison)
    return header + "\n\n" + body + "\n\n" + drag_text
