"""Narrative summaries for Streamlit tabs."""


def tab1_summary(params, moments):
    c_rel = 0.0
    if moments["var_y_hat"] > 1e-12:
        c_rel = (moments["var_c_hat"] / moments["var_y_hat"]) ** 0.5
    return (
        f"With habit h={params['habit']:.2f}, consumption volatility is {100*c_rel:.1f}% of output volatility. "
        f"Price stickiness (theta_p={params['theta_p']:.2f}) and wage stickiness (theta_w={params['theta_w']:.2f}) "
        f"shape inflation persistence, while debt feedback phi_b={params['phi_b']:.3f} stabilizes debt."
    )


def tab2_summary(shock, rule, mult):
    drag_text = (
        f"Fiscal drag begins in quarter {mult['fiscal_drag_horizon']}."
        if mult["fiscal_drag_horizon"] is not None
        else "No fiscal drag within the 40-quarter window."
    )
    return (
        f"For the {shock} under {rule}, the impact multiplier is {mult['impact_multiplier']:.3f} "
        f"and the cumulative multiplier is {mult['cumulative_multiplier']:.3f}. {drag_text}"
    )
