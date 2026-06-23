═══════════════════════════════════════════════════
ECO 317 PROJECT 3 — GRADING REPORT
═══════════════════════════════════════════════════


CATEGORY A: Technical Execution & Logic
Score: 3/4

Evidence:
- The project is unified in a single codebase with a clean modular structure:
  dsge/, solvers/, simulation/, policy/, utils/, tests/ (app.py lines 1–541).
- Heavy inline comments explaining both economic logic and mathematical
  derivations are present in every module (e.g., steady_state.py documents
  every step from Euler to fiscal block; state_space.py labels every
  equation row; econometrics.py documents the Taylor-rule recovery
  regression spec).
- @st.cache_data correctly separates the solve step from the simulate step
  (app.py lines 120–133). The solve_key excludes sigma_ parameters
  (line 183), so moving a shock-variance slider does NOT trigger a re-solve.
  The simulate cache uses A_bytes, B_bytes, T, seed, and shock_stds as keys.
- The solver returns clear "converged" / "indeterminacy" / "no_solution"
  flags (rational_expectations.py lines 146–159), and the UI surfaces
  st.error() with a diagnostic hint via likely_failure_hint() when BK fails
  (app.py lines 334–337, 411–414).
- requirements.txt is present with all dependencies (streamlit, numpy,
  scipy, pandas, matplotlib, plotly, pytest, statsmodels, fredapi,
  pandas-datareader).

Issues:
1. NO prompt_log.md found anywhere in the project. This is explicitly
   required by the assignment. (-0.25 points)
2. solve_model_objects() (dsge/model.py) does not return a
   "structural_system" key, but test_blanchard_kahn.py (lines 8, 51) and
   test_multipliers.py (line 31) reference model["structural_system"].
   These tests will raise KeyError at runtime. At least 3 tests are
   broken. (-0.25 points)
3. The _enforce_taylor_rule() and _enforce_government_budget() functions
   in rational_expectations.py (lines 92–147) overwrite rows of the A and
   B policy matrices AFTER the QZ solve. This is non-standard and suggests
   the QZ extraction has numerical issues that require post-hoc patching.
   While it produces stable results, it is fragile and not theoretically
   grounded.


CATEGORY B: Data Retrieval & Integrity
Score: 4/4

Evidence:
- simulation/empirical.py pulls real data from FRED programmatically
  (lines 64–95) using fredapi with fallback to pandas-datareader.
- All required FRED series are present (lines 36–42):
  GDPC1 (Real GDP), PCECC96 (Real PCE), PNFIC1 (Investment),
  HOABS (Hours), CPIAUCSL (CPI).
- FRED pull is cached to disk as CSV (CACHE_PATH, line 45) with a
  configurable TTL (default 24h, line 46). The Streamlit layer adds
  @st.cache_data(ttl=86400) on top (app.py lines 135–137).
- Raw FRED series are log-transformed (lines 107–115); CPI is
  log-differenced to get inflation. All series are HP-filtered at
  λ=1600 (line 127) before computing empirical moments.
- Tab 1 shows a side-by-side model-vs-empirical moments comparison
  table via build_comparison_table() (app.py lines 353–359), with
  columns: Moment, Model, Empirical, Difference, Ratio.
- Unit transformations are correct: log-levels for GDP/C/I/L,
  log-differences for CPI→inflation.

Issues:
- None significant. This category is well-executed.


CATEGORY C: Visualization & Presentation
Score: 3/4

Evidence:
- Styling: Dark mode uses #0e1a2b background (app.py line 67),
  EB Garamond font imported from Google Fonts (line 65), high-contrast
  #f0f2f6 text. Light mode also implemented with toggle (line 43).
  Sidebar uses #0a1420 (line 68). Tab labels and widgets use Garamond.
- Tab 1 charts: Time-series plot of ŷ, ĉ, î over 1,000 periods
  (lines 362–373) with labeled axes, legend, and zero-line.
- Tab 2 IRF subplots: Four subplots — Output, Consumption, Investment,
  Government Debt (lines 469–490). Labeled axes, zero-line, legend.
  Fiscal Drag Horizon is visually marked as a vertical dotted line
  with annotation (lines 481–483).
- Composite chart: Overlay of Output IRF across ALL five financing
  rules on the same axes with a legend (lines 493–528). This is the
  "bells and whistles" feature.
- Metric boxes: Impact Multiplier, Cumulative Multiplier, and Fiscal
  Drag Horizon displayed with st.metric (lines 460–466). Text is
  readable against background in both dark and light mode via CSS
  targeting [data-testid="stMetricValue"] (lines 78–79).
- Tables: Moments table (line 350), calibration table (lines 282–294),
  steady-state table (lines 298–310), Taylor-rule table (lines 313–318)
  all render cleanly. Unicode symbols used for Greek letters (β, σ, α,
  etc.) — renders properly.
- Model Specification tab (lines 218–329) displays all key equations
  with proper LaTeX rendering via st.latex().

Issues:
1. The Cumulative Multiplier comparison table by financing rule
   (lines 530–533) renders but could be more prominent.
2. No Plotly interactive charts — all static matplotlib. The
   requirements.txt includes plotly but it's never used. Interactive
   charts would earn the "4" here.


CATEGORY D: Econometric Analysis
Score: 3/4

Evidence:
- Taylor-rule recovery regression in simulation/econometrics.py
  (lines 102–155): Estimates î_t = c + ρ̂_i·î_{t-1} + φ̂_π·π̂_t +
  φ̂_y·ŷ_t + u_t. Reports coefficients, standard errors, t-stats,
  and R² via OLSResult.summary_rows() (lines 35–49).
  The regression divides raw coefficients by (1-ρ̂_i) to recover
  structural φ_π and φ_y (lines 139–147).
- Consumption smoothness regression (lines 161–183): Estimates
  ĉ_t = c + α₁·ŷ_t + α₂·ŷ_{t-1} + α₃·ĉ_{t-1} + u_t. Reports
  all required statistics.
- Variance ratios with bootstrapped standard errors and 95% CIs
  (lines 195–239): Var(ĉ)/Var(ŷ), Var(î)/Var(ŷ), etc. with
  n_boot=500 bootstrap replications.
- Regression results displayed in Tab 1 below the moments table
  (app.py lines 375–398) in a two-column layout.
- test_econometrics.py verifies coefficient recovery:
  ρ_i within 15% of 0.80 (line 89), φ_π within 15% of 1.50
  (line 95), φ_y within 25% of 0.125 (line 101).
  The "actual model" test (lines 136–156) checks recovery at
  machine precision when the monetary shock is subtracted.
- The commentary in summaries.py ties econometric results to
  economic theory (lines 199–223).

Issues:
1. The Taylor-rule recovery test uses 25% tolerance for φ_y (not the
   15% stated in the rubric) — test_econometrics.py line 101:
   `abs(phi_hat - 0.125) / 0.125 < 0.25`. This is looser than spec.
2. The econometric conclusions in summaries.py are somewhat generic
   (e.g., "phi_pi_hat is near/away from the calibration target").
   More detailed economic interpretation of what each coefficient
   means for policy transmission would strengthen this.


CATEGORY E: Demo Performance & Q&A Readiness
Score: 3/4

Evidence:
- Q1 ("Explain rule-of-thumb consumers"): The code implements ROT
  households with lambda_rot parameter. state_space.py lines 81–88
  define the ROT budget constraint: ĉ^rot depends on current wage
  income and transfers only (no savings). Lines 90–92 define
  aggregate consumption: ĉ = (1-λ)ĉ^o + λĉ^rot. The lambda_rot
  slider is wired in app.py (line 51) and flows through to the
  state-space system via params["lambda_rot"].
- Q2 ("Why does capital-tax financing crash the multiplier"): The
  code implements all five financing rules in policy/financing.py.
  debt_feedback_sign() (lines 50–62) correctly returns +1 for tax
  instruments and -1 for lump-sum/gc-cut. Under capital-tax
  financing, τ_K rises with debt, distorting r^k and discouraging
  investment. The composite overlay chart (app.py lines 493–528)
  visually demonstrates this effect. The automated policy briefing
  (lines 535–540) provides verbal commentary.

Issues:
1. The ROT budget in state_space.py (lines 81–88) uses aggregate C
   in the normalization denominators rather than C^rot — this is
   an approximation that holds only when C^rot ≈ C in steady state.
   Under extreme lambda_rot values, this could introduce error.
2. The Model Specification tab shows all equations but lacks a
   paragraph-level explanation of how rule-of-thumb consumers
   alter the aggregate Euler equation, which would be needed for
   Q1 in a live demo.


─────────────────────────────────────────────────
TOTAL: 16/20
─────────────────────────────────────────────────


ECONOMICS VERIFICATION CHECKLIST:

[!] Euler equation (full habit FOC, β placement)
    ISSUE FOUND — state_space.py line 67:
    Gamma_f[c_o_hat] = β·h/(1+h) — coefficient on E_t[ĉ_{t+1}^o]
    is β·h/(1+h) = 0.407, but the standard shortcut linearization
    (Smets-Wouters 2007 eq. 2) gives 1/(1+h) = 0.588. The plan
    locks in the FULL internal-habit FOC, which gives
    βh/(1+βh²) ≈ 0.414 — closer to the code but with (1+βh²)
    in the denominator, not (1+h). The code uses an inconsistent
    hybrid of shortcut and full-FOC coefficients.
    ADDITIONALLY: The Euler row includes a direct Gamma_f term on
    pi_hat (line 68) of -(1-h)/(σ(1+h)), while simultaneously
    using r_real_hat (line 70) which itself = î - E_t[π̂] via the
    Fisher equation. This DOUBLE-COUNTS expected inflation.
    The correct shortcut Euler using r_real_hat should have NO
    separate E_t[π̂] forward term.
    ADDITIONALLY: The coefficient on r_real_hat is 1/σ (line 70),
    but the correct shortcut coefficient is (1-h)/(σ(1+h)).

[✓] Rule-of-thumb aggregation (λ wired correctly)
    state_space.py lines 90–92: ĉ = (1-λ)ĉ^o + λĉ^rot ✓
    app.py line 51: lambda_rot slider connected ✓

[✓] Price Phillips curve (κ_p formula, limit behavior)
    state_space.py line 62:
    κ_p = (1-θ_p)(1-βθ_p)/θ_p ✓
    Limit: θ_p→0 gives κ_p→∞ (flexible) ✓
    Limit: θ_p→1 gives κ_p→0 (sticky) ✓
    NKPC: π̂_t = β·E_t[π̂_{t+1}] + κ_p·m̂c_t (lines 94–97) ✓

[!] Wage Phillips curve (κ_w formula)
    ISSUE FOUND — state_space.py lines 63–67:
    κ_w = (1-θ_w)(1-βθ_w) / (θ_w(1+φ_l·ε_w)) ✓ (formula correct)
    BUT the wage PC equation (lines 99–103) uses:
      -κ_w·(σ+φ_l)·ĉ_t  AND  -κ_w·(σ+φ_l)·l̂_t
    This applies the SAME coefficient (σ+φ_l) to both consumption
    and labor. The correct MRS linearization is:
      m̂rs_t = σ·ĉ_t + φ_l·l̂_t  (σ for c, φ_l for l)
    With baseline σ=1, φ_l=1 the code gives 2ĉ+2l̂ where it
    should give ĉ+l̂ — DOUBLING the wage PC slope.

[✓] Marginal cost equation
    state_space.py lines 107–111:
    m̂c_t = (1-α)·ŵ_t + α·r̂^k_t - â_t ✓

[✓] Capital timing (K_{t-1} in production)
    state_space.py line 114: Gamma_l[k_hat] = -alpha
    Production uses K_{t-1} via the lag matrix ✓
    Capital rental FOC also uses K_{t-1} (line 119) ✓

[✓] Government budget constraint (all sign conventions)
    state_space.py lines 129–142:
    b̂_t = (1+r)b̂_{t-1} + (G_c/B)ĝ_c + (G_i/B)ĝ_i + (|T|/B)t̂
           - (τ_c·C/B)(τ̂_c + ĉ) - (τ_l·wL/B)(τ̂_l + ŵ + l̂)
           - (τ_k·r^k·K/B)(τ̂_k + r̂^k + k̂)
    Signs: spending (+), transfers (+), taxes (−) ✓
    Note: uses |T| for transfer_scale — unconventional but
    internally consistent if t_hat is defined in |T| units.

[✓] Fiscal rule signs (instrument-specific φ_b)
    policy/financing.py lines 50–62:
    debt_feedback_sign("lump_sum") = -1.0 ✓
    debt_feedback_sign("gc_cut") = -1.0 ✓
    debt_feedback_sign("consumption_tax") = +1.0 ✓
    debt_feedback_sign("labor_tax") = +1.0 ✓
    debt_feedback_sign("capital_tax") = +1.0 ✓

[✓] Taylor rule present with φ_π > 1 enforcement
    state_space.py lines 145–150: Taylor rule equation present ✓
    dsge/model.py line 55: policy_ok = phi_pi > 1.0 ✓
    app.py line 58: st.error when phi_pi ≤ 1 ✓

[✓] Fisher relation connecting nominal and real rates
    state_space.py lines 92–95:
    r̂_t^{real} = î_t - E_t[π̂_{t+1}] ✓

[✓] Steady-state residuals ≈ 0
    dsge/steady_state.py lines 75–84: Three residuals computed ✓
    test_steady_state.py: All three verified < 1e-10 ✓

[✓] BK condition checked correctly
    rational_expectations.py uses scipy.linalg.ordqz (line 4) ✓
    Explosive eigenvalues counted correctly (lines 12–14) ✓
    BK = explosive == jump_count (lines 146–155) ✓
    Complex conjugate pairs counted individually ✓
    Returns converged/indeterminacy/no_solution flag ✓


AI-HALLUCINATION TRAP CHECKLIST:

[✓] Calvo intercept not dropped
    κ_p = (1-θ_p)(1-βθ_p)/θ_p — correct formula with no
    missing intercept. Limit behavior verified.

[!] Habit FOC version documented
    The code implements a HYBRID that matches neither the full
    internal-habit FOC nor the standard shortcut. The README does
    not document which version is used or flag the discrepancy.

[!] β on correct side of Euler
    The code places β·h/(1+h) on E_t[ĉ_{t+1}], which appears to
    confuse the full-FOC β placement with the shortcut form.
    The standard shortcut has 1/(1+h) (no β on consumption);
    the full FOC has βh/(1+βh²).

[✓] K_{t-1} timing in production
    state_space.py line 114: Gamma_l (lag matrix) used for k_hat ✓

[✓] Budget constraint signs correct
    Spending grows debt (+), taxes shrink debt (−) ✓

[✓] G_c-cut sign flip handled
    financing.py: debt_feedback_sign("gc_cut") = -1.0 ✓
    A positive debt impulse under gc_cut produces a decline in G_c.

[✓] Multiplier Y/G scaling present
    policy/multipliers.py lines 12–13: to_level_response() converts
    log-deviations to level terms using SS levels. impact_multiplier()
    computes (Y_ss * ŷ_0) / (G_ss * ĝ_0) ✓

[✓] Taylor rule not omitted
    Present in state_space.py lines 145–150, enforced in QZ output,
    and surfaced in app.py diagnostics.


CRITICAL ISSUES (would cause point deductions):

1. EULER EQUATION MISSPECIFICATION (state_space.py lines 67–72):
   Three distinct errors: (a) coefficient βh/(1+h) on forward
   consumption instead of 1/(1+h) (shortcut) or βh/(1+βh²) (full FOC);
   (b) double-counting of E_t[π̂] through both a direct Gamma_f term
   and through r_real_hat; (c) coefficient 1/σ on r_real_hat instead
   of (1-h)/(σ(1+h)). These affect all consumption dynamics and
   propagation of monetary policy shocks through the model.
   SEVERITY: HIGH — this is the core intertemporal equation.

2. WAGE PHILLIPS CURVE MRS ERROR (state_space.py lines 99–103):
   The marginal rate of substitution coefficient applies (σ+φ_l) to
   BOTH consumption and labor, instead of σ to consumption and φ_l
   to labor. At baseline (σ=φ_l=1), this doubles the wage PC slope,
   producing excessively rigid real wages relative to the correct
   specification.
   SEVERITY: MEDIUM — affects wage-inflation dynamics.

3. TEST SUITE BREAKAGE (test_blanchard_kahn.py lines 8, 51;
   test_multipliers.py line 31): Three tests reference
   model["structural_system"] which does not exist in the return
   value of solve_model_objects(). These tests will raise KeyError.
   SEVERITY: MEDIUM — tests exist but don't pass.

4. MISSING PROMPT_LOG.MD: The assignment explicitly requires a
   prompt_log.md documenting AI-assisted components. None found.
   SEVERITY: LOW-MEDIUM — documentation requirement.


RECOMMENDATIONS FOR IMPROVEMENT:

1. FIX THE EULER EQUATION: Replace the euler_optimizer row in
   state_space.py with the correct shortcut linearization:
     Gamma_f[c_o_hat] = 1.0 / (1.0 + habit)  [not βh/(1+h)]
     Remove Gamma_f[pi_hat] term entirely  [use r_real_hat only]
     Gamma_0[r_real_hat] = (1.0 - habit) / (sigma * (1.0 + habit))
   Or, if the plan's full internal-habit FOC is desired, derive and
   implement the correct linearization with (1+βh²) denominators.

2. FIX THE WAGE PHILLIPS CURVE: Split the mrs_coeff into separate
   coefficients for consumption and labor:
     Gamma_0[c_hat] = -kappa_w * sigma
     Gamma_0[l_hat] = -kappa_w * phi_l
   This recovers m̂rs = σĉ + φl̂ as required.

3. FIX THE TEST SUITE: Add "structural_system": structural to the
   return dict of solve_model_objects() in dsge/model.py, or update
   the three broken tests to use the correct keys.

4. ADD PROMPT_LOG.MD: Create a prompt_log.md documenting which AI
   tools were used, which components were AI-generated, and what
   modifications were made to AI output.

5. REMOVE THE ENFORCE FUNCTIONS: Once the Euler and wage PC are
   fixed, the _enforce_taylor_rule() and _enforce_government_budget()
   patches in rational_expectations.py should become unnecessary.
   Remove them and verify the QZ output directly satisfies these
   equations.

6. ADD INTERACTIVE CHARTS: Replace matplotlib figures with Plotly
   for interactivity (zoom, hover tooltips with exact values).
   The plotly dependency is already in requirements.txt.

7. STRENGTHEN ECONOMETRIC COMMENTARY: Tie the Taylor-rule recovery
   more explicitly to economic theory — e.g., "the recovered ρ̂_i ≈ 0.80
   confirms the Taylor-rule smoothing parameter, implying the central
   bank adjusts interest rates gradually, with only 20% of the desired
   adjustment occurring each quarter."

8. HANDLE FINANCING RULE IN TEST_IRF: The test
   test_gc_output_irfs_differ_across_financing_rules (test_irf.py
   line 42) calls simulate_irf() without passing financing_rule or
   shock_index, so all rules produce identical IRFs. Pass these
   parameters to actually test differentiation across rules.
