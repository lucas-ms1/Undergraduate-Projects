# Prompt Log — ECO 317 Project 3

## 2026-04-24: Critical bug-fix pass

**Prompt:** Fix critical issues identified in grading report.

**Changes made:**

### Critical Issue 1 — Euler Equation (`solvers/state_space.py`)

Three errors in the `euler_optimizer` row were corrected:

1. **Forward consumption coefficient:** Was `beta * habit / (1 + habit)`. Corrected to `-1.0 / (1 + habit)` (the standard shortcut-form habit-persistence Euler coefficient on E_t[c_{t+1}]).

2. **Double-counted expected inflation:** The equation had both a direct forward `pi_hat` term AND used `r_real_hat`, which itself embeds expected inflation via the Fisher relation (`r_real = i_nom - E[pi]`). Removed the direct `Gamma_f[pi_hat]` entry so expected inflation enters only once, through `r_real_hat`.

3. **Real rate coefficient:** Was `1/sigma`. Corrected to `(1 - habit) / (sigma * (1 + habit))`, which is the proper elasticity under external habit persistence.

### Critical Issue 2 — Wage Phillips Curve (`solvers/state_space.py`)

The MRS linearization applied `(sigma + phi_l)` to both consumption and labor. The correct linearization is `sigma` on consumption and `phi_l` on labor:
- `Gamma_0[c_hat] = -kappa_w * sigma`
- `Gamma_0[l_hat] = -kappa_w * phi_l`

### Test KeyError fix (`dsge/model.py`)

Three tests (`test_blanchard_kahn.py` lines 10 and 81, `test_multipliers.py` line 33) referenced `model["structural_system"]`, which was not in the `solve_model_objects()` return dict. Added `"structural_system": structural` to the return dictionary.

### Test assertion update (`tests/test_structural_state_space.py`)

Updated `test_euler_row_coefficients` to assert the corrected coefficients: forward consumption at `-1/(1+h)`, zero forward `pi_hat`, and real-rate coefficient at `(1-h)/(sigma*(1+h))`.
