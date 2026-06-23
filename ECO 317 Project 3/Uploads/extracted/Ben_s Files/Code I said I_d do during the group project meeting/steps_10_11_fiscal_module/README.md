# Steps 10-11 Fiscal Module

This folder implements the assignment's Step 10-11 core pieces in isolation:

- `policy/shocks.py`: fiscal shock impulses (`gc`, `gi`, `tau_l_cut`, `tau_k_cut`)
- `policy/financing.py`: financing-rule selector logic with instrument-specific debt-feedback signs
- `simulation/irf.py`: deterministic 40-quarter IRF simulation
- `policy/multipliers.py`: impact multiplier, discounted cumulative multiplier, fiscal drag horizon
- `tests/`: minimal tests for IRF behavior and multipliers

Run tests from project root:

```powershell
python -m pytest steps_10_11_fiscal_module/tests -q
```
