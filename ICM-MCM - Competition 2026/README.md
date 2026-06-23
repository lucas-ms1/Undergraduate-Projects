# ICM-MCM - Competition 2026

Competition project folder containing work for the 2026 ICM/MCM submissions. The folder currently includes a policy and workforce exposure project for ICM Problem F and a data/modeling project for MCM Problem C.

## Contents

- `Problem #2/`: ICM Problem F project with data construction, calibration, policy scoring, uncertainty analysis, tables, figures, and report artifacts.
- `Problem C Data With The Stars/`: MCM Problem C project with preprocessing, model fitting, counterfactual analysis, robustness checks, and report artifacts.
- `.gitignore`: original project ignore rules copied from the standalone repository.

## Problem #2

Key entry points include:

- `run_all.py`: full project pipeline.
- `run_scenarios.py`: scenario pipeline.
- `build_*.py`: individual data, calibration, policy, uncertainty, and report-build steps.
- `data/`: source, intermediate, and final data artifacts.
- `reports/`: LaTeX report, tables, and figures.

## Problem C Data With The Stars

Key entry points include:

- `main.py`: main analysis script.
- `run_gonogo_checks.py`: validation and readiness checks.
- `src/`: preprocessing, model, fitting, and analysis code.
- `data/`: competition data.
- `reports/`: report files, tables, and diagnostics.

## Notes

This is a copied project folder inside `Undergraduate-Projects`, not a nested Git repository. The original standalone local clone is outside this repo at `C:\Users\freew\Dropbox\Github\ICM-MCM-Competition-2026`.
