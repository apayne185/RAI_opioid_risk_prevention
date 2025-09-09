# ML4HL OD RAI Toolbox — Opioid Risk Prevention

This project provides a didactic, end‑to‑end workflow to explore opioid use disorder (OD) risk with a calibrated classification model and Responsible AI (RAI) tooling. It includes a Jupyter notebook, reusable utilities for model evaluation and threshold selection, and a small, synthetic dataset.

- Goal: generate non‑trivial, actionable insights for clinical decision making using machine learning paired with responsible AI practices.
- Notebook: `ML4HL_OD_RAI_toolbox.ipynb`
- Utilities: `utils.py` (AUC reporting, threshold selection policies, plots)
- Data: `Data/opiod_raw_data.csv` (1,000 rows, 20+ features)


**Table of Contents**
- Introduction
- Project Structure
- Install
- Quickstart: Run the Notebook
- Data Description
- Utilities: Evaluation, Thresholds, and Plots
- Optional: Responsible AI Dashboard
- Reproducibility and Notes
- Troubleshooting
- License and Acknowledgements


**Introduction**
- Purpose: analyze OD risk predictions, understand errors, evaluate performance, and explore threshold policies (workload, recall floors, and cost) to inform prevention strategies.
- Methods: scikit‑learn pipeline (preprocessing + logistic regression), probability calibration, transparent model reporting (ROC/PR AUC), and threshold trade‑offs. Optional RAI dashboard for model error analysis and interpretability.
- Notebook origin: adapted from Microsoft’s Responsible AI toolkit examples.


**Project Structure**
- `ML4HL_OD_RAI_toolbox.ipynb`: step‑by‑step walkthrough (data prep, modeling, calibration, thresholding, visualizations).
- `utils.py`: helper functions used by the notebook for evaluation, threshold policies, and visualizations.
- `Data/opiod_raw_data.csv`: sample dataset used by the notebook (1,000 rows).
- `environment.yml`: conda environment for reproducibility (Python 3.10, sklearn, lightgbm, imbalanced‑learn, matplotlib, RAI packages, etc.).
- `LICENSE`: license for this repository.
- `README.md`: this file.


**Install**
- Prereqs: conda/mamba, Git, and a working Jupyter setup.
- Create the environment (recommended via conda/mamba):
  - `conda env create -f environment.yml`
  - `conda activate od_rai_lgbm`
- Verify Jupyter + widgets:
  - `python -c "import IPython, ipywidgets; print('Jupyter OK')"`
  - If using JupyterLab and widgets don’t render, ensure Lab ≥ 3.x. No manual widget install is typically required with this env.


**Quickstart: Run the Notebook**
- Launch Jupyter:
  - `jupyter lab` (or `jupyter notebook`)
- Open `ML4HL_OD_RAI_toolbox.ipynb` and run cells top‑to‑bottom.
- What the notebook does:
  - Loads `Data/opiod_raw_data.csv` and performs light cleaning/renaming (e.g., `rx ds → rx_ds`, `SURG → Surgery`).
  - Splits data into train/validation/test.
  - Builds a preprocessing pipeline: median imputation + scaling for numeric, and imputation for binary features.
  - Trains a logistic regression baseline and applies probability calibration (`CalibratedClassifierCV`).
  - Reports discrimination (ROC/PR AUC), calibration diagnostics, and prevalence/lift.
  - Explores threshold selection policies: workload constraints (alerts per 1,000), recall floors, and cost‑based choices.
  - Visualizes precision/recall vs threshold, cumulative recall vs alerts, and top‑K highest‑risk cases.


**Data Description**
- Rows: patient‑level records.
- Target: `OD` (1 = opioid use disorder in the 2‑year window, 0 = otherwise).
- Key predictors in the raw CSV:
  - `Low_inc`: low income flag.
  - `SURG`: surgery within 2 years (renamed to `Surgery` in notebook).
  - `rx ds`: days of prescribed opioids in 2 years (renamed to `rx_ds`).
  - `A .. V`: binary flags (e.g., infectious diseases, circulatory, respiratory, injuries, trauma, etc.).
- Example prevalence in the notebook: ~0.18 on validation/test.


**Utilities: Evaluation, Thresholds, and Plots**
`utils.py` exposes small, composable helpers used in the notebook. Import them directly:

```python
from utils import (
    positive_scores, auc_report, tradeoff_table,
    pick_threshold_cost, pick_threshold_recall_floor, pick_threshold_workload,
    summary_at_threshold,
    plot_recall_floor_curves, plot_cumulative_recall_at_threshold, plot_topk_at_threshold,
)
```

- Evaluation
  - `positive_scores(estimator, X)`: returns positive‑class scores for classifiers supporting `predict_proba` or `decision_function`.
  - `auc_report(y_true, y_score, name="model", plot=True)`: prints ROC/PR AUC, prevalence, and lift; plots ROC/PR curves.
- Threshold trade‑offs
  - `tradeoff_table(y_true, y_score, thresholds=None)`: precision, recall, confusion counts, alerts/1k, TP/1k across thresholds.
  - `pick_threshold_workload(y_true, y_score, alerts_per_1000_max)`: best TP/1k under an alert budget (returns summary + table).
  - `pick_threshold_recall_floor(y_true, y_score, recall_floor)`: max precision subject to minimum recall (returns summary + table).
  - `pick_threshold_cost(y_true, y_score, C_FP, C_FN)`: minimizes expected cost (Bayes formula vs empirical minimum).
- Visualizations at a threshold
  - `summary_at_threshold(y_true, y_score, thr)`: one‑row summary at a specific threshold.
  - `plot_recall_floor_curves(...)`: precision/recall vs threshold with chosen recall floor and threshold highlighted.
  - `plot_cumulative_recall_at_threshold(...)`: cumulative capture vs number of alerts with vertical line at the implied alerts.
  - `plot_topk_at_threshold(...)`: bar chart of top‑K highest‑risk patients, coloring TPs/FPs with threshold line.

Minimal example (outside the notebook):

```python
# Given a fitted sklearn classifier `clf` and arrays y_val, X_val
y_score = positive_scores(clf, X_val)
auc_report(y_val, y_score, name="My Model", plot=True)

# Choose threshold under an alert budget of 100 per 1,000
res = pick_threshold_workload(y_val, y_score, alerts_per_1000_max=100.0)
print(res["summary"])   # chosen threshold and metrics

# Visualize at the chosen threshold
thr = res["threshold"]
plot_recall_floor_curves(y_val, y_score, recall_floor=0.30, chosen_threshold=thr)
plot_cumulative_recall_at_threshold(y_val, y_score, chosen_threshold=thr)
plot_topk_at_threshold(y_val, y_score, chosen_threshold=thr, top_k=30)
```


**Optional: Responsible AI Dashboard**
The environment includes `responsibleai` and `raiwidgets`. The notebook imports them; you can optionally create a dashboard for error analysis and explanations. Example pattern:

```python
from responsibleai import RAIInsights
from raiwidgets import ResponsibleAIDashboard

# X_train, y_train, X_test, y_test, and a fitted `calibrated_clf` exist from the notebook
features = X_train.columns.tolist()
rai = RAIInsights(
    model=calibrated_clf,
    train=X_train, test=X_test,
    target_column="OD",
    task_type="classification",
    categorical_features=[c for c in features if X_train[c].nunique() <= 10],
)

rai.explainer.add()
rai.error_analysis.add()
rai.compute()
ResponsibleAIDashboard(rai)
```

Notes:
- If the widget does not render, trust the notebook (File → Trust Notebook) and prefer JupyterLab ≥ 3.x.
- The dashboard can be heavy; run after the core analysis completes.


**Reproducibility and Notes**
- Random seed: the notebook sets `RANDOM_STATE = 42` for splits and modeling.
- Calibration: uses `CalibratedClassifierCV` over a logistic baseline pipeline (with imputation, scaling, and variance filtering).
- Reported examples in the notebook (will vary with random seeds/splits):
  - Baseline logistic (validation): PR AUC ≈ 0.495, ROC AUC ≈ 0.750, prevalence ≈ 0.18, lift ≈ 2.75×.
  - Final calibrated model (test): PR AUC ≈ 0.445, ROC AUC ≈ 0.764, prevalence ≈ 0.18, lift ≈ 2.47×.
- Data stewardship: this is a synthetic teaching dataset. In clinical settings, ensure governance, privacy, and bias auditing before deployment.


**Troubleshooting**
- Widget/dashboard not showing:
  - Trust the notebook. Try JupyterLab instead of classic Notebook.
  - Ensure the conda env is active where Jupyter runs (`which jupyter`).
- Import errors (e.g., `fairlearn`, `responsibleai`):
  - Recreate the environment: `conda env remove -n od_rai_lgbm && conda env create -f environment.yml`.
- Plots not appearing:
  - Ensure cells aren’t in skipped state and that Matplotlib backend is interactive (`%matplotlib inline` or default in Jupyter).


**License and Acknowledgements**
- License: see `LICENSE`.
- Based on/reference: Microsoft Responsible AI toolbox notebooks and standard scikit‑learn documentation.
