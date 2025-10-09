# ML4HL_LC_ResponsibleAI_Workflow

The project provides an **end-to-end pipeline for machine learning** used for investigating lung cancer survival prediction that is based on Responsible AI (RAI) practices.
It includes **Exploratory Data Analysis (EDA)** and **model/calibration** phases to depict how structured clinical data can be utilized to offer responsible early intervention and treatment information.

---

## Introduction and Purpose

### Clinical Motivation

Lung cancer remains one of the leading reasons for cancer related deaths. Early diagnosis and personalized treatment are key to improving survival outcomes.

This project mimics an actual healthcare workflow using **artificial patient-level data**, simulating survival likelihood to guide **clinical decision-making**, **error analysis**, and **interpretability**.


### Project Goals
1. **Understand patient-level survival drivers** in exact exploratory data analysis (EDA)  
2. **Construct and tune a predictive model** to predict survival probability with demographic, clinical, and behavioral predictors. 
3. **Investigate trade-offs and Responsible AI methods** for fairness, interpretability, and reliability in deployment of models.  
4. **Derive practical insights** for healthcare practitioners with realistic data assumptions.

### Responsible AI Integration
The modeling notebook integrates Microsoft’s **Responsible AI Toolbox**, featuring:
- **Interpretability and feature attribution**
- **Error and subgroup analysis**
- **Fairness diagnostics**
- **Transparency through calibrated probability modeling**

---

## Project Structure

| File | Purpose |
|------|----------|
| **ML4HL_LC_EDA.ipynb** | Exploratory Data Analysis (EDA) — data cleaning, feature summaries, and hypothesis validation. |
| **ML4HL_LC_RAI_toolbox.ipynb** | Modeling, calibration, evaluation, and Responsible AI interpretability. |
| **Data/lung_cancer_dataset_med.csv** | Synthetic dataset with ~890k patient records for lung cancer survival. |
| **utils.py** | Reusable utilities for model evaluation (AUC, lift), threshold selection (recall floor, workload, cost), and plotting. |
| **environment.yml** | Conda environment file with all dependencies. |
| **LICENSE** | License for the repository. |
| **README.md** | This documentation file. |

---

## Install and Environment

### Prerequisites
- Conda or Mamba
- Git
- Jupyter Notebook or JupyterLab (≥3.x)

### Setup Instructions
```bash
# Clone and create environment
git clone <repository_url>
cd ML4HL_LC_ResponsibleAI_Workflow
conda env create -f environment.yml
conda activate lc_rai_env
```

Verify setup:
```bash
python -c "import IPython, ipywidgets; print('Jupyter OK')"
```

If widgets do not display in JupyterLab:
```bash
# Ensure lab is trusted and widgets enabled
jupyter trust ML4HL_LC_RAI_toolbox.ipynb
```

---

## Quickstart Workflow

### Step 1: Run EDA
```bash
jupyter lab
# Open ML4HL_LC_EDA.ipynb
```
The EDA notebook:
- Loads and cleans the raw dataset.
- Standardizes schema, encodes categorical and binary variables.
- Analyzes missingness, distributions, and correlations.
- Validates clinical hypotheses:
  - Asthma may improve early detection.
  - Cirrhosis may lower survival during chemo/radiation.
  - Smoking history reduces survival likelihood.

### Step 2: Run Responsible AI Modeling
```bash
# From JupyterLab
Open ML4HL_LC_RAI_toolbox.ipynb
```
The modeling notebook:
1. Imports the processed dataset.
2. Constructs a preprocessing + logistic regression pipeline.
3. Tunes predicted probabilities.
4. Evaluates performance with **ROC AUC**, **PR AUC**, and **lift**.
5. Applies thresholding policies (recall floor, workload, cost).
6. Launches an optional Responsible AI Dashboard for explainability.

---

## Data Description

**Dataset:** `lung_cancer_dataset_med.csv`  
**Rows:** ~890,000 (synthetic, patient-level)  
**Target variable:**  
- `survived` — binary (1 = survived, 0 = died)

### Feature Overview

| Category | Example Features |
|-----------|------------------|
| **Demographics** | `age`, `gender`, `country`, `region` |
| **Clinical** | `cancer_stage`, `family_history`, `hypertension`, `asthma`, `cirrhosis` |
| **Behavioral** | `smoking_status` (current, former, passive, never) |
| **Treatment** | `treatment_type` (surgery, chemotherapy, radiation, combined) |
| **Derived/Engineered** | `treatment_duration_days`, `treat_chemo_comorbid`, `treat_radiation_smoked`, etc. |

**Class balance:** ~22% survived, ~78% died  
**Note:** This dataset is **synthetic** and designed for educational and Responsible AI demonstration purposes only.

---

## Evaluation and Thresholding

### Metrics
- **ROC AUC:** Measures model discrimination  
- **PR AUC:** Precision–recall balance (more robust for class imbalance)  
- **Lift:** Ratio of model precision to baseline prevalence  
- **Calibration:** Probability alignment (CalibratedClassifierCV)

### Example Results (may vary)
| Model | ROC AUC | PR AUC | Lift |
|--------|----------|--------|------|
| Logistic (validation) | 0.75 | 0.49 | 2.2× |
| Calibrated (test) | 0.76 | 0.45 | 2.1× |

### Threshold Policies
Implemented in `utils.py`:
- **Workload-based:** Limit alerts per 1,000 patients.  
- **Recall floor:** Optimize precision with a minimum recall.  
- **Cost-based:** Minimize expected cost using custom false-positive/false-negative costs.

### Visualizations
- Precision–recall vs. threshold  
- Cumulative recall vs. alert rate  
- Top-K risk cases (TP/FP visualization)

---

## Responsible AI and Reproducibility

### Interpretability and Error Analysis
Powered by the **Microsoft Responsible AI Toolbox**:
```python
from responsibleai import RAIInsights
from raiwidgets import ResponsibleAIDashboard
```

- **Feature importance:** Identify dominant predictors of survival.
- **Error clustering:** Analyze misclassifications and outliers.
- **Fairness metrics:** Explore subgroup differences (gender, region).
- **What-if analysis:** Understand counterfactual outcomes.

### Reproducibility
- Random seed: `RANDOM_STATE = 42`  
- Deterministic data splits: 80/15/5 train/validation/test  
- Reusable environment: `environment.yml`

### Governance and Ethics
- All data is **synthetic** and privacy-preserving.  
- Emphasis on **interpretability before deployment**.  
- Encourages transparent communication of model limitations and fairness assessments.

---

## License and Acknowledgements

- **License:** See `LICENSE` file in this repository.  
- **Acknowledgements:**  
  Adapted from Microsoft Responsible AI Toolkit and scikit-learn documentation.  
  Developed as part of a **Machine Learning for Healthcare (ML4HL)** educational exercise on Responsible AI.
