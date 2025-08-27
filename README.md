# **Counterfactual Estimation with Cross-World Assumptions**

[![R](https://img.shields.io/badge/R-available-blue)]()
[![Python](https://img.shields.io/badge/Python-available-green)]()

This repository provides **R and Python implementations** of the method for estimating **counterfactual outcomes and prediction intervals** under a cross-world assumption parameterized by ρ ∈ \[-1, 1].
The method combines Conformal Quantile Regression (CQR) with bootstrap aggregation for robust uncertainty quantification.

<img width="583" height="259" alt="image" src="https://github.com/user-attachments/assets/e40969a4-e702-4370-8eed-6206838e4bb7" />


## **What is C\_rho function and how to use it?**

`C_rho` estimates the **counterfactual mean** and **interval** for each observation given:

* Observed outcome `Y_obs`

* Treatment assignment `T ∈ {0, 1}`

* Covariates `X`

* A user-specified **cross-world parameter** `ρ` that controls dependence between potential outcomes:
  * `ρ = 0` → No cross-world coupling (independence)
  * `ρ = 1` → Strong coupling (`Y(1)` ≈ linear function of `Y(0)`)
  * Negative ρ values allow for anti-correlation scenarios (usually very unrealistic)


**Output:**

* `cf` — Predicted counterfactual mean
* `lower`, `upper` — Prediction intervals for the counterfactual outcome

---

## **Installation**

Clone the repository and install dependencies:

### **R**

```r
# Install required packages
install.packages(c("randomForest", "quantregForest", "mgcv", 'qgam'))
```

### **Python**

```bash
pip install numpy pandas scikit-learn lightgbm pygam scipy
```

---

## **R Usage**

Functions:

* `C_rho()` — Counterfactual estimation
* `data_synthetic()` — Synthetic data generator
* `show_all_results()` — Visualization & evaluation
  Defined in:
* **`Main_function.R`** (core C_rho() function)
* **`helpers.R`** (utilities)

### **Example**

```r
source("helpers.R")
source("Main_function.R")

n <- 500
rho <- 0.5
d <- 1

# 1) Generate synthetic data
data <- data_synthetic(n = n, d = d, rho = rho)

X <- data.frame(data[, 1:d])
treatment <- data$treatment
Y_obs <- data$Y_obs
Y_cf_true <- data$Y_cf

# 2) Run C_rho
result <- C_rho(
  X = X,
  treatment = treatment,
  Y_obs = Y_obs,
  rho = rho,
  CI = FALSE,
  lambda = NULL,
  bootstraps_for_stable_lambda = 1,
  bootstraps_for_mu = 10
)

# 3) Evaluate results
show_all_results(
  Y_cf_true = Y_cf_true,
  Y_cf_est = result$cf,
  lower = result$lower,
  upper = result$upper
)
```

---

## **Python Usage**

Functions:

* `C_rho()` — Counterfactual estimation
* `data_synthetic()` — Synthetic data generator
  Defined in:
* **`Main_function.py`**

### **Example**

```python
import numpy as np
import pandas as pd
from Main_function import C_rho, data_synthetic

# 1) Generate synthetic data
rho = 0
df = data_synthetic(n=500, d=1, rho=rho)

X_df = pd.DataFrame(
    df[[c for c in df.columns if c.startswith("X")]].values.astype("float32"),
    columns=[f"X{i+1}" for i in range(sum(col.startswith('X') for col in df.columns))]
)
T = df["T"].values.astype("int64")
Y = df["Y_obs"].values.astype("float32")

# 2) Run C_rho
out = C_rho(
    X=X_df,
    treatment=T,
    Y_obs=Y,
    rho=rho,
    bootstraps_for_stable_lambda=5,
    CI=False,
    bootstraps_for_mu=50,
    desired_coverage=0.9,
    CQR_qr='auto'
)

# 3) Inspect results
cf, lo, hi = out["cf"], out["lower"], out["upper"]
print("First 5 predictions:", cf[:5])

# MSE against true counterfactual
true_cf = np.where(T == 1, df["Y0"].values, df["Y1"].values)
mse = np.mean((cf - true_cf) ** 2)
print(f"MSE: {mse:.4f}")

# Empirical coverage
coverage = np.mean((true_cf >= lo) & (true_cf <= hi))
print(f"Coverage: {coverage:.3f}")
```

---

## **Key Parameters**

* `rho`: Cross-world correlation parameter (\[-1, 1])
* `CI`: Whether to compute extra bootstrap-based CI adjustment (`True` = better coverage, slower)
* `bootstraps_for_stable_lambda`: Stabilizes width ratio estimate (`λ`)
* `desired_coverage`: Target prediction interval coverage (e.g., 0.9)
* `CQR_qr`: Quantile learner (`'auto'` = qgam if d ≤ 5, else RF)

---

## **Outputs**

* `cf`: Counterfactual predictions
* `lower`, `upper`: Interval bounds

---

## **Notes**

* The Python version is a **direct translation** of the R implementation. If discrepancies arise, use R as the reference.
* Intervals improve with `CI=True` (slower).
* For large datasets, use `bootstraps_for_stable_lambda = 1` and `CI=False` for speed.

---

## **Experiments**

TODO
