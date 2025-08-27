"""
Counterfactual point & interval estimation using a cross-world parameter rho ∈ [-1, 1].

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import check_random_state
from scipy.stats import beta as beta_dist
import lightgbm as lgb
import pygam

__all__ = ["C_rho", "data_synthetic"]

# ---------------------- Helpers  ----------------------

def _is_binary(y: np.ndarray) -> bool:
    y = pd.Series(y).dropna().astype(float).values
    u = np.unique(y)
    return u.size == 2 and np.all(np.isin(u, [0.0, 1.0]))


def _enforce_min_width(lo: np.ndarray, hi: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    w = hi - lo
    too_small = ~np.isfinite(w) | (w < eps)
    if np.any(too_small):
        mid = (lo + hi) / 2.0
        lo[too_small] = mid[too_small] - eps / 2.0
        hi[too_small] = mid[too_small] + eps / 2.0
    return lo, hi


def _iqr(x: np.ndarray) -> float:
    x = pd.Series(x).dropna().values
    if x.size == 0:
        return 1.0
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

# ---------- Math utilities (no-scipy error function / inverse) ----------

def erf(x):
    # Abramowitz and Stegun formula 7.1.26 approximation
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def erfinv(x):
    # Approximate inverse error function
    a = 0.147
    ln_term = np.log(1 - x**2)
    first = (2/(np.pi*a) + ln_term/2)
    second = ln_term/a
    return np.sign(x) * np.sqrt(np.sqrt(first**2 - second) - first)

# ---------- Synthetic data helpers (optional) ----------

def qlaplace(p):
    return np.where(p < 0.5, np.log(2 * p), -np.log(2 * (1 - p)))


def gaussian_copula(n, rho):
    cov = np.array([[1, rho],[rho, 1]])
    z = np.random.multivariate_normal([0, 0], cov, size=n)
    u = 0.5 * (1 + erf(z / np.sqrt(2)))  # Normal CDF approx
    return u


def transform_marginal(u_vec, marginal):
    if marginal == "gaussian":
        return np.sqrt(2) * erfinv(2 * u_vec - 1)
    elif marginal == "t":
        from scipy.stats import t
        return t.ppf(u_vec, df=3)
    elif marginal == "laplace":
        return qlaplace(u_vec)
    elif marginal == "chisq":
        return np.sqrt(2) * erfinv(2 * u_vec - 1) ** 2
    else:
        raise ValueError("Unsupported marginal")


def smooth_random_function_1d(size=1001):
    x = np.linspace(-1, 1, size)
    freq = np.random.uniform(1, 3)
    phase = np.random.uniform(0, 2*np.pi)
    return 5 * np.sin(freq * x * np.pi + phase)


def smooth_random_function_2d(size=1001):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    freqx = np.random.uniform(1, 3)
    freqy = np.random.uniform(1, 3)
    phase = np.random.uniform(0, 2*np.pi)
    return 5 * (np.sin(freqx * X * np.pi + phase) + np.cos(freqy * Y * np.pi + phase)) / 2


def data_synthetic(n=1000, d=2, rho=0.5, sigma_1=1, sigma_2=4, constant_propensity=False, marginal="gaussian"):
    # Covariates & CATE
    if d == 1:
        X = np.random.uniform(-1, 1, n)
        CATE = 5 + np.interp(X, np.linspace(-1,1,1001), smooth_random_function_1d())
        mu = 5 + 5 * X
    else:
        Sigma = np.full((d, d), 0.25)
        np.fill_diagonal(Sigma, 1)
        tilde_X = np.random.multivariate_normal(np.zeros(d), Sigma, size=n)
        X = 0.5 * (1 + erf(tilde_X / np.sqrt(2)))  # normal CDF
        tau = smooth_random_function_2d()
        CATE = np.array([
            tau[int(x1 * 1000), int(x2 * 1000)] for x1, x2 in X[:, :2]
        ])
        beta = np.random.normal(size=d)
        mu = X @ beta

    # Errors via Gaussian copula
    u = gaussian_copula(n, rho)
    eps1 = transform_marginal(u[:,0], marginal) * np.sqrt(sigma_1)
    eps2 = transform_marginal(u[:,1], marginal) * np.sqrt(sigma_2)

    # Outcomes
    Y0 = mu + eps1
    Y1 = mu + CATE + eps2

    # Treatment assignment
    if not constant_propensity:
        if d == 1:
            propensity_score = (1 + np.abs(X)) / 4
        else:
            propensity_score = (1 + np.abs(X[:,0])) / 4
        T = np.random.binomial(1, 1 - propensity_score)
    else:
        T = np.random.choice([0,1], size=n)

    Y_obs = np.where(T == 1, Y1, Y0)

    # Output
    if d == 1:
        df = pd.DataFrame({"X": X, "Y0": Y0, "Y1": Y1, "Y_obs": Y_obs, "T": T})
    else:
        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(d)])
        df["Y0"] = Y0
        df["Y1"] = Y1
        df["Y_obs"] = Y_obs
        df["T"] = T
    return df

# ---------------------- Mean / Probability learners ----------------------

def _mean_wrapper(
    X: pd.DataFrame,
    y: np.ndarray,
    newX: pd.DataFrame,
    center_method: Literal['gam', 'rf'] = 'rf',
    ntree: int = 1000,
    nodesize: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Predict E[Y|X] with either RandomForest ('rf') or GAM ('gam', via pygam)."""
    if center_method == 'gam':
        gam = pygam.LinearGAM().fit(X.values, y)
        return gam.predict(newX.values).astype(float)
    rf = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=nodesize, random_state=random_state)
    rf.fit(X.values, y)
    return rf.predict(newX.values).astype(float)


def _prob_wrapper(
    X: pd.DataFrame,
    y: np.ndarray,
    newX: pd.DataFrame,
    method: Literal['gam', 'rf'] = 'rf',
    ntree: int = 1000,
    nodesize: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Predict P(Y=1|X) with RandomForestClassifier ('rf') or LogisticGAM ('gam')."""
    if method == 'gam':
        gam = pygam.LogisticGAM().gridsearch(X.values, y)
        return gam.predict_mu(newX.values).astype(float)
    clf = RandomForestClassifier(n_estimators=ntree, min_samples_leaf=nodesize, random_state=random_state)
    clf.fit(X.values, y.astype(int))
    proba = clf.predict_proba(newX.values)
    return proba[:, 1].astype(float)

# ---------------------- Quantile learner (two models) ----------------------

@dataclass
class _QRModel:
    model_lo: Any
    model_hi: Any

    def predict(self, newX: pd.DataFrame) -> Dict[str, np.ndarray]:
        lo = self.model_lo.predict(newX.values).astype(float)
        hi = self.model_hi.predict(newX.values).astype(float)
        return {"lo": lo, "hi": hi}


def _make_quantile_model(
    X: pd.DataFrame,
    y: np.ndarray,
    q_lo: float,
    q_hi: float,
    method: Literal['RF', 'qgam'] = 'RF',  # kept for API parity
    ntree: int = 1000,
    nodesize: int = 5,
    random_state: Optional[int] = None,
) -> _QRModel:
    """LightGBM quantile models for q_lo and q_hi (no fallback)."""
    params = dict(
        objective='quantile',
        min_child_samples=nodesize,   # use sklearn param name; avoids min_data_in_leaf warning
        n_estimators=ntree,
        random_state=random_state,
        verbose=-1,                  # silence training logs
        verbosity=-1,                # extra safety: suppress LightGBM info/warnings
        force_col_wise=True          # remove "Auto-choosing col-wise" info
    )
    m_lo = lgb.LGBMRegressor(**params, alpha=q_lo)
    m_hi = lgb.LGBMRegressor(**params, alpha=q_hi)
    m_lo.fit(X.values, y, callbacks=[lgb.log_evaluation(-1)])
    m_hi.fit(X.values, y, callbacks=[lgb.log_evaluation(-1)])
    return _QRModel(m_lo, m_hi)

# ---------------------- Conformal Quantile Regression ----------------------

def _cqr(
    X: pd.DataFrame,
    y: np.ndarray,
    newX: pd.DataFrame,
    desired_coverage: float = 0.9,
    train_calib_split: float = 0.8,
    CQR_qr: Literal['auto', 'RF', 'qgam'] = 'auto',
    ntree: int = 1000,
    nodesize: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    if CQR_qr == 'auto':
        CQR_qr = 'RF' if X.shape[1] > 5 else 'qgam'
    if CQR_qr not in ('RF', 'qgam'):
        raise ValueError("Unsupported CQR_qr method")

    n = X.shape[0]
    n_tr = int(np.floor(n * train_calib_split))
    tr_idx = np.arange(n_tr)
    cal_idx = np.arange(n_tr, n)

    X_tr, y_tr = X.iloc[tr_idx, :], y[tr_idx]
    X_cal, y_cal = X.iloc[cal_idx, :], y[cal_idx]

    alpha = 1.0 - desired_coverage
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

    qr_model = _make_quantile_model(X_tr, y_tr, q_lo, q_hi, method=CQR_qr, ntree=ntree, nodesize=nodesize, random_state=random_state)
    b_np = qr_model.predict(newX)
    b_cal = qr_model.predict(X_cal)

    center_method = 'gam' if CQR_qr == 'qgam' else 'rf'
    mu_np = _mean_wrapper(X_tr, y_tr, newX, center_method=center_method, ntree=ntree, nodesize=nodesize, random_state=random_state)

    scores = np.maximum(b_cal["lo"] - y_cal, y_cal - b_cal["hi"])  # nonconformity
    m = scores.size
    q_level = int(np.ceil((1 - alpha) * (m + 1))) / m if m > 0 else 0.0
    gamma = np.quantile(scores, q_level, method='nearest') if m > 0 else 0.0
    gamma = float(max(gamma, 1e-12 * max(1.0, _iqr(y_tr))))

    lower = b_np["lo"] - gamma
    upper = b_np["hi"] + gamma

    return {"hat_f": mu_np.astype(float), "lower": lower.astype(float), "upper": upper.astype(float)}

# ---------------------- Bernoulli CI helper ----------------------

def _bern_ci(p: np.ndarray, alpha: float = 0.10, m_eff: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    a = p * m_eff + 0.5
    b = (1 - p) * m_eff + 0.5
    lo = beta_dist.ppf(alpha / 2.0, a, b)
    hi = beta_dist.ppf(1 - alpha / 2.0, a, b)
    return lo, hi

# ============================= Main ==============================

def C_rho(
    X: pd.DataFrame,
    treatment: np.ndarray,
    Y_obs: np.ndarray,
    rho: float,
    bootstraps_for_stable_lambda: int = 5,
    bootstraps_for_mu: int = 50,
    lambda_: Optional[float] = None,
    CI: bool = False,
    desired_coverage: float = 0.9,
    train_calib_split: float = 0.8,
    CQR_qr: Literal['auto', 'RF', 'qgam'] = 'auto',
    ci_level: float = 0.95,
    ntree: int = 2000,
    nodesize: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict with keys: 'cf', 'lower', 'upper' (length-n arrays).
    For binary Y, 'cf' is the counterfactual mean probability in [0,1];
    'lower'/'upper' define a (1 - alpha) prediction set over {0,1}.
    For continuous Y, 'cf' is centered at the mean (not the median).
    """
    rng = check_random_state(random_state)

    X = pd.DataFrame(X).reset_index(drop=True)
    treatment = np.asarray(treatment).astype(int)
    Y_obs = np.asarray(Y_obs).astype(float)
    n = X.shape[0]

    lambda_min, lambda_max = 0.1, 10.0
    eps_width = 1e-6 * max(1.0, _iqr(Y_obs))

    if rho == 0:
        CI = False
    if lambda_ is not None:
        bootstraps_for_stable_lambda = 1
    if CQR_qr == 'auto':
        CQR_qr = 'RF' if X.shape[1] > 5 else 'qgam'
    if CQR_qr not in ('RF', 'qgam'):
        raise ValueError("Unsupported CQR_qr method")

    X_treated = X.loc[treatment == 1]
    Y_treated = Y_obs[treatment == 1]
    X_control = X.loc[treatment == 0]
    Y_control = Y_obs[treatment == 0]

    # ------------------ Binary branch ------------------
    if _is_binary(Y_obs):
        prob_method = 'gam' if CQR_qr == 'qgam' else 'rf'
        alpha = 1.0 - desired_coverage

        mu0 = _prob_wrapper(X_control, Y_control, X, method=prob_method, ntree=ntree, nodesize=nodesize, random_state=random_state)
        mu1 = _prob_wrapper(X_treated, Y_treated, X, method=prob_method, ntree=ntree, nodesize=nodesize, random_state=random_state)

        eps = 1e-8
        sd0 = np.sqrt(np.maximum(mu0 * (1 - mu0), eps))
        sd1 = np.sqrt(np.maximum(mu1 * (1 - mu1), eps))
        if lambda_ is None:
            lam = sd1 / sd0
        else:
            lam = np.full(n, float(lambda_), dtype=float)
        lam = np.clip(lam, lambda_min, lambda_max)
        inv_lam = 1.0 / np.maximum(lam, lambda_min)

        p_cf = np.empty(n, dtype=float)
        for j in range(n):
            if treatment[j] == 0:
                p_cf[j] = mu1[j] + lam[j] * rho * (Y_obs[j] - mu0[j])
            else:
                p_cf[j] = mu0[j] + inv_lam[j] * rho * (Y_obs[j] - mu1[j])
        p_cf = np.clip(p_cf, 0.0, 1.0)

        lower = np.zeros(n, dtype=float)
        upper = np.zeros(n, dtype=float)
        for j in range(n):
            if p_cf[j] <= alpha:
                lower[j] = upper[j] = 0.0
            elif p_cf[j] >= 1.0 - alpha:
                lower[j] = upper[j] = 1.0
            else:
                lower[j], upper[j] = 0.0, 1.0

        if not CI:
            return {"cf": p_cf, "lower": lower, "upper": upper}

        mean_CI_mat = np.full((n, bootstraps_for_mu), np.nan, dtype=float)
        for b in range(bootstraps_for_mu):
            i0 = rng.randint(0, X_control.shape[0], size=X_control.shape[0])
            i1 = rng.randint(0, X_treated.shape[0], size=X_treated.shape[0])
            mu0_b = _prob_wrapper(X_control.iloc[i0], Y_control[i0], X, method=prob_method, ntree=ntree, nodesize=nodesize, random_state=rng)
            mu1_b = _prob_wrapper(X_treated.iloc[i1], Y_treated[i1], X, method=prob_method, ntree=ntree, nodesize=nodesize, random_state=rng)
            for j in range(n):
                if treatment[j] == 0:
                    mean_CI_mat[j, b] = mu1_b[j] + lam[j] * rho * (Y_obs[j] - mu0_b[j])
                else:
                    mean_CI_mat[j, b] = mu0_b[j] + inv_lam[j] * rho * (Y_obs[j] - mu1_b[j])
        p_cf_boot = np.clip(np.nanmedian(mean_CI_mat, axis=1), 0.0, 1.0)
        return {"cf": p_cf_boot, "lower": lower, "upper": upper}

    # ------------------ Continuous branch ------------------
    B = int(bootstraps_for_stable_lambda)
    lam_mat = np.full((n, B), np.nan, dtype=float)
    mu0_mat = np.full((n, B), np.nan, dtype=float)
    mu1_mat = np.full((n, B), np.nan, dtype=float)
    lower0_mat = np.full((n, B), np.nan, dtype=float)
    upper0_mat = np.full((n, B), np.nan, dtype=float)
    lower1_mat = np.full((n, B), np.nan, dtype=float)
    upper1_mat = np.full((n, B), np.nan, dtype=float)

    for b in range(B):
        if B > 1:
            i0 = rng.randint(0, X_control.shape[0], size=X_control.shape[0])
            i1 = rng.randint(0, X_treated.shape[0], size=X_treated.shape[0])
            X0_b, y0_b = X_control.iloc[i0], Y_control[i0]
            X1_b, y1_b = X_treated.iloc[i1], Y_treated[i1]
        else:
            X0_b, y0_b = X_control, Y_control
            X1_b, y1_b = X_treated, Y_treated

        est0 = _cqr(X0_b, y0_b, X, desired_coverage, train_calib_split, CQR_qr, ntree, nodesize, random_state=rng)
        est1 = _cqr(X1_b, y1_b, X, desired_coverage, train_calib_split, CQR_qr, ntree, nodesize, random_state=rng)

        lo0, hi0 = _enforce_min_width(est0["lower"], est0["upper"], eps_width)
        lo1, hi1 = _enforce_min_width(est1["lower"], est1["upper"], eps_width)

        mu0_mat[:, b] = est0["hat_f"]
        mu1_mat[:, b] = est1["hat_f"]
        lower0_mat[:, b] = lo0
        upper0_mat[:, b] = hi0
        lower1_mat[:, b] = lo1
        upper1_mat[:, b] = hi1

        w0 = np.maximum(hi0 - lo0, eps_width)
        w1 = np.maximum(hi1 - lo1, eps_width)
        lam_mat[:, b] = np.exp(np.log(w1) - np.log(w0))

    mu0 = np.nanmedian(mu0_mat, axis=1)
    mu1 = np.nanmedian(mu1_mat, axis=1)
    lower0 = np.nanmedian(lower0_mat, axis=1)
    upper0 = np.nanmedian(upper0_mat, axis=1)
    lower1 = np.nanmedian(lower1_mat, axis=1)
    upper1 = np.nanmedian(upper1_mat, axis=1)

    lower0, upper0 = _enforce_min_width(lower0, upper0, eps_width)
    lower1, upper1 = _enforce_min_width(lower1, upper1, eps_width)

    if lambda_ is None:
        lam = np.nanmedian(lam_mat, axis=1)
    else:
        lam = np.full(n, float(lambda_), dtype=float)
    lam = np.clip(lam, lambda_min, lambda_max)
    inv_lam = 1.0 / np.maximum(lam, lambda_min)

    if not CI:
        mean = np.empty(n, dtype=float)
        lower = np.empty(n, dtype=float)
        upper = np.empty(n, dtype=float)
        s = np.sqrt(max(0.0, 1.0 - rho ** 2))
        for j in range(n):
            if treatment[j] == 0:
                mean[j] = mu1[j] + lam[j] * rho * (Y_obs[j] - mu0[j])
                lower[j] = mean[j] - s * (mu1[j] - lower1[j])
                upper[j] = mean[j] + s * (upper1[j] - mu1[j])
            else:
                mean[j] = mu0[j] + inv_lam[j] * rho * (Y_obs[j] - mu1[j])
                lower[j] = mean[j] - s * (mu0[j] - lower0[j])
                upper[j] = mean[j] + s * (upper0[j] - mu0[j])
        return {"cf": mean, "lower": lower, "upper": upper}

    total_B = B + int(bootstraps_for_mu)
    mean_CI_mat = np.full((n, total_B), np.nan, dtype=float)

    for b in range(B):
        for j in range(n):
            if treatment[j] == 0:
                mean_CI_mat[j, b] = mu1_mat[j, b] + lam_mat[j, b] * rho * (Y_obs[j] - mu0_mat[j, b])
            else:
                lam_b = np.clip(lam_mat[j, b], lambda_min, lambda_max)
                mean_CI_mat[j, b] = mu0_mat[j, b] + (1.0 / lam_b) * rho * (Y_obs[j] - mu1_mat[j, b])

    for b in range(B, total_B):
        i0 = rng.randint(0, X_control.shape[0], size=X_control.shape[0])
        i1 = rng.randint(0, X_treated.shape[0], size=X_treated.shape[0])
        center_method = 'gam' if CQR_qr == 'qgam' else 'rf'
        mu0_boot = _mean_wrapper(X_control.iloc[i0], Y_control[i0], X, center_method=center_method, ntree=ntree, nodesize=nodesize, random_state=rng)
        mu1_boot = _mean_wrapper(X_treated.iloc[i1], Y_treated[i1], X, center_method=center_method, ntree=ntree, nodesize=nodesize, random_state=rng)
        for j in range(n):
            if treatment[j] == 0:
                mean_CI_mat[j, b] = mu1_boot[j] + lam[j] * rho * (Y_obs[j] - mu0_boot[j])
            else:
                mean_CI_mat[j, b] = mu0_boot[j] + inv_lam[j] * rho * (Y_obs[j] - mu1_boot[j])

    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    lower_CI = np.nanquantile(mean_CI_mat, lo_q, axis=1)
    upper_CI = np.nanquantile(mean_CI_mat, hi_q, axis=1)
    mean = np.nanmedian(mean_CI_mat, axis=1)

    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    cscale = rho ** 2
    s = np.sqrt(max(0.0, 1.0 - rho ** 2))
    for j in range(n):
        if treatment[j] == 0:
            lower[j] = mean[j] - cscale * (mean[j] - lower_CI[j]) - s * (mu1[j] - lower1[j])
            upper[j] = mean[j] + cscale * (upper_CI[j] - mean[j]) + s * (upper1[j] - mu1[j])
        else:
            lower[j] = mean[j] - cscale * (mean[j] - lower_CI[j]) - s * (mu0[j] - lower0[j])
            upper[j] = mean[j] + cscale * (upper_CI[j] - mean[j]) + s * (upper0[j] - mu0[j])

    return {"cf": mean.astype(float), "lower": lower.astype(float), "upper": upper.astype(float)}
