# ----------------------------- Libraries -----------------------------
library(qgam)               # quantile GAMs when CQR_qr = 'qgam'
library(quantregForest)     # quantile RF when CQR_qr = 'RF'
library(mgcv)               # mean/prob GAM
library(randomForest)       # mean/prob RF


# ============================= Main ==============================

# --------------------------------------------------------------------
# C_rho
#   X:           n x d data.frame of covariates
#   treatment:   length-n binary vector (1 = treated, 0 = control)
#   Y_obs:       length-n observed outcomes (continuous or binary {0,1})
#   rho:         cross-world parameter: scalar in [-1, 1]
#   bootstraps_for_stable_lambda: bootstrap resamples for lambda for stability
#   bootstraps_for_mu: bootstrap resamples for CI when CI = TRUE
#   lambda:      if not NULL, use this fixed lambda instead of estimating it
#   CI:          whether to add CIs for μ based on bootstrap aggregation
#   desired_coverage: target coverage for CQR intervals (or prob intervals)
#   train_calib_split: fraction of data for CQR training (rest for calibration)
#   CQR_qr:      'auto' | 'qgam' | 'RF'
#   ci_level:    level for uncertainty on the mean via bootstrap aggregation
#   ntree:       number of trees for RF / quantile RF
#   nodesize:    minimum terminal node size for RF / quantile RF
# Returns:
#   list(cf, lower, upper) of length-n vectors
#   cf: counterfactual point estimates (centered at MEAN, not median)
#   lower/upper: counterfactual interval bounds
# --------------------------------------------------------------------
C_rho <- function(X, treatment, Y_obs, 
                  rho, 
                  bootstraps_for_stable_lambda = 1,
                  bootstraps_for_mu = 20, 
                  lambda = NULL, 
                  CI = FALSE, 
                  desired_coverage = 0.9,
                  train_calib_split = 0.8,
                  CQR_qr = 'auto',
                  ci_level = 0.95,
                  ntree = 2000,
                  nodesize = 5) {
  
  
  # ============================= Helpers ==============================
  
  # Common GAM formula helper
  create_formula <- function(X, smoothing = TRUE) {
    terms <- sapply(1:ncol(X), function(j) {
      if (length(unique(X[, j])) > 9 && smoothing) paste0("s(X", j, ")")
      else paste0("as.factor(X", j, ")")
    })
    reformulate(terms, response = "Y")
  }
  
  # Mean (center) regression wrapper (continuous Y)
  Mean_wrapper <- function(X, Y, new_points,
                           center_method = c('gam','rf'),
                           ntree = 1000,
                           nodesize = 5) {
    center_method <- match.arg(center_method)
    names(X) <- paste0("X", 1:ncol(X))
    names(new_points) <- paste0("X", 1:ncol(new_points))
    if (center_method == 'gam') {
      formula <- create_formula(X)
      invisible(capture.output({
        fit <- mgcv::gam(formula, data = data.frame(X, Y = Y), family = gaussian())
      }))
      pred <- predict(fit, newdata = new_points, type = "response")
    } else {
      rf <- randomForest::randomForest(x = X, y = Y, ntree = ntree, nodesize = nodesize)
      pred <- predict(rf, newdata = new_points)
    }
    as.numeric(pred)
  }
  
  # One-fit quantile model (lo & hi together) for continuous Y
  QR_wrapper <- function(X, Y, CQR_qr, q_lo, q_hi, ntree = 1000, nodesize = 5) {
    names(X) <- paste0("X", 1:ncol(X))
    if (CQR_qr == "RF") {
      qrf <- quantregForest(x = X, y = Y, ntree = ntree, nodesize = nodesize)
      pred_fun <- function(newdata) {
        names(newdata) <- paste0("X", 1:ncol(newdata))
        mat <- predict(qrf, newdata = newdata, what = c(q_lo, q_hi))
        list(lo = as.numeric(mat[, 1]), hi = as.numeric(mat[, 2]))
      }
      return(list(predict = pred_fun))
    } else if (CQR_qr == "qgam") {
      formula <- create_formula(X)
      invisible(capture.output({
        fit_lo <- qgam(formula, data = data.frame(X, Y = Y), qu = q_lo)
        fit_hi <- qgam(formula, data = data.frame(X, Y = Y), qu = q_hi)
      }))
      pred_fun <- function(newdata) {
        names(newdata) <- paste0("X", 1:ncol(newdata))
        lo <- predict(fit_lo, newdata = newdata)
        hi <- predict(fit_hi, newdata = newdata)
        list(lo = as.numeric(lo), hi = as.numeric(hi))
      }
      return(list(predict = pred_fun))
    } else stop("Unsupported CQR_qr in QR_wrapper")
  }
  
  # Conformal Quantile Regression (bounds + mean center) for continuous Y
  CQR <- function(X, Y, new_points,
                  desired_coverage = 0.9,
                  train_calib_split = 0.8,
                  CQR_qr = 'auto',
                  ntree = 1000,
                  nodesize = 5) {
    if (CQR_qr == 'auto') CQR_qr <- if (ncol(data.frame(X)) > 5) 'RF' else 'qgam'
    if (!CQR_qr %in% c('RF', 'qgam')) stop("Unsupported CQR_qr method")
    
    d <- ncol(data.frame(X))
    X <- data.frame(X); Y <- as.numeric(Y)
    new_points <- data.frame(new_points)
    names(X) <- names(new_points) <- paste0("X", 1:d)
    
    # Split train/calibration
    n <- nrow(X)
    n_tr <- floor(n * train_calib_split)
    tr_idx <- seq_len(n_tr)
    cal_idx <- (n_tr + 1):n
    X_tr <- X[tr_idx, , drop = FALSE]; Y_tr <- Y[tr_idx]
    X_cal <- X[cal_idx, , drop = FALSE]; Y_cal <- Y[cal_idx]
    
    alpha <- 1 - desired_coverage
    q_lo <- alpha / 2
    q_hi <- 1 - alpha / 2
    
    # Bounds via the quantile learner
    qr_model <- QR_wrapper(X_tr, Y_tr, CQR_qr, q_lo, q_hi, ntree, nodesize)
    b_np  <- qr_model$predict(new_points)
    b_cal <- qr_model$predict(X_cal)
    qlo_np <- b_np$lo; qhi_np <- b_np$hi
    qlo_cal <- b_cal$lo; qhi_cal <- b_cal$hi
    
    # Mean center via RF/GAM (not median)
    center_method <- if (CQR_qr == 'qgam') 'gam' else 'rf'
    mu_np <- Mean_wrapper(X_tr, Y_tr, new_points, center_method, ntree, nodesize)
    
    # Conformal calibration with a tiny floor
    scores <- pmax(qlo_cal - Y_cal, Y_cal - qhi_cal)
    m <- length(scores)
    q_level <- ceiling((1 - alpha) * (m + 1)) / m
    gamma <- as.numeric(quantile(scores, probs = q_level, type = 1))
    gamma <- max(gamma, 1e-12 * max(1, IQR(Y_tr, na.rm = TRUE)))
    
    lower <- qlo_np - gamma
    upper <- qhi_np + gamma
    
    list(hat_f = mu_np, lower = lower, upper = upper, new_points = new_points)
  }
  
  # Enforce a minimum interval width (to avoid zero-widths)
  enforce_min_width <- function(lo, hi, eps) {
    w <- hi - lo
    too_small <- !is.finite(w) | w < eps
    if (any(too_small)) {
      mid <- (lo + hi)/2
      lo[too_small] <- mid[too_small] - eps/2
      hi[too_small] <- mid[too_small] + eps/2
    }
    list(lo = lo, hi = hi)
  }
  
  # ------------------ Binary utilities ------------------
  is_binary <- function(y) {
    yu <- sort(unique(na.omit(as.numeric(y))))
    length(yu) == 2 && all(yu %in% c(0,1))
  }
  
  # Probabilistic learner μ(x) = P(Y=1 | X, arm)
  Prob_wrapper <- function(X, Y, new_points,
                           method = c('gam','rf'),
                           ntree = 1000, nodesize = 5) {
    method <- match.arg(method)
    names(X) <- paste0("X", 1:ncol(X))
    names(new_points) <- paste0("X", 1:ncol(new_points))
    if (method == 'gam') {
      formula <- reformulate(
        sapply(1:ncol(X), function(j) {
          if (length(unique(X[, j])) > 9) paste0("s(X", j, ")")
          else paste0("as.factor(X", j, ")")
        }),
        response = "Y"
      )
      invisible(capture.output({
        fit <- mgcv::gam(formula, data = data.frame(X, Y = Y), family = binomial())
      }))
      as.numeric(plogis(predict(fit, newdata = new_points, type = "link")))
    } else {
      rf <- randomForest::randomForest(x = X, y = as.factor(Y),
                                       ntree = ntree, nodesize = nodesize)
      as.numeric(predict(rf, newdata = new_points, type = "prob")[, "1"])
    }
  }
  
  # Jeffreys/Wilson-like per-x CI from an "effective sample size" m_eff
  bern_ci <- function(p, alpha = 0.10, m_eff = 100) {
    a <- p * m_eff + 0.5
    b <- (1 - p) * m_eff + 0.5
    lo <- qbeta(alpha/2, a, b)
    hi <- qbeta(1 - alpha/2, a, b)
    list(lo = lo, hi = hi)
  }
  
  #########################   MAIN   ######################################
  # ------------------ Input checks and initial setup ---------------------
  
  lambda_min <- 0.1
  lambda_max <- 10
  eps_width  <- 1e-6 * max(1, IQR(Y_obs, na.rm = TRUE))
  
  # Fast exit if rho==0 -> no CI aggregation needed
  if (rho == 0) CI <- FALSE
  if (!is.null(lambda)) bootstraps_for_stable_lambda <- 1
  if (CQR_qr == 'auto') CQR_qr <- if (ncol(data.frame(X)) > 5) 'RF' else 'qgam'
  if (!CQR_qr %in% c('RF', 'qgam')) stop("Unsupported CQR_qr method")
  
  n <- nrow(X)
  X <- data.frame(X)
  X_treated  <- data.frame(X[treatment == 1, , drop = FALSE]); Y_treated  <- Y_obs[treatment == 1]
  X_control  <- data.frame(X[treatment == 0, , drop = FALSE]); Y_control  <- Y_obs[treatment == 0]
  
  # ------------------ Binary branch (prediction-set intervals) ------------------
  if (is_binary(Y_obs)) {
    prob_method <- if (CQR_qr == 'qgam') 'gam' else 'rf'
    alpha <- 1 - desired_coverage
    
    # μ0(x), μ1(x): success probabilities per arm
    mu0 <- Prob_wrapper(X_control, Y_control, X, method = prob_method,
                        ntree = ntree, nodesize = nodesize)
    mu1 <- Prob_wrapper(X_treated, Y_treated, X, method = prob_method,
                        ntree = ntree, nodesize = nodesize)
    
    # λ via Bernoulli variance proxy + clipping
    eps <- 1e-8
    sd0 <- sqrt(pmax(mu0 * (1 - mu0), eps))
    sd1 <- sqrt(pmax(mu1 * (1 - mu1), eps))
    if (is.null(lambda)) {
      lambda <- sd1 / sd0
    } else {
      lambda <- rep(lambda, n)
    }
    lambda <- pmin(pmax(lambda, lambda_min), lambda_max)
    inv_lambda <- 1 / pmax(lambda, lambda_min)
    
    # Counterfactual mean on probability scale (this is p_cf in [0,1])
    p_cf <- numeric(n)
    for (j in 1:n) {
      if (treatment[j] == 0) {
        p_cf[j] <- mu1[j] + lambda[j] * rho * (Y_obs[j] - mu0[j])
      } else {
        p_cf[j] <- mu0[j] + inv_lambda[j] * rho * (Y_obs[j] - mu1[j])
      }
    }
    p_cf <- pmin(pmax(p_cf, 0), 1)
    
    # Map probability to a (1 - alpha) prediction set over {0,1}
    lower <- upper <- numeric(n)
    for (j in 1:n) {
      if (p_cf[j] <= alpha) {
        lower[j] <- 0; upper[j] <- 0   # confidently predict 0
      } else if (p_cf[j] >= 1 - alpha) {
        lower[j] <- 1; upper[j] <- 1   # confidently predict 1
      } else {
        lower[j] <- 0; upper[j] <- 1   # ambiguous: return {0,1}
      }
    }
    
    if (!CI) {
      return(list(cf = p_cf, lower = lower, upper = upper))
    }
    
    # --- Optional CI on the mean probability (does not change prediction-set logic) ---
    mean_CI_mat <- matrix(NA_real_, nrow = n, ncol = bootstraps_for_mu)
    for (b in 1:bootstraps_for_mu) {
      i0 <- sample.int(nrow(X_control), replace = TRUE)
      i1 <- sample.int(nrow(X_treated), replace = TRUE)
      mu0_b <- Prob_wrapper(X_control[i0, , drop = FALSE], Y_control[i0], X,
                            method = prob_method, ntree = ntree, nodesize = nodesize)
      mu1_b <- Prob_wrapper(X_treated[i1, , drop = FALSE], Y_treated[i1], X,
                            method = prob_method, ntree = ntree, nodesize = nodesize)
      for (j in 1:n) {
        if (treatment[j] == 0) {
          mean_CI_mat[j, b] <- mu1_b[j] + lambda[j] * rho * (Y_obs[j] - mu0_b[j])
        } else {
          mean_CI_mat[j, b] <- mu0_b[j] + inv_lambda[j] * rho * (Y_obs[j] - mu1_b[j])
        }
      }
    }
    p_cf_boot <- pmin(pmax(apply(mean_CI_mat, 1, median, na.rm = TRUE), 0), 1)
    # Prediction-set bounds remain based on alpha; we return the CI-improved cf if desired
    return(list(cf = p_cf_boot, lower = lower, upper = upper))
  }
  
  
  # ------------------ Continuous branch (your original logic) ------------------
  
  # Storage for bootstrap results
  lam_mat <- mu0_mat <- mu1_mat <- lower0_mat <- upper0_mat <- lower1_mat <- upper1_mat <- 
    matrix(NA_real_, nrow = n, ncol = bootstraps_for_stable_lambda)
  
  # Bootstrap loop for lambda estimation
  for (b in 1:bootstraps_for_stable_lambda) {
    if (bootstraps_for_stable_lambda > 1) {
      i0 <- sample.int(nrow(X_control), replace = TRUE)
      i1 <- sample.int(nrow(X_treated), replace = TRUE)
      X_boot_control <- X_control[i0, , drop = FALSE]; Y_boot_control <- Y_control[i0]
      X_boot_treated <- X_treated[i1, , drop = FALSE]; Y_boot_treated <- Y_treated[i1]
    } else {
      X_boot_control <- X_control; Y_boot_control <- Y_control
      X_boot_treated <- X_treated; Y_boot_treated <- Y_treated
    }
    
    est0 <- CQR(X_boot_control, Y_boot_control, X, desired_coverage, train_calib_split, CQR_qr, ntree, nodesize)
    est1 <- CQR(X_boot_treated, Y_boot_treated, X, desired_coverage, train_calib_split, CQR_qr, ntree, nodesize)
    
    # Enforce min width to avoid zero denominators
    adj0 <- enforce_min_width(est0$lower, est0$upper, eps_width)
    adj1 <- enforce_min_width(est1$lower, est1$upper, eps_width)
    
    mu0_mat[, b] <- est0$hat_f
    mu1_mat[, b] <- est1$hat_f
    lower0_mat[, b]  <- adj0$lo
    upper0_mat[, b]  <- adj0$hi
    lower1_mat[, b]  <- adj1$lo
    upper1_mat[, b]  <- adj1$hi
    
    # λ via log-width ratio (numerically safe)
    w0 <- pmax(adj0$hi - adj0$lo, eps_width)
    w1 <- pmax(adj1$hi - adj1$lo, eps_width)
    lam_mat[, b] <- exp(log(w1) - log(w0))  # = w1 / w0 safely
  }
  
  # Aggregate medians
  mu0 <- apply(mu0_mat, 1, median, na.rm = TRUE)
  mu1 <- apply(mu1_mat, 1, median, na.rm = TRUE)
  lower0 <- apply(lower0_mat, 1, median, na.rm = TRUE)
  upper0 <- apply(upper0_mat, 1, median, na.rm = TRUE)
  lower1 <- apply(lower1_mat, 1, median, na.rm = TRUE)
  upper1 <- apply(upper1_mat, 1, median, na.rm = TRUE)
  
  # Enforce min width after aggregation (extra safety)
  adj0 <- enforce_min_width(lower0, upper0, eps_width)
  adj1 <- enforce_min_width(lower1, upper1, eps_width)
  lower0 <- adj0$lo; upper0 <- adj0$hi
  lower1 <- adj1$lo; upper1 <- adj1$hi
  
  # Lambda: median across bootstraps, then clip
  if (is.null(lambda)) lambda <- apply(lam_mat, 1, median, na.rm = TRUE) else lambda <- rep(lambda, n)
  lambda <- pmin(pmax(lambda, lambda_min), lambda_max)
  inv_lambda <- 1 / pmax(lambda, lambda_min)
  
  # Without CI
  if (!CI) {
    mean  <- lower <- upper <- numeric(n)
    for (j in 1:n) {
      if (treatment[j] == 0) {
        mean[j]  <- mu1[j] + lambda[j] * rho * (Y_obs[j] - mu0[j])
        lower[j] <- mean[j] - sqrt(1 - rho^2) * (mu1[j] - lower1[j])
        upper[j] <- mean[j] + sqrt(1 - rho^2) * (upper1[j] - mu1[j])
      } else {
        mean[j]  <- mu0[j] + inv_lambda[j] * rho * (Y_obs[j] - mu1[j])
        lower[j] <- mean[j] - sqrt(1 - rho^2) * (mu0[j] - lower0[j])
        upper[j] <- mean[j] + sqrt(1 - rho^2) * (upper0[j] - mu0[j])
      }
    }
    return(list(cf = mean, lower = lower, upper = upper))
  }
  
  # With CI (continuous Y) — bootstrap aggregation
  mean_CI_mat <- matrix(NA_real_, nrow = n, ncol = bootstraps_for_stable_lambda + bootstraps_for_mu)
  
  # reuse stored bootstrap centers for stability part
  for (b in 1:bootstraps_for_stable_lambda) {
    for (j in 1:n) {
      if (treatment[j] == 0) {
        mean_CI_mat[j, b] <- mu1_mat[j, b] + lam_mat[j, b] * rho * (Y_obs[j] - mu0_mat[j, b])
      } else {
        # guard reciprocal with clipping
        lam_b <- pmin(pmax(lam_mat[j, b], lambda_min), lambda_max)
        mean_CI_mat[j, b] <- mu0_mat[j, b] + (1 / lam_b) * rho * (Y_obs[j] - mu1_mat[j, b])
      }
    }
  }
  
  # additional bootstraps for μ using MEAN estimators (not median)
  for (b in (bootstraps_for_stable_lambda + 1):(bootstraps_for_stable_lambda + bootstraps_for_mu)) {
    i0 <- sample.int(nrow(X_control), replace = TRUE)
    i1 <- sample.int(nrow(X_treated), replace = TRUE)
    X_boot_control <- X_control[i0, , drop = FALSE]; Y_boot_control <- Y_control[i0]
    X_boot_treated <- X_treated[i1, , drop = FALSE]; Y_boot_treated <- Y_treated[i1]
    
    center_method <- if (CQR_qr == 'qgam') 'gam' else 'rf'
    mu0_boot <- Mean_wrapper(X_boot_control, Y_boot_control, X, center_method, ntree, nodesize)
    mu1_boot <- Mean_wrapper(X_boot_treated, Y_boot_treated, X, center_method, ntree, nodesize)
    
    for (j in 1:n) {
      if (treatment[j] == 0) {
        mean_CI_mat[j, b] <- mu1_boot[j] + lambda[j] * rho * (Y_obs[j] - mu0_boot[j])
      } else {
        mean_CI_mat[j, b] <- mu0_boot[j] + inv_lambda[j] * rho * (Y_obs[j] - mu1_boot[j])
      }
    }
  }
  
  lower_CI <- apply(mean_CI_mat, 1, quantile, probs = (1 - ci_level) / 2, na.rm = TRUE)
  upper_CI <- apply(mean_CI_mat, 1, quantile, probs = 1 - (1 - ci_level) / 2, na.rm = TRUE)
  mean <- apply(mean_CI_mat, 1, median, na.rm = TRUE)
  
  lower <- upper <- numeric(n)
  cscale <- rho^2 # scaling factor for the intervals; hyperparameter of our choice
  for (j in 1:n) {
    if (treatment[j] == 0) {
      lower[j] <- mean[j] - cscale*(mean[j] - lower_CI[j]) - sqrt(1 - rho^2) * (mu1[j] - lower1[j])
      upper[j] <- mean[j] + cscale*(upper_CI[j] - mean[j]) + sqrt(1 - rho^2) * (upper1[j] - mu1[j])
    } else {
      lower[j] <- mean[j] - cscale*(mean[j] - lower_CI[j]) - sqrt(1 - rho^2) * (mu0[j] - lower0[j])
      upper[j] <- mean[j] + cscale*(upper_CI[j] - mean[j]) + sqrt(1 - rho^2) * (upper0[j] - mu0[j])
    }
  }
  
  list(cf = mean, lower = lower, upper = upper)
}
