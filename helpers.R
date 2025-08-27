library(MASS)
library(akima)
library(rgl)
library(raster)
library(ambient)
library(dplyr)      
library(plotly)
library(ggplot2)
library(gridExtra)
library(forcats)
library(patchwork)
library(purrr)
library(grid)
library(copula)
library(qgam)               # quantile GAMs when CQR_qr = 'qgam'
library(quantregForest)     # quantile RF when CQR_qr = 'RF'
library(mgcv)               # mean/prob GAM
library(randomForest)       # mean/prob RF
library(readxl)

## ====================================================================
## estimate_counterfactual_literature: unified interface
## - strategy = "Y(1)=mu_1(X)" OR "Y(1)=Y(0)+tau(X)"
## - method   = "T_learnaer" | "S_learner" | "grf" | "DR"
## ====================================================================



estimate_counterfactual_literature <- function(
    Y, T, X,
    strategy = c("Y(1)=mu_1(X)", "Y(1)=Y(0)+tau(X)"),
    method   = c("T_learnaer", "grf", "S_learner", "DR"),
    learner  = c("randomForest", "ranger", "lm"),
    # DR options
    propensity_learner = c("ranger_cls", "logit"),
    prop_clip   = 1e-3,
    dr_smooth   = TRUE,
    # Interval controls
    intervals   = FALSE,
    # CQR (used ONLY when strategy == "Y(1)=mu_1(X)" AND Y is continuous)
    desired_coverage = 0.90,
    train_calib_split = 0.80,
    CQR_qr = "auto",   # "auto" | "qgam" | "RF"
    ntree = 2000,
    nodesize = 5,
    seed = NULL,
    # τ CI (used ONLY when strategy == "Y(1)=Y(0)+tau(X)" AND Y is continuous)
    ci_level = 0.95,
    ...
){
  stopifnot(is.data.frame(X), length(Y) == nrow(X), length(T) == nrow(X))
  strategy <- match.arg(strategy)
  method   <- match.arg(method)
  learner  <- match.arg(learner)
  
  # ---------- binary detector ----------
  .is_binary <- function(y) {
    yu <- sort(unique(na.omit(as.numeric(y))))
    length(yu) == 2 && all(yu %in% c(0,1))
  }
  Y_binary <- .is_binary(Y)
  alpha <- 1 - desired_coverage
  
  ## ---------- helpers ----------
  .fit_model <- function(X, y, learner, y_is_binary = FALSE, ...) {
    df <- data.frame(y = y, X, check.names = FALSE)
    if (y_is_binary) {
      # probabilistic models for P(Y=1|X)
      if (learner == "ranger") {
        if (!requireNamespace("ranger", quietly = TRUE)) stop("Need 'ranger'.")
        m <- ranger::ranger(y ~ ., data = transform(df, y = factor(y)),
                            probability = TRUE, ...)
        pred <- function(newX) {
          pr <- predict(m, data = data.frame(newX))$predictions
          if (is.null(colnames(pr))) as.numeric(pr[,2]) else as.numeric(pr[, which(colnames(pr) %in% c("1","TRUE"))[1]])
        }
      } else if (learner == "randomForest") {
        if (!requireNamespace("randomForest", quietly = TRUE)) stop("Need 'randomForest'.")
        m <- randomForest::randomForest(y ~ ., data = transform(df, y = factor(y)), ...)
        pred <- function(newX) as.numeric(predict(m, newdata = data.frame(newX), type = "prob")[, "1"])
      } else if (learner == "lm") {
        m <- stats::glm(y ~ ., data = df, family = binomial())
        pred <- function(newX) as.numeric(stats::predict(m, newdata = data.frame(newX), type = "response"))
      } else stop("Unsupported learner: ", learner)
    } else {
      # continuous regression
      if (learner == "ranger") {
        if (!requireNamespace("ranger", quietly = TRUE)) stop("Need 'ranger'.")
        m <- ranger::ranger(y ~ ., data = df, ...)
        pred <- function(newX) as.numeric(predict(m, data = data.frame(newX))$predictions)
      } else if (learner == "randomForest") {
        if (!requireNamespace("randomForest", quietly = TRUE)) stop("Need 'randomForest'.")
        m <- randomForest::randomForest(y ~ ., data = df, ...)
        pred <- function(newX) as.numeric(predict(m, newdata = data.frame(newX)))
      } else if (learner == "lm") {
        m <- stats::lm(y ~ ., data = df, ...)
        pred <- function(newX) as.numeric(predict(m, newdata = data.frame(newX)))
      } else stop("Unsupported learner: ", learner)
    }
    list(model = m, predict = pred)
  }
  
  .fit_propensity <- function(X, T, propensity_learner, ...) {
    propensity_learner <- match.arg(propensity_learner, c("logit", "ranger_cls"))
    if (propensity_learner == "logit") {
      df <- data.frame(T = as.numeric(T), X, check.names = FALSE)
      m <- stats::glm(T ~ ., data = df, family = binomial())
      pred <- function(newX) as.numeric(stats::predict(m, newdata = data.frame(newX), type = "response"))
    } else {
      if (!requireNamespace("ranger", quietly = TRUE)) stop("Need 'ranger' for propensity.")
      df <- data.frame(T = factor(T), X, check.names = FALSE)
      m <- ranger::ranger(T ~ ., data = df, probability = TRUE, ...)
      pred <- function(newX) {
        pr <- predict(m, data = data.frame(newX))$predictions
        if (is.null(colnames(pr))) as.numeric(pr[,2]) else as.numeric(pr[, which(colnames(pr) %in% c("1","TRUE"))[1]])
      }
    }
    list(model = m, predict = pred)
  }
  
  .assemble_from_mu  <- function(mu0_hat, mu1_hat, T) list(cf = as.numeric(ifelse(T==1, mu0_hat, mu1_hat)),
                                                           mu0_hat = as.numeric(mu0_hat), mu1_hat = as.numeric(mu1_hat))
  .assemble_from_tau <- function(Y, T, tau_hat) {
    cf <- ifelse(T==1, Y - tau_hat, Y + tau_hat)
    list(cf = as.numeric(cf), tau_hat = as.numeric(tau_hat))
  }
  
  # ==== CQR machinery (continuous Y only) ====
  .create_formula <- function(X, smoothing=TRUE){
    terms <- sapply(1:ncol(X), function(j){
      if (length(unique(X[,j])) > 9 && smoothing) paste0("s(X",j,")") else paste0("as.factor(X",j,")")
    })
    reformulate(terms, response = "Y")
  }
  .Mean_wrapper <- function(X, Y, new_points, center_method=c("gam","rf"), ntree=1000, nodesize=5){
    center_method <- match.arg(center_method)
    names(X) <- paste0("X",1:ncol(X)); names(new_points) <- paste0("X",1:ncol(new_points))
    if (center_method=="gam"){
      if (!requireNamespace("mgcv", quietly=TRUE)) stop("Need 'mgcv'.")
      formula <- .create_formula(X)
      invisible(capture.output({ fit <- mgcv::gam(formula, data=data.frame(X,Y=Y), family=gaussian()) }))
      as.numeric(predict(fit, newdata=new_points, type="response"))
    } else {
      if (!requireNamespace("randomForest", quietly=TRUE)) stop("Need 'randomForest'.")
      rf <- randomForest::randomForest(x=X, y=Y, ntree=ntree, nodesize=nodesize)
      as.numeric(predict(rf, newdata=new_points))
    }
  }
  .QR_wrapper <- function(X, Y, CQR_qr, q_lo, q_hi, ntree=1000, nodesize=5){
    names(X) <- paste0("X",1:ncol(X))
    if (CQR_qr=="RF"){
      if (!requireNamespace("quantregForest", quietly=TRUE)) stop("Need 'quantregForest'.")
      qrf <- quantregForest::quantregForest(x=X, y=Y, ntree=ntree, nodesize=nodesize)
      pred_fun <- function(newdata){
        names(newdata) <- paste0("X",1:ncol(newdata))
        mat <- predict(qrf, newdata=newdata, what=c(q_lo,q_hi))
        list(lo=as.numeric(mat[,1]), hi=as.numeric(mat[,2]))
      }
      return(list(predict=pred_fun))
    } else if (CQR_qr=="qgam"){
      if (!requireNamespace("qgam", quietly=TRUE)) stop("Need 'qgam'.")
      formula <- .create_formula(X)
      invisible(capture.output({
        fit_lo <- qgam::qgam(formula, data=data.frame(X,Y=Y), qu=q_lo)
        fit_hi <- qgam::qgam(formula, data=data.frame(X,Y=Y), qu=q_hi)
      }))
      pred_fun <- function(newdata){
        names(newdata) <- paste0("X",1:ncol(newdata))
        list(lo=as.numeric(predict(fit_lo, newdata=newdata)),
             hi=as.numeric(predict(fit_hi, newdata=newdata)))
      }
      return(list(predict=pred_fun))
    } else stop("Unsupported CQR_qr")
  }
  .CQR <- function(X, Y, new_points, desired_coverage=0.9, train_calib_split=0.8,
                   CQR_qr="auto", ntree=1000, nodesize=5, seed=NULL){
    if (CQR_qr=="auto") CQR_qr <- if (ncol(data.frame(X))>5) "RF" else "qgam"
    d <- ncol(data.frame(X))
    X <- data.frame(X); Y <- as.numeric(Y); new_points <- data.frame(new_points)
    names(X) <- names(new_points) <- paste0("X",1:d)
    n <- nrow(X); n_tr <- floor(n*train_calib_split)
    if (!is.null(seed)) set.seed(seed)
    tr_idx <- seq_len(n_tr); cal_idx <- (n_tr+1):n
    X_tr <- X[tr_idx,,drop=FALSE]; Y_tr <- Y[tr_idx]
    X_cal <- X[cal_idx,,drop=FALSE]; Y_cal <- Y[cal_idx]
    alpha_loc <- 1 - desired_coverage; q_lo <- alpha_loc/2; q_hi <- 1 - alpha_loc/2
    qr_model <- .QR_wrapper(X_tr, Y_tr, CQR_qr, q_lo, q_hi, ntree, nodesize)
    b_np  <- qr_model$predict(new_points)
    b_cal <- qr_model$predict(X_cal)
    center_method <- if (CQR_qr=="qgam") "gam" else "rf"
    mu_np <- .Mean_wrapper(X_tr, Y_tr, new_points, center_method, ntree, nodesize)
    scores <- pmax(b_cal$lo - Y_cal, Y_cal - b_cal$hi)
    m <- length(scores); q_level <- ceiling((1 - alpha_loc)*(m+1))/m
    gamma <- as.numeric(stats::quantile(scores, probs=q_level, type=1))
    list(hat_f = mu_np, lower = b_np$lo - gamma, upper = b_np$hi + gamma)
  }
  
  ## ---------- point estimators ----------
  if (strategy == "Y(1)=mu_1(X)") {
    # need mu0_hat, mu1_hat
    if (method == "T_learnaer") {
      fit1 <- .fit_model(X[T==1,,drop=FALSE], Y[T==1], learner=learner, y_is_binary=Y_binary, ...)
      fit0 <- .fit_model(X[T==0,,drop=FALSE], Y[T==0], learner=learner, y_is_binary=Y_binary, ...)
      mu1_hat <- fit1$predict(X); mu0_hat <- fit0$predict(X)
      cf_point <- .assemble_from_mu(mu0_hat, mu1_hat, T)$cf
    } else if (method == "S_learner") {
      X_aug <- data.frame(X, T=as.numeric(T))
      # fit single model on (X,T); if binary Y, this is probabilistic
      fit <- .fit_model(X_aug, Y, learner=learner, y_is_binary=Y_binary, ...)
      X1 <- X_aug; X1$T <- 1
      X0 <- X_aug; X0$T <- 0
      mu1_hat <- fit$predict(X1); mu0_hat <- fit$predict(X0)
      cf_point <- .assemble_from_mu(mu0_hat, mu1_hat, T)$cf
    } else if (method == "grf") {
      if (!requireNamespace("grf", quietly = TRUE)) stop("Need 'grf'.")
      if (Y_binary) {
        # probability forests per arm
        rf1 <- grf::probability_forest(X[T==1,,drop=FALSE], Y[T==1], ...)
        rf0 <- grf::probability_forest(X[T==0,,drop=FALSE], Y[T==0], ...)
        mu1_hat <- as.numeric(predict(rf1, X)$predictions)
        mu0_hat <- as.numeric(predict(rf0, X)$predictions)
      } else {
        rf1 <- grf::regression_forest(X[T==1,,drop=FALSE], Y[T==1], ...)
        rf0 <- grf::regression_forest(X[T==0,,drop=FALSE], Y[T==0], ...)
        mu1_hat <- as.numeric(predict(rf1, X)$predictions)
        mu0_hat <- as.numeric(predict(rf0, X)$predictions)
      }
      cf_point <- .assemble_from_mu(mu0_hat, mu1_hat, T)$cf
    } else if (method == "DR") {
      fit1 <- .fit_model(X[T==1,,drop=FALSE], Y[T==1], learner=learner, y_is_binary=Y_binary, ...)
      fit0 <- .fit_model(X[T==0,,drop=FALSE], Y[T==0], learner=learner, y_is_binary=Y_binary, ...)
      mu1_b <- fit1$predict(X); mu0_b <- fit0$predict(X)
      prop <- .fit_propensity(X, T, propensity_learner, ...)
      ehat <- pmin(pmax(prop$predict(X), prop_clip), 1 - prop_clip)
      # DR pseudo-outcomes (works for both cases; with binary Y these are stabilized residuals)
      mu1_hat <- mu1_b + T      *(Y - mu1_b)/ehat
      mu0_hat <- mu0_b + (1 - T)*(Y - mu0_b)/(1 - ehat)
      cf_point <- .assemble_from_mu(mu0_hat, mu1_hat, T)$cf
    } else stop("Unknown method")
    
    if (!intervals) {
      # point estimate only
      if (Y_binary) cf_point <- pmin(pmax(cf_point, 0), 1)
      return(list(cf = as.numeric(cf_point)))
    }
    
    # ----- intervals -----
    if (Y_binary) {
      # prediction-set intervals over {0,1} based on cf probability
      p_cf <- pmin(pmax(cf_point, 0), 1)
      lower <- upper <- numeric(length(p_cf))
      for (i in seq_along(p_cf)) {
        if (p_cf[i] <= alpha) { lower[i] <- 0; upper[i] <- 0 }
        else if (p_cf[i] >= 1 - alpha) { lower[i] <- 1; upper[i] <- 1 }
        else { lower[i] <- 0; upper[i] <- 1 }
      }
      return(list(cf = as.numeric(p_cf), lower = as.numeric(lower), upper = as.numeric(upper)))
    } else {
      # continuous: CQR from counterfactual arm
      Xdf <- data.frame(X)
      est0 <- .CQR(Xdf[T==0,,drop=FALSE], Y[T==0], Xdf, desired_coverage, train_calib_split, CQR_qr, ntree, nodesize, seed)
      est1 <- .CQR(Xdf[T==1,,drop=FALSE], Y[T==1], Xdf, desired_coverage, train_calib_split, CQR_qr, ntree, nodesize, seed)
      lower <- ifelse(T==1, est0$lower, est1$lower)
      upper <- ifelse(T==1, est0$upper, est1$upper)
      cf    <- cf_point
      return(list(cf = as.numeric(cf), lower = as.numeric(lower), upper = as.numeric(upper)))
    }
  }
  
  # ===== strategy == "Y(1)=Y(0)+tau(X)" =====
  # tau point estimation
  if (method == "T_learnaer") {
    fit1 <- .fit_model(X[T==1,,drop=FALSE], Y[T==1], learner=learner, y_is_binary=Y_binary, ...)
    fit0 <- .fit_model(X[T==0,,drop=FALSE], Y[T==0], learner=learner, y_is_binary=Y_binary, ...)
    mu1_hat <- fit1$predict(X); mu0_hat <- fit0$predict(X)
    tau_hat <- as.numeric(mu1_hat - mu0_hat)
  } else if (method == "S_learner") {
    X_aug <- data.frame(X, T=as.numeric(T))
    fit <- .fit_model(X_aug, Y, learner=learner, y_is_binary=Y_binary, ...)
    X1 <- X_aug; X1$T <- 1
    X0 <- X_aug; X0$T <- 0
    tau_hat <- as.numeric(fit$predict(X1) - fit$predict(X0))
  } else if (method == "grf") {
    if (!requireNamespace("grf", quietly = TRUE)) stop("Need 'grf'.")
    if (Y_binary) {
      # estimate tau via difference of probability forests (simple & robust)
      rf1 <- grf::probability_forest(X[T==1,,drop=FALSE], Y[T==1], ...)
      rf0 <- grf::probability_forest(X[T==0,,drop=FALSE], Y[T==0], ...)
      tau_hat <- as.numeric(predict(rf1, X)$predictions - predict(rf0, X)$predictions)
    } else {
      cf <- grf::causal_forest(X, Y, T, ...)
      pred <- predict(cf, estimate.variance = TRUE)
      tau_hat <- as.numeric(pred$predictions)
      se_tau  <- sqrt(as.numeric(pred$variance.estimates))
    }
  } else if (method == "DR") {
    fit1 <- .fit_model(X[T==1,,drop=FALSE], Y[T==1], learner=learner, y_is_binary=Y_binary, ...)
    fit0 <- .fit_model(X[T==0,,drop=FALSE], Y[T==0], learner=learner, y_is_binary=Y_binary, ...)
    mu1_hat <- fit1$predict(X); mu0_hat <- fit0$predict(X)
    prop <- .fit_propensity(X, T, propensity_learner, ...)
    ehat <- pmin(pmax(prop$predict(X), prop_clip), 1 - prop_clip)
    phi <- (mu1_hat - mu0_hat) + T*(Y - mu1_hat)/ehat - (1 - T)*(Y - mu0_hat)/(1 - ehat)
    if (dr_smooth) {
      sm <- .fit_model(X, phi, learner=learner, y_is_binary=FALSE, ...)
      tau_hat <- as.numeric(sm$predict(X))
    } else tau_hat <- as.numeric(phi)
  } else stop("Unknown method")
  
  # Map τ to counterfactuals
  cf <- ifelse(T==1, Y - tau_hat, Y + tau_hat)
  
  if (!intervals) {
    if (Y_binary) cf <- pmin(pmax(cf, 0), 1)
    return(list(cf = as.numeric(cf)))
  }
  
  # ----- intervals for tau strategy -----
  if (Y_binary) {
    # Use prediction-set intervals from cf probability
    p_cf <- pmin(pmax(cf, 0), 1)
    lower <- upper <- numeric(length(p_cf))
    for (i in seq_along(p_cf)) {
      if (p_cf[i] <= alpha) { lower[i] <- 1; upper[i] <- 1 }  # if prob of 1 is tiny, confident 0 -> set {0}
      else if (p_cf[i] >= 1 - alpha) { lower[i] <- 0; upper[i] <- 0 } # (swap if you prefer interpreting p_cf as P(Y=1))
      else { lower[i] <- 0; upper[i] <- 1 }
    }
    # NOTE: If you interpret cf as P(Y=1), use the same rule as above in the μ-strategy block:
    # if (p_cf <= alpha) -> {0}; if (p_cf >= 1-alpha) -> {1}; else {0,1}
    # Adjusted below to the same rule for consistency:
    lower <- upper <- numeric(length(p_cf))
    for (i in seq_along(p_cf)) {
      if (p_cf[i] <= alpha) { lower[i] <- 0; upper[i] <- 0 }
      else if (p_cf[i] >= 1 - alpha) { lower[i] <- 1; upper[i] <- 1 }
      else { lower[i] <- 0; upper[i] <- 1 }
    }
    return(list(cf = as.numeric(p_cf), lower = as.numeric(lower), upper = as.numeric(upper)))
  } else {
    # continuous-Y CI using a simple homoscedastic proxy if needed
    z <- stats::qnorm( (1 + ci_level) / 2 )
    if (!exists("se_tau")) {
      # generic proxy SE via calibration residuals
      n <- nrow(X); n_tr <- max(2L, floor(train_calib_split * n))
      idx <- seq_len(n)
      tr_idx <- idx[seq_len(n_tr)]; cal_idx <- idx[(n_tr+1):n]
      # build a proxy for tau for smoothing in non-GRF methods
      phi <- tau_hat
      sm_tau <- .fit_model(X[tr_idx,,drop=FALSE], phi[tr_idx], learner=learner, y_is_binary=FALSE, ...)
      r_cal  <- phi[cal_idx] - sm_tau$predict(X[cal_idx,,drop=FALSE])
      se_tau <- rep(stats::sd(r_cal, na.rm=TRUE) + 1e-12, n)  # small ridge
    }
    lower <- cf - z * se_tau
    upper <- cf + z * se_tau
    return(list(cf = as.numeric(cf), lower = as.numeric(lower), upper = as.numeric(upper)))
  }
}


# --- Matching-based counterfactuals (uniform kernel, standardized X) ---------
estimate_counterfactual_matching <- function(
    Y, T, X,
    distance = c("ps", "mahalanobis"),
    K = 1,
    replace = TRUE,
    caliper = NULL,
    propensity_learner = c("logit", "ranger_cls"),
    prop_clip = 1e-3,
    alpha = 0.10,
    ...
) {
  stopifnot(is.data.frame(X), length(Y) == nrow(X), length(T) == nrow(X))
  distance <- match.arg(distance)
  propensity_learner <- match.arg(propensity_learner)
  n <- nrow(X)
  T <- as.integer(T)
  
  .fit_propensity <- function(X, T, propensity_learner, ...) {
    if (propensity_learner == "logit") {
      df <- data.frame(T = as.numeric(T), X, check.names = FALSE)
      m <- stats::glm(T ~ ., data = df, family = binomial())
      pred <- function(newX) as.numeric(stats::predict(m, newdata = data.frame(newX), type = "response"))
      return(pred)
    } else {
      if (!requireNamespace("ranger", quietly = TRUE)) stop("Need 'ranger' for propensity.")
      df <- data.frame(T = factor(T), X, check.names = FALSE)
      m <- ranger::ranger(T ~ ., data = df, probability = TRUE, ...)
      pred <- function(newX) {
        pr <- predict(m, data = data.frame(newX))$predictions
        if (is.null(colnames(pr))) as.numeric(pr[, 2]) else as.numeric(pr[, which(colnames(pr) %in% c("1","TRUE"))[1]])
      }
      return(pred)
    }
  }
  
  # ----- distances -----
  if (distance == "ps") {
    prop_pred <- .fit_propensity(X, T, propensity_learner, ...)
    e_hat <- pmin(pmax(prop_pred(X), prop_clip), 1 - prop_clip)
    lp <- qlogis(e_hat)
    D <- abs(outer(lp, lp, "-"))
    if (!is.null(caliper)) {
      sdlp <- stats::sd(lp)
      D[D > caliper * sdlp] <- Inf
    }
  } else {
    Xmat <- scale(as.matrix(X))
    S <- stats::cov(Xmat)
    Sinv <- tryCatch(solve(S), error = function(e) MASS::ginv(S))
    D <- matrix(0, n, n)
    for (j in 1:n) {
      diff <- t(t(Xmat) - Xmat[j, ])
      D[, j] <- sqrt(rowSums((diff %*% Sinv) * diff))
    }
  }
  
  opp_mask <- outer(T, T, function(a, b) a != b)
  D[!opp_mask] <- Inf
  diag(D) <- Inf
  
  # ----- select donors -----
  donors_idx <- vector("list", n)
  if (replace) {
    for (i in 1:n) {
      ord <- order(D[i, ], decreasing = FALSE)
      finite_ord <- ord[is.finite(D[i, ord])]
      donors_idx[[i]] <- head(finite_ord, K)
    }
  } else {
    idx_t1 <- which(T == 1); idx_t0 <- which(T == 0)
    avail_t1 <- idx_t1; avail_t0 <- idx_t0
    assign_block <- function(target_idx, donor_pool_idx) {
      for (i in target_idx) {
        cand <- donor_pool_idx
        drow <- D[i, cand]
        ord <- order(drow, decreasing = FALSE)
        finite_ord <- cand[ord][is.finite(drow[ord])]
        kix <- head(finite_ord, K)
        donors_idx[[i]] <<- kix
        if (length(kix) > 0 && K == 1) {
          if (T[i] == 1) avail_t0 <<- setdiff(avail_t0, kix) else avail_t1 <<- setdiff(avail_t1, kix)
        }
      }
      list(avail_t1 = avail_t1, avail_t0 = avail_t0)
    }
    res <- assign_block(idx_t1, avail_t0); avail_t1 <- res$avail_t1; avail_t0 <- res$avail_t0
    res <- assign_block(idx_t0, avail_t1)
  }
  
  # ----- compute cf + PI -----
  cf <- lower <- upper <- rep(NA_real_, n)
  for (i in 1:n) {
    kix <- donors_idx[[i]]
    if (length(kix) == 0) next
    y_don <- Y[kix]
    cf[i] <- mean(y_don)
    if (length(y_don) >= 2) {
      s <- stats::sd(y_don)
      tcrit <- stats::qt(1 - alpha/2, df = length(y_don) - 1)
      half <- tcrit * s / sqrt(length(y_don))
    } else {
      half <- 0
    }
    lower[i] <- cf[i] - half
    upper[i] <- cf[i] + half
  }
  
  list(cf = as.numeric(cf), lower = as.numeric(lower), upper = as.numeric(upper))
}






####################################################################################################
################# Generating benchmark datasets ####################################################
####################################################################################################
data_synthetic <- function(n = 1000, 
                           d = 2, 
                           rho, 
                           sigma_1 = 1, 
                           sigma_2 = 4, 
                           constant_propensity = FALSE, 
                           copula_type = "gaussian", # "gaussian","clayton","t","gumbel"
                           marginal = "gaussian"){     # "gaussian","t","laplace","chisq"
  # --- small helpers ----
  qlaplace <- function(p) ifelse(p < 0.5, log(2*p), -log(2*(1-p)))
  plaplace <- function(x) ifelse(x < 0, 0.5*exp(x), 1 - 0.5*exp(-x))
  
  q_marg <- function(u, type) switch(type,
                                     "gaussian" = qnorm(u),
                                     "t"       = qt(u, df = 3),
                                     "laplace" = qlaplace(u),
                                     "chisq"   = qchisq(u, df = 3),
                                     stop("Unsupported marginal type.")
  )
  p_marg <- function(z, type) switch(type,
                                     "gaussian" = pnorm(z),
                                     "t"       = pt(z, df = 3),
                                     "laplace" = plaplace(z),
                                     "chisq"   = pchisq(z, df = 3),
                                     stop("Unsupported marginal type.")
  )
  build_copula <- function(rho, copula_type){
    tau <- (2/pi) * asin(rho)
    if(tau <= -1) tau <- -0.99
    if(tau >=  1) tau <-  0.99
    switch(copula_type,
           "gaussian" = normalCopula(param = rho, dim = 2),
           "t"        = tCopula(param = rho, dim = 2, df = 3),
           "clayton"  = claytonCopula(param = iTau(claytonCopula(), tau), dim = 2),
           "gumbel"   = gumbelCopula(param  = iTau(gumbelCopula(),  tau), dim = 2),
           stop("Unsupported copula type.")
    )
  }
  
  # --- random smooth functions ---
  random_function_1d <- function(freq = 0.1) {
    s = seq(-10, 10, length.out = 1001)
    f <- long_grid(s)
    f$noise <- rep(0, length(s))
    while (all(f$noise[300:600] == 0)) {
      f$noise <- gen_perlin(f$x, frequency = freq, fractal = 'rigid-multi')
    }
    amplitude = 10 / (max(f$noise) + 1)
    f$noise * amplitude
  }
  evaluation_of_f_1d <- function(f, X1) {
    minim = min(X1); maxim = max(X1)
    sapply(X1, function(x) f[round((x - minim) * 1000 / (maxim - minim) + 1)])
  }
  random_function_2d <- function(freq = 0.001) {
    f = noise_perlin(c(1001, 1001), frequency = freq, fractal = 'fbm', octaves = 2, lacunarity = 2, gain = 0.4)
    f = f^2
    amplitude = 10 / max(f)
    f * amplitude
  }
  evaluation_of_f_2d <- function(f, X1, X2) {
    minim_x = min(X1); maxim_x = max(X1)
    minim_y = min(X2); maxim_y = max(X2)
    mapply(function(x, y) {
      xx = round((x - minim_x) * 1000 / (maxim_x - minim_x) + 1)
      yy = round((y - minim_y) * 1000 / (maxim_y - minim_y) + 1)
      f[xx, yy]
    }, X1, X2)
  }
  
  # --- covariates & signal ---
  if (d == 1) {
    X = runif(n, -1, 1)
    CATE = 5 + evaluation_of_f_1d(random_function_1d(), X)
    mu = 5 + 5 * X
  } else {
    Sigma <- matrix(0.25, nrow = d, ncol = d); diag(Sigma) <- 1
    tilde_X <- MASS::mvrnorm(n, mu = rep(0, d), Sigma = Sigma)
    X <- pnorm(tilde_X)
    tau = random_function_2d()
    CATE = evaluation_of_f_2d(tau, X[,1], X[,2])
    beta = rnorm(d)
    mu = as.vector(X %*% beta)
  }
  
  # --- errors via copula ---
  generate_errors <- function(n, rho, sigma_1 , sigma_2, copula_type = "gaussian", marginal = "gaussian") {
    cop <- build_copula(rho, copula_type)
    u <- rCopula(n, cop)
    transform_marginal <- function(u_vec, type) switch(type,
                                                       "gaussian" = qnorm(u_vec),
                                                       "t"       = qt(u_vec, df = 3),
                                                       "laplace" = qlaplace(u_vec),
                                                       "chisq"   = qchisq(u_vec, df = 3),
                                                       stop("Unsupported marginal type.")
    )
    eps1 <- transform_marginal(u[, 1], marginal) * sqrt(sigma_1)
    eps2 <- transform_marginal(u[, 2], marginal) * sqrt(sigma_2)
    data.frame(eps1, eps2)
  }
  eps <- generate_errors(n, rho, sigma_1, sigma_2, copula_type, marginal)
  epsilon1 <- eps[,1]; epsilon2 <- eps[,2]
  
  Y0 <- mu + epsilon1
  Y1 <- mu + CATE + epsilon2
  
  if (!constant_propensity) {
    propensity_score = if (d == 1) (1 + abs(X)) / 4 else (1 + abs(X[,1])) / 4
    treatment = rbinom(n, 1, 1 - propensity_score)
  } else {
    treatment = sample(c(0, 1), n, replace = TRUE)
  }
  
  Y_obs = ifelse(treatment == 1, Y1, Y0)
  Y_cf  = ifelse(treatment == 0, Y1, Y0)
  
  # ---------------- ORACLE: E[Y_cf | X, Y_obs, T] ----------------
  cop <- build_copula(rho, copula_type)
  # uniforms corresponding to observed epsilons
  u1 <- p_marg(epsilon1 / sqrt(sigma_1), marginal)
  u2 <- p_marg(epsilon2 / sqrt(sigma_2), marginal)
  u1 <- pmin(pmax(u1, .Machine$double.eps), 1 - .Machine$double.eps)
  u2 <- pmin(pmax(u2, .Machine$double.eps), 1 - .Machine$double.eps)
  
  # fast closed form if (copula in {gaussian,t}) AND (marginal in {gaussian,t})
  use_closed_form <- (copula_type %in% c("gaussian","t")) && (marginal %in% c("gaussian","t"))
  
  if (use_closed_form){
    # bivariate normal / t: conditional mean is linear
    E_eps2_given_eps1 <- rho * sqrt(sigma_2 / sigma_1) * epsilon1
    E_eps1_given_eps2 <- rho * sqrt(sigma_1 / sigma_2) * epsilon2
  } else {
    # numerical: E[ q_marg(U2) | U1=u1 ] = \int_0^1 q_marg(u2) * c(u1,u2) du2
    # and symmetrically for E[ q_marg(U1) | U2=u2 ]
    E_eps2_given_eps1 <- vapply(u1, function(u1i){
      integrand <- function(v) {
        q_marg(v, marginal) * dCopula(cbind(rep(u1i, length(v)), v), cop)
      }
      val <- integrate(integrand, lower = 0, upper = 1, rel.tol = 1e-6, subdivisions = 200L)$value
      sqrt(sigma_2) * val
    }, numeric(1))
    
    E_eps1_given_eps2 <- vapply(u2, function(u2i){
      integrand <- function(v) {
        q_marg(v, marginal) * dCopula(cbind(v, rep(u2i, length(v))), cop)
      }
      val <- integrate(integrand, lower = 0, upper = 1, rel.tol = 1e-6, subdivisions = 200L)$value
      sqrt(sigma_1) * val
    }, numeric(1))
  }
  
  oracle_cf <- ifelse(treatment == 0,
                      mu + CATE + E_eps2_given_eps1, # T=0: observed Y0 => need E[Y1|...]
                      mu + E_eps1_given_eps2)        # T=1: observed Y1 => need E[Y0|...]
  
  return(data.frame(X, Y0 = Y0, Y1 = Y1, Y_obs = Y_obs, Y_cf = Y_cf, 
                    treatment = treatment, oracle_cf = oracle_cf))
}




show_all_results <- function(Y_cf_true, Y_cf_est, lower=NA, upper=NA, desired_coverage = 0.9) {
  alpha <- 1 - desired_coverage
  
  # Remove rows where Y_cf_est is NA, Inf, or -Inf
  keep_idx <- !(is.na(Y_cf_est) | is.infinite(Y_cf_est))
  Y_cf_true <- Y_cf_true[keep_idx]
  Y_cf_est  <- Y_cf_est[keep_idx]
  lower     <- lower[keep_idx]
  upper     <- upper[keep_idx]
  
  # Helper: QL_tau(y, q)
  QL_tau <- function(y, q, tau) {
    (y - q) * (tau - as.numeric(y <= q))
  }
  
  # Helper: Interval Score for a single y,l,u at level 1-alpha
  IS_point <- function(y, l, u, alpha){
    wid <- pmax(0, u - l)
    wid + (2/alpha) * pmax(0, l - y) + (2/alpha) * pmax(0, y - u)
  }
  
  # Check if intervals exist
  has_intervals <- !(all(is.na(lower)) || all(is.na(upper)))
  
  metrics_list <- list()
  
  # Always compute MSE
  mse <- mean((Y_cf_est - Y_cf_true)^2, na.rm = TRUE)
  metrics_list$mse <- mse
  
  metrics_table <- data.frame(
    Metric = c("MSE"),
    Value  = c(sprintf("%.3f", mse)),
    stringsAsFactors = FALSE
  )
  
  plot <- NULL
  
  if (has_intervals) {
    covered    <- (Y_cf_true >= lower & Y_cf_true <= upper)
    width_vec  <- pmax(0, upper - lower)
    coverage   <- mean(covered, na.rm = TRUE)
    avg_width  <- mean(width_vec, na.rm = TRUE)
    
    # Quantile loss (symmetric, using lower/upper at alpha/2 and 1-alpha/2)
    ql_vals <- QL_tau(Y_cf_true, lower, alpha/2) +
      QL_tau(Y_cf_true, upper, 1 - alpha/2)
    quantile_loss <- mean(ql_vals / 2, na.rm = TRUE)
    
    # Interval Score
    is_vals <- IS_point(Y_cf_true, lower, upper, alpha)
    interval_score <- mean(is_vals, na.rm = TRUE)
    
    metrics_list$coverage       <- coverage
    metrics_list$avg_width      <- avg_width
    metrics_list$quantile_loss  <- quantile_loss
    metrics_list$interval_score <- interval_score
    
    metrics_table <- rbind(
      data.frame(
        Metric = c(
          "Coverage",
          "Average Width",
          "Interval Score",
          "Quantile Loss"
        ),
        Value  = c(
          sprintf("%.3f", coverage),
          sprintf("%.3f", avg_width),
          sprintf("%.3f", interval_score),
          sprintf("%.3f", quantile_loss)
        ),
        stringsAsFactors = FALSE
      ),
      metrics_table
    )
    
    plot_df <- data.frame(
      Y_true  = Y_cf_true,
      pred    = Y_cf_est,
      lower   = lower,
      upper   = upper,
      covered = covered
    )
    
    plot <- ggplot(plot_df, aes(x = pred, y = Y_true)) +
      geom_errorbarh(aes(xmin = lower, xmax = upper, color = covered), height = 0.002) +
      geom_point(alpha = 0.5, size = 0.6) +
      geom_abline(slope = 1, intercept = 0) +
      scale_color_manual(values = c("FALSE" = "red", "TRUE" = "grey50"),
                         labels = c("FALSE" = "No", "TRUE" = "Yes")) +
      labs(
        title = "Counterfactual Estimation with Prediction Intervals",
        x = "Predicted Counterfactual (mean & interval)",
        y = "True Counterfactual",
        color = "Interval covers truth:"
      ) +
      theme_minimal()
  } else {
    # If no intervals, still record NA for interval metrics for consistency
    metrics_list$coverage       <- NA_real_
    metrics_list$avg_width      <- NA_real_
    metrics_list$quantile_loss  <- NA_real_
    metrics_list$interval_score <- NA_real_
  }
  
  return(list(metrics = metrics_list, table = metrics_table, plot = plot))
}



IHDP_with_rho <- function(rho, setup = 'B', load_csv_file=FALSE) {
  if(load_csv_file){ihdp  =  read.csv("ihdp_data.csv")}
  data = ihdp
  data$treatment = as.integer(as.logical(data$treatment) )
  X = as.matrix(data[, -c(1,2,3,4,5)])
  
  p <- ncol(X)
  n <- nrow(X)
  
  if (setup == 'A') {
    # ---------------------------
    # Response Surface A (Linear)
    # ---------------------------
    
    beta_A <- sample(0:4, p, replace = TRUE, prob = c(0.5, 0.2, 0.15, 0.1, 0.05))
    
    mu0 <- X %*% beta_A
    mu1 <- mu0 + 4  # constant treatment effect
    
  } else if (setup == 'B') {
    # ------------------------------
    # Response Surface B (Nonlinear)
    # ------------------------------
    
    beta_B <- sample(c(0, 0.1, 0.2, 0.3, 0.4), p, replace = TRUE, prob = c(0.6, 0.1, 0.1, 0.1, 0.1))
    
    W <- matrix(0.5, nrow = n, ncol = p)
    
    mu0_raw <- exp((X + W) %*% beta_B)
    mu1_raw <- X %*% beta_B
    
    # Center treatment effect to have ATE = 4
    omega_B <- mean(mu1_raw - mu0_raw) - 4
    mu0 <- mu0_raw
    mu1 <- mu1_raw - omega_B
    
  } else {
    stop("setup must be 'A' or 'B'")
  }
  
  # Correlated noise
  epsilons <- MASS::mvrnorm(n, mu = c(0, 0), 
                            Sigma = matrix(c(1, rho, rho, 1), nrow = 2))
  
  Y0 <- as.numeric(mu0 + epsilons[,1])
  Y1 <- as.numeric(mu1 + epsilons[,2])
  Y_obs = ifelse(data$treatment == 1, Y1, Y0)
  Y_cf = ifelse(data$treatment == 1, Y0 , Y1)
  X1 = data$x1; X2 = data$x2; X3 = data$x3; X4 = data$x4; X5 = data$x5; X6 = data$x6; X7 = data$x7; X8 = data$x8; X9 = data$x9; X10 = data$x10
  X11= data$x11; X12 = data$x12; X13 = data$x13; X14 = data$x14; X15 = data$x15; X16 = data$x16; X17 = data$x17; X18 = data$x18; X19 = data$x19; X20 = data$x20
  X21= data$x21; X22 = data$x22; X23 = data$x23; X24 = data$x24; X25 = data$x25
  return(data.frame(X1=X1,X2=X2,X3=X3,X4=X4,X5=X5,X6=X6,X7=X7,X8=X8,X9=X9,X10=X10,
                    X11=X11,X12=X12,X13=X13,X14=X14,X15=X15,X16=X16,X17=X17,X18=X18,X19=X19,X20=X20,
                    X21=X21,X22=X22,X23=X23,X24=X24,X25=X25, treatment = data$treatment, Y_obs = Y_obs, Y0 = Y0, Y1 = Y1, Y_cf=Y_cf))
}







Twins_upload <- function(path = "twins.csv", n=11984, d=74) {
  # ---- 1) Read file (supports .csv, .csv.gz, .csv.gr) ----
  read_fast <- function(p) {
    if (requireNamespace("readr", quietly = TRUE)) {
      suppressMessages(
        suppressWarnings(
          readr::read_csv(p, show_col_types = FALSE)
        )
      )
    } else {
      if (grepl("\\.(gz|gr)$", p, ignore.case = TRUE)) {
        suppressWarnings(utils::read.csv(gzfile(p)))
      } else {
        suppressWarnings(utils::read.csv(p))
      }
    }
  }
  
  data <- as.data.frame(read_fast(path))[1:n, ]
  
  # ---- 2) Detect key columns ----
  treat_candidates <- c("T", "treatment", "A", "Z")
  t_col <- treat_candidates[treat_candidates %in% names(data)]
  if (length(t_col) == 0) stop("Couldn't find treatment column.")
  treatment <- as.integer(as.logical(data[[t_col[1]]]))
  
  y0_candidates   <- c("y0", "Y0", "Y_0")
  y1_candidates   <- c("y1", "Y1", "Y_1")
  yf_candidates   <- c("yf", "Yf", "Y_obs", "y", "Y")
  ycf_candidates  <- c("y_cf", "Y_cf", "Ycf")
  
  pick <- function(cands) cands[cands %in% names(data)][1]
  y0_col  <- pick(y0_candidates)
  y1_col  <- pick(y1_candidates)
  yf_col  <- pick(yf_candidates)
  ycf_col <- pick(ycf_candidates)
  
  if (is.na(y0_col) || is.na(y1_col)) {
    stop("Couldn't find both y0 and y1 columns.")
  }
  
  Y0 <- as.numeric(data[[y0_col]])
  Y1 <- as.numeric(data[[y1_col]])
  Y_obs <- if (!is.na(yf_col)) as.numeric(data[[yf_col]]) else ifelse(treatment == 1, Y1, Y0)
  Y_cf  <- if (!is.na(ycf_col)) as.numeric(data[[ycf_col]]) else ifelse(treatment == 1, Y0, Y1)
  
  # ---- 3) Build feature matrix X ----
  drop_cols <- unique(na.omit(c(t_col[1], y0_col, y1_col, yf_col, ycf_col, 
                                "Propensity", "propensity", "X", "id", "ID")))
  feature_names <- setdiff(names(data), drop_cols)
  
  X_raw <- data[feature_names]
  X <- as.data.frame(lapply(X_raw, function(v) {
    if (is.factor(v)) as.numeric(v)
    else if (is.logical(v)) as.integer(v)
    else as.numeric(v)
  }))
  keep <- vapply(X, function(col) !all(is.na(col)), logical(1))
  X <- X[keep]
  
  # Rename to X1..Xp, then drop the first one (ID)
  colnames(X) <- paste0("X", seq_len(ncol(X)))
  if (ncol(X) > 1) {
    X <- X[, -1, drop = FALSE]  # remove first column (ID)
  }
  
  ########################
  X=X[,1:d]  # Keep only the first d columns
  # ---- 4) Return ----
  out <- cbind(X,
               treatment = treatment,
               Y_obs = Y_obs,
               Y0 = Y0,
               Y1 = Y1,
               Y_cf = Y_cf)
  rownames(out) <- NULL
  as.data.frame(out)
}
