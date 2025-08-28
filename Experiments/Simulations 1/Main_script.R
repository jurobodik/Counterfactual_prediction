# ============================================================
# Counterfactual Simulation Study — Restricted Grid + Checkpoints
# EXACT setups (order preserved) are defined up-front below.
# Includes extra method: C_rho_rhoPerturbed (ρ randomly shifted by ±0.25, clipped to [0,1])
# ============================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr)
  library(stringr); library(tibble); library(glue)
  library(progress); library(readr); library(rlang)
})
source("Main_function.R")
source("helpers.R")
# -------------------------
# User-configurable
# -------------------------
n_reps         <- 50           # Synthetic & IHDP; Twins uses reps=1 via reps_for()
base_seed      <- 123
INCLUDE_GANITE <- TRUE         # set FALSE if Python GANITE isn't available

# --- DEFINE COMBINATIONS HERE (edit freely) ---
# Order is preserved. n for Twins/IHDP is read from data; the 'n' here is just informative.
setups <- list(
  # Twins (ρ = 0.75 only; two d values)
  list(DGP = "Twins",     n = 11984L, d = 10L, rho = 0.5),
  list(DGP = "Twins",     n = 11984L, d = 10L, rho = 0.5),
  list(DGP = "Twins",     n = 11984L, d =  1L, rho = 0.5),
  
  # IHDP (d = 10 then d = 1; ρ in 1, .75, .5, .25, 0)
  list(DGP = "IHDP",      n =  747L,  d = 10L, rho = 1),
  list(DGP = "IHDP",      n =  747L,  d = 10L, rho = 0.75),
  list(DGP = "IHDP",      n =  747L,  d = 10L, rho = 0.5),
  list(DGP = "IHDP",      n =  747L,  d = 10L, rho = 0.25),
  list(DGP = "IHDP",      n =  747L,  d = 10L, rho = 0),
  list(DGP = "IHDP",      n =  747L,  d =  1L, rho = 1),
  list(DGP = "IHDP",      n =  747L,  d =  1L, rho = 0.75),
  list(DGP = "IHDP",      n =  747L,  d =  1L, rho = 0.5),
  list(DGP = "IHDP",      n =  747L,  d =  1L, rho = 0.25),
  list(DGP = "IHDP",      n =  747L,  d =  1L, rho = 0),
  
  # Synthetic (n fixed = 1000; d = 10 then d = 1; same ρ order)
  list(DGP = "Synthetic", n = 1000L,  d = 10L, rho = 1),
  list(DGP = "Synthetic", n = 1000L,  d = 10L, rho = 0.75),
  list(DGP = "Synthetic", n = 1000L,  d = 10L, rho = 0.5),
  list(DGP = "Synthetic", n = 1000L,  d = 10L, rho = 0.25),
  list(DGP = "Synthetic", n = 1000L,  d = 10L, rho = 0),
  list(DGP = "Synthetic", n = 1000L,  d =  1L, rho = 1),
  list(DGP = "Synthetic", n = 1000L,  d =  1L, rho = 0.75),
  list(DGP = "Synthetic", n = 1000L,  d =  1L, rho = 0.5),
  list(DGP = "Synthetic", n = 1000L,  d =  1L, rho = 0.25),
  list(DGP = "Synthetic", n = 1000L,  d =  1L, rho = 0)
)
# --- END COMBINATIONS ---

# Output dir & tag
timestamp_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
outdir <- file.path(getwd(), glue("sim_results_{timestamp_tag}"))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Files (incremental)
raw_csv_path    <- file.path(outdir, "all_runs_long_incremental.csv")
agg_csv_path    <- file.path(outdir, "agg_means_se_incremental.csv")
raw_rds_path    <- file.path(outdir, "all_runs_long_incremental.rds")
last_checkpoint <- file.path(outdir, "checkpoint_info.txt")

# -------------------------
# Robust IO helpers
# -------------------------
write_csv_atomic <- function(df, path) {
  tmp <- paste0(path, ".tmp")
  readr::write_csv(df, tmp, na = "")
  file.rename(tmp, path)
}

append_csv_rows <- function(df, path) {
  if (!file.exists(path)) {
    readr::write_csv(df, path, na = "")
  } else {
    suppressWarnings(readr::write_csv(df, path, na = "", append = TRUE, col_names = FALSE))
  }
}

safe_save_rds <- function(object, path) {
  tmp <- paste0(path, ".tmp")
  saveRDS(object, tmp)
  file.rename(tmp, path)
}

write_checkpoint <- function(text) {
  writeLines(text, con = last_checkpoint)
}

# -------------------------
# Metric helpers (MSE first)
# -------------------------
.extract_metrics <- function(res_table) {
  m <- setNames(as.list(res_table$Value), res_table$Metric)
  list(
    MSE           = suppressWarnings(as.numeric(m[["MSE"]])),
    IntervalScore = suppressWarnings(as.numeric(m[["Interval Score"]])),
    Coverage      = suppressWarnings(as.numeric(m[["Coverage"]])),
    AvgWidth      = suppressWarnings(as.numeric(m[["Average Width"]])),
    QuantileLoss  = suppressWarnings(as.numeric(m[["Quantile Loss"]]))
  )
}

.safe <- function(expr) tryCatch(suppressWarnings(expr), error = function(e) NULL)

.get_metrics <- function(Y_cf_est, Y_cf_true, lower = NULL, upper = NULL) {
  tab <- if (!is.null(lower) && !is.null(upper)) {
    .safe(show_all_results(
      Y_cf_est = Y_cf_est, Y_cf_true = Y_cf_true,
      lower = lower, upper = upper
    )$table)
  } else {
    .safe(show_all_results(
      Y_cf_est = Y_cf_est, Y_cf_true = Y_cf_true
    )$table)
  }
  if (is.null(tab)) {
    return(list(MSE=NA_real_, IntervalScore=NA_real_, Coverage=NA_real_, AvgWidth=NA_real_, QuantileLoss=NA_real_))
  }
  .extract_metrics(tab)
}

# -------------------------
# Data builders
#   NOTE: For Twins & IHDP we DO NOT subset by 'd' here; we subset later.
# -------------------------
.make_data <- function(DGP, n, d, rho) {
  if (DGP == "Synthetic") {
    data <- data_synthetic(n = n, d = d, rho = rho)
    X <- data.frame(data[, 1:d, drop = FALSE])
  } else if (DGP == "Twins") {
    data <- suppressMessages(suppressWarnings(Twins_upload()))
    non_feature_cols <- intersect(c("Y_obs", "treatment", "Y_cf", "Y0", "Y1"), colnames(data))
    feat_cols <- setdiff(colnames(data), non_feature_cols)
    X <- data[, feat_cols, drop = FALSE]
  } else if (DGP == "IHDP") {
    data <- IHDP_with_rho(rho, load_csv_file = TRUE)
    non_feature_cols <- intersect(c("Y_obs", "treatment", "Y_cf", "Y0", "Y1"), colnames(data))
    feat_cols <- setdiff(colnames(data), non_feature_cols)
    X <- data[, feat_cols, drop = FALSE]
  } else stop("Unknown DGP: ", DGP)
  
  list(
    data = data,
    X = as.data.frame(X),
    Y = data$Y_obs,
    T = data$treatment,
    Y_cf_true = data$Y_cf
  )
}

# Cache Twins once (full matrix; we subset to first d columns in loop)
twins_data_cached <- .make_data("Twins", n = NA, d = NA, rho = NA)

# -------------------------
# Methods runner (single rep)
# -------------------------
.run_all_methods_once <- function(X, Y, T, Y_cf_true, rho) {
  rows <- list()
  
  # 1) C_rho (true rho)
  cr <- .safe(C_rho(
    X = X, treatment = T, Y_obs = Y, rho = rho,
    bootstraps_for_stable_lambda = 5, bootstraps_for_mu = 50,
    CI = TRUE
  ))
  met <- if (!is.null(cr)) .get_metrics(cr$cf, Y_cf_true, cr$lower, cr$upper) else list(MSE=NA,IntervalScore=NA,Coverage=NA,AvgWidth=NA,QuantileLoss=NA)
  rows[["C_rho"]] <- met
  
  # 1b) C_rho with perturbed rho (rho_modified)
  rho_mod <- pmin(1, pmax(0, rho + runif(1, -0.5, 0.5)))  # clip to [0,1]
  crp <- .safe(C_rho(
    X = X, treatment = T, Y_obs = Y, rho = rho_mod,
    bootstraps_for_stable_lambda = 5, bootstraps_for_mu = 50,
    CI = TRUE))
  met <- if (!is.null(crp)) .get_metrics(crp$cf, Y_cf_true, crp$lower, crp$upper) else list(MSE=NA,IntervalScore=NA,Coverage=NA,AvgWidth=NA,QuantileLoss=NA)
  rows[["C_rho_rhoPerturbed"]] <- met
  
  # 2) DO
  dofit <- .safe(estimate_counterfactual_literature(
    Y = Y, T = T, X = X,
    strategy = "Y(1)=mu_1(X)", method = "T_learnaer",
    intervals = TRUE, desired_coverage = 0.90,
    train_calib_split = 0.80, CQR_qr = "auto",
    ntree = 2000, nodesize = 5
  ))
  met <- if (!is.null(dofit)) .get_metrics(dofit$cf, Y_cf_true, dofit$lower, dofit$upper) else list(MSE=NA,IntervalScore=NA,Coverage=NA,AvgWidth=NA,QuantileLoss=NA)
  rows[["DO"]] <- met
  
  # 3) CATE_adj
  cate <- .safe(estimate_counterfactual_literature(
    Y = Y, T = T, X = X,
    strategy = "Y(1)=Y(0)+tau(X)", method = "T_learnaer",
    intervals = TRUE, ci_level = 0.95,
    train_calib_split = 0.80, ntree = 2000, nodesize = 5
  ))
  met <- if (!is.null(cate)) .get_metrics(cate$cf, Y_cf_true, cate$lower, cate$upper) else list(MSE=NA,IntervalScore=NA,Coverage=NA,AvgWidth=NA,QuantileLoss=NA)
  rows[["CATE_adj"]] <- met
  
  # 4) Matching
  matchfit <- .safe(estimate_counterfactual_matching(
    Y = Y, T = T, X = X,
    distance = "mahalanobis", K = 5, replace = TRUE,
    propensity_learner = "ranger_cls", caliper = NULL
  ))
  met <- if (!is.null(matchfit)) .get_metrics(matchfit$cf, Y_cf_true, matchfit$lower, matchfit$upper) else list(MSE=NA,IntervalScore=NA,Coverage=NA,AvgWidth=NA,QuantileLoss=NA)
  rows[["matching"]] <- met
  
  # 5) GANITE (optional)
  if (INCLUDE_GANITE) {
    gan <- .safe(ganite_counterfactual(X = X, Y = Y, T = T))
    met <- if (!is.null(gan)) {
      out <- .get_metrics(gan, Y_cf_true)
      lapply(out, function(x) if (length(x) == 0) NA_real_ else x)
    } else {
      list(MSE=NA, IntervalScore=NA, Coverage=NA, AvgWidth=NA, QuantileLoss=NA)
    }
    rows[["ganite"]] <- met
  }
  
  bind_rows(lapply(names(rows), function(mth) {
    as_tibble(rows[[mth]]) |> mutate(Method = mth, .before = 1)
  }))
}

# -------------------------
# Aggregation + wide format (MSE first)
# -------------------------
method_levels <- c("C_rho", "C_rho_rhoPerturbed", "DO", "CATE_adj", "matching", "ganite")

.aggregate_metrics <- function(df_long) {
  df_long %>%
    mutate(Method = factor(Method, levels = method_levels)) %>%
    group_by(DGP, n, d, rho, Method) %>%
    summarise(
      MSE_mean           = mean(MSE, na.rm = TRUE),
      MSE_se             = sd(MSE, na.rm = TRUE) / sqrt(sum(!is.na(MSE))),
      Coverage_mean      = mean(Coverage, na.rm = TRUE),
      Coverage_se        = sd(Coverage, na.rm = TRUE) / sqrt(sum(!is.na(Coverage))),
      AvgWidth_mean      = mean(AvgWidth, na.rm = TRUE),
      AvgWidth_se        = sd(AvgWidth, na.rm = TRUE) / sqrt(sum(!is.na(AvgWidth))),
      IntervalScore_mean = mean(IntervalScore, na.rm = TRUE),
      IntervalScore_se   = sd(IntervalScore, na.rm = TRUE) / sqrt(sum(!is.na(IntervalScore))),
      QuantileLoss_mean  = mean(QuantileLoss, na.rm = TRUE),
      QuantileLoss_se    = sd(QuantileLoss, na.rm = TRUE) / sqrt(sum(!is.na(QuantileLoss))),
      .groups = "drop"
    ) %>%
    arrange(DGP, rho, n, d, Method)
}

to_wide <- function(agg, metric_name) {
  mu <- paste0(metric_name, "_mean")
  se <- paste0(metric_name, "_se")
  out_col <- sym(metric_name)
  agg %>%
    select(DGP, n, d, rho, Method, all_of(mu), all_of(se)) %>%
    unite(!!out_col, all_of(c(mu, se)), sep = " ± ", na.rm = FALSE) %>%
    pivot_wider(names_from = Method, values_from = !!out_col) %>%
    arrange(DGP, rho, n, d)
}

# -------------------------
# Progress bar setup (Twins uses only 1 repetition per setup)
# -------------------------
reps_for <- function(dgp) { if (identical(dgp, "Twins")) 1L else n_reps }
total_reps <- sum(vapply(setups, function(s) as.integer(reps_for(s$DGP)), integer(1)))
pb <- progress_bar$new(format = "Running [:bar] :current/:total (:percent) ETA: :eta",
                       total = total_reps, clear = FALSE, width = 80)

# Raw results column order (MSE first)
col_order <- c("Method","DGP","rho","n","d","rep","MSE","IntervalScore","Coverage","AvgWidth","QuantileLoss")

# -------------------------
# Twins cache (loaded once)
# -------------------------
twins_data_cached <- .make_data("Twins", n = NA, d = NA, rho = NA)

# -------------------------
# Main loop
# -------------------------
setup_index <- 0L
for (setup in setups) {
  setup_index <- setup_index + 1L
  DGP  <- setup$DGP
  rho0 <- setup$rho
  d_in <- setup$d
  n_in <- setup$n
  
  reps_this <- reps_for(DGP)
  temp_results <- vector("list", reps_this)
  
  for (rep in seq_len(reps_this)) {
    # Reproducible seed index even with non-uniform reps across setups
    idx_seed <- (setup_index - 1L) * max(1L, n_reps) + rep
    set.seed(base_seed + idx_seed)
    
    dat <- if (DGP == "Synthetic") {
      .make_data(DGP, n_in, d_in, rho0)        # already subset to d_in
    } else if (DGP == "Twins") {
      twins_data_cached                         # full X; subset below
    } else {
      .make_data(DGP, NA, NA, rho0)             # full X; subset below
    }
    
    # Subset to first d predictors for real data
    if (DGP == "Synthetic") {
      X <- dat$X
    } else {
      d_target <- as.integer(d_in)
      d_use <- min(d_target, ncol(dat$X))
      X <- dat$X[, seq_len(d_use), drop = FALSE]
    }
    
    Y <- dat$Y; T <- dat$T; Y_cf_true <- dat$Y_cf_true
    n_actual <- nrow(X); d_actual <- ncol(X)
    
    res_long <- .run_all_methods_once(X, Y, T, Y_cf_true, rho0) %>%
      mutate(DGP = DGP, rho = rho0,
             n = ifelse(DGP == "Synthetic", n_in, n_actual),
             d = ifelse(DGP == "Synthetic", d_in, d_actual),
             rep = rep, .after = Method) %>%
      select(all_of(col_order))
    
    temp_results[[rep]] <- res_long
    append_csv_rows(res_long, raw_csv_path)
    
    # Save to RDS checkpoint
    if (file.exists(raw_rds_path)) {
      current_all <- tryCatch(readRDS(raw_rds_path), error = function(e) NULL)
      if (is.null(current_all)) {
        safe_save_rds(res_long, raw_rds_path)
      } else {
        safe_save_rds(bind_rows(current_all, res_long), raw_rds_path)
      }
    } else {
      safe_save_rds(res_long, raw_rds_path)
    }
    
    write_checkpoint(glue("Last completed: DGP={DGP}, rho={rho0}, rep={rep}, n={ifelse(DGP=='Synthetic', n_in, n_actual)}, d={ifelse(DGP=='Synthetic', d_in, d_actual)} at {Sys.time()}"))
    pb$tick()
  }
  
  # Per-setup summary (MSE first)
  temp_df <- bind_rows(temp_results)
  temp_summary <- .aggregate_metrics(temp_df)
  cat("\n\n=== Partial results after completed SETUP ===\n")
  cat(glue("DGP: {DGP}, rho: {rho0}, n: {unique(temp_df$n)}, d: {unique(temp_df$d)}\n"))
  print(temp_summary, n = Inf)
  cat("============================================\n\n")
  
  # Save per-setup raw + agg
  per_setup_tag <- glue("{DGP}_rho{rho0}_n{unique(temp_df$n)}_d{unique(temp_df$d)}")
  write_csv_atomic(temp_df,     file.path(outdir, glue("raw_{per_setup_tag}.csv")))
  write_csv_atomic(temp_summary,file.path(outdir, glue("agg_{per_setup_tag}.csv")))
  
  # Update global aggregated CSV and wide tables
  cumulative_raw <- readr::read_csv(raw_csv_path, show_col_types = FALSE)
  cumulative_agg <- .aggregate_metrics(cumulative_raw)
  write_csv_atomic(cumulative_agg, agg_csv_path)
  
  if (nrow(cumulative_agg) > 0) {
    mse_wide           <- to_wide(cumulative_agg, "MSE")
    coverage_wide      <- to_wide(cumulative_agg, "Coverage")
    avgwidth_wide      <- to_wide(cumulative_agg, "AvgWidth")
    intervalscore_wide <- to_wide(cumulative_agg, "IntervalScore")
    qloss_wide         <- to_wide(cumulative_agg, "QuantileLoss")
    
    write_csv_atomic(mse_wide,           file.path(outdir, "MSE_results.csv"))
    write_csv_atomic(coverage_wide,      file.path(outdir, "coverage_results.csv"))
    write_csv_atomic(avgwidth_wide,      file.path(outdir, "avgwidth_results.csv"))
    write_csv_atomic(intervalscore_wide, file.path(outdir, "intervalscore_results.csv"))
    write_csv_atomic(qloss_wide,         file.path(outdir, "quantileloss_results.csv"))
  }
}

cat("\nAll incremental files in:\n", outdir, "\n", sep = "")
cat("Raw per-rep CSV: ", raw_csv_path, "\n", sep = "")
cat("Cumulative agg CSV: ", agg_csv_path, "\n", sep = "")
cat("Checkpoint info: ", last_checkpoint, "\n", sep = "")

mse_wide           

