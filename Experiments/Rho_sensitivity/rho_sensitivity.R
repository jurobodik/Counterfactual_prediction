# --- packages ---
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)
library(ggplot2)
library(knitr)
library(readr)

# --- output directory ---
out_dir <- "rho_sensitivity_output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- experiment settings ---
set.seed(42)
d    <- 1
reps <- 50
n1 <- 200
n2 <- 2000

rho_dgp_vals <- seq(0, 1, by = 0.1)
rho_est_vals <- seq(0, 1, by = 0.1)

# --- helper for ETA (reset per run) ---
make_eta <- function(total_cells) {
  start_time <- Sys.time()
  function(done) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    if (done == 0) {
      cat(sprintf("[progress] 0/%d cells done. ETA: unknown\n", total_cells))
      return(invisible(NULL))
    }
    rate <- elapsed / done
    remaining <- (total_cells - done) * rate
    cat(sprintf("[progress] %d/%d cells (%.1f%%). Elapsed: %.1fs. ETA: %.1fs (~%.1f min)\n",
                done, total_cells, 100*done/total_cells, elapsed, remaining, remaining/60))
    invisible(NULL)
  }
}

# --- single run for one (rho_dgp, rho_est) ---
run_once <- function(rho_dgp, rho_est, n, d) {
  data <- data_synthetic(n = n, d = d, rho = rho_dgp)
  
  X         <- as.data.frame(data[, 1:d, drop = FALSE])
  treatment <- data$treatment
  Y_obs     <- data$Y_obs
  Y_cf_true <- data$Y_cf
  
  result <- C_rho(X = X, treatment = treatment, Y_obs = Y_obs, rho = rho_est, CI = FALSE)
  
  mse_our <- show_all_results(
    Y_cf_true = Y_cf_true,
    Y_cf_est  = result$cf,
    lower     = result$lower,
    upper     = result$upper
  )[[1]]$mse
  
  mse_oracle <- show_all_results(
    Y_cf_true = Y_cf_true,
    Y_cf_est  = data$oracle_cf,
    lower     = result$lower,
    upper     = result$upper
  )[[1]]$mse
  
  tibble(mse_our = mse_our, mse_oracle = mse_oracle, gap = mse_our - mse_oracle)
}

# --- average over reps for one cell ---
run_cell <- function(rho_dgp, rho_est, n, d, reps) {
  out <- map_dfr(1:reps, ~ run_once(rho_dgp, rho_est, n, d))
  out %>%
    summarise(
      mse_our_mean    = mean(mse_our),
      mse_oracle_mean = mean(mse_oracle),
      gap_mean        = mean(gap),
      mse_our_sd      = sd(mse_our),
      mse_oracle_sd   = sd(mse_oracle),
      gap_sd          = sd(gap)
    ) %>%
    mutate(rho_dgp = rho_dgp, rho_est = rho_est, reps = reps) %>%
    select(rho_dgp, rho_est, reps, everything())
}

# --- run the full grid for a given n ---
run_for_n <- function(n, d, reps, rho_dgp_vals, rho_est_vals) {
  grid_df <- expand.grid(rho_dgp = rho_dgp_vals, rho_est = rho_est_vals) %>% as_tibble()
  total_cells <- nrow(grid_df)
  eta <- make_eta(total_cells)
  
  results_list <- vector("list", length = total_cells)
  for (i in seq_len(total_cells)) {
    row <- grid_df[i, ]
    results_list[[i]] <- run_cell(row$rho_dgp, row$rho_est, n, d, reps)
    eta(i)
  }
  bind_rows(results_list)
}

# --- run for n1 ---
cat(sprintf("\nRunning experiments for n = %d\n", n1))
results_n1 <- run_for_n(n1, d, reps, rho_dgp_vals, rho_est_vals)
csv_n1 <- file.path(out_dir, sprintf("rho_sensitivity_results_n%d.csv", n1))
write_csv(results_n1, csv_n1)
cat(sprintf("✅ Results saved to: %s\n", csv_n1))

# --- run for n2 ---
cat(sprintf("\nRunning experiments for n = %d\n", n2))
results_n2 <- run_for_n(n2, d, reps, rho_dgp_vals, rho_est_vals)
csv_n2 <- file.path(out_dir, sprintf("rho_sensitivity_results_n%d.csv", n2))
write_csv(results_n2, csv_n2)
cat(sprintf("✅ Results saved to: %s\n", csv_n2))

# --- plots ---
# --- combined plot across n (facets side-by-side) ---
library(forcats)   # for factor ordering if needed
library(grid)      # for unit()
library(patchwork)
library(scales)

results_n1 <- results_n1 %>% mutate(n = n1)
results_n2 <- results_n2 %>% mutate(n = n2)

combined <- bind_rows(results_n1, results_n2) %>%
  mutate(
    rho_est = factor(rho_est, levels = rho_est_vals),   # ensure increasing order
    rho_dgp = factor(rho_dgp, levels = rho_dgp_vals)
  )

fill_limits <- range(combined$gap_mean, na.rm = TRUE)


# custom power transform: x^p
pow_trans <- function(p = 0.7) {
  trans_new(
    name      = paste0("pow-", p),
    transform = function(x) x^p,
    inverse   = function(x) x^(1/p),
    domain    = c(0, Inf)
  )
}

p_facets <- ggplot(combined, aes(x = rho_est, y = rho_dgp, fill = gap_mean)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", gap_mean)), size = 3) +
  scale_fill_gradientn(
    name   = "Gap",
    colours = c("white", "white", "steelblue"),  # flat white, then transition
    values  = scales::rescale(c(0, 0.2, max(fill_limits))),  # threshold at 0.10
    limits  = fill_limits,
    trans   = pow_trans(0.7)   # optional transform on top
  ) +
  labs(
    x = expression(rho ~ " used in the estimator"),
    y = expression(rho ~ " (true, used in DGP)")
  ) +
  coord_equal() +
  facet_wrap(
    ~ n, nrow = 1,
    labeller = as_labeller(function(n) paste("Sample size n =", n))
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.spacing.x = unit(12, "pt"),
    plot.title = element_text(hjust = 0.5)
  )

print(p_facets)

ggsave("rho_misspec_plot.pdf", p_facets, width = 6, height = 3.2)