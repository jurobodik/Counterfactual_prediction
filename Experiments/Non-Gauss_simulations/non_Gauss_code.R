library(dplyr)
library(Metrics)
library(purrr)
library(lubridate)

# Grid
n_vals <- c(2000, 500, 300, 100)
marginals <- c("gaussian","t","laplace","chisq")
copulas <- c("gaussian", "gumbel")
rho_vals <- c(0, 0.5, 1)
dims <- c(1)
replications <- 50

results <- expand.grid(n = n_vals, marginal = marginals, copula_type = copulas,
                       rho = rho_vals, d = dims, stringsAsFactors = FALSE)

# Safe evaluation with retries
evaluate_one_safe <- function(n, marginal, copula_type, rho, d, max_retries = 3){
  for (attempt in 1:max_retries){
    tryCatch({
      data <- data_synthetic(n = n, d = d, rho = rho,
                             marginal = marginal, copula_type = copula_type)
      X <- data[,1:d, drop = FALSE]
      treat <- data$treatment
      Y_obs <- data$Y_obs
      Y_cf_true <- data$Y_cf
      
      # Our estimator
      est <- C_rho(X = X, treatment = treat, Y_obs = Y_obs, rho = rho,
                   CI = FALSE)
      
      mse_our <- mse(Y_cf_true, est$cf)
      mse_oracle <- mse(Y_cf_true, data$oracle_cf)
      
      return(data.frame(mse_our = mse_our, mse_oracle = mse_oracle))
    }, error = function(e){
      if (attempt == max_retries){
        warning(sprintf("Failed after %d attempts for (n=%d, marg=%s, cop=%s, rho=%.2f, d=%d)",
                        max_retries, n, marginal, copula_type, rho, d))
        return(data.frame(mse_our = NA, mse_oracle = NA))
      }
    })
  }
}

# Progress tracker
total_configs <- nrow(results)
start_time <- Sys.time()

results_full <- list()

for (i in seq_len(total_configs)){
  cfg <- results[i, ]
  cat(sprintf("\n[%d/%d] Running config: n=%d, marg=%s, cop=%s, rho=%.2f, d=%d\n",
              i, total_configs, cfg$n, cfg$marginal, cfg$copula_type, cfg$rho, cfg$d))
  
  cfg_results <- replicate(replications, 
                           evaluate_one_safe(cfg$n, cfg$marginal, cfg$copula_type, cfg$rho, cfg$d),
                           simplify = FALSE) %>%
    bind_rows() %>%
    mutate(n = cfg$n, marginal = cfg$marginal, copula_type = cfg$copula_type,
           rho = cfg$rho, d = cfg$d)
  
  results_full[[i]] <- cfg_results
  
  # Estimate remaining time
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  avg_time_per_cfg <- elapsed / i
  remaining_secs <- avg_time_per_cfg * (total_configs - i)
  cat(sprintf("Completed %d configs in %.1f secs. ETA: %s\n",
              i, elapsed, seconds_to_period(remaining_secs)))
}

results_full <- bind_rows(results_full)

# Summarize
summary_results <- results_full %>%
  group_by(n,marginal,copula_type,rho,d) %>%
  summarise(mean_mse_our = mean(mse_our, na.rm = TRUE),
            mean_mse_oracle = mean(mse_oracle, na.rm = TRUE),
            sd_mse_our = sd(mse_our, na.rm = TRUE),
            sd_mse_oracle = sd(mse_oracle, na.rm = TRUE),
            .groups = 'drop')%>%
  mutate(mean_mse_oracle = if_else(rho == 1, 0, mean_mse_oracle))%>%
  mutate(gap = if_else(rho == 1,
                       mean_mse_our,
                       mean_mse_our - mean_mse_oracle))%>%
  mutate(marg_copula = paste(marginal, copula_type, sep = "_"))

# Save  summary results to CSV
write.csv(summary_results, "counterfactual_experiments_summary.csv", row.names = FALSE)
cat("\n✅ Results saved to:\n- counterfactual_experiments_raw.csv\n- counterfactual_experiments_summary.csv\n")



# ============================================================
# Post-processing script for experiments
# ============================================================
library(dplyr)
library(ggplot2)
library(forcats)
library(grid)

# Determine ordering based on gap for rho = 0.5 and n = 100
order_df <- summary_results %>%
  filter(rho == 0.5, n == 100) %>%
  mutate(marg_copula = paste(marginal, copula_type, sep = " × ")) %>%
  select(marg_copula, order_value = gap)

plot_df <- summary_results %>%
  mutate(
    n = factor(n, levels = c("100","300","500","2000")),
    marg_copula = paste(marginal, copula_type, sep = " × "),
    rho = factor(rho, levels = c("0","0.5","1")),
    rho_lab = factor(rho, levels = c("0","0.5","1"),
                     labels = c("rho==0","rho==0.5","rho==1"))
  ) %>%
  left_join(order_df, by = "marg_copula") %>%
  mutate(marg_copula = fct_reorder(marg_copula, order_value, .desc = TRUE))

# Cap color scale at 90th percentile
cap <- quantile(plot_df$gap, 0.9, na.rm = TRUE)

p <- ggplot(plot_df, aes(x = n, y = marg_copula, fill = pmin(gap, cap))) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", gap)), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue",
                      name = "Gap", limits = c(min(plot_df$gap, na.rm = TRUE), cap)) +
  facet_wrap(~ rho_lab, ncol = 3, labeller = label_parsed) +
  labs(
    x = "Sample size (n)", y = "Marginal × Copula") +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold"),
    panel.spacing = unit(1, "lines"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
print(p)
ggsave("nonGauss_gap_heatmap.pdf", plot = p, width = 7, height = 3.5)
