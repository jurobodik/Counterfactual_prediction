# ==== MSE rank heatmap (legend right + flattened rows) ====

mse_final <- read.csv("MSE_results.csv", stringsAsFactors = FALSE)

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

# ---------- 1) Wide → long ----------
mse_long <- mse_final %>%
  pivot_longer(cols = -c(DGP, n, d, rho),
               names_to = "method", values_to = "value") %>%
  mutate(method = case_when(
    method %in% c("C_rho_rhoPerturbed", "C_rhoperturbed") ~ "mu_rho-misspecified",
    TRUE ~ method
  ))

# ---------- 2) Parse "mean ± SE" ----------
split <- str_split_fixed(as.character(mse_long$value), "\\s*(±|\\+/-|\\+-)\\s*", 2)
mse_long <- mse_long %>%
  mutate(
    mean_MSE = suppressWarnings(as.numeric(trimws(split[,1]))),
    se_MSE   = suppressWarnings(as.numeric(na_if(trimws(split[,2]), "NA"))),
    n   = as.numeric(n),
    d   = as.numeric(d),
    rho = as.numeric(rho)
  )

# ---------- 3) Merge C_rho & misspecified into one pseudo-method ----------
combo <- mse_long %>%
  filter(method %in% c("C_rho","mu_rho-misspecified")) %>%
  select(DGP, n, d, rho, method, mean_MSE) %>%
  pivot_wider(names_from = method, values_from = mean_MSE) %>%
  mutate(
    method    = "mu_rho (combined)",
    mean_MSE  = `mu_rho-misspecified`,                 # <<< rank by misspecified
    label_str = sprintf("%.2f / %.2f", `C_rho`, `mu_rho-misspecified`)
  ) %>%
  select(DGP, n, d, rho, method, mean_MSE, label_str)

others <- mse_long %>%
  filter(!method %in% c("C_rho","mu_rho-misspecified")) %>%
  mutate(label_str = sprintf("%.2f", mean_MSE)) %>%
  select(DGP, n, d, rho, method, mean_MSE, label_str)

mse <- bind_rows(combo, others)

# ---------- 4) Keep methods/rho of interest; drop Synthetic n=500 ----------
methods_order <- c("mu_rho (combined)","DO","CATE_adj","matching","ganite")  # μρ first
rho_grid <- c(0, 0.25, 0.5, 0.75, 1)

mse <- mse %>%
  filter(method %in% methods_order) %>%
  filter(is.na(rho) | rho %in% rho_grid) %>%
  filter(!(DGP == "Synthetic" & n == 500))

# ---------- 5) Scenario labels (remove n=..., keep d and ρ) ----------
fmt_rho <- function(x) {
  ifelse(is.na(x), "NA", formatC(as.numeric(x), format = "f", digits = 2))
}

mse <- mse %>%
  mutate(scenario = sprintf("%s | d=%s, ρ=%s", DGP, d, fmt_rho(rho)))

# ---------- 6) Rank (1 = best … 5 = worst) ----------
ranked <- mse %>%
  group_by(scenario) %>%
  mutate(
    rank_tmp = base::rank(mean_MSE, ties.method = "min"),
    rank_int = dplyr::dense_rank(rank_tmp),
    rank_int = pmin(rank_int, 5L)
  ) %>%
  ungroup() %>%
  mutate(rank_cat = factor(rank_int, levels = 1:5,
                           labels = c("1 (best)","2","3","4","5 (worst)")))

# ---------- 7) Order rows: Synthetic → IHDP → Twins (top→bottom) ----------
scenario_meta <- mse %>%
  distinct(scenario, DGP, d, rho) %>%
  mutate(DGP_key = case_when(
    DGP == "Synthetic" ~ 1L,
    DGP == "IHDP"      ~ 2L,
    DGP == "Twins"     ~ 3L,
    TRUE               ~ 99L
  )) %>%
  arrange(DGP_key, d, rho)

ranked$scenario <- factor(ranked$scenario,
                          levels = rev(scenario_meta$scenario))

# ---------- 8) Convert row labels to plotmath expressions with bold rho ----------
make_label_expr <- function(s) {
  parts <- strsplit(s, "\\|")[[1]]
  dgp <- trimws(parts[1])
  details <- trimws(parts[2])
  d_val <- sub(".*d=([0-9]+).*", "\\1", details)
  rho_val <- sub(".*ρ=(.*)", "\\1", details)
  sprintf('"%s"~"| d="~%s*","~bold(rho)~"="~"%s"', dgp, d_val, rho_val)
}
scenario_levels <- levels(ranked$scenario)
lab_expr <- sapply(scenario_levels, make_label_expr)

# ---------- 9) Separators ----------
scenario_meta_plot <- ranked %>%
  distinct(scenario, DGP, rho) %>%
  mutate(y_index = match(scenario, levels(ranked$scenario)))

# Thick lines between datasets
dgp_sep <- scenario_meta_plot %>%
  group_by(DGP) %>%
  summarise(last_idx = max(y_index), .groups = "drop") %>%
  transmute(yintercept = last_idx + 0.5)

# Thin lines between rho=1 and rho=0 (consider reversed order!)
rho_sep <- scenario_meta_plot %>%
  group_by(DGP) %>%
  arrange(desc(y_index), .by_group = TRUE) %>%
  mutate(next_rho = lead(rho)) %>%
  filter(rho == 1 & next_rho == 0) %>%
  transmute(yintercept = y_index - 0.5)

# ---------- 10) Force column order ----------
ranked$method <- factor(ranked$method, levels = methods_order)

# ---------- 11) Panel assignment: Synthetic (left) vs Real data (right), separators, and plot ----------
library(patchwork)   # for side-by-side & shared legend; install if needed

# Attach panel directly from DGP and set panel order (Synthetic left, Real data right)
ranked <- ranked %>%
  dplyr::mutate(
    panel = dplyr::if_else(DGP == "Synthetic", "Synthetic", "Real data"),
    panel = factor(panel, levels = c("Synthetic", "Real data"))
  )

# Build a meta table with per-panel y indices (respecting the existing factor order of 'scenario')
scenario_meta_plot <- ranked %>%
  dplyr::distinct(scenario, DGP, rho, panel) %>%
  dplyr::mutate(y_index = match(scenario, levels(ranked$scenario))) %>%
  dplyr::arrange(panel, y_index)

# Thick lines between datasets (per panel)
dgp_sep <- scenario_meta_plot %>%
  dplyr::group_by(panel, DGP) %>%
  dplyr::summarise(last_idx = max(y_index), .groups = "drop") %>%
  dplyr::transmute(panel, yintercept = last_idx + 0.5)

# Remove the first horizontal thick line (the top line) within each panel
dgp_sep <- dgp_sep %>%
  dplyr::group_by(panel) %>%
  dplyr::filter(yintercept != max(yintercept)) %>%
  dplyr::ungroup()

# Thin dashed lines between rho=1 and rho=0 (per panel; account for reversed y order)
rho_sep <- scenario_meta_plot %>%
  dplyr::group_by(panel, DGP) %>%
  dplyr::arrange(dplyr::desc(y_index), .by_group = TRUE) %>%
  dplyr::mutate(next_rho = dplyr::lead(rho)) %>%
  dplyr::filter(rho == 1 & next_rho == 0) %>%
  dplyr::transmute(panel, yintercept = y_index - 0.5) %>%
  dplyr::ungroup()

# ---------- Custom y-axis labels: remove "Synthetic" text for Synthetic panel ----------
scenario_levels <- levels(ranked$scenario)
panel_map_df <- ranked %>% dplyr::distinct(scenario, panel)
panel_lookup <- setNames(as.character(panel_map_df$panel), as.character(panel_map_df$scenario))

lab_expr <- vapply(scenario_levels, function(s) {
  parts <- strsplit(s, "\\|")[[1]]
  dgp <- trimws(parts[1])
  details <- trimws(parts[2])
  d_val <- sub(".*d=([0-9]+).*", "\\1", details)
  rho_val <- sub(".*ρ=(.*)", "\\1", details)
  if (identical(panel_lookup[[s]], "Synthetic")) {
    sprintf('"d="~%s*","~bold(rho)~"="~"%s"', d_val, rho_val)
  } else {
    sprintf('"%s"~"| d="~%s*","~bold(rho)~"="~"%s"', dgp, d_val, rho_val)
  }
}, character(1L))

# ---------- 12) Plot ----------
rank_pal <- c("1 (best)"="#1a9850","2"="#91cf60","3"="#d9ef8b","4"="#fee08b","5 (worst)"="#d73027")

p_split <- ggplot(ranked, aes(x = method, y = scenario, fill = rank_cat)) +
  geom_tile(color = "white", linewidth = 0.2) +
  geom_hline(data = dgp_sep, aes(yintercept = yintercept),
             inherit.aes = FALSE, color = "black", linewidth = 1.2) +
  geom_hline(data = rho_sep, aes(yintercept = yintercept),
             inherit.aes = FALSE, color = "grey40", linewidth = 0.6, linetype = "dashed") +
  geom_text(aes(label = label_str), size = 2.6) +
  scale_x_discrete(labels = function(x) {
    out <- dplyr::case_when(
      x == "mu_rho (combined)" ~ "atop(mu[rho], rho[correct]~'/'~rho[misspec])",
      x == "DO"                ~ "atop('DO', scriptstyle(italic('(CQR)')))",
      x == "CATE_adj"          ~ "atop('cate-adj', scriptstyle(italic('(T-learner)'))) ",
      x == "matching"          ~ "atop('matching', scriptstyle(italic('(Mah. dist.)')))",
      TRUE ~ paste0("'", x, "'")
    )
    parse(text = out)
  }) +
  scale_y_discrete(labels = setNames(parse(text = lab_expr), scenario_levels)) +
  scale_fill_manual(values = rank_pal, name = "Rank (MSE)", drop = FALSE, na.value = "gray80") +
  labs(title = "MSE of Counterfactual Estimators (Ranked)", x = NULL, y = NULL) +
  facet_wrap(~ panel, ncol = 2, scales = "free_y") +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.y = element_text(size = 8),
    axis.text.x = element_text(size = 9),
    legend.text = element_text(size = 11),
    legend.title = element_text(size = 12, face = "bold"),
    plot.title = element_text(face = "bold", margin = ggplot2::margin(b = 6), hjust = 0.5),
    strip.text = element_text(face = "bold"),
    strip.background = element_blank(),
    panel.grid  = element_blank(),
    legend.position = "bottom",
    plot.margin = ggplot2::margin(4, 6, 4, 6)
  )

# Combine panels with shared legend
p_split + plot_layout(guides = "collect") & theme(legend.position = "bottom")

# Save plot
#ggsave("MSE_heat_v2.pdf", width = 8.5, height = 6, dpi = 500)

