mse_final <- read.csv("MSE_results.csv", stringsAsFactors = FALSE)

# --- Packages ---
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(patchwork)

# Assumes you already have `mse_final` in your environment

# ---------- 1) Wide → long ----------
mse_long <- mse_final %>%
  pivot_longer(cols = -c(DGP, n, d, rho),
               names_to = "method", values_to = "value") %>%
  mutate(method = case_when(
    method %in% c("C_rho_rhoPerturbed", "C_rhoperturbed") ~ "mu_rho-misspecified",
    TRUE ~ method
  ))

# ---------- 2) Parse "mean ± SD" ----------
split <- str_split_fixed(as.character(mse_long$value), "\\s*(±|\\+/-|\\+-)\\s*", 2)
mse_long <- mse_long %>%
  mutate(
    mean_MSE = suppressWarnings(as.numeric(trimws(split[,1]))),
    sd_MSE   = suppressWarnings(as.numeric(na_if(trimws(split[,2]), "NA"))),
    n   = as.numeric(n),
    d   = as.numeric(d),
    rho = as.numeric(rho)
  )

# ---------- 3) Create TWO μρ columns ----------
mu_two <- mse_long %>%
  filter(method %in% c("C_rho", "mu_rho-misspecified")) %>%
  select(DGP, n, d, rho, method, mean_MSE, sd_MSE) %>%
  mutate(method = recode(method,
                         "C_rho" = "mu_rho (rho_correct)",
                         "mu_rho-misspecified" = "mu_rho (rho_misspec)")) %>%
  mutate(label_str = ifelse(is.na(sd_MSE),
                            sprintf("%.2f±NA", mean_MSE),
                            sprintf("%.2f±%.2f", mean_MSE, sd_MSE))) %>%
  select(DGP, n, d, rho, method, mean_MSE, label_str)

# ---------- 4) Other methods ----------
others <- mse_long %>%
  filter(!method %in% c("C_rho", "mu_rho-misspecified")) %>%
  mutate(label_str = ifelse(is.na(sd_MSE),
                            sprintf("%.2f±NA", mean_MSE),
                            sprintf("%.2f±%.2f", mean_MSE, sd_MSE))) %>%
  select(DGP, n, d, rho, method, mean_MSE, label_str)

mse <- bind_rows(mu_two, others)

# ---------- 5) Keep methods ----------
methods_order <- c("mu_rho (rho_correct)", "mu_rho (rho_misspec)", "DO", "CATE_adj", "matching", "ganite")
rho_grid <- c(0, 0.25, 0.5, 0.75, 1)

mse <- mse %>%
  filter(method %in% methods_order) %>%
  filter(is.na(rho) | rho %in% rho_grid) %>%
  filter(!(DGP == "Synthetic" & n == 500))

# ---------- 6) Scenario labels ----------
fmt_rho <- function(x) {
  ifelse(is.na(x), "NA", formatC(as.numeric(x), format = "f", digits = 2))
}

mse <- mse %>%
  mutate(scenario = sprintf("%s | d=%s, ρ=%s", DGP, d, fmt_rho(rho)))

# ---------- 7) Rank ----------
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

# ---------- 8) Order rows ----------
scenario_meta <- mse %>%
  distinct(scenario, DGP, d, rho) %>%
  mutate(DGP_key = case_when(
    DGP == "Synthetic" ~ 1L,
    DGP == "IHDP"      ~ 2L,
    DGP == "Twins"     ~ 3L,
    TRUE               ~ 99L
  )) %>%
  arrange(DGP_key, d, rho)

ranked$scenario <- factor(ranked$scenario, levels = rev(scenario_meta$scenario))

# ---------- 9) Panel assignment ----------
ranked <- ranked %>%
  mutate(
    panel = if_else(DGP == "Synthetic", "Synthetic", "Real data"),
    panel = factor(panel, levels = c("Synthetic", "Real data"))
  )

scenario_meta_plot <- ranked %>%
  distinct(scenario, DGP, rho, panel) %>%
  mutate(y_index = match(scenario, levels(ranked$scenario))) %>%
  arrange(panel, y_index)

# Separators
dgp_sep <- scenario_meta_plot %>%
  group_by(panel, DGP) %>%
  summarise(last_idx = max(y_index), .groups = "drop") %>%
  transmute(panel, yintercept = last_idx + 0.5) %>%
  group_by(panel) %>%
  filter(yintercept != max(yintercept)) %>%
  ungroup()

rho_sep <- scenario_meta_plot %>%
  group_by(panel, DGP) %>%
  arrange(desc(y_index), .by_group = TRUE) %>%
  mutate(next_rho = lead(rho)) %>%
  filter(rho == 1 & next_rho == 0) %>%
  transmute(panel, yintercept = y_index - 0.5) %>%
  ungroup()

# ---------- 10) Y-axis labels ----------
scenario_levels <- levels(ranked$scenario)
panel_map_df <- ranked %>% distinct(scenario, panel)
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

# ---------- 11) Plot ----------
rank_pal <- c("1 (best)"="#1a9850","2"="#91cf60","3"="#d9ef8b","4"="#fee08b","5 (worst)"="#d73027")
ranked$method <- factor(ranked$method, levels = methods_order)

p_split <- ggplot(ranked, aes(x = method, y = scenario, fill = rank_cat)) +
  geom_tile(color = "white", linewidth = 0.2) +
  geom_hline(data = dgp_sep, aes(yintercept = yintercept), color = "black", linewidth = 1.2) +
  geom_hline(data = rho_sep, aes(yintercept = yintercept), color = "grey40", linewidth = 0.6, linetype = "dashed") +
  geom_text(aes(label = label_str), size = 3.8, lineheight = 0.95) +   # bigger cell text
  scale_x_discrete(labels = function(x) {
    out <- dplyr::case_when(
      x == "mu_rho (rho_correct)" ~ "atop(mu[rho], rho[correct])",
      x == "mu_rho (rho_misspec)" ~ "atop(mu[rho], rho[misspec])",
      x == "DO"                   ~ "atop('DO', scriptstyle(italic('(CQR)')))",
      x == "CATE_adj"             ~ "atop('cate-adj', scriptstyle(italic('(T-learner)')))",
      x == "matching"             ~ "atop('matching', scriptstyle(italic('(Mah. dist.)')))",
      TRUE ~ paste0("'", x, "'")
    )
    parse(text = out)
  }) +
  scale_y_discrete(labels = setNames(parse(text = lab_expr), scenario_levels)) +
  scale_fill_manual(values = rank_pal, name = "Rank (MSE)", drop = FALSE, na.value = "gray80") +
  labs(title = "MSE of Counterfactual Estimators (Ranked)", x = NULL, y = NULL) +
  facet_wrap(~ panel, ncol = 1, scales = "free_y") +  # stacked panels
  theme_minimal(base_size = 11) +
  theme(
    axis.text.y = element_text(size = 11),         # larger row names
    axis.text.x = element_text(size = 11),         # larger column names
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 13, face = "bold"),
    plot.title = element_text(size = 15, face = "bold", margin = ggplot2::margin(b = 6), hjust = 0.5),
    strip.text = element_text(size = 12, face = "bold"),
    strip.background = element_blank(),
    panel.grid  = element_blank(),
    legend.position = "bottom",
    plot.margin = ggplot2::margin(4, 6, 4, 6)
  )

# Shared legend
p_split + plot_layout(guides = "collect") & theme(legend.position = "bottom")


ggsave("mse_heat_figure_with_sd.pdf", width = 8, height = 12, dpi = 500)
