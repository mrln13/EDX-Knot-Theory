# ============================================================
# EWMA-CLR Convergence Sensitivity Analysis
#
# Implements the Monte Carlo sensitivity study described in the manuscript
# (Section "Sensitivity Analysis of the Convergence Criterion"):
#   - N = 81 hyperparameter sets (baseline + 80 randomly sampled)
#   - n = 320 starting positions per set (stratified over r/c bins)
#   - analysis horizon X_max = 300 tiles
#
# Inputs: aggregate JSON files produced by analyze_orders.py for the RASTER
# scan pattern (one JSON per dataset).
# Outputs (written to OUT_DIR):
#   - sensitivity_results.csv              per-parameter-set summary
#   - paramset_convergence_summary.csv     per dataset x param_id
#   - parameter_importance_tstats_by_dataset.csv
#   - baseline_subset_ewma_clr_ribbons.png
#
# Run with Rscript; edit the USER SETTINGS block below before first use.
# ============================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(readr)
  library(ggplot2)
})

# ---------------------------
# USER SETTINGS
# ---------------------------

# Named vector of aggregate JSON files produced by analyze_orders.py.
# The sensitivity analysis is defined on the RASTER scan pattern (see
# manuscript), so each path should point to a raster-pattern JSON.
# The vector names ("Diorite", "Schist") become dataset labels in the output.
AGGREGATE_JSONS <- c(
  Diorite = "path/to/Diorite_Raster.json",
  Schist  = "path/to/Schist_Raster.json"
)

OUT_DIR <- getwd()
DISPLAY_PLOTS <- TRUE

# Subset selection (stratified over r/c bins), applied PER DATASET
SET_SEED <- 1
BX <- 8
BY <- 8
PER_BIN <- 5

# Analysis horizon
X_MAX <- 300

# Major elements handling
major_elems <- c("O","Si","Al","Fe","Ca","Mg","Na","K","Ti")
k_majors <- 9

# Sensitivity design
N_PARAM_SETS <- 80

CHOICES_lambda <- c(0.05, 0.10, 0.20, 0.30, 0.40, 0.50)

R_theta_med    <- c(0.01, 0.08)
R_theta_spread <- c(0.01, 0.10)
R_theta_slope  <- c(1e-4, 2e-3)  # log-uniform

CHOICES_w_min      <- c(10, 20, 30)
CHOICES_w_max      <- c(60, 100, 140)
CHOICES_alpha_grow <- c(1.03, 1.06, 1.08, 1.10)
CHOICES_L_ok       <- c(10, 15, 20, 30)

BASELINE_PARAMS <- list(
  lambda = 0.30,
  w_min = 20, w_max = 100, alpha_grow = 1.08, L_ok = 20,
  theta_med_base = 0.03, theta_spread_base = 0.03, theta_slope_base = 0.0005
)

WRITE_CACHE_RDS <- TRUE
CACHE_RDS_NAME <- "subset_Yclr_cache.rds"

WRITE_RESULTS_CSV <- TRUE
RESULTS_CSV_NAME <- "sensitivity_results.csv"

DO_IMPACT_REPORT <- TRUE

MAKE_BASELINE_PLOT <- TRUE
BASELINE_PLOT_NAME <- "baseline_subset_ewma_clr_ribbons.png"
BINWIDTH <- 10
Y_MAX <- 0.75
HIST_MAX_FRAC <- 0.5

# ---------------------------
# HELPERS
# ---------------------------
file_stem <- function(path_or_name) {
  b <- basename(path_or_name)
  sub("\\.[^.]*$", "", b)
}

extract_rc <- function(path_or_name) {
  stem <- file_stem(as.character(path_or_name))
  m <- str_match(stem, "r(\\d+)_c(\\d+)$")
  if (any(is.na(m))) m <- str_match(stem, "r(\\d+)_c(\\d+)")
  tibble(stem = stem, r = as.integer(m[2]), c = as.integer(m[3]))
}

closure_vec <- function(x) {
  x <- as.numeric(pmax(0, x))
  s <- sum(x)
  if (!is.finite(s) || s <= 0) rep(1/length(x), length(x)) else x / s
}

clr_vec <- function(p) {
  eps <- 1e-12
  p <- pmax(eps, p)
  gm <- exp(mean(log(p)))
  log(p / gm)
}

ewma_clr <- function(Y, lambda) {
  Tn <- nrow(Y); D <- ncol(Y)
  Z  <- matrix(NA_real_, Tn, D)
  d  <- numeric(Tn)
  Z[1, ] <- Y[1, ]
  d[1]   <- 0
  if (Tn >= 2) {
    for (t in 2:Tn) {
      d[t]   <- sqrt(sum((Y[t, ] - Z[t-1, ])^2))
      Z[t, ] <- lambda * Y[t, ] + (1 - lambda) * Z[t-1, ]
    }
  }
  list(d = d, Z = Z)
}

dyn_window_size <- function(t, w_min, w_max, alpha) {
  w <- floor(w_min * (alpha ^ floor(t / w_min)))
  min(w, w_max)
}

slope_fast <- function(y) {
  n <- length(y)
  if (n < 2) return(NA_real_)
  x <- seq_len(n)
  xbar <- (n + 1) / 2
  ybar <- mean(y)
  num <- sum((x - xbar) * (y - ybar))
  den <- sum((x - xbar)^2)
  if (den <= 0) return(NA_real_)
  num / den
}

roll_stats_ok <- function(d, t, w, thr_med, thr_spread, thr_slope) {
  if (t < 2 || t - w + 1 < 1) return(FALSE)
  idx <- (t - w + 1):t
  ww  <- d[idx]
  ww  <- ww[is.finite(ww)]
  if (length(ww) < 2) return(FALSE)
  
  med <- median(ww, na.rm = TRUE)
  p10 <- as.numeric(quantile(ww, 0.10, names = FALSE, na.rm = TRUE))
  p90 <- as.numeric(quantile(ww, 0.90, names = FALSE, na.rm = TRUE))
  spr <- p90 - p10
  
  sl <- slope_fast(ww)
  if (!is.finite(sl)) return(FALSE)
  
  (med < thr_med) && (spr < thr_spread) && (abs(sl) < thr_slope)
}

comp_list_to_matrix <- function(comp_list) {
  elem_union <- sort(unique(unlist(lapply(comp_list, names))))
  if (!length(elem_union)) return(NULL)
  Tn <- length(comp_list); D <- length(elem_union)
  M <- matrix(0.0, nrow = Tn, ncol = D)
  colnames(M) <- elem_union
  for (i in seq_len(Tn)) {
    di <- comp_list[[i]]
    for (j in seq_along(elem_union)) {
      el <- elem_union[j]
      v <- di[[el]]
      if (!is.null(v)) M[i, j] <- as.numeric(v)
    }
  }
  rs <- rowSums(M)
  for (i in seq_len(Tn)) {
    M[i, ] <- if (!is.finite(rs[i]) || rs[i] <= 0) rep(1/ncol(M), ncol(M)) else M[i, ]/rs[i]
  }
  M
}

compute_Yclr_for_run <- function(run_obj, X_MAX) {
  if (is.null(run_obj$file) || is.null(run_obj$composition)) return(NULL)
  
  stem <- file_stem(as.character(run_obj$file))
  rc <- extract_rc(stem)
  
  M <- comp_list_to_matrix(run_obj$composition)
  if (is.null(M) || nrow(M) == 0) return(NULL)
  
  if (is.null(major_elems)) {
    means <- colMeans(M, na.rm = TRUE)
    ord_ix <- order(means, decreasing = TRUE)
    majors <- colnames(M)[ord_ix][seq_len(min(k_majors, length(means)))]
  } else {
    majors <- intersect(major_elems, colnames(M))
    if (!length(majors)) {
      means <- colMeans(M, na.rm = TRUE)
      ord_ix <- order(means, decreasing = TRUE)
      majors <- colnames(M)[ord_ix][seq_len(min(k_majors, length(means)))]
    }
  }
  
  other <- pmax(0, 1 - rowSums(M[, majors, drop = FALSE]))
  G <- cbind(M[, majors, drop = FALSE], Other = other)
  
  S <- rowSums(G); S[!is.finite(S) | S <= 0] <- 1
  G <- sweep(G, 1, S, "/")
  
  Y_clr <- t(apply(G, 1, function(p) clr_vec(closure_vec(p))))
  if (nrow(Y_clr) > X_MAX) Y_clr <- Y_clr[seq_len(X_MAX), , drop = FALSE]
  
  tibble(file = stem, r = rc$r, c = rc$c, Y = list(Y_clr))
}

conv_from_d <- function(d, w_min, w_max, alpha_grow,
                        theta_med_base, theta_spread_base, theta_slope_base, L_ok) {
  if (!length(d)) return(NA_integer_)
  
  base_window <- min(w_min, length(d))
  d0 <- d[seq_len(base_window)]
  d0 <- d0[is.finite(d0)]
  if (!length(d0)) d0 <- 0
  
  base_med    <- median(d0, na.rm = TRUE)
  base_spread <- as.numeric(quantile(d0, 0.9, na.rm = TRUE) -
                              quantile(d0, 0.1, na.rm = TRUE))
  
  thr_med    <- max(theta_med_base,    0.5 * base_med)
  thr_spread <- max(theta_spread_base, 0.5 * base_spread)
  thr_slope  <- theta_slope_base
  
  streak <- 0L
  for (t in seq_along(d)) {
    w <- dyn_window_size(t, w_min, w_max, alpha_grow)
    ok <- roll_stats_ok(d, t, w, thr_med, thr_spread, thr_slope)
    if (ok) {
      streak <- streak + 1L
      if (streak >= L_ok) return(as.integer(t - L_ok + 1L))
    } else {
      streak <- 0L
    }
  }
  NA_integer_
}

conv_from_Y <- function(Y, lambda,
                        w_min, w_max, alpha_grow,
                        theta_med_base, theta_spread_base, theta_slope_base, L_ok) {
  d <- ewma_clr(Y, lambda)$d
  if (length(d) > X_MAX) d <- d[seq_len(X_MAX)]
  conv_from_d(d, w_min, w_max, alpha_grow, theta_med_base, theta_spread_base, theta_slope_base, L_ok)
}

d_from_Y <- function(Y, lambda) {
  d <- ewma_clr(Y, lambda)$d
  if (length(d) > X_MAX) d <- d[seq_len(X_MAX)]
  d
}

save_plot <- function(p, filename, w = 7.5, h = 5.2) {
  if (DISPLAY_PLOTS) print(p)
  ggsave(file.path(OUT_DIR, filename), p, width = w, height = h, dpi = 300)
  cat("Wrote: ", file.path(OUT_DIR, filename), "\n", sep = "")
}

# ---------------------------
# LOAD + SUBSET PER DATASET, THEN COMBINE
# ---------------------------

### NEW/CHANGED: helper to load+subset one dataset ###
load_subset_cache_one <- function(dataset_name, json_path) {
  cat("\n=== Dataset: ", dataset_name, " ===\n", sep = "")
  cat("Reading JSON: ", json_path, "\n", sep = "")
  
  agg <- tryCatch(fromJSON(json_path, simplifyVector = FALSE),
                  error = function(e) stop("Failed to read ", json_path, ": ", e$message))
  if (is.null(agg$runs) || !length(agg$runs)) stop("Aggregated JSON has no 'runs': ", json_path)
  
  cat("Building run metadata...\n")
  run_meta <- map_dfr(agg$runs, \(r) tibble(file = as.character(r$file))) %>%
    mutate(row = row_number()) %>%
    mutate(tmp = map(file, extract_rc)) %>%
    unnest(tmp)
  
  if (any(!is.finite(run_meta$r)) || any(!is.finite(run_meta$c))) {
    bad <- sum(!is.finite(run_meta$r) | !is.finite(run_meta$c))
    stop("Failed to parse r/c for ", bad, " runs in dataset ", dataset_name)
  }
  
  set.seed(SET_SEED)
  meta2 <- run_meta %>%
    mutate(
      rbin = ntile(r, BX),
      cbin = ntile(c, BY),
      bin  = paste0(rbin, "_", cbin)
    )
  
  sub_rows <- meta2 %>%
    group_by(bin) %>%
    group_modify(~ .x %>% slice_sample(n = min(PER_BIN, nrow(.x)))) %>%
    ungroup() %>%
    pull(row)
  
  runs_sub <- agg$runs[sub_rows]
  cat("Subset size: ", length(runs_sub),
      " runs (target approx ", BX * BY * PER_BIN, ")\n", sep = "")
  
  cat("Computing per-run CLR series (cached)...\n")
  cache_df <- map_dfr(runs_sub, compute_Yclr_for_run, X_MAX = X_MAX)
  if (!nrow(cache_df)) stop("No valid runs produced CLR series in dataset ", dataset_name)
  
  cache_df %>%
    mutate(dataset = dataset_name) %>%
    select(dataset, everything())
}

cat("Loading all datasets...\n")
cache_df <- purrr::imap_dfr(AGGREGATE_JSONS, ~ load_subset_cache_one(.y, .x))

if (WRITE_CACHE_RDS) {
  saveRDS(cache_df, file.path(OUT_DIR, CACHE_RDS_NAME))
  cat("\nWrote cache: ", file.path(OUT_DIR, CACHE_RDS_NAME), "\n", sep = "")
}

# Combined lists, but dataset label retained
Y_list <- cache_df$Y
n_runs_total <- nrow(cache_df)
cat("\nTotal subset runs across datasets: ", n_runs_total, "\n", sep = "")

# ---------------------------
# PARAMETER SETS (baseline + random)
# ---------------------------
cat("\nGenerating parameter sets...\n")
set.seed(SET_SEED + 1)

logu <- function(n, lo, hi) exp(runif(n, log(lo), log(hi)))

param_sets <- tibble(
  param_id = seq_len(N_PARAM_SETS),
  lambda = sample(CHOICES_lambda, N_PARAM_SETS, replace = TRUE),
  theta_med_base    = runif(N_PARAM_SETS, R_theta_med[1],    R_theta_med[2]),
  theta_spread_base = runif(N_PARAM_SETS, R_theta_spread[1], R_theta_spread[2]),
  theta_slope_base  = logu(N_PARAM_SETS, R_theta_slope[1],   R_theta_slope[2]),
  w_min      = sample(CHOICES_w_min,      N_PARAM_SETS, replace = TRUE),
  w_max      = sample(CHOICES_w_max,      N_PARAM_SETS, replace = TRUE),
  alpha_grow = sample(CHOICES_alpha_grow, N_PARAM_SETS, replace = TRUE),
  L_ok       = sample(CHOICES_L_ok,       N_PARAM_SETS, replace = TRUE)
) %>%
  mutate(w_max = pmax(w_max, w_min))

baseline_row <- tibble(
  param_id = 0L,
  lambda = BASELINE_PARAMS$lambda,
  theta_med_base = BASELINE_PARAMS$theta_med_base,
  theta_spread_base = BASELINE_PARAMS$theta_spread_base,
  theta_slope_base = BASELINE_PARAMS$theta_slope_base,
  w_min = BASELINE_PARAMS$w_min,
  w_max = BASELINE_PARAMS$w_max,
  alpha_grow = BASELINE_PARAMS$alpha_grow,
  L_ok = BASELINE_PARAMS$L_ok
)

param_sets <- bind_rows(baseline_row, param_sets) %>% arrange(param_id)

# ---------------------------
# SENSITIVITY SUMMARY (per parameter set AND per dataset)
# ---------------------------
cat("\nEvaluating sensitivity summaries...\n")

### NEW/CHANGED: returns per-dataset summaries ###
eval_one <- function(ps_row) {
  conv <- vapply(
    Y_list,
    function(Y) conv_from_Y(
      Y = Y, lambda = ps_row$lambda,
      w_min = ps_row$w_min, w_max = ps_row$w_max, alpha_grow = ps_row$alpha_grow,
      theta_med_base = ps_row$theta_med_base,
      theta_spread_base = ps_row$theta_spread_base,
      theta_slope_base = ps_row$theta_slope_base,
      L_ok = ps_row$L_ok
    ),
    integer(1)
  )
  
  tmp <- cache_df %>%
    transmute(dataset, conv_step = conv) %>%
    mutate(conv_ok = !is.na(conv_step) & is.finite(conv_step) & conv_step <= X_MAX)
  
  tmp %>%
    group_by(dataset) %>%
    summarise(
      param_id = ps_row$param_id,
      n_runs = dplyr::n(),
      converged = sum(conv_ok),
      conv_rate = converged / n_runs,
      conv_med = if (sum(conv_ok) > 0) median(conv_step[conv_ok]) else NA_real_,
      conv_p90 = if (sum(conv_ok) > 0) as.numeric(quantile(conv_step[conv_ok], 0.9, names = FALSE)) else NA_real_,
      .groups = "drop"
    )
}

results <- map_dfr(split(param_sets, param_sets$param_id), eval_one) %>%
  left_join(param_sets, by = "param_id") %>%
  arrange(dataset, desc(conv_rate), conv_med)

if (WRITE_RESULTS_CSV) {
  write_csv(results, file.path(OUT_DIR, RESULTS_CSV_NAME))
  cat("Wrote results: ", file.path(OUT_DIR, RESULTS_CSV_NAME), "\n", sep = "")
}

cat("\nTop parameter sets per dataset:\n")
print(results %>%
        group_by(dataset) %>%
        arrange(desc(conv_rate), conv_med) %>%
        slice_head(n = 8) %>%
        ungroup() %>%
        select(dataset, param_id, conv_rate, conv_med, conv_p90, lambda,
               theta_med_base, theta_spread_base, theta_slope_base,
               w_min, w_max, alpha_grow, L_ok))

# ---------------------------
# OPTIONAL: BASELINE PLOT (faceted by dataset)
# ---------------------------
if (MAKE_BASELINE_PLOT) {
  cat("\nBuilding baseline plot (faceted by dataset)...\n")
  
  baseline_conv <- vapply(
    Y_list,
    function(Y) conv_from_Y(
      Y = Y, lambda = BASELINE_PARAMS$lambda,
      w_min = BASELINE_PARAMS$w_min, w_max = BASELINE_PARAMS$w_max, alpha_grow = BASELINE_PARAMS$alpha_grow,
      theta_med_base = BASELINE_PARAMS$theta_med_base,
      theta_spread_base = BASELINE_PARAMS$theta_spread_base,
      theta_slope_base = BASELINE_PARAMS$theta_slope_base,
      L_ok = BASELINE_PARAMS$L_ok
    ),
    integer(1)
  )
  
  df_all <- cache_df %>%
    mutate(d = map(Y, ~ d_from_Y(.x, BASELINE_PARAMS$lambda))) %>%
    select(dataset, file, d) %>%
    unnest_longer(d, indices_to = "step", values_to = "dist") %>%
    mutate(step = as.integer(step)) %>%
    filter(step <= X_MAX)
  
  summ <- df_all %>%
    group_by(dataset, step) %>%
    summarise(
      min_v = {v <- dist[is.finite(dist)]; if (length(v)) min(v) else NA_real_},
      max_v = {v <- dist[is.finite(dist)]; if (length(v)) max(v) else NA_real_},
      p10   = {v <- dist[is.finite(dist)]; if (length(v)) as.numeric(quantile(v, 0.10, names = FALSE)) else NA_real_},
      med   = {v <- dist[is.finite(dist)]; if (length(v)) median(v) else NA_real_},
      p90   = {v <- dist[is.finite(dist)]; if (length(v)) as.numeric(quantile(v, 0.90, names = FALSE)) else NA_real_},
      .groups = "drop"
    )
  
  conv_df <- cache_df %>%
    transmute(dataset, file, conv_step = baseline_conv) %>%
    filter(!is.na(conv_step), is.finite(conv_step), conv_step <= X_MAX)
  
  # Histogram scaling per dataset
  htab <- conv_df %>%
    mutate(bin = floor(conv_step / BINWIDTH) * BINWIDTH) %>%
    count(dataset, bin, name = "count")
  
  max_count <- htab %>%
    group_by(dataset) %>%
    summarise(max_count = max(count), .groups = "drop")
  
  # For plotting: scale each dataset separately using after_stat(..count..) * scale_y
  # We'll approximate with a single scale_y based on global max (keeps comparable)
  max_count_global <- max(htab$count, na.rm = TRUE)
  if (!is.finite(max_count_global) || max_count_global <= 0) max_count_global <- 1
  scale_y <- (Y_MAX * HIST_MAX_FRAC) / max_count_global
  
  p <- ggplot() +
    stat_bin(
      data = conv_df,
      aes(x = pmin(conv_step, X_MAX), y = after_stat(..count.. * scale_y)),
      binwidth = BINWIDTH, boundary = 0, closed = "left",
      fill = "grey80", alpha = 0.35
    ) +
    stat_bin(
      data = conv_df,
      aes(x = pmin(conv_step, X_MAX), y = after_stat(..count.. * scale_y)),
      binwidth = BINWIDTH, boundary = 0, closed = "left",
      fill = NA, color = "black", linewidth = 0.4
    ) +
    geom_ribbon(data = summ, aes(x = step, ymin = min_v, ymax = max_v),
                fill = "grey85", alpha = 0.35) +
    geom_ribbon(data = summ, aes(x = step, ymin = p10, ymax = p90),
                fill = "grey55", alpha = 0.35) +
    geom_line(data = summ, aes(x = step, y = med),
              linewidth = 0.9, color = "black") +
    coord_cartesian(xlim = c(0, X_MAX), ylim = c(0, Y_MAX)) +
    scale_y_continuous(
      name     = "Distance to EWMA state (clr space)",
      sec.axis = sec_axis(~ . / scale_y, name = "Runs converged")
    ) +
    labs(x = "Tile (step)",
         title = bquote("Baseline (" * lambda == .(BASELINE_PARAMS$lambda) * ")")) +
    facet_wrap(~dataset, ncol = 1, scales = "free_y") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none", plot.margin = margin(6, 12, 6, 12))
  
  save_plot(p, BASELINE_PLOT_NAME, w = 10, h = 7.5)
}

# ---------------------------
# IMPACT REPORT (per-run × param-set, dataset-aware)
# ---------------------------
if (DO_IMPACT_REPORT) {
  cat("\nComputing full run × parameter-set conv_step table (impact report)...\n")
  
  run_tbl <- cache_df %>%
    group_by(dataset) %>%
    mutate(run_id = row_number()) %>%
    ungroup() %>%
    mutate(run_uid = paste(dataset, run_id, sep = ":"))
  
  conv_long <- purrr::map_dfr(seq_len(nrow(param_sets)), function(i) {
    ps <- param_sets[i, ]
    
    conv <- vapply(
      Y_list,
      function(Y) conv_from_Y(
        Y = Y, lambda = ps$lambda,
        w_min = ps$w_min, w_max = ps$w_max, alpha_grow = ps$alpha_grow,
        theta_med_base = ps$theta_med_base,
        theta_spread_base = ps$theta_spread_base,
        theta_slope_base = ps$theta_slope_base,
        L_ok = ps$L_ok
      ),
      integer(1)
    )
    
    tibble(
      param_id  = ps$param_id,
      dataset   = run_tbl$dataset,
      run_id    = run_tbl$run_id,
      run_uid   = run_tbl$run_uid,
      file      = run_tbl$file,
      r         = run_tbl$r,
      c         = run_tbl$c,
      conv_step = conv
    )
  }) %>%
    left_join(param_sets, by = "param_id") %>%
    mutate(conv_step_cens = if_else(is.na(conv_step), as.integer(X_MAX + 1L), as.integer(conv_step)))
  
  write_csv(conv_long, file.path(OUT_DIR, "conv_steps_long.csv.gz"))
  cat("Wrote: ", file.path(OUT_DIR, "conv_steps_long.csv.gz"), "\n", sep = "")
  
  byset <- conv_long %>%
    group_by(dataset, param_id) %>%
    summarise(
      n_runs        = dplyr::n(),
      converged     = sum(!is.na(conv_step) & conv_step <= X_MAX),
      conv_rate     = converged / n_runs,
      conv_med      = if (sum(!is.na(conv_step)) > 0) median(conv_step, na.rm = TRUE) else NA_real_,
      conv_p90      = if (sum(!is.na(conv_step)) > 0) as.numeric(quantile(conv_step, 0.9, na.rm = TRUE, names = FALSE)) else NA_real_,
      conv_med_cens = median(conv_step_cens),
      .groups = "drop"
    ) %>%
    left_join(param_sets, by = "param_id") %>%
    arrange(dataset, desc(conv_rate), conv_med_cens)
  
  write_csv(byset, file.path(OUT_DIR, "paramset_convergence_summary.csv"))
  cat("Wrote: ", file.path(OUT_DIR, "paramset_convergence_summary.csv"), "\n", sep = "")
  
  # Baseline delta (within each dataset + run)
  base_per_run <- conv_long %>%
    filter(param_id == 0L) %>%
    select(dataset, run_uid, base_cens = conv_step_cens)
  
  delta_long <- conv_long %>%
    left_join(base_per_run, by = c("dataset", "run_uid")) %>%
    mutate(delta_step = conv_step_cens - base_cens)
  
  delta_byset <- delta_long %>%
    group_by(dataset, param_id) %>%
    summarise(
      delta_med = median(delta_step),
      delta_p10 = as.numeric(quantile(delta_step, 0.10, names = FALSE)),
      delta_p90 = as.numeric(quantile(delta_step, 0.90, names = FALSE)),
      .groups = "drop"
    ) %>%
    left_join(param_sets, by = "param_id") %>%
    left_join(byset %>% select(dataset, param_id, conv_rate, conv_med_cens), by = c("dataset", "param_id")) %>%
    arrange(dataset, delta_med)
  
  write_csv(delta_byset, file.path(OUT_DIR, "baseline_delta_summary.csv"))
  cat("Wrote: ", file.path(OUT_DIR, "baseline_delta_summary.csv"), "\n", sep = "")
  
  # ---------------------------
  # PUBLICATION-READY LABELS
  # ---------------------------
  lab_lambda <- expression(lambda)
  lab_theta_med    <- expression(theta[med])
  lab_theta_spread <- expression(theta[spread])
  lab_theta_slope  <- expression(theta[slope])
  lab_Lok   <- expression(L[ok])
  lab_wmin  <- expression(w[min])
  lab_wmax  <- expression(w[max])
  lab_alpha <- expression(alpha)
  
  y_med_cens <- "Median convergence step (censored)"
  y_delta    <- "Δ median convergence step vs baseline"
  
  # ---------------------------
  # EFFECT PLOTS (both datasets together)
  # ---------------------------
  theme_minimal(base_size = 14)  
  
  pL <- ggplot(byset, aes(x = factor(lambda), y = conv_med_cens, fill = dataset)) +
    geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.8)) +
    labs(x = lab_lambda, y = y_med_cens, fill = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(pL, "effect_lambda.png")
  ggsave("lambda.pdf", pL, width = 6, height = 4)
  
  p1 <- ggplot(byset, aes(x = theta_med_base, y = conv_med_cens, color = dataset)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "loess", se = FALSE) +
    labs(x = lab_theta_med, y = y_med_cens, color = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(p1, "effect_theta_med.png")
  
  p2 <- ggplot(byset, aes(x = theta_spread_base, y = conv_med_cens, color = dataset)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "loess", se = FALSE) +
    labs(x = lab_theta_spread, y = y_med_cens, color = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(p2, "effect_theta_spread.png")
  
  p3 <- ggplot(byset, aes(x = theta_slope_base, y = conv_med_cens, color = dataset)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "loess", se = FALSE) +
    scale_x_log10() +
    labs(x = as.expression(lab_theta_slope),
         y = y_med_cens, color = "Dataset") +
    theme_minimal(base_size = 12)
  # Put the math symbol in subtitle (keeps it clean with log note)
  #p3 <- p3 + labs(subtitle = as.expression(lab_theta_slope))
  save_plot(p3, "effect_theta_slope.png")
  ggsave("theta_slope.pdf", p3, width = 6, height = 4)
  
  p4 <- ggplot(byset, aes(x = factor(L_ok), y = conv_med_cens, fill = dataset)) +
    geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.8)) +
    labs(x = lab_Lok, y = y_med_cens, fill = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(p4, "effect_L_ok.png")
  
  p5 <- ggplot(byset, aes(x = factor(w_min), y = conv_med_cens, fill = dataset)) +
    geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.8)) +
    labs(x = lab_wmin, y = y_med_cens, fill = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(p5, "effect_w_min.png")
  
  p6 <- ggplot(byset, aes(x = factor(w_max), y = conv_med_cens, fill = dataset)) +
    geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.8)) +
    labs(x = lab_wmax, y = y_med_cens, fill = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(p6, "effect_w_max.png")
  
  p7 <- ggplot(byset, aes(x = factor(alpha_grow), y = conv_med_cens, fill = dataset)) +
    geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.8)) +
    labs(x = lab_alpha, y = y_med_cens, fill = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(p7, "effect_alpha.png")
  
  # ---------------------------
  # DELTA PLOTS (both datasets together)
  # ---------------------------
  
  pdL_scatter <- ggplot(delta_byset %>% filter(param_id != 0L),
                        aes(x = lambda, y = delta_med, color = dataset)) +
    geom_point(alpha = 0.6,
               position = position_jitter(width = 0.01, height = 0)) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    geom_smooth(method = "loess", se = FALSE) +
    labs(x = lab_lambda, y = y_delta, color = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(pdL_scatter, "delta_vs_lambda_scatter_loess.png")
  
  pd1 <- ggplot(delta_byset %>% filter(param_id != 0L),
                aes(x = theta_med_base, y = delta_med, color = dataset)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    labs(x = lab_theta_med, y = y_delta, color = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(pd1, "delta_vs_theta_med.png")
  
  pd2 <- ggplot(delta_byset %>% filter(param_id != 0L),
                aes(x = factor(L_ok), y = delta_med, fill = dataset)) +
    geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.8)) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    labs(x = lab_Lok, y = y_delta, fill = "Dataset") +
    theme_minimal(base_size = 12)
  save_plot(pd2, "delta_vs_L_ok.png")
  
  # ---------------------------
  # IMPORTANCE (separate model per dataset; avoids mixing scales)
  # ---------------------------
  cat("\nFitting separate linear models per dataset for rough importance...\n")
  
  imp_all <- byset %>%
    group_by(dataset) %>%
    group_modify(~{
      m <- lm(
        conv_med_cens ~ factor(lambda) + theta_med_base + theta_spread_base + log(theta_slope_base) +
          factor(w_min) + factor(w_max) + factor(alpha_grow) + factor(L_ok),
        data = .x
      )
      coef_tab <- summary(m)$coefficients
      tibble(
        term = rownames(coef_tab),
        estimate = coef_tab[, "Estimate"],
        t_value  = coef_tab[, "t value"],
        abs_t    = abs(coef_tab[, "t value"])
      ) %>% arrange(desc(abs_t))
    }) %>%
    ungroup()
  
  write_csv(imp_all, file.path(OUT_DIR, "parameter_importance_tstats_by_dataset.csv"))
  cat("Wrote: ", file.path(OUT_DIR, "parameter_importance_tstats_by_dataset.csv"), "\n", sep = "")
  
  cat("\nTop terms by |t| per dataset:\n")
  print(imp_all %>% group_by(dataset) %>% slice_head(n = 10))
}

cat("\nAll done.\n")
