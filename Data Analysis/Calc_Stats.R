# ============================================================
# EWMA–CLR convergence (aggregated ribbons + conv-step histogram)
# - Input: single aggregated JSON: { "runs": [ {file, composition:[{El:frac,...}, ...]}, ... ] }
# - Output: one combined figure with min–max & P10–P90 ribbons, median line,
#           and a compact histogram of convergence steps (no legend, no titles)
# ============================================================

# -------- user settings --------
AGGREGATE_JSON    <- "path/to/file.json"   # Data produced by Python batch processing script
SAVE_PNG_COMBINED <- FALSE
COMBINED_PNG_NAME <- "output.png"

# majors: if NULL, pick top-k per run by mean fraction
major_elems  <- c("O","Si","Al","Fe","Ca","Mg","Na","K","Ti")   # set to NULL to auto-pick top-k
k_majors     <- 10

# EWMA + window params
lambda       <- 0.30          # EWMA smoothing factor (0<lambda<=1)
w_min        <- 20            # minimum rolling window length
w_max        <- 100           # maximum rolling window length
alpha_grow   <- 1.08          # window growth factor
L_ok         <- 20            # require L_ok consecutive OK windows for convergence

# thresholds (EWMA–CLR scale; adapt if needed)
theta_med_base    <- 0.03
theta_spread_base <- 0.03
theta_slope_base  <- 0.0005

# write per-run CSV summaries/time series?
WRITE_PER_RUN_CSV <- FALSE  # FALSE when summarizing many runs

# -------- libraries --------
suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(purrr)
  library(tibble)
  library(cowplot)  
})

# -------- helpers --------
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

roll_stats_ok <- function(d, t, w, thr_med, thr_spread, thr_slope) {
  # guard: enough history?
  if (t < 2 || t - w + 1 < 1) return(FALSE)
  idx <- (t - w + 1):t
  ww  <- d[idx]
  # remove non-finite values
  ww  <- ww[is.finite(ww)]
  if (length(ww) < 2) return(FALSE)
  
  med <- median(ww, na.rm = TRUE)
  p10 <- as.numeric(quantile(ww, 0.10, names = FALSE, na.rm = TRUE))
  p90 <- as.numeric(quantile(ww, 0.90, names = FALSE, na.rm = TRUE))
  spr <- p90 - p10
  
  x   <- seq_along(ww)
  # linear slope; if lm fails (rare), treat as non-OK
  sl <- tryCatch(as.numeric(coef(lm(ww ~ x))[2]), error = function(e) NA_real_)
  if (!is.finite(sl)) return(FALSE)
  
  (med < thr_med) && (spr < thr_spread) && (abs(sl) < thr_slope)
}

first_run_of_true <- function(x, run_len) {
  if (length(x) == 0) return(NA_integer_)
  r <- rle(x)
  ends <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1
  i <- which(r$values & r$lengths >= run_len)
  if (length(i) == 0) return(NA_integer_)
  starts[i[1]]
}

file_stem <- function(path_or_name) {
  b <- basename(path_or_name)
  sub("\\.[^.]*$", "", b)
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
cat("\nCreated helpers\n")
# -------- per-run processor → returns per-step distances + conv step --------
process_run <- function(run_obj) {
  if (is.null(run_obj$file) || is.null(run_obj$composition)) return(NULL)
  stem <- file_stem(as.character(run_obj$file))
  M <- comp_list_to_matrix(run_obj$composition)
  if (is.null(M) || nrow(M) == 0) return(NULL)
  
  # majors
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
  
  # re-closure
  S <- rowSums(G); S[!is.finite(S) | S <= 0] <- 1
  G <- sweep(G, 1, S, "/")
  
  # CLR + EWMA
  Y_clr <- t(apply(G, 1, function(p) clr_vec(closure_vec(p))))
  ew <- ewma_clr(Y_clr, lambda = lambda)
  d  <- ew$d
  
  # thresholds from early window (clean non-finites)
  base_window <- min(w_min, length(d))
  d0 <- d[seq_len(base_window)]
  d0 <- d0[is.finite(d0)]
  if (!length(d0)) d0 <- 0
  
  base_med    <- median(d0, na.rm = TRUE)
  base_spread <- as.numeric(quantile(d0, 0.9, na.rm = TRUE) -
                              quantile(d0, 0.1, na.rm = TRUE))
  theta_med    <- max(theta_med_base,    0.5 * base_med)
  theta_spread <- max(theta_spread_base, 0.5 * base_spread)
  theta_slope  <- theta_slope_base
  
  ok <- logical(length(d))
  for (t in seq_along(d)) {
    w <- dyn_window_size(t, w_min, w_max, alpha_grow)
    ok[t] <- roll_stats_ok(d, t, w, theta_med, theta_spread, theta_slope)
  }
  conv_step <- first_run_of_true(ok, L_ok)
  
  tibble(file = stem, step = seq_along(d), dist = d, ok = ok,
         conv_step = if (is.na(conv_step)) NA_integer_ else conv_step)
}
cat("\nCreated main processor\n")
# -------- read aggregated JSON --------
agg <- tryCatch(fromJSON(AGGREGATE_JSON, simplifyVector = FALSE),
                error = function(e) { stop("Failed to read ", AGGREGATE_JSON, ": ", e$message) })
if (is.null(agg$runs) || !length(agg$runs)) stop("Aggregated JSON has no 'runs'.")
cat("\nJSON read\n")
# -------- run all --------
results <- map(agg$runs, process_run)
results <- results[!vapply(results, is.null, logical(1))]
if (!length(results)) stop("No valid runs processed.")
df_all <- bind_rows(results)
cat("\nAggregated results\n")


# ----- display controls -----
X_MAX     <- 300     # show up to this many tiles on x-axis
Y_MAX     <- 0.75    # primary y max (EWMA–CLR distance)
BINWIDTH  <- 10      # tiles for histogram bin width
HIST_MAX_FRAC <- 0.5 # fraction of primary y-range the histogram can occupy (visual height)

#---------calculate confidence intervals---------
summ <- df_all %>%
  dplyr::filter(step <= X_MAX) %>%
  dplyr::group_by(step) %>%
  dplyr::summarise(
    n     = sum(is.finite(dist)),
    min_v = {v <- dist[is.finite(dist)]; if (length(v)) min(v) else NA_real_},
    max_v = {v <- dist[is.finite(dist)]; if (length(v)) max(v) else NA_real_},
    p10   = {v <- dist[is.finite(dist)]; if (length(v)) as.numeric(quantile(v, 0.10, names = FALSE)) else NA_real_},
    med   = {v <- dist[is.finite(dist)]; if (length(v)) median(v) else NA_real_},
    p90   = {v <- dist[is.finite(dist)]; if (length(v)) as.numeric(quantile(v, 0.90, names = FALSE)) else NA_real_},
    .groups = "drop"
  )

# ----- convergence steps (per run), truncated to X_MAX -----
conv_df <- df_all %>%
  dplyr::group_by(file) %>%
  dplyr::summarise(conv_step = unique(na.omit(conv_step))[1], .groups = "drop") %>%
  dplyr::filter(!is.na(conv_step), conv_step <= X_MAX)

# Build histogram (bin convergence steps in [0, X_MAX])
htab <- if (nrow(conv_df) > 0) {
  conv_df %>%
    dplyr::filter(conv_step >= 0, conv_step <= X_MAX) %>%
    dplyr::mutate(bin = floor(conv_step / BINWIDTH) * BINWIDTH) %>%
    dplyr::count(bin, name = "count") %>%
    dplyr::mutate(
      xmin  = bin,
      xmax  = bin + BINWIDTH,
      xmid  = bin + BINWIDTH / 2
    )
} else {
  tibble::tibble(xmin = numeric(0), xmax = numeric(0), xmid = numeric(0), count = numeric(0))
}

# Scale histogram counts into the primary y-range [0, Y_MAX]
max_count <- if (nrow(htab) > 0) max(htab$count, na.rm = TRUE) else 1
if (!is.finite(max_count) || max_count <= 0) max_count <- 1

scale_y <- (Y_MAX * HIST_MAX_FRAC) / max_count  # sec.axis mapping: count = y / scale_y

# ----- build plot (no data capping; visual clip only via coord_cartesian) -----
p <- ggplot() +
  # Histogram (filled) scaled to primary axis
  stat_bin(
    data = conv_df,
    aes(x = pmin(conv_step, X_MAX), y = after_stat(..count.. * scale_y)),
    binwidth = BINWIDTH, boundary = 0, closed = "left",
    fill = "grey80", alpha = 0.35
  ) +
  # Histogram outline
  stat_bin(
    data = conv_df,
    aes(x = pmin(conv_step, X_MAX), y = after_stat(..count.. * scale_y)),
    binwidth = BINWIDTH, boundary = 0, closed = "left",
    fill = NA, color = "black", linewidth = 0.5
  ) +
  # Min–max ribbon (true stats; *not* capped in data)
  geom_ribbon(
    data = summ,
    aes(x = step, ymin = min_v, ymax = max_v),
    fill = "grey85", alpha = 0.35
  ) +
  # P10–P90 ribbon (true stats)
  geom_ribbon(
    data = summ,
    aes(x = step, ymin = p10, ymax = p90),
    fill = "grey55", alpha = 0.35
  ) +
  # Median line (true stats)
  geom_line(
    data = summ,
    aes(x = step, y = med),
    linewidth = 0.9, color = "black"
  ) +
  # Visual clip only
  coord_cartesian(xlim = c(0, X_MAX), ylim = c(0, Y_MAX)) +
  scale_y_continuous(
    name     = "Distance to EWMA state (clr-space)",
    sec.axis = sec_axis(~ . / scale_y, name = "Runs converged")
  ) +
  labs(x = "Tile (step)") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.margin     = margin(6, 12, 6, 12)
  )

print(p)

COMBINED_PNG_NAME <- "C23S1A_SpiralCCW_edges.pdf"

if (SAVE_PNG_COMBINED) {
  ggsave(COMBINED_PNG_NAME, p, width = 10, height = 5.8, dpi = 300)
  cat("\nCombined plot saved:", file.path(getwd(), COMBINED_PNG_NAME), "\n")
}

cat("\nAll done.\n")
