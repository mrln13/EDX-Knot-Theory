# ============================================================
# Big Beautiful Figure: 3 rows x 2 cols (A left, B right)
# - Bottom row has its own histogram scale (fills HIST_MAX_FRAC)
# - Global left y-axis title (cowplot)
# - Global right y-axis title (cowplot) — shown ONCE
# - Per-panel mean (solid) and mean±1 SD (dashed) lines
# - Dataset name labels inside panels REMOVED
# ============================================================

GROUP_A_FILES <- c(
  "path/to/dataset1a.RData",
  "path/to/dataset1b.RData",
  "path/to/dataset1c.RData"
)
GROUP_B_FILES <- c(
  "path/to/dataset2a.RData",
  "path/to/dataset2b.RData",
  "path/to/dataset2c.RData"
)

SAVE_FIGURE <- TRUE
OUT_NAME    <- "BigBeautifulFigure.pdf"

# ----- display controls -----
X_MAX     <- 300
Y_MAX     <- 0.75
BINWIDTH  <- 10
HIST_MAX_FRAC <- 0.5

# -------- libraries --------
suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(purrr); library(tibble)
  library(patchwork)  
  library(cowplot)    
})

# -------- helpers --------
clean_name <- function(x) {
  x <- basename(x)
  x <- sub("\\.RData$", "", x, ignore.case = TRUE)
  x <- sub("_newparams$", "", x, ignore.case = TRUE)
  x
}

load_summ_conv <- function(path, dataset_id = NULL) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  if (!exists("summ", envir = e)) stop("Missing 'summ' in: ", path)
  if (!exists("conv_df", envir = e)) stop("Missing 'conv_df' in: ", path)
  s <- get("summ", envir = e) %>% as_tibble()
  c <- get("conv_df", envir = e) %>% as_tibble()
  req_s <- c("step","min_v","max_v","p10","med","p90")
  if (!all(req_s %in% names(s))) stop("Bad 'summ' columns in: ", path)
  if (!("conv_step" %in% names(c))) stop("Missing 'conv_step' in conv_df: ", path)
  id <- if (is.null(dataset_id)) basename(path) else dataset_id
  s$dataset <- id; c$dataset <- id
  list(summ = s, conv_df = c)
}

# Load group, assign within-group slots (1..3) in the order of files
load_group <- function(files, group_label) {
  lst  <- map(files, ~ load_summ_conv(.x, dataset_id = clean_name(.x)))
  order_tbl <- tibble(dataset = clean_name(files), slot = seq_along(files))
  summ <- bind_rows(map(lst, "summ")) %>%
    left_join(order_tbl, by = "dataset") %>%
    mutate(group = group_label)
  conv <- bind_rows(map(lst, "conv_df")) %>%
    left_join(order_tbl, by = "dataset") %>%
    mutate(group = group_label)
  list(summ = summ, conv = conv, order = order_tbl)
}

ga <- load_group(GROUP_A_FILES, "A")
gb <- load_group(GROUP_B_FILES, "B")

summ_all <- bind_rows(ga$summ, gb$summ) %>% filter(step <= X_MAX)
conv_all <- bind_rows(ga$conv, gb$conv) %>% filter(is.finite(conv_step), conv_step >= 0, conv_step <= X_MAX)

# Ensure slot factor is 1..3 (top to bottom)
slot_levels <- as.character(1:3)
summ_all$slot <- factor(as.character(summ_all$slot), levels = slot_levels)
conv_all$slot <- factor(as.character(conv_all$slot), levels = slot_levels)

# Collapse duplicates per panel (if any)
summ_plot <- summ_all %>%
  group_by(group, slot, step) %>%
  summarise(
    min_v = min(min_v, na.rm = TRUE),
    max_v = max(max_v, na.rm = TRUE),
    p10   = min(p10,   na.rm = TRUE),
    p90   = max(p90,   na.rm = TRUE),
    med   = median(med, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    min_v = ifelse(is.finite(min_v), min_v, NA_real_),
    max_v = ifelse(is.finite(max_v), max_v, NA_real_),
    p10   = ifelse(is.finite(p10),   p10,   NA_real_),
    p90   = ifelse(is.finite(p90),   p90,   NA_real_),
    med   = ifelse(is.finite(med),   med,   NA_real_)
  )

conv_plot <- conv_all %>% select(group, slot, conv_step)

# -------- compute histogram scales: rows 1–2 share; bottom row separate --------
conv_counts <- conv_plot %>%
  mutate(bin = floor(conv_step / BINWIDTH) * BINWIDTH) %>%
  count(slot, bin, name = "count")

slot_lvls   <- levels(conv_plot$slot)
bottom_slot <- tail(slot_lvls, 1)          # "3"
upper_slots <- setdiff(slot_lvls, bottom_slot)

upper_max <- conv_counts %>% filter(slot %in% upper_slots) %>% summarise(m = max(count, na.rm = TRUE)) %>% pull(m)
bottom_max <- conv_counts %>% filter(slot == bottom_slot) %>% summarise(m = max(count, na.rm = TRUE)) %>% pull(m)
if (!is.finite(upper_max)  || upper_max  <= 0) upper_max  <- 1
if (!is.finite(bottom_max) || bottom_max <= 0) bottom_max <- 1

upper_scale_y  <- (Y_MAX * HIST_MAX_FRAC) / upper_max
bottom_scale_y <- (Y_MAX * HIST_MAX_FRAC) / bottom_max

# ---- Pre-bin histograms per (group, slot) with the proper row scale ----
prep_hist_rects <- function(df_conv, scale_y, binwidth) {
  df_conv %>%
    mutate(bin = floor(conv_step / binwidth) * binwidth) %>%
    count(group, slot, bin, name = "count") %>%
    mutate(
      y    = count * scale_y,
      xmin = bin,
      xmax = bin + binwidth,
      ymin = 0,
      ymax = y
    )
}
hist_rects_topmid <- prep_hist_rects(conv_plot %>% filter(slot %in% upper_slots),
                                     upper_scale_y, BINWIDTH)
hist_rects_bottom <- prep_hist_rects(conv_plot %>% filter(slot == bottom_slot),
                                     bottom_scale_y, BINWIDTH)

# -------- per-panel mean and SD of conv_step for vertical lines --------
stats_lines <- conv_plot %>%
  group_by(group, slot) %>%
  summarise(
    mu = mean(conv_step, na.rm = TRUE),
    sd = sd(conv_step, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    mu     = pmin(pmax(mu, 0), X_MAX),
    mu_m1s = pmin(pmax(mu - sd, 0), X_MAX),
    mu_p1s = pmin(pmax(mu + sd, 0), X_MAX)
  )

# -------- function to build a single row plot with its own secondary axis --------
make_row_plot <- function(slot_value, hist_rects, sec_axis_scale) {
  # Filter data for this slot
  summ_row   <- summ_plot   %>% filter(slot == slot_value)
  rects_row  <- hist_rects  %>% filter(slot == slot_value)
  stats_row  <- stats_lines %>% filter(slot == slot_value)
  
  ggplot() +
    # Histogram (filled + outline)
    geom_rect(
      data = rects_row,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill = "grey80", alpha = 0.35
    ) +
    geom_rect(
      data = rects_row,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill = NA, color = "black", linewidth = 0.5
    ) +
    # Ribbons + median
    geom_ribbon(
      data = summ_row,
      aes(x = step, ymin = min_v, ymax = max_v),
      fill = "grey85", alpha = 0.35
    ) +
    geom_ribbon(
      data = summ_row,
      aes(x = step, ymin = p10, ymax = p90),
      fill = "grey55", alpha = 0.35
    ) +
    geom_line(
      data = summ_row,
      aes(x = step, y = med),
      linewidth = 0.9, color = "black"
    ) +
    # Mean (solid) and ±1 SD (dashed)
    geom_vline(data = stats_row, aes(xintercept = mu,     group = group), linewidth = 0.7, linetype = "solid") +
    geom_vline(data = stats_row, aes(xintercept = mu_m1s, group = group), linewidth = 0.6, linetype = "dashed") +
    geom_vline(data = stats_row, aes(xintercept = mu_p1s, group = group), linewidth = 0.6, linetype = "dashed") +
    coord_cartesian(xlim = c(0, X_MAX), ylim = c(0, Y_MAX)) +
    scale_y_continuous(
      name     = NULL,  # left title is added globally
      sec.axis = sec_axis(~ . / sec_axis_scale, name = NULL)  # <- no per-row right title
    ) +
    labs(x = NULL) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position   = "none",
      plot.margin       = margin(4, 12, 4, 12),
      strip.text        = element_text(face = "bold"),
      strip.background  = element_rect(fill = "grey95", color = NA),
      axis.title.x      = element_text(margin = margin(t = 6))
    ) +
    facet_grid(cols = vars(group),
               labeller = labeller(group = as_labeller(c(A = "C23-T2-1A", B = "C23-S1-A"))))
}

# Build the three row plots (top & middle share axis scale; bottom has its own)
p_top    <- make_row_plot(slot_levels[1], hist_rects_topmid, upper_scale_y)
p_middle <- make_row_plot(slot_levels[2], hist_rects_topmid, upper_scale_y)
p_bottom <- make_row_plot(slot_levels[3], hist_rects_bottom, bottom_scale_y) + labs(x = "Tile (step)")

# Stack the rows (align widths)
p_stack <- p_top / p_middle / p_bottom + plot_layout(heights = c(1,1,1))

# Add single, centered left & right axis titles
p_final <- ggdraw() +
  # Left label
  draw_label("Distance to EWMA state (clr-space)",
             x = 0.015, y = 0.5, angle = 90, vjust = 0.5, hjust = 0.5) +
  # Right label (generic since rows use different scales; ticks remain per-row)
  draw_label("Runs converged",
             x = 0.985, y = 0.5, angle = -90, vjust = 0.5, hjust = 0.5) +
  # Main plot area
  draw_plot(p_stack, x = 0.06, y = 0, width = 0.88, height = 1)

print(p_final)

if (SAVE_FIGURE) {
  ggsave(OUT_NAME, p_final, width = 8.5, height = 7, dpi = 300)
  cat("\nCombined plot saved:", file.path(getwd(), OUT_NAME), "\n")
}

cat("\nAll done.\n")
