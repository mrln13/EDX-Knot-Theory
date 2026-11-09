# ============================================================
# Big Beautiful Figure: 3 rows x (1, 2, or 3) cols
# - Top column strip shows the dataset (set via GROUP_*_LABEL); on the other
#   rows the strip space is preserved with transparent text for alignment
# - Right-side row strip shows the pattern (Billiard / Raster / Spiral),
#   placed between the panel and the secondary-axis tick labels
# - Bottom row has its own histogram scale (fills HIST_MAX_FRAC)
# - Global left y-axis title (cowplot)
# - Global right y-axis title (cowplot) — shown ONCE
# - Per-panel mean (solid) and mean±1 SD (dashed) lines
# - Set PLOT_GROUPS to any subset of c("A","B","C") to render the
#   corresponding columns left-to-right
# ============================================================

GROUP_B_FILES <- c(
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/R_projects/EDX_SC_data_vis/C23S1A/Billiard_all_newparams.RData",
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/R_projects/EDX_SC_data_vis/C23S1A/Raster_snake_all_newparams.RData",
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/R_projects/EDX_SC_data_vis/C23S1A/Spiral_CCW_edges_newparams.RData"
)
GROUP_A_FILES <- c(
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/R_projects/EDX_SC_data_vis/C23T2-1A/Billiard_all_newparams.RData",
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/R_projects/EDX_SC_data_vis/C23T2-1A/Column_snake_all_newparams.RData",
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/R_projects/EDX_SC_data_vis/C23T2-1A/Spiral_CW_edges_newparams.RData"
)
GROUP_C_FILES <- c(
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Granulite_Rdata/Granulite_Billiard_All_460.RData",
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Granulite_Rdata/Granulite_Raster_All_460.RData",
  "C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Granulite_Rdata/Granulite_Spiral_Edges_460.RData"
)

GROUP_B_LABEL <- "Schist"
GROUP_A_LABEL <- "Diorite"
GROUP_C_LABEL <- "Granulite"

# Row labels — slot order matches the file order in GROUP_*_FILES above
SLOT_LABELS <- c("1" = "Billiard", "2" = "Raster", "3" = "Spiral")

# Which groups to plot. c("A","B") = both columns; "A" or "B" = single column.
PLOT_GROUPS <- c("A", "B", "C")

SAVE_FIGURE <- TRUE
# Output name auto-built from PLOT_GROUPS — override below if you prefer.
.label_for_group <- c(A = GROUP_A_LABEL, B = GROUP_B_LABEL, C = GROUP_C_LABEL)
OUT_NAME <- paste0(
  "BigBeautifulFigure_",
  paste(.label_for_group[PLOT_GROUPS], collapse = "_"),
  ".pdf"
)

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

# Only load the groups we plan to plot, so single-group runs don't require
# both groups' .RData files to exist.
ga <- if ("A" %in% PLOT_GROUPS) load_group(GROUP_A_FILES, "A") else NULL
gb <- if ("B" %in% PLOT_GROUPS) load_group(GROUP_B_FILES, "B") else NULL
gc <- if ("C" %in% PLOT_GROUPS) load_group(GROUP_C_FILES, "C") else NULL

summ_all <- bind_rows(
  if (!is.null(ga)) ga$summ,
  if (!is.null(gb)) gb$summ,
  if (!is.null(gc)) gc$summ
) %>% filter(step <= X_MAX)

conv_all <- bind_rows(
  if (!is.null(ga)) ga$conv,
  if (!is.null(gb)) gb$conv,
  if (!is.null(gc)) gc$conv
) %>% filter(is.finite(conv_step), conv_step >= 0, conv_step <= X_MAX)

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
    p10   = median(p10, na.rm = TRUE),
    p90   = median(p90, na.rm = TRUE),
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
    mu_m1s = mu - sd,
    mu_p1s = mu + sd
  )

# -------- function to build a single row plot with its own secondary axis --------
make_row_plot <- function(slot_value, hist_rects, sec_axis_scale, show_col_strip = FALSE) {
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
    facet_grid(rows = vars(slot), cols = vars(group),
               labeller = labeller(
                 group = as_labeller(c(A = GROUP_A_LABEL, B = GROUP_B_LABEL, C = GROUP_C_LABEL)),
                 slot  = as_labeller(SLOT_LABELS)
               )) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position    = "none",
      plot.margin        = margin(4, 12, 4, 12),
      # Column strip (Kibo / Granulite): only show text on the top row,
      # but keep the strip space on other rows so panels stay vertically aligned.
      strip.text.x       = if (show_col_strip) element_text(face = "bold")
                           else element_text(color = "transparent"),
      strip.background.x = if (show_col_strip) element_rect(fill = "grey95", color = NA)
                           else element_rect(fill = NA, color = NA),
      # Row strip (Billiard / Raster / Spiral) on the right, vertical text,
      # placed between the panel and the secondary-axis tick labels.
      strip.text.y.right = element_text(face = "bold", angle = -90),
      strip.background.y = element_rect(fill = "grey95", color = NA),
      strip.placement    = "inside",
      axis.title.x       = element_text(margin = margin(t = 6))
    )
}

# Build the three row plots (top & middle share axis scale; bottom has its own)
# Only the top plot renders the column strip; middle and bottom keep the strip
# space (transparent) so the three panel grids align vertically.
p_top    <- make_row_plot(slot_levels[1], hist_rects_topmid, upper_scale_y, show_col_strip = TRUE)
p_middle <- make_row_plot(slot_levels[2], hist_rects_topmid, upper_scale_y, show_col_strip = FALSE)
p_bottom <- make_row_plot(slot_levels[3], hist_rects_bottom, bottom_scale_y, show_col_strip = FALSE) + labs(x = "Tile (step)")

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
  fig_width <- c("1" = 5.0, "2" = 8.5, "3" = 12.0)[as.character(length(PLOT_GROUPS))]
  ggsave(OUT_NAME, p_final, width = fig_width, height = 7, dpi = 300)
  cat("\nCombined plot saved:", file.path(getwd(), OUT_NAME), "\n")
}

cat("\nAll done.\n")
