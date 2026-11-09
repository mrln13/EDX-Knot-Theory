# ============================================================
# Manuscript Figure 4 (pooled-orientation version)
# 3 rows (Billiard / Raster / Spiral) x 3 cols (Schist / Diorite / Granulite)
#
# Differences from plot_manuscript.R:
# - All three rows POOL both scan orientations (row + column serpentine
#   raster; clockwise + counter-clockwise inward spiral; the billiard path for
#   (p, q) and its left-right mirror image), matching Table 1 of the manuscript.
#   The second billiard path is the left-right mirror for the schist/diorite and
#   the top-bottom mirror for the granulite (the other mirror image is the
#   original path reversed). Runs come from Data Analysis/scan_orientation/*_billiard{mirror,flip}_dfall.RData
#   (built by Data Analysis/scan_orientation/scripts/json_to_dfall.R; schist/diorite orders were
#   truncated at 600 tiles, beyond every convergence step and the plotted range).
# - Ribbons and histograms are rebuilt from per-run data (df_all), not from
#   the saved per-variant 'summ'/'conv_df', because quantiles cannot be pooled.
# - No 300-tile cut on the convergence steps: every run that converged is
#   counted; runs that never converged are excluded (as in Table 1).
# - Each column gets its own x-range, long enough to show every converged run
#   (the granulite needs the full 460-tile sequence).
# - Column order Schist, Diorite, Granulite (as in the caption and Table 1).
#
# Sources are either Calc_Stats.R workspaces (.RData containing df_all) or
# aggregate JSONs from analyze_orders.py, which are processed with the
# unchanged Calc_Stats.R helpers (lines 1-215: settings + process_run).
# Run from the repository root:  Rscript "Data Analysis/plot_manuscript_pooled.R" <out.pdf>
# ============================================================

suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(purrr); library(tibble)
  library(patchwork); library(cowplot); library(jsonlite)
})

args <- commandArgs(TRUE)
OUT_PDF <- if (length(args)) args[1] else "Fig4_pooled.pdf"

RP <- "../R_projects/EDX_SC_data_vis/"
SOURCES <- list(
  Schist = list(
    Billiard = c(paste0(RP, "C23S1A/billiard_all_newparams.RData"),
                 "Data Analysis/scan_orientation/schist_billiardmirror_dfall.RData"),
    Raster   = c(paste0(RP, "C23S1A/Raster_snake_all_newparams.RData"),
                 paste0(RP, "C23S1A/column_snake_all_newparams.RData")),
    Spiral   = c(paste0(RP, "C23S1A/Spiral_CW_edges_newparams.RData"),
                 paste0(RP, "C23S1A/Spiral_CCW_edges_newparams.RData"))),
  Diorite = list(
    Billiard = c(paste0(RP, "C23T2-1A/Billiard_all_newparams.RData"),
                 "Data Analysis/scan_orientation/diorite_billiardmirror_dfall.RData"),
    Raster   = c(paste0(RP, "C23T2-1A/Raster_snake_all_newparams.RData"),
                 paste0(RP, "C23T2-1A/Column_snake_all_newparams.RData")),
    Spiral   = c(paste0(RP, "C23T2-1A/Spiral_CW_edges_newparams.RData"),
                 paste0(RP, "C23T2-1A/Spiral_CCW_edges_newparams.RData"))),
  Granulite = list(
    Billiard = c("Granulite_Rdata/Granulite_Billiard_All_460.RData",
                 "Data Analysis/scan_orientation/granulite_billiardflip_dfall.RData"),
    Raster   = c("Granulite_Rdata/Granulite_Raster_All_460.RData",
                 "Granulite_Column_All/granulite_column_metrics.json"),
    Spiral   = c("Granulite_Rdata/Granulite_Spiral_Edges_460.RData",
                 "Granulite_SpiralCCW_Edges/granulite_spiralccw_edge_metrics.json"))
)
SLOTS <- c("Billiard", "Raster", "Spiral")

# ----- display controls (as in plot_manuscript.R) -----
X_MIN_RANGE <- 300   # minimum x-range per column; extended to cover all runs
Y_MAX       <- 0.75
BINWIDTH    <- 10
HIST_MAX_FRAC <- 0.5

# ----- Calc_Stats.R helpers, only needed for JSON sources -----
.need_json <- any(grepl("\\.json$", unlist(SOURCES)))
if (.need_json) {
  src <- readLines("Data Analysis/Calc_Stats.R", warn = FALSE)
  eval(parse(text = src[1:215]), envir = globalenv())
}

read_df_all <- function(path) {
  if (grepl("\\.json$", path)) {
    agg <- fromJSON(path, simplifyVector = FALSE)
    d <- bind_rows(map(agg$runs, process_run))
  } else {
    e <- new.env(); load(path, envir = e)
    d <- get("df_all", envir = e)
  }
  # per-run conv step (first non-NA), plus per-step distances
  conv <- d %>% group_by(file) %>%
    summarise(conv_step = unique(na.omit(conv_step))[1], .groups = "drop") %>%
    mutate(file = paste(basename(path), file, sep = "::"))
  steps <- d %>% transmute(file = paste(basename(path), file, sep = "::"), step, dist)
  list(conv = conv, steps = steps)
}

conv_all <- list(); steps_all <- list()
for (g in names(SOURCES)) for (s in SLOTS) {
  parts <- map(SOURCES[[g]][[s]], read_df_all)
  conv_all[[length(conv_all) + 1]]   <- bind_rows(map(parts, "conv"))  %>% mutate(group = g, slot = s)
  steps_all[[length(steps_all) + 1]] <- bind_rows(map(parts, "steps")) %>% mutate(group = g, slot = s)
  cat("loaded:", g, s, "\n")
}
conv_all  <- bind_rows(conv_all)
steps_all <- bind_rows(steps_all)

# per-column x-range: at least X_MIN_RANGE, rounded up to cover every converged run
x_rng <- conv_all %>% filter(!is.na(conv_step)) %>% group_by(group) %>%
  summarise(xmax = max(X_MIN_RANGE, ceiling(max(conv_step) / 50) * 50), .groups = "drop")
print(as.data.frame(x_rng))

summ_plot <- steps_all %>% inner_join(x_rng, by = "group") %>% filter(step <= xmax) %>%
  group_by(group, slot, step) %>%
  summarise(min_v = min(dist, na.rm = TRUE), max_v = max(dist, na.rm = TRUE),
            p10 = quantile(dist, .10, na.rm = TRUE, names = FALSE),
            med = median(dist, na.rm = TRUE),
            p90 = quantile(dist, .90, na.rm = TRUE, names = FALSE), .groups = "drop")
rm(steps_all); invisible(gc())

conv_plot <- conv_all %>% filter(!is.na(conv_step))
never <- conv_all %>% group_by(group, slot) %>% summarise(n = n(), never = sum(is.na(conv_step)), .groups = "drop")
print(as.data.frame(never))

stats_lines <- conv_plot %>% group_by(group, slot) %>%
  summarise(mu = mean(conv_step), sd = sd(conv_step), .groups = "drop") %>%
  mutate(mu_m1s = mu - sd, mu_p1s = mu + sd)
print(as.data.frame(stats_lines %>% mutate(across(where(is.numeric), ~ round(.x, 1)))))

# ----- histograms: fraction of runs per (group, slot); rows 1-2 share a scale -----
fracs <- conv_plot %>% mutate(bin = floor(conv_step / BINWIDTH) * BINWIDTH) %>%
  count(group, slot, bin, name = "count") %>% group_by(group, slot) %>%
  mutate(frac = count / sum(count)) %>% ungroup()
upper_max  <- max(fracs$frac[fracs$slot %in% SLOTS[1:2]])
bottom_max <- max(fracs$frac[fracs$slot == SLOTS[3]])
scale_for <- function(s) (Y_MAX * HIST_MAX_FRAC) / if (s == SLOTS[3]) bottom_max else upper_max
hist_rects <- fracs %>% rowwise() %>% mutate(ymax = frac * scale_for(slot)) %>% ungroup() %>%
  mutate(xmin = bin, xmax = bin + BINWIDTH, ymin = 0)

lvl <- function(df) df %>% mutate(group = factor(group, levels = names(SOURCES)),
                                  slot = factor(slot, levels = SLOTS))
summ_plot <- lvl(summ_plot); hist_rects <- lvl(hist_rects); stats_lines <- lvl(stats_lines)
x_blank <- lvl(bind_rows(x_rng %>% mutate(x = 0), x_rng %>% mutate(x = xmax)) %>%
                 tidyr::crossing(slot = SLOTS))

make_row_plot <- function(slot_value, show_col_strip = FALSE) {
  sr <- summ_plot %>% filter(slot == slot_value)
  hr <- hist_rects %>% filter(slot == slot_value)
  st <- stats_lines %>% filter(slot == slot_value)
  xb <- x_blank %>% filter(slot == slot_value)
  ggplot() +
    geom_blank(data = xb, aes(x = x)) +
    geom_rect(data = hr, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), fill = "grey80", alpha = 0.35) +
    geom_rect(data = hr, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), fill = NA, color = "black", linewidth = 0.5) +
    geom_ribbon(data = sr, aes(x = step, ymin = min_v, ymax = max_v), fill = "grey85", alpha = 0.35) +
    geom_ribbon(data = sr, aes(x = step, ymin = p10, ymax = p90), fill = "grey55", alpha = 0.35) +
    geom_line(data = sr, aes(x = step, y = med), linewidth = 0.9, color = "black") +
    geom_vline(data = st, aes(xintercept = mu), linewidth = 0.7, linetype = "solid") +
    geom_vline(data = st, aes(xintercept = mu_m1s), linewidth = 0.6, linetype = "dashed") +
    geom_vline(data = st, aes(xintercept = mu_p1s), linewidth = 0.6, linetype = "dashed") +
    coord_cartesian(ylim = c(0, Y_MAX)) +
    scale_y_continuous(name = NULL, sec.axis = sec_axis(~ . / scale_for(slot_value), name = NULL)) +
    scale_x_continuous(expand = expansion(mult = 0.02)) +
    labs(x = NULL) +
    facet_grid(rows = vars(slot), cols = vars(group), scales = "free_x") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none", plot.margin = margin(4, 12, 4, 12),
          panel.spacing.x = unit(1.5, "lines"),  # keeps "300" and "0" tick labels apart at column joins
          strip.text.x = if (show_col_strip) element_text(face = "bold") else element_text(color = "transparent"),
          strip.background.x = if (show_col_strip) element_rect(fill = "grey95", color = NA) else element_rect(fill = NA, color = NA),
          strip.text.y.right = element_text(face = "bold", angle = -90),
          strip.background.y = element_rect(fill = "grey95", color = NA),
          strip.placement = "inside", axis.title.x = element_text(margin = margin(t = 6)))
}

p_stack <- make_row_plot(SLOTS[1], TRUE) / make_row_plot(SLOTS[2]) /
  (make_row_plot(SLOTS[3]) + labs(x = "Tile (step)")) + plot_layout(heights = c(1, 1, 1))
p_final <- ggdraw() +
  draw_label("Distance to EWMA state (clr-space)", x = 0.015, y = 0.5, angle = 90) +
  draw_label("Fraction of runs converged", x = 0.985, y = 0.5, angle = -90) +
  draw_plot(p_stack, x = 0.06, y = 0, width = 0.88, height = 1)
ggsave(OUT_PDF, p_final, width = 12.0, height = 7, dpi = 300)
cat("\nSaved:", OUT_PDF, "\n")
