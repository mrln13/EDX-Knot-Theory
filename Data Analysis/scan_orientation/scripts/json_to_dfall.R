# usage: Rscript json_to_dfall.R IN.json OUT.RData {fixed|auto}
# Runs Calc_Stats.R's process_run (lines 1-215, unchanged) on every run of an
# aggregate JSON from analyze_orders.py and saves a compact df_all
# (file, step, dist, conv_step), the same columns plot_manuscript_pooled.R
# reads from the Calc_Stats.R workspaces.
# 'fixed' uses the 9 major elements of the schist/diorite analysis (as stored in
# Sensitivity_analysis.RData); 'auto' keeps Calc_Stats.R's top-10 selection
# (used for the granulite).
suppressPackageStartupMessages({library(dplyr); library(jsonlite); library(purrr)})
a <- commandArgs(TRUE)
src <- readLines("C:/Users/4029771/OneDrive - Universiteit Utrecht/Dev/EDX-Knot-Theory/Data Analysis/Calc_Stats.R", warn = FALSE)
eval(parse(text = src[1:215]), envir = globalenv())
if (a[3] == "fixed") {
  e <- new.env(); load("C:/Users/4029771/OneDrive - Universiteit Utrecht/Data/EDX Sparse/Sensitivity_analysis.RData", envir = e)
  major_elems <- e$major_elems
  cat("majors:", major_elems, "\n")
}
agg <- fromJSON(a[1], simplifyVector = FALSE)
df_all <- map_dfr(agg$runs, function(r) process_run(r) %>% select(file, step, dist, conv_step))
rm(agg); invisible(gc())
save(df_all, file = a[2])
cs <- df_all %>% group_by(file) %>% summarise(c = unique(na.omit(conv_step))[1], .groups = "drop")
cat("runs", nrow(cs), " converged", sum(!is.na(cs$c)),
    " mean", round(mean(cs$c, na.rm = TRUE), 2), " sd", round(sd(cs$c, na.rm = TRUE), 2), "\n")
