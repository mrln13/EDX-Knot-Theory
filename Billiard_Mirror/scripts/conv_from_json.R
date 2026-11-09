# usage: Rscript conv_from_json.R IN.json OUT.csv {fixed|auto}
# Runs Calc_Stats.R's process_run (lines 1-215, unchanged) on every run and
# writes one t* per run. 'fixed' uses the 9 major elements of the schist/diorite
# analysis (as stored in Sensitivity_analysis.RData); 'auto' keeps top-10 per run.
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
res <- map_dfr(agg$runs, function(r) {
  d <- process_run(r)
  tibble(file = d$file[1], n = nrow(d), conv_step = d$conv_step[1])
})
write.csv(res, a[2], row.names = FALSE)
cat("runs", nrow(res), " converged", sum(!is.na(res$conv_step)),
    " mean", round(mean(res$conv_step, na.rm = TRUE), 1), " sd", round(sd(res$conv_step, na.rm = TRUE), 1), "\n")
