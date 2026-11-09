# Export per-run convergence steps for every source used in Table 1 / Figure 4.
# Reuses SOURCES and read_df_all() from plot_manuscript_pooled.R (lines 28-89),
# so the run sets are exactly those behind the pooled manuscript statistics.
# Run from the repository root.
src <- readLines("Data Analysis/plot_manuscript_pooled.R", warn = FALSE)
eval(parse(text = src[28:89]), envir = globalenv())
OUT <- commandArgs(TRUE)[1]
res <- list()
for (g in names(SOURCES)) for (s in SLOTS) for (path in SOURCES[[g]][[s]]) {
  d <- read_df_all(path)$conv
  d$group <- g; d$slot <- s; d$source <- basename(path)
  res[[length(res) + 1]] <- d
  cat(g, s, basename(path), nrow(d), "runs,", sum(is.na(d$conv_step)), "never\n")
  invisible(gc())
}
out <- dplyr::bind_rows(res)
write.csv(out, OUT, row.names = FALSE)
cat("wrote", nrow(out), "rows to", OUT, "\n")
