suppressPackageStartupMessages({library(dplyr); library(jsonlite); library(purrr)})
OUT <- commandArgs(TRUE)[1]
# Load settings + helpers + process_run from Calc_Stats.R unchanged (lines 1-215)
src <- readLines("Data Analysis/Calc_Stats.R", warn = FALSE)
eval(parse(text = src[1:215]), envir = globalenv())
summ_runs <- function(cs, ntiles) {           # cs: one conv_step per run (NA = never)
  a <- cs[!is.na(cs)]; t1 <- a[a <= 300]
  tibble(n_runs = length(cs), n_tiles = ntiles, never = sum(is.na(cs)), after300 = sum(a > 300),
         t1_mean = mean(t1), t1_sd = sd(t1),
         all_mean = mean(a), all_sd = sd(a), median = median(a),
         p10 = unname(quantile(a, .1)), p90 = unname(quantile(a, .9)))
}
from_df <- function(d) { r <- d %>% group_by(file) %>% summarise(cs = unique(na.omit(conv_step))[1], n = n(), .groups="drop"); list(cs = r$cs, nt = max(r$n)) }
from_rdata <- function(f) { e <- new.env(); load(f, envir = e); from_df(get("df_all", envir = e)) }
from_json <- function(f) { agg <- fromJSON(f, simplifyVector = FALSE); from_df(bind_rows(map(agg$runs, process_run))) }
RP <- "../R_projects/EDX_SC_data_vis/"
jobs <- list(
  list("Schist","Billiard","all starts","rd", paste0(RP,"C23S1A/billiard_all_newparams.RData"), TRUE),
  list("Schist","Raster row snake","all starts","rd", paste0(RP,"C23S1A/Raster_snake_all_newparams.RData"), TRUE),
  list("Schist","Raster column snake","all starts","rd", paste0(RP,"C23S1A/column_snake_all_newparams.RData"), FALSE),
  list("Schist","Spiral CW","edge starts","rd", paste0(RP,"C23S1A/Spiral_CW_edges_newparams.RData"), FALSE),
  list("Schist","Spiral CCW","edge starts","rd", paste0(RP,"C23S1A/Spiral_CCW_edges_newparams.RData"), TRUE),
  list("Diorite","Billiard","all starts","rd", paste0(RP,"C23T2-1A/Billiard_all_newparams.RData"), TRUE),
  list("Diorite","Raster row snake","all starts","rd", paste0(RP,"C23T2-1A/Raster_snake_all_newparams.RData"), FALSE),
  list("Diorite","Raster column snake","all starts","rd", paste0(RP,"C23T2-1A/Column_snake_all_newparams.RData"), TRUE),
  list("Diorite","Spiral CW","edge starts","rd", paste0(RP,"C23T2-1A/Spiral_CW_edges_newparams.RData"), TRUE),
  list("Diorite","Spiral CCW","edge starts","rd", paste0(RP,"C23T2-1A/Spiral_CCW_edges_newparams.RData"), FALSE),
  list("Granulite","Billiard","all starts","rd", "Granulite_Rdata/Granulite_Billiard_All_460.RData", TRUE),
  list("Granulite","Raster row snake","all starts","rd", "Granulite_Rdata/Granulite_Raster_All_460.RData", TRUE),
  list("Granulite","Raster column snake","all starts","js", "Granulite_Column_All/granulite_column_metrics.json", FALSE),
  list("Granulite","Spiral CW","edge starts","rd", "Granulite_Rdata/Granulite_Spiral_Edges_460.RData", TRUE),
  list("Granulite","Spiral CCW","edge starts","js", "Granulite_SpiralCCW_Edges/granulite_spiralccw_edge_metrics.json", FALSE),
  list("CHECK","Granulite row snake (reprocessed JSON)","all starts","js", "Granulite_Raster_All/granulite_raster_metrics.json", NA)
)
res <- list()
for (j in jobs) {
  x <- if (j[[4]] == "rd") from_rdata(j[[5]]) else from_json(j[[5]])
  res[[length(res)+1]] <- bind_cols(tibble(dataset=j[[1]], variant=j[[2]], starts=j[[3]], in_table1=j[[6]]), summ_runs(x$cs, x$nt))
  cat("done:", j[[1]], j[[2]], "\n")
}
out <- bind_rows(res); write.csv(out, OUT, row.names = FALSE)
print(as.data.frame(out %>% mutate(across(where(is.numeric), ~round(.x, 1)))), width = 250)
