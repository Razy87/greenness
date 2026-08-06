#' ============================================================================
#' Figure: raw vs. GCV-smoothed greenness curves
#' ============================================================================
#'
#' R-side companion to `scripts/make_figures.py`'s Figure 1 (raw vs smoothed
#' curves for a sample of cameras), giving the R analysis the same visual
#' sanity check as the Python port. The Python script's Figure 2
#' (leave-one-camera-out UTrK vs UCoK) already has an R equivalent, produced
#' by `06_run_analysis_2015.R` -> `results/utrk_vs_ucok_example_R.pdf`, so
#' it is not repeated here.
#'
#' Source 00_setup.R before this file (it also sources 02 and 03 itself).
#' Safe to run either standalone, or right after `06_run_analysis_2015.R`
#' in the same session -- it reuses `greenness` / `smoothed` if they're
#' already in your workspace, and builds them fresh otherwise.
#'
#' Run from the repository root, e.g.:
#'   source("R/08_figures.R")
#' or, to reuse an already-loaded dataset:
#'   source("R/06_run_analysis_2015.R")
#'   source("R/08_figures.R")

source("R/00_setup.R")
source("R/02_load_and_clean_data.R")
source("R/03_functional_smoothing.R")

#' Plot raw (faint) vs GCV-smoothed (bold) curves for a sample of cameras.
#'
#' @param greenness (n_time x n_camera) matrix, columns named by camera id.
#' @param smoothed_fd The `fd` object from `smooth_greenness_gcv()`, fit on
#'   the same `greenness` matrix.
#' @param time_grid The x-axis `greenness` is sampled on (default 1:365).
#' @param n_sample Number of cameras to plot (default 8, taken from the
#'   start of `colnames(greenness)`, matching `make_figures.py`).
#' @param out_file Output PNG path.
#' @return `out_file`, invisibly.
plot_smoothed_curves <- function(greenness, smoothed_fd, time_grid = DOY_RANGE,
                                  n_sample = 8, out_file = "results/smoothed_curves_2015_R.png") {
  sample_cams <- colnames(greenness)[seq_len(min(n_sample, ncol(greenness)))]
  fitted <- fda::eval.fd(time_grid, smoothed_fd)  # (n_time x n_camera)
  palette <- grDevices::rainbow(length(sample_cams))

  grDevices::png(out_file, width = 9, height = 5, units = "in", res = 150)
  on.exit(grDevices::dev.off(), add = TRUE)

  matplot(time_grid, greenness[, sample_cams],
          type = "l", lty = 1, lwd = 1,
          col = grDevices::adjustcolor(palette, alpha.f = 0.3),
          xlab = "Day of year (2015)", ylab = "Relative greenness",
          main = "Raw (faint) vs GCV B-spline-smoothed (bold) greenness curves")
  matlines(time_grid, fitted[, sample_cams], lty = 1, lwd = 2, col = palette)
  legend("topright", legend = sprintf("cam %s", sample_cams),
         col = palette, lty = 1, lwd = 2, ncol = 2, cex = 0.75, bty = "n")

  invisible(out_file)
}

# ---- Auto-run when sourced -----------------------------------------------
# Reuse `greenness` / `smoothed` from the calling session if already present
# (e.g. right after 06_run_analysis_2015.R), otherwise build them fresh.
if (!exists("greenness", envir = .GlobalEnv) || !exists("smoothed", envir = .GlobalEnv)) {
  message("`greenness`/`smoothed` not found in this session -- loading and smoothing 2015 data...")
  dataset <- load_greenness_dataset(year = 2015)
  greenness <- dataset$greenness
  smoothed <- smooth_greenness_gcv(greenness, nbasis = 90, norder = 6)
}

out_path <- plot_smoothed_curves(greenness, smoothed)
message(sprintf("Wrote %s", out_path))
