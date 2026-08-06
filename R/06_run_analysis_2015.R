#' ============================================================================
#' End-to-end example: 2015 greenness curves -> UTrK & UCoK predictions
#' ============================================================================
#'
#' Reproduces the paper's core workflow top to bottom:
#'   load data -> GCV smoothing -> functional PCA -> UTrK & UCoK prediction
#'   -> Monte Carlo RMSPE evaluation
#'
#' This replaces the original `UCoK& UTrK.R` as the main runnable example
#' (that script is kept privately, outside this repo, for reference).
#'
#' Run from the repository root, e.g.:
#'   Rscript R/06_run_analysis_2015.R

source("R/00_setup.R")
source("R/01_data_acquisition.R")
source("R/02_load_and_clean_data.R")
source("R/03_functional_smoothing.R")
source("R/04_kriging_models.R")
source("R/05_monte_carlo_evaluation.R")

# ---- 1. Load & clean ---------------------------------------------------------
message("Loading and cleaning the 2015 greenness dataset...")
dataset <- load_greenness_dataset(year = 2015)
greenness <- dataset$greenness
coords <- dataset$coords
message(sprintf("%d cameras retained after cleaning.", ncol(greenness)))

# ---- 2. Smooth + FPCA preview -------------------------------------------------
message("Smoothing with GCV-selected lambda...")
smoothed <- smooth_greenness_gcv(greenness, nbasis = 90, norder = 6)
message(sprintf("Selected lambda = %.4g", attr(smoothed, "lambda")))

fpca <- rotated_fpca(smoothed, n_components = 3)
message(sprintf(
  "Rotated FPCA variance explained (PC1-3): %s",
  paste(sprintf("%.1f%%", 100 * fpca$varprop), collapse = ", ")
))

# ---- 3. A single leave-one-camera-out UTrK / UCoK prediction ------------------
set.seed(2024)
test_camera <- sample(colnames(greenness), 1)
train_ids <- setdiff(colnames(greenness), test_camera)

train_coords <- coords[match(train_ids, coords$camera_id), c("lon", "lat")]
test_coords <- coords[coords$camera_id == test_camera, c("lon", "lat")]
rownames(train_coords) <- train_ids
rownames(test_coords) <- test_camera

utrk <- fit_utrk(
  greenness[, train_ids], train_coords, test_coords,
  model = gstat::vgm("Exp"), n_lags = 15, lag_max = 50
)
ucok <- fit_ucok(
  greenness[, train_ids], train_coords = train_coords, new_coords = test_coords,
  nbasis = 90, norder = 6, n_components = 3,
  model = gstat::vgm("Exp"), n_lags = 15, lag_max = 50
)

true_curve <- greenness[, test_camera]
message(sprintf(
  "Held-out camera %s: RMSPE UTrK = %.5f, RMSPE UCoK = %.5f",
  test_camera,
  sqrt(mean((true_curve - utrk$prediction)^2)),
  sqrt(mean((true_curve - ucok$prediction)^2))
))

grDevices::pdf("results/utrk_vs_ucok_example_R.pdf", width = 9, height = 5)
matplot(DOY_RANGE, greenness[, train_ids], type = "l", col = "grey", lty = 1,
        xlab = "Day of year (2015)", ylab = "Relative greenness",
        main = "Leave-one-camera-out prediction: UTrK vs UCoK")
lines(DOY_RANGE, true_curve, col = "black", lwd = 2)
lines(DOY_RANGE, utrk$prediction, col = "firebrick", lwd = 2, lty = 2)
lines(DOY_RANGE, ucok$prediction, col = "royalblue", lwd = 2, lty = 2)
legend("topleft",
       legend = c(sprintf("true curve (camera %s, held out)", test_camera), "UTrK prediction", "UCoK prediction"),
       col = c("black", "firebrick", "royalblue"), lty = c(1, 2, 2), lwd = 2, bty = "n")
grDevices::dev.off()

# ---- 4. Monte Carlo evaluation (small run -- see README for the full 300-iteration study) ----
message("Running a small Monte Carlo evaluation (5 iterations, 80% train)...")
mc <- run_monte_carlo_study(greenness, coords, train_frac = 0.8, n_iter = 5, seed = 1)
print(summarize_monte_carlo(mc))

message("Done.")
