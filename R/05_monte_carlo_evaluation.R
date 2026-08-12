#' ============================================================================
#' Monte Carlo train/test RMSPE evaluation
#' ============================================================================
#'
#' Repeatedly holds out a random subset of cameras, predicts their curves
#' from the remaining ("training") cameras via UTrK and UCoK, and computes
#' the root-mean-squared prediction error (RMSPE) per held-out camera.
#'
#' Source 00_setup.R, 02_load_and_clean_data.R, 03_functional_smoothing.R,
#' and 04_kriging_models.R before this file.

#' Root-mean-squared prediction error, per column (camera).
rmspe <- function(true_curves, predicted_curves) {
  sqrt(colMeans((true_curves - predicted_curves)^2))
}

#' One Monte Carlo train/test split + UTrK/UCoK fit + RMSPE.
#'
#' @param greenness (n_time x n_camera) matrix, columns named by camera id.
#' @param coords data.frame with `camera_id`, `lat`, `lon`, aligned to
#'   `colnames(greenness)`.
#' @param train_frac Fraction of cameras used for training (the rest are
#'   held out and predicted).
#' @param utrk_args,ucok_args Named lists of extra arguments passed to
#'   `fit_utrk()` / `fit_ucok()` (model, n_lags, lag_max, nbasis, norder,
#'   n_components, ...).
#' @return A list with `rmspe` (data.frame: camera_id, RMSPE_UTrK,
#'   RMSPE_UCoK) and `train_ids` / `test_ids`.
monte_carlo_iteration <- function(greenness, coords, train_frac, utrk_args = list(), ucok_args = list()) {
  camera_ids <- colnames(greenness)
  n_train <- floor(train_frac * length(camera_ids))
  train_ids <- sample(camera_ids, n_train)
  test_ids <- setdiff(camera_ids, train_ids)

  train_curves <- greenness[, train_ids, drop = FALSE]
  test_curves <- greenness[, test_ids, drop = FALSE]
  train_coords <- coords[match(train_ids, coords$camera_id), c("lon", "lat")]
  test_coords <- coords[match(test_ids, coords$camera_id), c("lon", "lat")]
  rownames(train_coords) <- train_ids
  rownames(test_coords) <- test_ids

  utrk <- do.call(fit_utrk, c(list(greenness = train_curves, train_coords = train_coords, new_coords = test_coords), utrk_args))
  ucok <- do.call(fit_ucok, c(list(greenness = train_curves, train_coords = train_coords, new_coords = test_coords), ucok_args))

  list(
    rmspe = data.frame(
      camera_id = test_ids,
      RMSPE_UTrK = rmspe(test_curves, utrk$prediction),
      RMSPE_UCoK = rmspe(test_curves, ucok$prediction)
    ),
    train_ids = train_ids,
    test_ids = test_ids
  )
}

#' Model settings used per training fraction in the paper.
MC_SETTINGS_BY_TRAIN_FRAC <- list(
  `0.8` = list(
    utrk = list(model = gstat::vgm("Sph"), n_lags = 100, lag_max = 60),
    ucok = list(nbasis = 100, norder = 6, n_components = 3, model = gstat::vgm("Sph"), n_lags = 200, lag_max = 50)
  ),
  `0.7` = list(
    utrk = list(model = gstat::vgm("Sph"), n_lags = 200, lag_max = 80),
    ucok = list(nbasis = 100, norder = 6, n_components = 3, model = gstat::vgm("Exp"), n_lags = 300, lag_max = 80)
  ),
  `0.6` = list(
    utrk = list(model = gstat::vgm("Sph"), n_lags = 200, lag_max = 80),
    ucok = list(nbasis = 100, norder = 6, n_components = 3, model = gstat::vgm("Exp"), n_lags = 300, lag_max = 80)
  ),
  `0.5` = list(
    utrk = list(model = gstat::vgm("Sph"), n_lags = 200, lag_max = 80),
    ucok = list(nbasis = 100, norder = 6, n_components = 3, model = gstat::vgm("Exp"), n_lags = 300, lag_max = 80)
  )
)

#' Run the full Monte Carlo evaluation at one training fraction.
#'
#' @param greenness,coords See `monte_carlo_iteration()`.
#' @param train_frac Training fraction (0.5, 0.6, 0.7, or 0.8 use the
#'   settings in `MC_SETTINGS_BY_TRAIN_FRAC` by default).
#' @param n_iter Number of Monte Carlo iterations (300 in the paper).
#' @param seed RNG seed for reproducibility.
#' @param utrk_args,ucok_args Override the default settings looked up by
#'   `train_frac` if supplied.
#' @return A data.frame, one row per iteration, with mean/median RMSPE for
#'   UTrK and UCoK.
run_monte_carlo_study <- function(greenness, coords, train_frac = 0.8, n_iter = 300, seed = 2024,
                                   utrk_args = NULL, ucok_args = NULL) {
  settings <- MC_SETTINGS_BY_TRAIN_FRAC[[as.character(train_frac)]]
  if (is.null(utrk_args)) utrk_args <- settings$utrk
  if (is.null(ucok_args)) ucok_args <- settings$ucok

  set.seed(seed)
  iterations <- vector("list", n_iter)
  for (i in seq_len(n_iter)) {
    result <- monte_carlo_iteration(greenness, coords, train_frac, utrk_args, ucok_args)
    iterations[[i]] <- data.frame(
      iter = i,
      mean_RMSPE_UTrK = mean(result$rmspe$RMSPE_UTrK),
      mean_RMSPE_UCoK = mean(result$rmspe$RMSPE_UCoK),
      median_RMSPE_UTrK = stats::median(result$rmspe$RMSPE_UTrK),
      median_RMSPE_UCoK = stats::median(result$rmspe$RMSPE_UCoK)
    )
    if (i %% 25 == 0) message(sprintf("Monte Carlo iteration %d / %d done.", i, n_iter))
  }

  do.call(rbind, iterations)
}

#' Summarise a `run_monte_carlo_study()` result across all iterations.
summarize_monte_carlo <- function(mc_result) {
  data.frame(
    mean_RMSPE_UTrK = mean(mc_result$mean_RMSPE_UTrK),
    mean_RMSPE_UCoK = mean(mc_result$mean_RMSPE_UCoK),
    sd_RMSPE_UTrK = stats::sd(mc_result$mean_RMSPE_UTrK),
    sd_RMSPE_UCoK = stats::sd(mc_result$mean_RMSPE_UCoK)
  )
}
