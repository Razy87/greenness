#' ============================================================================
#' Universal Trace-Kriging (UTrK) and Universal Cokriging of FPC scores (UCoK)
#' ============================================================================
#'
#' Both predictors are built on the `fdagstat` package:
#'   - UTrK fits a single scalar "trace" variogram directly on the raw
#'     greenness curves, then krige's with the resulting weights applied to
#'     the whole curve at once.
#'   - UCoK reduces each curve to a handful of varimax-rotated functional
#'     principal component (FPC) scores (see 03_functional_smoothing.R),
#'     krige's each score, and reconstructs the predicted curve from the
#'     kriged scores.
#'
#' NOTE on a fix relative to the original scripts: `Fit_Utrk()` and
#' `Fit_Ucok()` originally ended with a leftover debugging
#' `return(Var_Utrk)` / `return(Sigma2_Ucok)` that discarded the predicted
#' curves and returned only the kriging variance. Both functions below
#' return the full result (prediction *and* variance) as intended.
#'
#' Source 00_setup.R and 03_functional_smoothing.R before this file.

#' Universal Trace-Kriging of greenness curves.
#'
#' @param greenness (n_time x n_train) matrix of training curves.
#' @param train_coords (n_train x 2) data.frame/matrix of (lon, lat) or
#'   (lat, lon) -- must match the column order used consistently elsewhere.
#' @param new_coords (n_test x 2) coordinates to predict at.
#' @param model A `gstat::vgm()` variogram model specification.
#' @param n_lags,lag_max Passed to `fdagstat::fvariogram()`.
#' @return A list with `prediction` (n_time x n_test matrix) and `variance`
#'   (length n_test).
fit_utrk <- function(greenness, train_coords, new_coords, model, n_lags, lag_max) {
  g <- fdagstat::fstat(NULL, vName = "Greenness", train_coords, Functions = as.data.frame(greenness), scalar = FALSE)
  g <- fdagstat::estimateDrift("~.", g, Intercept = TRUE)
  g <- fdagstat::fvariogram("~.", g, n_lags, lag_max, ArgStep = 1, useResidual = FALSE, comments = FALSE)
  g <- fdagstat::fitVariograms(g, model = model)
  g <- fdagstat::addCovariance(g, type = "omni")

  forecast <- fdagstat::predictFstat(g, .newCoordinates = new_coords, .what = "Greenness", .type = "UK")

  list(prediction = forecast$Forecast, variance = forecast$Variance, variogram = g)
}

#' Universal (co)kriging of varimax-rotated FPC scores.
#'
#' @param greenness (n_time x n_train) matrix of training curves.
#' @param time_grid The x-axis `greenness` is sampled on (default 1:365).
#' @param train_coords (n_train x 2) coordinates matching `greenness` columns.
#' @param new_coords (n_test x 2) coordinates to predict at.
#' @param nbasis,norder Passed to `smooth_greenness_gcv()`.
#' @param n_components Number of rotated FPC scores to krige (2, 3, or 4).
#' @param model A `gstat::vgm()` variogram model specification.
#' @param n_lags,lag_max Passed to `fdagstat::fvariogram()`.
#' @param force_nugget If not NULL, the nugget is fixed at this value rather
#'   than estimated (the original analysis fixes it at 0 for the FPC-score
#'   variograms).
#' @return A list with `prediction` (n_time x n_test matrix, reconstructed
#'   from the kriged scores) and `score_variance` (n_test x n_components).
fit_ucok <- function(greenness, time_grid = DOY_RANGE, train_coords, new_coords,
                      nbasis, norder = 6, n_components = 3,
                      model, n_lags, lag_max, force_nugget = 0) {
  stopifnot(n_components %in% 2:4)

  smoothed <- smooth_greenness_gcv(greenness, time_grid, nbasis, norder)
  fpca <- rotated_fpca(smoothed, n_components)

  mean_vals <- fda::eval.fd(time_grid, fpca$meanfd)
  harmonic_vals <- fda::eval.fd(time_grid, fpca$harmonics)  # (n_time x n_components)

  score_cols <- paste0("rpc", seq_len(n_components))
  scores <- as.data.frame(fpca$scores)
  colnames(scores) <- score_cols

  g <- fdagstat::fstat(NULL, score_cols[1], train_coords, scores[score_cols[1]], scalar = TRUE)
  for (col in score_cols[-1]) {
    g <- fdagstat::fstat(g, col, train_coords, scores[col], scalar = TRUE)
  }
  g <- fdagstat::estimateDrift("~.", g, Intercept = TRUE)
  g <- fdagstat::fvariogram("~.", g, n_lags, lag_max, useResidual = TRUE, comments = FALSE)
  g <- fdagstat::fitVariograms(g, model, forceNugget = !is.null(force_nugget), fitRanges = FALSE)
  g <- fdagstat::addCovariance(g, "omni")

  score_forecasts <- lapply(score_cols, function(col) {
    fdagstat::predictFstat(g, new_coords, .what = col, .type = "UcoK", algIndependent = TRUE)
  })
  names(score_forecasts) <- score_cols

  # NOTE: built with cbind(), not vapply()/sapply() -- when new_coords has
  # exactly one row (a single held-out camera), vapply()'s per-call result
  # (length 1) collapses to a plain vector instead of a 1-row matrix, which
  # silently transposes predicted_scores and makes the matrix multiply
  # below fail with "non-conformable arguments". cbind() always keeps the
  # (n_test x n_components) shape regardless of n_test.
  predicted_scores <- do.call(cbind, lapply(score_forecasts, function(f) f$Forecast))
  score_variance <- do.call(cbind, lapply(score_forecasts, function(f) f$Variance))
  colnames(predicted_scores) <- score_cols
  colnames(score_variance) <- score_cols

  # Reconstruct curves: mean(t) + harmonics(t) %*% predicted_scores'
  prediction <- mean_vals[, 1] + harmonic_vals %*% t(predicted_scores)
  colnames(prediction) <- rownames(new_coords)

  list(prediction = prediction, score_variance = score_variance, fpca = fpca, variogram = g)
}
