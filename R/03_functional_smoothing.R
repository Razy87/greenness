#' ============================================================================
#' Functional smoothing (B-splines + GCV) and functional PCA
#' ============================================================================
#'
#' Represents each camera's daily greenness curve as a smooth function via
#' penalized B-spline regression (`fda::smooth.basis`), with the roughness
#' penalty weight (lambda) chosen by generalized cross-validation. Also
#' provides the varimax-rotated functional-PCA step used by the UCoK model.
#'
#' Source 00_setup.R before this file.

#' Grid-search the GCV-optimal smoothing parameter lambda.
#'
#' @param greenness A (n_time x n_camera) numeric matrix.
#' @param time_grid The x-axis the rows of `greenness` are sampled on
#'   (typically 1:365).
#' @param nbasis Number of B-spline basis functions.
#' @param norder Spline order (order = degree + 1); norder = 6 gives quintic
#'   splines, matching the original analysis.
#' @param log10_lambda_grid Candidate log10(lambda) values to search.
#' @return A data.frame with columns log10_lambda, df, gcv (one row per grid
#'   point), sorted by log10_lambda.
gcv_lambda_search <- function(greenness, time_grid = DOY_RANGE, nbasis, norder = 6,
                               log10_lambda_grid = seq(-3, 8, by = 0.25)) {
  basis <- fda::create.bspline.basis(range(time_grid), nbasis, norder)

  results <- lapply(log10_lambda_grid, function(log10_lambda) {
    lambda <- 10^log10_lambda
    fd_par <- fda::fdPar(basis, Lfdobj = 4, lambda = lambda)
    fit <- fda::smooth.basis(time_grid, greenness, fd_par)
    data.frame(log10_lambda = log10_lambda, df = fit$df, gcv = sum(fit$gcv))
  })

  do.call(rbind, results)
}

#' Smooth a greenness matrix with the GCV-optimal roughness penalty.
#'
#' Equivalent to the original `Smooth_GVS()`, with the GCV search folded in
#' and the camera-id labelling made explicit.
#'
#' @param greenness A (n_time x n_camera) numeric matrix with camera ids as
#'   column names.
#' @param time_grid The x-axis `greenness` is sampled on (default 1:365).
#' @param nbasis Number of B-spline basis functions.
#' @param norder Spline order (default 6, quintic).
#' @return An `fd` object (see `fda::fd`) representing the smoothed curves.
smooth_greenness_gcv <- function(greenness, time_grid = DOY_RANGE, nbasis, norder = 6) {
  gcv_table <- gcv_lambda_search(greenness, time_grid, nbasis, norder)
  best_lambda <- 10^gcv_table$log10_lambda[which.min(gcv_table$gcv)]

  basis <- fda::create.bspline.basis(range(time_grid), nbasis, norder)
  fd_par <- fda::fdPar(basis, Lfdobj = 4, lambda = best_lambda)
  fit <- fda::smooth.basis(time_grid, greenness, fd_par)

  fit$fd$fdnames <- list("Day of year", "camera_id" = as.list(colnames(greenness)), "Greenness")
  attr(fit$fd, "lambda") <- best_lambda
  fit$fd
}

#' Varimax-rotated functional PCA of a smoothed greenness `fd` object.
#'
#' Thin, documented wrapper around `fda::pca.fd()` + `fda::varmx.pca.fd()`.
#'
#' @param smoothed_fd An `fd` object, e.g. from `smooth_greenness_gcv()`.
#' @param n_components Number of functional principal components to retain.
#' @return The rotated `pca.fd` result (see `fda::varmx.pca.fd`), with
#'   `$scores`, `$harmonics`, `$meanfd`, `$varprop` as usual.
rotated_fpca <- function(smoothed_fd, n_components = 3) {
  pca <- fda::pca.fd(smoothed_fd, nharm = n_components)
  fda::varmx.pca.fd(pca)
}
