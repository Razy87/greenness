#' ============================================================================
#' Optimal sampling design: where should the next camera go?
#' ============================================================================
#'
#' An extension beyond the published paper: given the existing camera
#' network, use simulated annealing to search candidate locations for a new
#' camera that minimizes the mean UTrK/UCoK kriging variance -- i.e. finds
#' where a new observation would most reduce prediction uncertainty. This is
#' a small, self-contained annealer with no external dependencies beyond the
#' packages loaded in 00_setup.R.
#'
#' Source 00_setup.R and 04_kriging_models.R before this file.

#' Mean UTrK kriging variance for a candidate set of new locations.
#'
#' @param candidate_coords data.frame(lon, lat) of candidate new locations.
#' @param greenness,train_coords Training data (as in `fit_utrk()`).
#' @param ... Passed to `fit_utrk()` (model, n_lags, lag_max).
objective_utrk_variance <- function(candidate_coords, greenness, train_coords, ...) {
  fit <- fit_utrk(greenness, train_coords, candidate_coords, ...)
  mean(fit$variance)
}

#' Mean total UCoK score kriging variance for a candidate set of new locations.
#'
#' @param candidate_coords data.frame(lon, lat) of candidate new locations.
#' @param greenness,train_coords Training data (as in `fit_ucok()`).
#' @param ... Passed to `fit_ucok()` (nbasis, norder, n_components, model, n_lags, lag_max).
objective_ucok_variance <- function(candidate_coords, greenness, train_coords, ...) {
  fit <- fit_ucok(greenness, train_coords = train_coords, new_coords = candidate_coords, ...)
  mean(rowSums(fit$score_variance))
}

#' Propose a neighbouring candidate solution by swapping one site for its
#' nearest not-yet-selected neighbour in `candidate_pool`.
#'
#' @param current_sites data.frame(lon, lat) of the current candidate set.
#' @param candidate_pool data.frame(lon, lat) of all eligible replacement sites.
#' @param i Index (row of `current_sites`) to swap out.
propose_swap <- function(current_sites, candidate_pool, i) {
  pool <- candidate_pool[!(rownames(candidate_pool) %in% rownames(current_sites)), , drop = FALSE]
  if (nrow(pool) == 0) return(current_sites)

  dists <- sp::spDists(as.matrix(current_sites[i, , drop = FALSE]), as.matrix(pool))
  nearest <- pool[which.min(dists), , drop = FALSE]

  updated <- current_sites
  updated[i, ] <- nearest
  updated
}

#' Simulated annealing search for the variance-minimizing new-site layout.
#'
#' @param initial_sites data.frame(lon, lat): starting candidate set (one row
#'   per new site to place).
#' @param candidate_pool data.frame(lon, lat): all eligible replacement sites.
#' @param objective_fn One of `objective_utrk_variance` / `objective_ucok_variance`.
#' @param objective_args Named list of extra arguments for `objective_fn`
#'   (greenness, train_coords, model, n_lags, lag_max, ...).
#' @param n_iter Number of annealing iterations.
#' @param alpha Cooling rate (temperature multiplier per iteration).
#' @param p0 Target initial acceptance probability, used to calibrate the
#'   starting temperature.
#' @param track If TRUE, also return the fitness trace for diagnostics.
#' @return A list with `sites` (best layout found), `fitness` (its
#'   objective value), and optionally `trace`.
optimize_sampling_design <- function(initial_sites, candidate_pool, objective_fn, objective_args,
                                      n_iter = 50, alpha = 0.95, p0 = 0.9, track = FALSE) {
  evaluate <- function(sites) do.call(objective_fn, c(list(candidate_coords = sites), objective_args))

  n_sites <- nrow(initial_sites)
  current <- initial_sites
  best <- current
  current_fit <- evaluate(current)
  best_fit <- current_fit

  # Calibrate the initial temperature from two random single-site swaps.
  probe_a <- evaluate(propose_swap(current, candidate_pool, sample.int(n_sites, 1)))
  probe_b <- evaluate(propose_swap(current, candidate_pool, sample.int(n_sites, 1)))
  temperature <- -mean(abs(probe_a - probe_b)) / log(p0)

  trace <- if (track) numeric(n_iter) else NULL

  for (iter in seq_len(n_iter)) {
    move_site <- sample.int(n_sites, 1)
    candidate <- propose_swap(current, candidate_pool, move_site)
    candidate_fit <- evaluate(candidate)

    if (exp(-(candidate_fit - current_fit) / temperature) > stats::runif(1)) {
      current <- candidate
      current_fit <- candidate_fit
    }
    if (candidate_fit <= best_fit) {
      best <- candidate
      best_fit <- candidate_fit
    }
    if (track) trace[iter] <- current_fit
    temperature <- alpha * temperature
  }

  result <- list(sites = best, fitness = best_fit)
  if (track) result$trace <- trace
  result
}
