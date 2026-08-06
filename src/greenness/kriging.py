"""
Universal Trace-Kriging (UTrK) and Universal (Co)Kriging of rotated FPC
scores (UCoK).

Mirrors ``Fit_Utrk`` and ``Fit_Ucok_rpc`` in ``R/my_kriging_functions.R``,
built on the `fdagstat` package's ``fstat`` / ``estimateDrift`` /
``fvariogram`` / ``fitVariograms`` / ``predictFstat`` pipeline.

Two simplifying facts about the original R implementation carried over
here on purpose (documented in the README):

1. ``estimateDrift("~.", g, Intercept = TRUE)`` with no extra covariates
   only ever estimates a constant (intercept) drift, so "universal"
   kriging here is mathematically Ordinary Kriging with a global mean.
2. ``predictFstat(..., .type = "UcoK", algIndependent = TRUE)`` krige's
   each rotated FPC score independently (not a true multivariate
   cross-covariance cokriging) -- this is what "UCoK" means in this
   codebase, and it's what we reproduce here.

Because Ordinary/Universal-Kriging weights depend only on the training
locations, the prediction locations, and the variogram model (not on the
values themselves), a single weight matrix is computed once and can be
applied either to raw curves (UTrK) or to FPC scores (UCoK).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from .fpca import FPCAResult, pca_fd, varimax_rotate
from .smoothing import smooth_with_gcv
from .variogram import FittedVariogram, empirical_variogram, fit_variogram


@dataclass
class KrigingWeights:
    weights: np.ndarray  # (n_test, n_train)
    variance: np.ndarray  # (n_test,) kriging variance (per scalar field used to fit the variogram)


def ordinary_kriging_weights(
    train_coords: np.ndarray, test_coords: np.ndarray, variogram: FittedVariogram
) -> KrigingWeights:
    """Solve the Ordinary/Universal-(intercept-only)-Kriging linear system.

    Standard system (Lagrange multiplier form):

        [ Gamma   1 ] [ w ]   [ gamma0 ]
        [ 1'      0 ] [ mu] = [   1    ]

    solved for every test location at once.
    """
    train_coords = np.asarray(train_coords, dtype=float)
    test_coords = np.asarray(test_coords, dtype=float)
    n = train_coords.shape[0]

    D = cdist(train_coords, train_coords)
    Gamma = variogram(D)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Gamma
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A_inv = np.linalg.pinv(A)  # pinv: robust to a near-singular variogram matrix

    d0 = cdist(test_coords, train_coords)
    Gamma0 = variogram(d0)
    rhs = np.hstack([Gamma0, np.ones((test_coords.shape[0], 1))])  # (m, n+1)

    sol = rhs @ A_inv.T  # (m, n+1)
    weights = sol[:, :n]
    mu = sol[:, n]
    variance = np.sum(weights * Gamma0, axis=1) + mu
    return KrigingWeights(weights=weights, variance=variance)


@dataclass
class UTrKResult:
    prediction: np.ndarray  # (n_time, n_test) predicted greenness curves
    variance: np.ndarray  # (n_test,) trace-kriging variance
    variogram: FittedVariogram
    weights: KrigingWeights


def fit_utrk(
    curves: np.ndarray,
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    model: str = "Exp",
    n_lags: int = 15,
    max_dist: float | None = None,
) -> UTrKResult:
    """Universal Trace-Kriging: one scalar trace-variogram fit on the raw
    curves, whose kriging weights are then applied directly to the curves
    themselves. Equivalent to ``Fit_Utrk`` in the R code.

    curves: (n_time, n_train) matrix, e.g. the smoothed 365 x n greenness matrix.
    """
    emp = empirical_variogram(train_coords, curves.T, n_lags=n_lags, max_dist=max_dist)
    vgm = fit_variogram(emp, model=model)
    kw = ordinary_kriging_weights(train_coords, test_coords, vgm)
    prediction = curves @ kw.weights.T  # (n_time, n_test)
    return UTrKResult(prediction=prediction, variance=kw.variance, variogram=vgm, weights=kw)


@dataclass
class UCoKResult:
    prediction: np.ndarray  # (n_time, n_test) reconstructed predicted curves
    score_variance: np.ndarray  # (n_test, n_components) per-component kriging variance
    total_variance: np.ndarray  # (n_test,) summed component variance (matches Sigma2_Ucok in R)
    fpca: FPCAResult
    variograms: list[FittedVariogram]


def fit_ucok(
    curves: np.ndarray,
    doy: np.ndarray,
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    nbasis: int = 100,
    norder: int = 6,
    n_components: int = 3,
    model: str = "Exp",
    n_lags: int = 15,
    max_dist: float | None = None,
    force_nugget: bool = True,
) -> UCoKResult:
    """Universal (Co)Kriging of varimax-rotated FPC scores. Equivalent to
    ``Fit_Ucok_rpc`` in the R code (with ``Pc_Number`` = ``n_components``).

    curves: (n_time, n_train) matrix.
    doy: (n_time,) the day-of-year (or other) x-axis the curves are sampled on.
    """
    smooth = smooth_with_gcv(curves, doy, nbasis=nbasis, norder=norder)
    fpca = pca_fd(smooth, n_components=n_components)
    fpca_rot = varimax_rotate(fpca)

    score_preds = np.zeros((test_coords.shape[0], n_components))
    score_vars = np.zeros((test_coords.shape[0], n_components))
    variograms = []
    for k in range(n_components):
        scores_k = fpca_rot.scores[:, k]
        emp = empirical_variogram(train_coords, scores_k, n_lags=n_lags, max_dist=max_dist)
        vgm = fit_variogram(emp, model=model, force_nugget=0.0 if force_nugget else None)
        kw = ordinary_kriging_weights(train_coords, test_coords, vgm)
        score_preds[:, k] = kw.weights @ scores_k
        score_vars[:, k] = kw.variance
        variograms.append(vgm)

    prediction = fpca_rot.reconstruct(doy, score_preds)  # (n_time, n_test)
    total_variance = score_vars.sum(axis=1)

    return UCoKResult(
        prediction=prediction,
        score_variance=score_vars,
        total_variance=total_variance,
        fpca=fpca_rot,
        variograms=variograms,
    )
