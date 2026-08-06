"""
Empirical variogram estimation and model fitting.

Mirrors the ``fvariogram`` / ``fitVariograms`` calls on the `fdagstat`
package object in the R scripts (``models <- c("Gau","Mat","Sph","Exp")``,
``vgm(...)``). `fdagstat` computes a single scalar "trace" (semi)variogram
from the functional data (or from a scalar field, for the FPC-score
kriging step) and fits one of these classical isotropic models to it by
weighted least squares.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma as gamma_fn
from scipy.special import kv


def _exponential(h, nugget, sill, rng):
    return nugget + (sill - nugget) * (1 - np.exp(-h / rng))


def _spherical(h, nugget, sill, rng):
    hs = np.clip(h / rng, 0, 1)
    return nugget + (sill - nugget) * (1.5 * hs - 0.5 * hs**3)


def _gaussian(h, nugget, sill, rng):
    return nugget + (sill - nugget) * (1 - np.exp(-((h / rng) ** 2)))


def _matern(h, nugget, sill, rng, nu=0.5):
    # nu=0.5 reduces exactly to the exponential model, matching gstat's
    # default kappa=0.5 for vgm("Mat") -- the R scripts never override kappa.
    if nu == 0.5:
        return _exponential(h, nugget, sill, rng)
    h = np.asarray(h, dtype=float)
    out = np.full_like(h, sill, dtype=float)
    mask = h > 0
    scaled = np.sqrt(2 * nu) * h[mask] / rng
    out[mask] = nugget + (sill - nugget) * (
        1 - (2 ** (1 - nu) / gamma_fn(nu)) * (scaled**nu) * kv(nu, scaled)
    )
    return out


MODELS: dict[str, Callable] = {
    "Exp": _exponential,
    "Sph": _spherical,
    "Gau": _gaussian,
    "Mat": _matern,
}


@dataclass
class EmpiricalVariogram:
    lags: np.ndarray
    gamma: np.ndarray
    n_pairs: np.ndarray


def empirical_variogram(
    coords: np.ndarray, values: np.ndarray, n_lags: int = 15, max_dist: float | None = None
) -> EmpiricalVariogram:
    """Classical (Matheron) empirical semivariogram, binned by distance.

    ``values`` may be:
      * a 1-D array of length n (scalar field, e.g. one FPC score), or
      * a 2-D array of shape (n, n_time) (a "trace" variogram over curves,
        matching `fdagstat`'s functional variogram: semivariance is the
        mean squared L2 difference between curve pairs).
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = coords.shape[0]

    dist = squareform(pdist(coords))
    iu = np.triu_indices(n, k=1)
    h = dist[iu]

    if values.ndim == 1:
        diffs = values[:, None] - values[None, :]
        sq = (diffs[iu]) ** 2
    else:
        diffs = values[:, None, :] - values[None, :, :]  # (n, n, n_time)
        sq = np.mean(diffs[iu[0], iu[1], :] ** 2, axis=1)  # mean over time -> "trace" semivariance

    if max_dist is None:
        max_dist = h.max()
    bins = np.linspace(0, max_dist, n_lags + 1)
    bin_idx = np.digitize(h, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_lags - 1)

    lags, gammas, counts = [], [], []
    for b in range(n_lags):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        lags.append(h[mask].mean())
        gammas.append(0.5 * sq[mask].mean())
        counts.append(mask.sum())

    return EmpiricalVariogram(np.array(lags), np.array(gammas), np.array(counts))


@dataclass
class FittedVariogram:
    model: str
    nugget: float
    sill: float
    range_: float
    nu: float = 0.5

    def __call__(self, h):
        fn = MODELS[self.model]
        if self.model == "Mat":
            return fn(h, self.nugget, self.sill, self.range_, self.nu)
        return fn(h, self.nugget, self.sill, self.range_)


def fit_variogram(
    emp: EmpiricalVariogram, model: str = "Exp", force_nugget: float | None = None
) -> FittedVariogram:
    """Weighted least-squares fit of a variogram model to empirical bins,
    weighting each bin by its pair count (matching gstat's default
    ``fit.method = 7`` weighting up to a constant)."""
    fn = MODELS[model]
    h, g, w = emp.lags, emp.gamma, emp.n_pairs.astype(float)
    sill0 = float(np.max(g)) if len(g) else 1.0
    range0 = float(np.max(h)) / 2 if len(h) else 1.0
    nugget0 = float(np.min(g)) if len(g) else 0.0

    if force_nugget is not None:

        def resid(params):
            sill, rng = params
            pred = fn(h, force_nugget, sill, rng) if model != "Mat" else fn(h, force_nugget, sill, rng, 0.5)
            return np.sqrt(w) * (pred - g)

        from scipy.optimize import least_squares

        res = least_squares(resid, x0=[sill0, range0], bounds=([1e-12, 1e-6], [np.inf, np.inf]))
        sill, rng = res.x
        return FittedVariogram(model, force_nugget, float(sill), float(rng))

    def resid(params):
        nugget, sill, rng = params
        pred = fn(h, nugget, sill, rng) if model != "Mat" else fn(h, nugget, sill, rng, 0.5)
        return np.sqrt(w) * (pred - g)

    from scipy.optimize import least_squares

    res = least_squares(
        resid, x0=[nugget0, sill0, range0], bounds=([0, 1e-12, 1e-6], [np.inf, np.inf, np.inf])
    )
    nugget, sill, rng = res.x
    return FittedVariogram(model, float(nugget), float(sill), float(rng))
