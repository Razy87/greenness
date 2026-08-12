"""
B-spline smoothing with a roughness penalty and GCV-selected smoothing
parameter.

Mirrors ``gcv_lambda_search`` and ``smooth_greenness_gcv`` in
``R/03_functional_smoothing.R``, which are thin wrappers around the R
`fda` package's ``create.bspline.basis`` / ``fdPar`` / ``smooth.basis``.
Here the same smoothing-spline problem

    minimize_c  ||y - B c||^2 + lambda * c' R c

is solved directly with numpy/scipy, where ``B`` is the B-spline design
matrix and ``R`` is the roughness penalty matrix built from the spline's
4th derivative (``Lfdobj = 4`` in the R implementation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class BSplineBasis:
    """A clamped, evenly-knotted B-spline basis on ``[domain[0], domain[1]]``.

    Equivalent to ``fda::create.bspline.basis(domain, nbasis, norder)``.
    """

    nbasis: int
    norder: int = 6  # order = degree + 1; norder=6 -> quintic splines (matches the R scripts)
    domain: tuple[float, float] = (1.0, 365.0)

    def __post_init__(self):
        degree = self.norder - 1
        n_interior = self.nbasis - self.norder
        if n_interior < 0:
            raise ValueError("nbasis must be >= norder")
        lo, hi = self.domain
        interior = np.linspace(lo, hi, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
        self.knots = np.concatenate([[lo] * self.norder, interior, [hi] * self.norder])
        self.degree = degree
        # A BSpline whose "coefficients" are the identity matrix evaluates to
        # the full basis design matrix -- convenient for both the basis
        # itself and (via .derivative) for any derivative order.
        self._basis_spline = BSpline(self.knots, np.eye(self.nbasis), degree, extrapolate=False)

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Return the (len(x), nbasis) matrix B with B[i, j] = phi_j(x[i])."""
        x = np.clip(x, self.domain[0], self.domain[1])
        return self._basis_spline(x)

    def roughness_matrix(self, deriv: int = 4, n_quad: int = 2000) -> np.ndarray:
        """Return R[j, k] = integral of phi_j^(deriv)(t) * phi_k^(deriv)(t) dt.

        Computed by Simpson's rule on a fine grid, which is accurate to
        machine precision for the smooth piecewise-polynomial integrands
        here and avoids needing a symbolic/analytic Gram-matrix derivation.
        """
        if deriv > self.degree:
            # The deriv-th derivative of a degree-`degree` piecewise polynomial
            # is identically zero almost everywhere (ignoring knot jump
            # discontinuities), so the roughness penalty is (numerically) zero.
            return np.zeros((self.nbasis, self.nbasis))
        dspline = self._basis_spline.derivative(deriv)
        lo, hi = self.domain
        grid = np.linspace(lo, hi, n_quad)
        D = dspline(grid)  # (n_quad, nbasis)
        return _pairwise_simpson(D, grid)


def _pairwise_simpson(D: np.ndarray, grid: np.ndarray) -> np.ndarray:
    from scipy.integrate import simpson

    nbasis = D.shape[1]
    R = np.empty((nbasis, nbasis))
    for j in range(nbasis):
        prod = D[:, j : j + 1] * D  # (n_quad, nbasis), column j against all columns
        R[j, :] = simpson(prod, x=grid, axis=0)
    return R


def gcv_lambda_search(
    y: np.ndarray,
    x: np.ndarray,
    nbasis: int,
    norder: int = 6,
    domain: tuple[float, float] | None = None,
    log10_lambda_grid: np.ndarray | None = None,
    deriv: int = 4,
) -> tuple[float, "np.ndarray"]:
    """Search a grid of smoothing parameters and return the GCV-optimal lambda.

    Equivalent to ``gcv_lambda_search`` in ``R/03_functional_smoothing.R``
    (``log10_lambda_grid = seq(-3, 8, 0.25)`` by default there).

    Parameters
    ----------
    y : array, shape (n_obs, n_curves)
    x : array, shape (n_obs,)

    Returns
    -------
    (best_lambda, table)
        ``table`` has columns [log10_lambda, df, gcv], one row per grid point.
    """
    if domain is None:
        domain = (float(np.min(x)), float(np.max(x)))
    if log10_lambda_grid is None:
        log10_lambda_grid = np.arange(-3, 8 + 1e-9, 0.25)

    basis = BSplineBasis(nbasis=nbasis, norder=norder, domain=domain)
    B = basis.design_matrix(x)
    R = basis.roughness_matrix(deriv=deriv)
    BtB = B.T @ B
    Bty = B.T @ y
    n = x.shape[0]

    rows = []
    for loglam in log10_lambda_grid:
        lam = 10.0**loglam
        A = BtB + lam * R
        # hat matrix trace (degrees of freedom), computed once per lambda
        H_diag_trace = np.trace(np.linalg.solve(A, BtB))
        coefs = np.linalg.solve(A, Bty)
        fitted = B @ coefs
        rss = np.sum((y - fitted) ** 2, axis=0)  # per curve
        df = H_diag_trace
        denom = max(n - df, 1e-6)
        gcv = np.sum((n * rss) / (denom**2))  # summed over curves, matching sum(smoothlist$gcv)
        rows.append((loglam, df, gcv))

    table = np.array(rows)
    best_idx = np.argmin(table[:, 2])
    best_lambda = 10.0 ** table[best_idx, 0]
    return best_lambda, table


@dataclass
class SmoothResult:
    basis: BSplineBasis
    coefficients: np.ndarray  # (nbasis, n_curves)
    lambda_: float
    x: np.ndarray

    def eval(self, x: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the smoothed curves at ``x`` (defaults to the fit grid)."""
        if x is None:
            x = self.x
        return self.basis.design_matrix(x) @ self.coefficients


def smooth_with_gcv(
    y: np.ndarray,
    x: np.ndarray,
    nbasis: int,
    norder: int = 6,
    deriv: int = 4,
    domain: tuple[float, float] | None = None,
) -> SmoothResult:
    """One-call convenience wrapper: GCV-select lambda, then fit.

    y : array, shape (n_obs, n_curves) e.g. the (365 x n_camera) greenness matrix.
    x : array, shape (n_obs,) e.g. day-of-year 1..365.
    """
    if domain is None:
        domain = (float(np.min(x)), float(np.max(x)))
    lam, _table = gcv_lambda_search(y, x, nbasis, norder, domain, deriv=deriv)
    basis = BSplineBasis(nbasis=nbasis, norder=norder, domain=domain)
    B = basis.design_matrix(x)
    R = basis.roughness_matrix(deriv=deriv)
    A = B.T @ B + lam * R
    coefs = np.linalg.solve(A, B.T @ y)
    return SmoothResult(basis=basis, coefficients=coefs, lambda_=lam, x=x)
