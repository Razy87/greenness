"""
Functional PCA + varimax rotation on B-spline-smoothed curves.

Mirrors ``pca.fd`` / ``varmx.pca.fd`` from the R `fda` package, as used in
``R/my_kriging_functions.R`` (``Fit_Ucok_rpc``) and
``R/UCoK& UTrK.R``.

Algorithm (standard functional-PCA-via-basis-expansion, Ramsay & Silverman
2005, ch. 8): if curves are represented by B-spline coefficients ``C``
(nbasis x n_curves) and ``W`` is the basis Gram matrix (``W[i,j] = integral
phi_i * phi_j``), then functional PCA of the curves is equivalent to
ordinary PCA of ``L' C`` where ``W = L L'`` (Cholesky). Harmonics and
scores are mapped back to coefficient space / curve space afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .smoothing import BSplineBasis, SmoothResult


@dataclass
class FPCAResult:
    mean_coefs: np.ndarray  # (nbasis,)
    harmonic_coefs: np.ndarray  # (nbasis, n_components)
    values: np.ndarray  # (n_components,) eigenvalues
    varprop: np.ndarray  # (n_components,) proportion of variance explained
    scores: np.ndarray  # (n_curves, n_components)
    basis: BSplineBasis

    def eval_mean(self, x: np.ndarray) -> np.ndarray:
        return self.basis.design_matrix(x) @ self.mean_coefs

    def eval_harmonics(self, x: np.ndarray) -> np.ndarray:
        return self.basis.design_matrix(x) @ self.harmonic_coefs  # (len(x), n_components)

    def reconstruct(self, x: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Reconstruct curves at ``x`` from (possibly predicted) scores.

        scores: (n_curves, n_components) -> returns (len(x), n_curves).
        """
        mean_vals = self.eval_mean(x)[:, None]
        harm_vals = self.eval_harmonics(x)  # (len(x), n_components)
        return mean_vals + harm_vals @ scores.T


def pca_fd(smooth: SmoothResult, n_components: int) -> FPCAResult:
    """Functional PCA of the curves represented by ``smooth.coefficients``."""
    C = smooth.coefficients  # (nbasis, n_curves)
    nbasis, n_curves = C.shape
    W = smooth.basis.roughness_matrix(deriv=0)  # Gram / mass matrix

    mean_coefs = C.mean(axis=1)
    Cc = C - mean_coefs[:, None]

    L = np.linalg.cholesky(W)  # W = L L'
    Z = L.T @ Cc  # (nbasis, n_curves)

    S = (Z @ Z.T) / max(n_curves - 1, 1)  # (nbasis, nbasis) covariance in transformed space
    eigvals, eigvecs = np.linalg.eigh(S)  # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.clip(eigvals[:n_components], 0, None)
    U = eigvecs[:, :n_components]  # (nbasis, n_components), orthonormal columns

    harmonic_coefs = np.linalg.solve(L.T, U)  # L^{-T} U -> L2-orthonormal harmonics in coef space
    scores = Z.T @ U  # (n_curves, n_components)

    total_var = np.trace(S)
    varprop = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)

    return FPCAResult(
        mean_coefs=mean_coefs,
        harmonic_coefs=harmonic_coefs,
        values=eigvals,
        varprop=varprop,
        scores=scores,
        basis=smooth.basis,
    )


def varimax(Phi: np.ndarray, normalize: bool = True, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Classic Kaiser varimax rotation. Returns the (k x k) rotation matrix R
    such that ``Phi @ R`` is the varimax-rotated loading matrix.

    Equivalent to R's ``stats::varimax`` (used internally by
    ``fda::varmx.pca.fd``), with the default ``normalize = TRUE``
    (Kaiser row-normalization while searching for the rotation).
    """
    p, k = Phi.shape
    if k < 2:
        return np.eye(k)

    if normalize:
        h = np.sqrt(np.sum(Phi**2, axis=1))
        h[h == 0] = 1.0
        X = Phi / h[:, None]
    else:
        X = Phi

    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Lambda = X @ R
        u, s, vt = np.linalg.svd(
            X.T @ (Lambda**3 - (1.0 / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vt
        d_new = np.sum(s)
        if d_new < d * (1 + tol):
            break
        d = d_new
    return R


def varimax_rotate(fpca: FPCAResult) -> FPCAResult:
    """Varimax-rotate the harmonics of an FPCAResult (equivalent to
    ``varmx.pca.fd``). Scores and harmonics are both rotated by the same
    orthogonal matrix so that ``mean + harmonics_rot @ scores_rot.T``
    reconstructs the same curves as before rotation.
    """
    R = varimax(fpca.harmonic_coefs)
    rotated_harmonics = fpca.harmonic_coefs @ R
    rotated_scores = fpca.scores @ R
    # Re-derive an eigenvalue-like "variance explained" for the rotated
    # components (their scores are no longer independent/ordered, but the
    # marginal variance of each rotated score is still a useful summary).
    rotated_values = rotated_scores.var(axis=0, ddof=1)
    total_var = fpca.values.sum() if fpca.values.sum() > 0 else 1.0
    rotated_varprop = rotated_values / total_var

    return FPCAResult(
        mean_coefs=fpca.mean_coefs,
        harmonic_coefs=rotated_harmonics,
        values=rotated_values,
        varprop=rotated_varprop,
        scores=rotated_scores,
        basis=fpca.basis,
    )
