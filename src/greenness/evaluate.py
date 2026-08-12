"""
Monte Carlo train/test RMSPE evaluation.

Mirrors ``monte_carlo_iteration`` / ``run_monte_carlo_study`` in
``R/05_monte_carlo_evaluation.R``: repeatedly hold out a random subset of
cameras, predict their curves from the remaining ("training") cameras via
UTrK and UCoK, and compute the root-mean-squared prediction error (RMSPE)
per held-out camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .kriging import fit_ucok, fit_utrk


@dataclass
class MCResult:
    per_iteration: pd.DataFrame  # one row per MC iteration: mean RMSPE for UTrK & UCoK
    per_camera: list[pd.DataFrame] = field(default_factory=list)  # per-iteration, per-held-out-camera RMSPE


def rmspe(true_curves: np.ndarray, pred_curves: np.ndarray) -> np.ndarray:
    """Per-column (per-camera) root-mean-squared prediction error."""
    return np.sqrt(np.mean((true_curves - pred_curves) ** 2, axis=0))


def monte_carlo_rmspe(
    curves: pd.DataFrame,
    coords: pd.DataFrame,
    train_frac: float = 0.8,
    n_iter: int = 300,
    seed: int = 2024,
    utrk_kwargs: dict | None = None,
    ucok_kwargs: dict | None = None,
) -> MCResult:
    """Equivalent to running ``monte_carlo_iteration()`` (R) ``n_iter`` times.

    curves: (n_time, n_camera) DataFrame, columns = camera_id, index = DOY.
    coords: (n_camera, [lat, lon]) DataFrame aligned to curves.columns.
    """
    utrk_kwargs = utrk_kwargs or {}
    ucok_kwargs = ucok_kwargs or {}
    rng = np.random.default_rng(seed)

    cams = list(curves.columns)
    n_cam = len(cams)
    n_train = int(np.floor(train_frac * n_cam))
    doy = curves.index.to_numpy(dtype=float)

    rows = []
    per_camera_frames = []
    for it in range(n_iter):
        train_idx = rng.choice(n_cam, size=n_train, replace=False)
        train_mask = np.zeros(n_cam, dtype=bool)
        train_mask[train_idx] = True

        train_cams = [c for c, m in zip(cams, train_mask) if m]
        test_cams = [c for c, m in zip(cams, train_mask) if not m]
        if not test_cams:
            continue

        train_curves = curves[train_cams].to_numpy()
        test_curves = curves[test_cams].to_numpy()
        train_coords = coords.loc[train_cams].to_numpy()
        test_coords = coords.loc[test_cams].to_numpy()

        utrk = fit_utrk(train_curves, train_coords, test_coords, **utrk_kwargs)
        ucok = fit_ucok(train_curves, doy, train_coords, test_coords, **ucok_kwargs)

        rmspe_utrk = rmspe(test_curves, utrk.prediction)
        rmspe_ucok = rmspe(test_curves, ucok.prediction)

        per_camera_frames.append(
            pd.DataFrame(
                {"camera_id": test_cams, "RMSPE_UTrK": rmspe_utrk, "RMSPE_UCoK": rmspe_ucok, "iter": it}
            )
        )
        rows.append(
            {
                "iter": it,
                "mean_RMSPE_UTrK": rmspe_utrk.mean(),
                "mean_RMSPE_UCoK": rmspe_ucok.mean(),
                "median_RMSPE_UTrK": np.median(rmspe_utrk),
                "median_RMSPE_UCoK": np.median(rmspe_ucok),
            }
        )

    return MCResult(per_iteration=pd.DataFrame(rows), per_camera=per_camera_frames)


def summarize(mc: MCResult) -> pd.DataFrame:
    """Overall mean/median/std across all MC iterations, mirroring the
    ``mean_RMSPE`` / ``median_RMSPE`` / ``Std.dev_RMSPE`` outputs of the R
    implementation."""
    df = mc.per_iteration
    summary = {
        "mean_RMSPE_UTrK": df["mean_RMSPE_UTrK"].mean(),
        "mean_RMSPE_UCoK": df["mean_RMSPE_UCoK"].mean(),
        "median_RMSPE_UTrK": df["mean_RMSPE_UTrK"].median(),
        "median_RMSPE_UCoK": df["mean_RMSPE_UCoK"].median(),
        "std_RMSPE_UTrK": df["mean_RMSPE_UTrK"].std(),
        "std_RMSPE_UCoK": df["mean_RMSPE_UCoK"].std(),
    }
    return pd.DataFrame([summary])
