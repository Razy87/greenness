"""
End-to-end demo of the full pipeline on the 2015 greenness dataset:

    load data -> smooth (GCV) -> FPCA + varimax -> UTrK & UCoK prediction
    -> Monte Carlo RMSPE evaluation

Mirrors the overall flow of ``R/Functional Analysis for year 2015.R`` +
``R/UCoK& UTrK.R`` + ``R/my_kriging_functions.R`` combined.

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from greenness.data import build_dataset
from greenness.evaluate import monte_carlo_rmspe, summarize
from greenness.kriging import fit_ucok, fit_utrk
from greenness.smoothing import smooth_with_gcv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    print("1) Loading data...")
    curves, coords = build_dataset(
        raw_dir=RAW / "greenness_curves",
        camera_id_csv=RAW / "CameraID.csv",
        locations_xlsx=RAW / "amos_locations.xlsx",
        year=2015,
        fill_missing="median",
    )
    print(f"   greenness matrix: {curves.shape[0]} days x {curves.shape[1]} cameras")
    print(f"   coords: {coords.shape}")

    doy = curves.index.to_numpy(dtype=float)

    print("2) Smoothing with GCV-selected lambda (nbasis=90, norder=6)...")
    smooth = smooth_with_gcv(curves.to_numpy(), doy, nbasis=90, norder=6)
    print(f"   selected lambda = {smooth.lambda_:.4g}")

    print("3) A single held-out-camera UTrK/UCoK prediction sanity check...")
    rng = np.random.default_rng(0)
    cams = list(curves.columns)
    test_cam = rng.choice(cams)
    train_cams = [c for c in cams if c != test_cam]

    train_curves = curves[train_cams].to_numpy()
    train_coords = coords.loc[train_cams].to_numpy()
    test_coords = coords.loc[[test_cam]].to_numpy()
    true_curve = curves[[test_cam]].to_numpy()

    utrk = fit_utrk(train_curves, train_coords, test_coords, model="Exp", n_lags=15)
    ucok = fit_ucok(train_curves, doy, train_coords, test_coords, nbasis=90, norder=6, n_components=3)

    rmspe_utrk = float(np.sqrt(np.mean((true_curve[:, 0] - utrk.prediction[:, 0]) ** 2)))
    rmspe_ucok = float(np.sqrt(np.mean((true_curve[:, 0] - ucok.prediction[:, 0]) ** 2)))
    print(f"   held-out camera {test_cam}: RMSPE UTrK={rmspe_utrk:.5f}  UCoK={rmspe_ucok:.5f}")

    print("4) Small Monte Carlo evaluation (5 iterations, 80% train) — smoke test only...")
    mc = monte_carlo_rmspe(
        curves,
        coords,
        train_frac=0.8,
        n_iter=5,
        seed=1,
        utrk_kwargs=dict(model="Exp", n_lags=15),
        ucok_kwargs=dict(nbasis=90, norder=6, n_components=3),
    )
    print(summarize(mc).to_string(index=False))
    print("\nDone -- pipeline runs end-to-end.")


if __name__ == "__main__":
    main()
