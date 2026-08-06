"""Generate a couple of illustrative figures for the README / results folder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from greenness.data import build_dataset
from greenness.kriging import fit_ucok, fit_utrk
from greenness.smoothing import smooth_with_gcv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def main():
    curves, coords = build_dataset(
        raw_dir=RAW / "greenness_curves",
        camera_id_csv=RAW / "CameraID.csv",
        locations_xlsx=RAW / "amos_locations.xlsx",
        year=2015,
        fill_missing="median",
    )
    doy = curves.index.to_numpy(dtype=float)

    # Figure 1: raw vs smoothed curves for a handful of cameras
    smooth = smooth_with_gcv(curves.to_numpy(), doy, nbasis=90, norder=6)
    fitted = smooth.eval()

    fig, ax = plt.subplots(figsize=(9, 5))
    sample_cams = list(curves.columns[:8])
    for i, cam in enumerate(sample_cams):
        j = list(curves.columns).index(cam)
        ax.plot(doy, curves[cam], color=f"C{i}", alpha=0.25, lw=1)
        ax.plot(doy, fitted[:, j], color=f"C{i}", lw=2, label=f"cam {cam}")
    ax.set_xlabel("Day of year (2015)")
    ax.set_ylabel("Relative greenness")
    ax.set_title("Raw (faint) vs GCV B-spline-smoothed (bold) greenness curves")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "smoothed_curves_2015.png", dpi=150)
    plt.close(fig)

    # Figure 2: UTrK vs UCoK prediction for one held-out camera
    rng = np.random.default_rng(3)
    cams = list(curves.columns)
    test_cam = rng.choice(cams)
    train_cams = [c for c in cams if c != test_cam]

    train_curves = curves[train_cams].to_numpy()
    train_coords = coords.loc[train_cams].to_numpy()
    test_coords = coords.loc[[test_cam]].to_numpy()
    true_curve = curves[test_cam].to_numpy()

    utrk = fit_utrk(train_curves, train_coords, test_coords, model="Exp", n_lags=15)
    ucok = fit_ucok(train_curves, doy, train_coords, test_coords, nbasis=90, norder=6, n_components=3)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(doy, train_curves, color="grey", alpha=0.15, lw=1)
    ax.plot(doy, true_curve, color="black", lw=2, label=f"true curve (camera {test_cam}, held out)")
    ax.plot(doy, utrk.prediction[:, 0], color="crimson", lw=2, ls="--", label="UTrK prediction")
    ax.plot(doy, ucok.prediction[:, 0], color="royalblue", lw=2, ls="--", label="UCoK prediction")
    ax.set_xlabel("Day of year (2015)")
    ax.set_ylabel("Relative greenness")
    ax.set_title("Leave-one-camera-out prediction: UTrK vs UCoK")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "utrk_vs_ucok_example.png", dpi=150)
    plt.close(fig)

    print("Wrote:", list(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
