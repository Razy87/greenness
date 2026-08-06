"""Basic sanity tests -- run with: pytest tests/"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from greenness.data import build_dataset
from greenness.fpca import pca_fd, varimax_rotate
from greenness.kriging import fit_ucok, fit_utrk
from greenness.smoothing import smooth_with_gcv
from greenness.variogram import empirical_variogram, fit_variogram

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _load():
    return build_dataset(
        raw_dir=RAW / "greenness_curves",
        camera_id_csv=RAW / "CameraID.csv",
        locations_xlsx=RAW / "amos_locations.xlsx",
        year=2015,
        fill_missing="median",
    )


def test_build_dataset_shapes():
    curves, coords = _load()
    assert curves.shape[0] == 365
    assert curves.shape[1] == coords.shape[0]
    assert not curves.isna().any().any()


def test_smoothing_reduces_roughness():
    curves, coords = _load()
    doy = curves.index.to_numpy(dtype=float)
    smooth = smooth_with_gcv(curves.to_numpy(), doy, nbasis=40, norder=6)
    fitted = smooth.eval()
    raw_roughness = np.mean(np.diff(curves.to_numpy(), n=2, axis=0) ** 2)
    smoothed_roughness = np.mean(np.diff(fitted, n=2, axis=0) ** 2)
    assert smoothed_roughness < raw_roughness


def test_fpca_reconstructs_mean():
    curves, coords = _load()
    doy = curves.index.to_numpy(dtype=float)
    smooth = smooth_with_gcv(curves.to_numpy(), doy, nbasis=40, norder=6)
    fpca = pca_fd(smooth, n_components=3)
    fpca_rot = varimax_rotate(fpca)
    # Reconstructing with the actual (non-rotated / rotated) scores should
    # approximate the smoothed curves reasonably well with only 3 PCs.
    recon = fpca_rot.reconstruct(doy, fpca_rot.scores)
    fitted = smooth.eval()
    rel_err = np.linalg.norm(recon - fitted) / np.linalg.norm(fitted)
    assert rel_err < 0.5


def test_variogram_fit_is_finite():
    curves, coords = _load()
    emp = empirical_variogram(coords.to_numpy(), curves.iloc[0].to_numpy(), n_lags=10)
    vgm = fit_variogram(emp, model="Exp")
    assert np.isfinite(vgm.sill) and np.isfinite(vgm.range_) and vgm.range_ > 0


def test_utrk_and_ucok_predict_finite_curves():
    curves, coords = _load()
    doy = curves.index.to_numpy(dtype=float)
    cams = list(curves.columns)
    test_cam = cams[0]
    train_cams = cams[1:]

    train_curves = curves[train_cams].to_numpy()
    train_coords = coords.loc[train_cams].to_numpy()
    test_coords = coords.loc[[test_cam]].to_numpy()

    utrk = fit_utrk(train_curves, train_coords, test_coords, model="Exp", n_lags=15)
    ucok = fit_ucok(train_curves, doy, train_coords, test_coords, nbasis=40, norder=6, n_components=3)

    assert utrk.prediction.shape == (365, 1)
    assert ucok.prediction.shape == (365, 1)
    assert np.all(np.isfinite(utrk.prediction))
    assert np.all(np.isfinite(ucok.prediction))
