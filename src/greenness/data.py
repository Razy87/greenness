"""
Data loading utilities.

Mirrors ``load_greenness_dataset`` in ``R/02_load_and_clean_data.R``: read
one CSV per camera, pull out the greenness column for the requested year,
reindex to a full day-of-year (DOY) axis, and stitch everything into a
single (day x camera) matrix. Camera coordinates are then attached via the
case-number -> AMOS-camera-ID -> (lat, lon) lookup chain (``CameraID.csv``
and ``amos_locations.xlsx``).

The raw per-camera pipeline that produces these CSVs from AMOS webcam
images (the "uRoI" processing) is *not* reproduced here -- it is outside
the scope of the paper itself, which starts from these compiled daily
greenness time series.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DOY_RANGE = np.arange(1, 366)  # 1..365, non-leap-year day-of-year axis used throughout the paper


def _case_number(path: Path) -> int:
    """Extract the numeric case id from a filename like 'DFcamera37new.csv'."""
    m = re.search(r"DFcamera(\d+)new?\.csv", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse a case number from {path.name!r}")
    return int(m.group(1))


def load_camera_curves(raw_dir: str | Path, year: int = 2015) -> pd.DataFrame:
    """Load every per-camera CSV in ``raw_dir`` and assemble a (DOY x case) matrix.

    Each source file has a row id column, a ``DOY`` column, and one greenness
    column per year it happened to cover (not every camera has all three of
    2013/2014/2015). We keep only the column matching ``year`` and reindex
    every camera onto the full 1..365 DOY axis, inserting NaN for missing
    days -- exactly what the R script does implicitly via the DOY match/rbind
    loop in ``grFUN`` / the ``sapply`` + ``ldply`` block.

    Returns
    -------
    pd.DataFrame
        Index: DOY (1..365). Columns: case number (int), sorted ascending.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("DFcamera*new.csv")) or sorted(raw_dir.glob("DFcamera*.csv"))
    if not files:
        raise FileNotFoundError(f"No per-camera CSV files found in {raw_dir}")

    series = {}
    year_pat = re.compile(rf"Year{year}\b")
    for f in files:
        case = _case_number(f)
        df = pd.read_csv(f, index_col=0)
        year_cols = [c for c in df.columns if year_pat.search(c)]
        if not year_cols:
            continue  # this camera has no data at all for the requested year
        s = df.set_index("DOY")[year_cols[0]]
        s = s.reindex(DOY_RANGE)  # fill missing DOYs with NaN, drop any stray >365 rows
        series[case] = s

    mat = pd.DataFrame(series)
    mat = mat.reindex(sorted(mat.columns), axis=1)
    mat.index.name = "DOY"
    return mat


def load_case_to_camera_map(camera_id_csv: str | Path) -> pd.Series:
    """Load ``CameraID.csv`` (columns: "Case Number", "Camera Number")."""
    df = pd.read_csv(camera_id_csv)
    df.columns = [c.strip() for c in df.columns]
    return df.set_index("Case Number")["Camera Number"]


def load_locations(locations_xlsx: str | Path) -> pd.DataFrame:
    """Load the AMOS camera_id -> (lat, lon) table (``Amos locations.xlsx``)."""
    df = pd.read_excel(locations_xlsx)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.set_index("camera_id")[["lat", "lon"]]


def build_dataset(
    raw_dir: str | Path,
    camera_id_csv: str | Path,
    locations_xlsx: str | Path,
    year: int = 2015,
    fill_missing: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the analysis-ready greenness matrix and matching coordinates.

    Parameters
    ----------
    fill_missing:
        ``"median"`` fills remaining NaNs per-camera with that camera's
        median (matches the quick-and-dirty imputation used for the raw
        exploratory plots in the R scripts). ``"ffill"`` does a forward-fill
        (Last Observation Carried Forward), matching the paper's stated
        missing-value handling. ``None`` leaves NaNs in place (needed if you
        want to do the imputation yourself, e.g. before smoothing).

    Returns
    -------
    (greenness, coords):
        ``greenness`` is a (DOY x camera_id) DataFrame, columns renamed from
        case numbers to the real AMOS ``camera_id``.
        ``coords`` is a (camera_id x [lat, lon]) DataFrame aligned to the
        same columns, in the same order.
    """
    curves = load_camera_curves(raw_dir, year=year)
    case_to_cam = load_case_to_camera_map(camera_id_csv)
    locations = load_locations(locations_xlsx)

    # Rename columns from "case number" to the real AMOS camera id.
    missing_cases = [c for c in curves.columns if c not in case_to_cam.index]
    if missing_cases:
        raise ValueError(f"CameraID.csv has no mapping for case numbers: {missing_cases}")
    curves = curves.rename(columns=case_to_cam.to_dict())

    # Keep only cameras we can also georeference.
    have_coords = [c for c in curves.columns if c in locations.index]
    dropped = sorted(set(curves.columns) - set(have_coords))
    curves = curves[have_coords]
    coords = locations.loc[have_coords]

    if fill_missing == "median":
        curves = curves.apply(lambda s: s.fillna(s.median()))
    elif fill_missing == "ffill":
        curves = curves.ffill().bfill()  # bfill covers any camera missing its first day(s)
    elif fill_missing is not None:
        raise ValueError(f"Unknown fill_missing={fill_missing!r}")

    if dropped:
        import warnings

        warnings.warn(f"Dropped {len(dropped)} camera(s) with no location info: {dropped}")

    return curves, coords
