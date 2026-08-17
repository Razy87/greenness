#' ============================================================================
#' Setup: packages and global constants
#' ============================================================================
#'
#' Source this file first. It loads the packages used by the scripts in this
#' folder and sets up shared paths and constants.
#'
#' Package roles:
#'   fda, fda.usc      - B-spline bases, smoothing, functional PCA
#'   gstat, sp         - variogram estimation/fitting, spatial coordinates
#'   fdagstat          - functional-data kriging (trace-variogram UTrK, UCoK)
#'                       remotes::install_github("ogru/fdagstat")
#'   dplyr, tidyr, readr, purrr - data wrangling
#'   readxl / openxlsx - reading Amos locations.xlsx
#'   ggplot2, naniar   - plotting and missing-data visualisation
#'   imputeTS          - na_locf() last-observation-carried-forward imputation
#'   googledrive       - downloading raw AMOS webcam images (01_data_acquisition.R only)

required_packages <- c(
  "dplyr", "tidyr", "readr", "purrr", "tibble",
  "readxl", "openxlsx",
  "ggplot2", "naniar", "imputeTS",
  "fda", "fda.usc",
  "gstat", "sp",
  "fdagstat",
  "googledrive", "jpeg"
)

missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  message(
    "The following packages are not installed and are required:\n  ",
    paste(missing_packages, collapse = ", "),
    "\nMost are on CRAN via install.packages(); `fdagstat` is on GitHub, e.g.\n",
    "  remotes::install_github(\"ogru/fdagstat\")"
  )
}

invisible(lapply(intersect(required_packages, rownames(installed.packages())), function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))

# ---- Project-relative paths -------------------------------------------------
# All scripts assume the working directory is the repository root (or that
# `here::here()` resolves there). Adjust PROJECT_ROOT if you source these
# files from elsewhere.
PROJECT_ROOT <- getwd()
RAW_DATA_DIR <- file.path(PROJECT_ROOT, "data", "raw")
GREENNESS_CURVES_DIR <- file.path(RAW_DATA_DIR, "greenness_curves")
CAMERA_ID_CSV <- file.path(RAW_DATA_DIR, "CameraID.csv")
LOCATIONS_XLSX <- file.path(RAW_DATA_DIR, "amos_locations.xlsx")

# ---- Shared analysis constants ----------------------------------------------
DOY_RANGE <- 1:365           # day-of-year axis used throughout (non-leap-year)
VARIOGRAM_MODELS <- c(Gau = "Gau", Mat = "Mat", Sph = "Sph", Exp = "Exp")
