#' ============================================================================
#' Load and clean the compiled per-camera greenness curves
#' ============================================================================
#'
#' Builds the analysis-ready (DOY x camera) greenness matrix and matching
#' (lat, lon) coordinates, from the compiled CSVs in data/raw/. The cleaning
#' steps are:
#'
#'   1. Read one CSV per camera, keep only the requested year's column.
#'   2. Map "case number" (the file-name index) to the real AMOS camera id.
#'   3. Drop cameras with no matching entry in the locations table.
#'   4. Drop cameras whose (lat, lon) is a boxplot outlier of the retained
#'      set (guards against clearly wrong geocoding, e.g. a camera with
#'      US-range imagery but coordinates in Finland).
#'   5. Drop cameras with more than 50% missing days (> 250 / 365 NA).
#'   6. Impute remaining gaps via Last-Observation-Carried-Forward (LOCF),
#'      matching the missing-data handling described in the paper.
#'   7. Nudge apart any cameras left sharing an exact (lat, lon) (kriging
#'      requires unique locations -- see `jitter_duplicate_coordinates()`).
#'
#' Source 00_setup.R before this file.

#' Load one camera's per-DOY greenness CSV and pull out a single year.
#'
#' @param path Path to a `DFcamera<case>new.csv` file.
#' @param year Year whose column to extract (matches a column name
#'   containing "Year<year>", e.g. "green relative Year2015.C37").
#' @return A named numeric vector of length 365 (NA for missing days), or
#'   NULL if this camera has no column for `year`.
read_camera_curve <- function(path, year) {
  df <- readr::read_csv(path, show_col_types = FALSE)
  year_col <- grep(sprintf("Year%s\\b", year), names(df), value = TRUE)
  if (length(year_col) == 0) return(NULL)

  full_doy <- tibble::tibble(DOY = DOY_RANGE)
  merged <- dplyr::left_join(full_doy, df[, c("DOY", year_col[1])], by = "DOY")
  stats::setNames(merged[[year_col[1]]], NULL)
}

#' Extract the numeric case id from a "DFcamera<case>new.csv" file name.
case_number_from_filename <- function(path) {
  as.integer(gsub(".*DFcamera(\\d+)new?\\.csv$", "\\1", basename(path)))
}

#' Assemble and clean the full analysis-ready dataset for one year.
#'
#' @param curves_dir Directory of per-camera CSVs (data/raw/greenness_curves).
#' @param camera_id_csv Path to CameraID.csv (case number -> AMOS camera id).
#' @param locations_xlsx Path to amos_locations.xlsx (camera id -> lat, lon).
#' @param year Year to extract (default 2015, matching the paper).
#' @param max_missing_frac Drop cameras with more than this fraction of
#'   missing days (default 0.5, i.e. > 250 / 365, matching the paper).
#' @param drop_location_outliers If TRUE, drop cameras whose lat or lon is a
#'   boxplot.stats() outlier of the retained set (default TRUE).
#' @param jitter_duplicate_coords If TRUE (default), nudge cameras that share
#'   an exact (lat, lon) with another camera apart by a negligible random
#'   offset -- see `jitter_duplicate_coordinates()`. Needed because the
#'   kriging step requires unique locations; without this, `fit_utrk()` /
#'   `fit_ucok()` fail with "system is exactly singular" whenever two
#'   cameras share a registered site (this does happen in the AMOS
#'   locations table, e.g. cameras 1306 and 1593).
#' @param jitter_degrees Maximum absolute jitter applied to duplicate
#'   coordinates, in degrees (default 1e-4, ~10 m -- negligible next to the
#'   spacing between distinct camera sites).
#' @param jitter_seed RNG seed used for the jitter, for reproducibility.
#' @return A list with `greenness` (365 x n_camera matrix, columns named by
#'   camera id) and `coords` (data.frame: camera_id, lat, lon; row order
#'   matches the matrix columns).
load_greenness_dataset <- function(curves_dir = GREENNESS_CURVES_DIR,
                                    camera_id_csv = CAMERA_ID_CSV,
                                    locations_xlsx = LOCATIONS_XLSX,
                                    year = 2015,
                                    max_missing_frac = 0.5,
                                    drop_location_outliers = TRUE,
                                    jitter_duplicate_coords = TRUE,
                                    jitter_degrees = 1e-4,
                                    jitter_seed = 1) {
  files <- list.files(curves_dir, pattern = "DFcamera.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) stop("No per-camera CSV files found in ", curves_dir)

  case_to_camera <- readr::read_csv(camera_id_csv, show_col_types = FALSE)
  names(case_to_camera) <- trimws(names(case_to_camera))
  locations <- openxlsx::read.xlsx(locations_xlsx)
  names(locations) <- tolower(trimws(names(locations)))

  cases <- vapply(files, case_number_from_filename, integer(1))
  curves <- lapply(files, read_camera_curve, year = year)
  keep <- !vapply(curves, is.null, logical(1))
  cases <- cases[keep]
  curves <- curves[keep]

  camera_ids <- as.integer(case_to_camera$`Camera Number`[match(cases, case_to_camera$`Case Number`)])
  if (anyNA(camera_ids)) {
    stop("CameraID.csv has no mapping for case number(s): ", paste(cases[is.na(camera_ids)], collapse = ", "))
  }

  # camera_id is treated as character everywhere downstream (matrix column
  # names are always character in R) -- normalize both sides explicitly here
  # so `match()` can't silently fail on a numeric-vs-character mismatch.
  greenness <- do.call(cbind, curves)
  colnames(greenness) <- as.character(camera_ids)
  rownames(greenness) <- DOY_RANGE
  greenness <- greenness[, order(as.integer(colnames(greenness))), drop = FALSE]

  locations$camera_id <- as.character(as.integer(locations$camera_id))
  coords <- locations[match(colnames(greenness), locations$camera_id), c("camera_id", "lat", "lon")]
  have_coords <- !is.na(coords$camera_id)
  if (any(!have_coords)) {
    message(sprintf(
      "Dropping %d camera(s) with no entry in %s: %s",
      sum(!have_coords), basename(locations_xlsx),
      paste(colnames(greenness)[!have_coords], collapse = ", ")
    ))
  }
  greenness <- greenness[, have_coords, drop = FALSE]
  coords <- coords[have_coords, ]

  if (drop_location_outliers) {
    lat_outliers <- grDevices::boxplot.stats(coords$lat)$out
    lon_outliers <- grDevices::boxplot.stats(coords$lon)$out
    is_outlier <- coords$lat %in% lat_outliers | coords$lon %in% lon_outliers
    if (any(is_outlier)) {
      message(sprintf(
        "Dropping %d camera(s) with outlier coordinates: %s",
        sum(is_outlier), paste(coords$camera_id[is_outlier], collapse = ", ")
      ))
      greenness <- greenness[, !is_outlier, drop = FALSE]
      coords <- coords[!is_outlier, ]
    }
  }

  missing_frac <- colMeans(is.na(greenness))
  too_sparse <- missing_frac > max_missing_frac
  if (any(too_sparse)) {
    message(sprintf(
      "Dropping %d camera(s) with > %.0f%% missing days: %s",
      sum(too_sparse), 100 * max_missing_frac,
      paste(colnames(greenness)[too_sparse], collapse = ", ")
    ))
    greenness <- greenness[, !too_sparse, drop = FALSE]
    coords <- coords[!too_sparse, ]
  }

  # Last-Observation-Carried-Forward, then back-fill any still-missing
  # leading days -- matches the paper's stated missing-value treatment.
  greenness <- imputeTS::na_locf(greenness, option = "locf", na_remaining = "rev")

  if (jitter_duplicate_coords) {
    coords <- jitter_duplicate_coordinates(coords, jitter_degrees, jitter_seed)
  }

  rownames(coords) <- NULL
  list(greenness = greenness, coords = coords)
}

#' Nudge exactly-co-located cameras apart by a negligible random offset.
#'
#' Kriging (UTrK and UCoK alike) solves a linear system built from the
#' pairwise covariance between camera locations. Two cameras at the exact
#' same (lat, lon) are at distance zero, which gives them identical rows in
#' that covariance matrix -- the system becomes exactly singular and
#' `solve()` fails with "Lapack routine dgesv: system is exactly singular".
#' This is a real property of the AMOS locations table (several cameras
#' share a registered site), not a data error, so cameras are not dropped;
#' instead all but the first camera in each duplicate-location group are
#' nudged by a uniform random offset of at most `jitter_degrees` (default
#' 1e-4 degrees, on the order of ~10 m) -- far smaller than the spacing
#' between distinct camera sites, so it does not meaningfully change the
#' kriging result, but it is enough to make the covariance matrix
#' non-singular.
#'
#' @param coords data.frame with `camera_id`, `lat`, `lon`.
#' @param jitter_degrees Maximum absolute jitter applied to lat and lon,
#'   in degrees.
#' @param seed RNG seed, for reproducibility across runs.
#' @return `coords` with duplicate-location rows (beyond the first in each
#'   group) nudged apart.
jitter_duplicate_coordinates <- function(coords, jitter_degrees = 1e-4, seed = 1) {
  is_dup <- duplicated(coords[, c("lat", "lon")])
  if (!any(is_dup)) return(coords)

  message(sprintf(
    "Nudging %d camera(s) that share an exact (lat, lon) with another camera by <= %g degrees (~%d m) so the kriging covariance matrix is non-singular: %s",
    sum(is_dup), jitter_degrees, round(jitter_degrees * 111000),
    paste(coords$camera_id[is_dup], collapse = ", ")
  ))

  # Save/restore the RNG state so this doesn't disturb reproducibility of
  # any sampling the caller does later (e.g. Monte Carlo train/test splits).
  had_seed <- exists(".Random.seed", envir = .GlobalEnv)
  old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
  on.exit({
    if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(seed)

  n_dup <- sum(is_dup)
  coords$lat[is_dup] <- coords$lat[is_dup] + stats::runif(n_dup, -jitter_degrees, jitter_degrees)
  coords$lon[is_dup] <- coords$lon[is_dup] + stats::runif(n_dup, -jitter_degrees, jitter_degrees)
  coords
}
