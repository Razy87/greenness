#' ============================================================================
#' Data acquisition: AMOS webcam images -> per-camera greenness time series
#' ============================================================================
#'
#' These helpers download the raw AMOS (amos.cse.wustl.edu) webcam image
#' archives for a given camera/year/month from Google Drive, and read back
#' the resulting greenness time series once the uRoI (unsupervised
#' Region-of-Interest) pipeline has produced them.
#'
#' This is the *upstream* data-collection step. The greenness-extraction
#' pipeline itself (uRoI image processing) is not included in this repo --
#' it produces the per-camera .RData files that `read_greenness_result()`
#' below reads back in. If you already have the compiled per-camera CSVs
#' (as in data/raw/greenness_curves/), you do not need this file at all;
#' start from 02_load_and_clean_data.R instead.
#'
#' Source 00_setup.R before this file.

#' Resolve a Google Drive sub-folder by name under a parent folder.
#'
#' @param parent_id A Google Drive folder id (or dribble) to search under.
#' @param name The (exact) sub-folder or file name to find.
#' @return The Drive id of the matched item.
resolve_drive_path <- function(parent_id, name) {
  match <- googledrive::drive_ls(path = parent_id, pattern = name, recursive = FALSE)
  if (nrow(match) == 0) {
    stop(sprintf("No Drive item named '%s' found under the given parent.", name))
  }
  match$id[[1]]
}

#' Download and unzip one camera's AMOS images for a single year/month.
#'
#' @param camera Camera id (numeric or character).
#' @param year Four-digit year.
#' @param month Month number (1-12).
#' @param drive_root_url The root AMOS Drive folder URL/id to search under.
#' @param dest_dir Local directory images are extracted into.
download_amos_camera_month <- function(camera, year, month, drive_root_url, dest_dir) {
  camera <- as.character(camera)
  year <- as.character(year)
  month_2digit <- formatC(as.integer(month), width = 2, flag = "0")

  camera_2digit <- formatC(as.integer(camera), width = 2, flag = "0")
  camera_4digit <- formatC(as.integer(camera), width = 4, flag = "0")
  camera_5digit <- formatC(as.integer(camera), width = 5, flag = "0")
  camera_8digit <- formatC(as.integer(camera), width = 8, flag = "0")

  year_folder <- resolve_drive_path(drive_root_url, year)
  cam2_folder <- resolve_drive_path(year_folder, camera_2digit)
  cam4_folder <- resolve_drive_path(cam2_folder, camera_4digit)
  cam8_folder <- resolve_drive_path(cam4_folder, camera_8digit)

  zip_pattern <- sprintf("%s.%s.zip", year, month_2digit)
  matches <- googledrive::drive_ls(path = cam8_folder, pattern = zip_pattern, recursive = FALSE)
  if (nrow(matches) == 0) {
    message(sprintf("No archive '%s' for camera %s -- skipping.", zip_pattern, camera))
    return(invisible(NULL))
  }

  out_dir <- file.path(dest_dir, paste0("000", camera_5digit), paste0(year, ".", month_2digit))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  tmp_zip <- tempfile(fileext = ".zip")
  googledrive::drive_download(googledrive::as_id(matches$id[[1]]), path = tmp_zip, overwrite = TRUE)
  file_list <- utils::unzip(tmp_zip, list = TRUE)
  utils::unzip(tmp_zip, files = file_list$Name, exdir = out_dir)

  message(sprintf(
    "[%s] downloaded camera %s, %s-%s -> %s",
    format(Sys.time()), camera, year, month_2digit, out_dir
  ))
  invisible(out_dir)
}

#' Download every camera/year/month combination in the given vectors.
download_amos_batch <- function(cameras, years, months, drive_root_url, dest_dir) {
  grid <- expand.grid(camera = cameras, year = years, month = months, stringsAsFactors = FALSE)
  for (i in seq_len(nrow(grid))) {
    download_amos_camera_month(grid$camera[i], grid$year[i], grid$month[i], drive_root_url, dest_dir)
  }
  invisible(grid)
}

#' Read back one camera's already-computed greenness time series.
#'
#' Looks for `bestgreennesstimeseries_uroi.RData` under
#' `results_dir/000<camera 5-digit>/<processing_date>/`, tries every date in
#' `processing_dates` until one exists, and reindexes onto a full 1..365 DOY
#' axis (missing days become NA).
#'
#' @param camera Camera id.
#' @param year Year the results belong to (only used for logging).
#' @param results_dir Directory containing one sub-folder per camera.
#' @param processing_dates Candidate processing-run date folders to try, in order.
#' @return A data.frame with columns DOY and <camera id>, or NULL if no
#'   result file was found for this camera.
read_greenness_result <- function(camera, year, results_dir, processing_dates) {
  camera <- as.character(camera)
  camera_5digit <- formatC(as.integer(camera), width = 5, flag = "0")
  camera_dir <- file.path(results_dir, paste0("000", camera_5digit))

  if (!dir.exists(camera_dir)) {
    message(sprintf("[%s] no results directory for camera %s, year %s.", format(Sys.time()), camera, year))
    return(NULL)
  }

  for (processing_date in processing_dates) {
    result_file <- file.path(camera_dir, processing_date, "bestgreennesstimeseries_uroi.RData")
    if (!file.exists(result_file)) next

    env <- new.env()
    load(result_file, envir = env)  # expected to define `grTS`
    daily <- as.data.frame(env$grTS)
    daily <- tibble::rownames_to_column(daily, "DOY")
    daily$DOY <- as.integer(daily$DOY)

    full_doy <- data.frame(DOY = DOY_RANGE)
    merged <- dplyr::left_join(full_doy, daily, by = "DOY")
    colnames(merged) <- c("DOY", camera)

    message(sprintf(
      "[%s] loaded greenness series for camera %s, year %s.", format(Sys.time()), camera, year
    ))
    return(merged)
  }

  message(sprintf(
    "[%s] no result file found for camera %s, year %s (tried %d processing date(s)).",
    format(Sys.time()), camera, year, length(processing_dates)
  ))
  NULL
}
