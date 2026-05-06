# ============================================================================
# Utility functions shared across the e2tree package
# ============================================================================

#' Population Variance
#' @keywords internal
e2_variance <- function(x) {
  sum((x - mean(x))^2) / length(x)
}

#' Check Availability of Suggested Packages
#' @keywords internal
check_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      sprintf("Package '%s' is required but not installed. Please install it with: install.packages('%s')", pkg, pkg),
      call. = FALSE
    )
  }
}

#' Identify the canonical class of a supported ensemble model
#'
#' Returns one of \code{"randomForest"}, \code{"ranger"}, \code{"xgb.Booster"},
#' \code{"lgb.Booster"}, \code{"gbm"}, \code{"catboost.CatBoost"} or
#' \code{"catboost.Model"} (the same class used by the S3 adapter dispatch),
#' or \code{NA_character_} when no supported class is matched.
#'
#' @keywords internal
ensemble_backend <- function(ensemble) {
  supported <- c("randomForest", "ranger", "xgb.Booster",
                 "lgb.Booster", "gbm",
                 "catboost.CatBoost", "catboost.Model")
  hit <- intersect(supported, class(ensemble))
  if (length(hit) == 0L) return(NA_character_)
  hit[1L]
}

#' Validate the output of \code{extract_terminal_nodes()}
#'
#' Boosting backends store their tree structures in opaque containers; a tiny
#' API change can silently produce a malformed leaf matrix (e.g. all zeros),
#' yielding a degenerate dissimilarity matrix without raising any error.
#' This function asserts the shape and type contract so problems surface
#' immediately at extraction time rather than much later, after the C++
#' co-occurrence call has already produced nonsense.
#'
#' Contract: \code{nodes} must be a \code{data.frame} with \code{nrow(data)}
#' rows and at least one column; every column must be coercible to integer;
#' at least one column must contain more than one distinct value.
#'
#' @keywords internal
validate_terminal_nodes <- function(nodes, data, backend = NA_character_) {
  if (!is.data.frame(nodes)) {
    stop(sprintf(
      "extract_terminal_nodes(%s) must return a data.frame; got '%s'.",
      backend, paste(class(nodes), collapse = "/")
    ), call. = FALSE)
  }
  if (nrow(nodes) != nrow(data)) {
    stop(sprintf(
      "extract_terminal_nodes(%s) returned %d rows but data has %d.",
      backend, nrow(nodes), nrow(data)
    ), call. = FALSE)
  }
  if (ncol(nodes) < 1L) {
    stop(sprintf(
      "extract_terminal_nodes(%s) returned 0 columns; expected one column per tree.",
      backend
    ), call. = FALSE)
  }
  numeric_cols <- vapply(nodes, function(col) is.numeric(col) || is.integer(col),
                         logical(1L))
  if (!all(numeric_cols)) {
    bad <- names(nodes)[!numeric_cols]
    stop(sprintf(
      "extract_terminal_nodes(%s): non-numeric leaf columns detected (%s).",
      backend, paste(bad, collapse = ", ")
    ), call. = FALSE)
  }
  # Detect the GBM-style silent failure: every leaf column collapses to a
  # single value, which turns the dissimilarity matrix into all-zeros.
  uniq <- vapply(nodes, function(col) length(unique(col)), integer(1L))
  if (all(uniq <= 1L)) {
    stop(sprintf(
      paste0("extract_terminal_nodes(%s) produced degenerate output: every ",
             "tree maps all observations to the same leaf. This usually ",
             "signals an internal API mismatch with the upstream package."),
      backend
    ), call. = FALSE)
  }
  invisible(TRUE)
}
