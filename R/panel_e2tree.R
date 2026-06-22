# ============================================================================
# Panel E2Tree: explaining tree-ensemble structure on PANEL (longitudinal)
# data by decomposing the FEATURE REPRESENTATION into a BETWEEN (unit-mean)
# and a WITHIN (unit-demeaned) component, in the spirit of the Mundlak (1978)
# decomposition. One ensemble and one e2tree surrogate are grown per source of
# variation and the panel reconstruction is additive:
#
#     y_hat_it = between(mean of unit i) [+ time effect t] + within(deviation)
#
# WHY two representations (and not one model). On panel data with high
# intraclass correlation the between variance dominates: a single pooled
# explanation conflates "what distinguishes units" with "what moves them over
# time" and the within signal is crowded out. Decomposing the INPUT and
# explaining each part separately removes the competition between the two
# sources. The PERMANOVA decomposition of a pooled ensemble's proximity
# (the natural single-model alternative) does not recover the outcome's
# between/within structure, so the decomposition must act on the
# representation; see panel_e2tree() details.
#
# WHAT IS EXPLAINED. With target = "outcome" (default) panel_e2tree() is a
# modelling-and-explanation protocol: it refits one ensemble per
# representation and explains each. With target = "pooled" it explains a
# *given* pooled ensemble by decomposing that ensemble's predictions into
# between/within components and explaining each component.
#
# This module orchestrates the package primitives -- createDisMatrix(),
# e2tree(), predict.e2tree(), get_ensemble_predictions(), vimp() -- through
# the adapter layer, so it inherits multi-backend ensemble support. e2tree is
# an EXPLANATION method, not a predictor: component quality is fidelity to
# the ensemble it explains, cor(e2tree, ensemble)^2 computed against
# FULL-ENSEMBLE predictions.
# ============================================================================

utils::globalVariables(c(".unit", ".time"))

## ----------------------------------------------------------------------------
## helper: drop zero-variance predictors. Unit-invariant (time-constant)
## features become constant after within-demeaning and MUST be removed before
## e2tree() (which errors on a constant predictor); `keep` is never dropped.
## The tolerance is relative to `ref_scale` (sd of the original feature):
## demeaning a time-invariant feature measured on a large scale leaves float
## noise whose absolute sd can exceed any fixed absolute tolerance.
## ----------------------------------------------------------------------------
.drop_constant <- function(x, keep, ref_scale = NULL) {
  ok <- vapply(names(x), function(nm) {
    if (nm == keep) return(TRUE)
    s <- stats::sd(x[[nm]], na.rm = TRUE)
    if (!is.finite(s)) return(FALSE)
    ref <- 1
    if (!is.null(ref_scale) && nm %in% names(ref_scale) && is.finite(ref_scale[[nm]])) {
      ref <- max(ref_scale[[nm]], 1)
    }
    s > 1e-8 * ref
  }, logical(1))
  x[, ok, drop = FALSE]
}

## ----------------------------------------------------------------------------
## helper: mean share of within-unit variance across features. A low value
## means the features barely move within units, so the within tree degenerates
## (see the feature-set dependence note in the details). Returns a scalar in
## [0, 1]; features with (near-)zero total variance are skipped.
## ----------------------------------------------------------------------------
.within_signal <- function(d, features, unit_means_aligned) {
  shares <- vapply(features, function(f) {
    tot <- stats::var(d[[f]], na.rm = TRUE)
    if (!is.finite(tot) || tot < 1e-12) return(NA_real_)
    dev <- d[[f]] - unit_means_aligned[[f]]
    stats::var(dev, na.rm = TRUE) / tot
  }, numeric(1))
  mean(shares, na.rm = TRUE)
}

## ----------------------------------------------------------------------------
## helper: ANOVA-style intraclass correlation of x given grouping g:
## between-group share of total variance.
## ----------------------------------------------------------------------------
.icc <- function(x, g) {
  m  <- tapply(x, g, mean)
  ng <- tapply(x, g, length)
  vb <- sum(ng * (m - mean(x))^2) / length(x)
  vw <- sum((x - m[match(g, names(m))])^2) / length(x)
  if (vb + vw < 1e-12) return(NA_real_)
  vb / (vb + vw)
}

## ----------------------------------------------------------------------------
## helper: fit one ensemble (shared by the component fits and the internal
## pooled fit of target = "pooled").
## ----------------------------------------------------------------------------
.fit_ensemble <- function(df, outcome, engine, ntree, ...) {
  preds <- setdiff(names(df), outcome)
  form  <- stats::reformulate(preds, response = outcome)

  if (engine == "ranger") {
    if (!requireNamespace("ranger", quietly = TRUE)) {
      stop("engine = 'ranger' requires the 'ranger' package.", call. = FALSE)
    }
    ranger::ranger(form, data = df, num.trees = ntree, importance = "none", ...)
  } else {
    if (!requireNamespace("randomForest", quietly = TRUE)) {
      stop("engine = 'randomForest' requires the 'randomForest' package.", call. = FALSE)
    }
    randomForest::randomForest(form, data = df, ntree = ntree, ...)
  }
}

## ----------------------------------------------------------------------------
## helper: fit ensemble + dissimilarity + e2tree on one representation.
## Routes everything through the adapter-aware package primitives. Fidelity is
## computed against FULL-ENSEMBLE predictions (oob = FALSE): OOB predictions
## are noisier and measure a different quantity than the surrogate's target.
## ----------------------------------------------------------------------------
.panel_fit_component <- function(df, outcome, engine, ntree, setting, label,
                                 dis_args = list(), ...) {
  ens <- .fit_ensemble(df, outcome, engine, ntree, ...)

  preds <- setdiff(names(df), outcome)
  form  <- stats::reformulate(preds, response = outcome)

  da <- utils::modifyList(list(parallel = list(active = FALSE, no_cores = 1)),
                          dis_args)
  D    <- do.call(createDisMatrix,
                  c(list(ensemble = ens, data = df, label = outcome), da))
  tree <- e2tree(form, df, D, ens, setting)
  fit  <- as.numeric(predict(tree, newdata = df)$fit)
  ep   <- get_ensemble_predictions(ens, df, type = "regression", oob = FALSE)

  degenerate <- stats::sd(fit) < 1e-12
  fidelity   <- if (degenerate) NA_real_ else stats::cor(fit, ep)^2

  vi <- tryCatch(vimp(tree, df), error = function(e) {
    warning(sprintf("vimp() failed for the %s component: %s",
                    label, conditionMessage(e)), call. = FALSE)
    NULL
  })

  list(
    tree       = tree,
    ensemble   = ens,
    D          = D,
    fidelity   = fidelity,
    degenerate = degenerate,
    pred       = fit,
    ens_pred   = ep,
    variables  = unique(stats::na.omit(tree$tree$variable)),
    varimp     = vi,
    data       = df
  )
}

## empty component placeholder (used when no time-varying feature survives)
.empty_component <- function(n) {
  list(tree = NULL, ensemble = NULL, D = NULL, fidelity = NA_real_,
       degenerate = TRUE, pred = rep(0, n), ens_pred = rep(0, n),
       variables = character(0), varimp = NULL, data = NULL)
}


# ============================================================================
# panel_e2tree(): main entry point
# ============================================================================

#' Explainable Ensemble Tree for Panel (Longitudinal) Data
#'
#' `panel_e2tree()` explains tree-ensemble structure on **panel** (unit
#' \eqn{\times} time) data by decomposing the *feature representation* into
#' sources of variation, in the spirit of the Mundlak (1978) decomposition,
#' and growing a separate \code{\link{e2tree}} surrogate for each:
#'
#' \describe{
#'   \item{**between**}{unit means (one row per unit) -- what distinguishes the
#'     units, i.e. differences between unit averages;}
#'   \item{**within**}{unit-demeaned deviations -- what moves the outcome over
#'     time within a unit. With \code{within = "twoway"} the deviations are
#'     additionally purged of common period effects (two-way demeaning), so
#'     the within tree explains *idiosyncratic* movements and common shocks
#'     are reported separately as descriptive period effects.}
#' }
#'
#' The panel reconstruction is additive:
#' \deqn{\hat y_{it} = \mathrm{between}(\bar x_i) \;[+\; \hat\tau_t]\; +
#'   \mathrm{within}(\tilde x_{it}),}
#' where \eqn{\hat\tau_t} (period effects of the outcome) appears only under
#' \code{within = "twoway"}.
#'
#' @details
#' **What is explained.** With \code{target = "outcome"} (default) the
#' function is a *modelling-and-explanation protocol*: it fits one ensemble
#' per representation and explains each with an e2tree. With
#' \code{target = "pooled"} it explains a **given pooled ensemble**: the
#' pooled model's full-ensemble predictions \eqn{\hat f(x_{it})} replace the
#' outcome, are decomposed into between/within components, and each component
#' is explained. All reported metrics then quantify how faithfully the
#' decomposed explanation reproduces the *pooled model's behaviour*.
#'
#' **Why two representations.** On panel data with high intraclass correlation
#' (ICC) the between variance dominates: a single *pooled* e2tree conflates the
#' two sources and the within signal is crowded out, yielding low fidelity.
#' Single-model shortcuts (a pooled e2tree on raw features; a post-hoc PERMANOVA
#' decomposition of a pooled proximity; a single e2tree on Mundlak-augmented
#' features) were tested and discarded. Decomposing the *input* -- and
#' explaining each part with its own surrogate -- is what separates the two
#' sources.
#'
#' **What "within" is and is not.** The within representation is invariant to
#' permuting periods within a unit: it captures *what moves the outcome within
#' a unit*, not temporal ordering (lags, persistence). Common period shocks
#' are part of the within variation under \code{within = "unit"} and are
#' removed (and reported as period effects) under \code{within = "twoway"}.
#'
#' **Interpretive, not predictive.** e2tree is an explanation method. Each
#' component's quality is its *fidelity* to the ensemble it explains,
#' \eqn{\mathrm{cor}(\text{e2tree}, \text{ensemble})^2}, computed against
#' full-ensemble predictions. The returned \code{outcome_var_recovered} and
#' \code{r2_panel} quantify how much of the explanandum's variance the additive
#' reconstruction accounts for -- a descriptive diagnostic, never a predictive
#' claim about e2tree.
#'
#' **Feature-set dependence of the within component.** The within tree can only
#' be as good as the *within signal* of the features: with few time-varying
#' predictors it may degenerate. \code{panel_e2tree()} reports the mean
#' within-variance share of the features and warns when it is low.
#'
#' **Constant columns.** After within-demeaning, time-invariant features become
#' constant and are dropped from the within representation (e2tree errors on a
#' constant predictor); the same guard is applied to the between representation.
#' The tolerance is relative to each feature's original scale. Dropped features
#' are recorded and shown by \code{summary()}.
#'
#' **Missing data.** With \code{na.action = "listwise"} (default) rows with any
#' missing value are dropped first and unit means are computed on the complete
#' rows. With \code{na.action = "unit.available"} unit means are computed
#' per variable on all available observations (less biased in unbalanced
#' panels with missingness); fitting still uses complete rows. Note that
#' listwise deletion can materially change the panel's ICC -- compare
#' \code{$icc} across the two settings.
#'
#' **Scope.** This version supports **regression** outcomes only. The
#' between component generalises naturally to classification (unit proportions),
#' but a "within deviation" of a categorical target is an open direction; a
#' factor outcome therefore raises an error. Predictors must be numeric (the
#' between/within decomposition is defined on numeric features).
#'
#' @param formula A model formula \code{y ~ x1 + x2 + ...} (no interaction
#'   terms). \code{y ~ .} expands to every column except \code{unit} and
#'   \code{time}.
#' @param data A \code{data.frame} in long panel format.
#' @param unit Character. Name of the unit (group id) column, e.g. country.
#' @param time Character or \code{NULL}. Name of the time column. Required for
#'   \code{within = "twoway"}; otherwise used to check for duplicated
#'   (unit, time) rows and to annotate predictions.
#' @param engine Character. Ensemble backend fitted internally on each derived
#'   representation: \code{"ranger"} (default, fast) or \code{"randomForest"}.
#' @param ntree Integer. Number of trees per ensemble. Default \code{500}.
#' @param setting_between,setting_within Lists of e2tree stopping rules
#'   (\code{impTotal}, \code{maxDec}, \code{n}, \code{level}, optionally
#'   \code{max_thresholds}) for the between and within surrogate, respectively.
#'   See \code{\link{e2tree}}.
#' @param target \code{"outcome"} (default) explains the outcome's panel
#'   structure (refit protocol); \code{"pooled"} explains a pooled ensemble's
#'   predictions (see Details).
#' @param within \code{"unit"} (default) demeans by unit; \code{"twoway"}
#'   additionally removes common period effects (requires \code{time}).
#' @param na.action \code{"listwise"} (default) or \code{"unit.available"};
#'   see Details.
#' @param min_periods Integer \eqn{\ge 1}. Units observed fewer than
#'   \code{min_periods} times (after row filtering) are excluded. Units with a
#'   single observation contribute all-zero rows to the within representation;
#'   their count is reported by \code{summary()}.
#' @param pooled_ensemble Optional. A fitted ensemble (any supported backend)
#'   to be explained when \code{target = "pooled"}; if \code{NULL}, a pooled
#'   ensemble is fitted internally with \code{engine}/\code{ntree}.
#' @param keep_D Logical. Keep the per-component dissimilarity matrices in the
#'   returned object (size \eqn{O(n^2)} each). Default \code{FALSE}.
#' @param keep_data Logical. Keep a copy of each derived representation in the
#'   returned object. Default \code{FALSE} (the e2tree objects already store
#'   what their methods need).
#' @param dis_args Named list of additional arguments passed to
#'   \code{\link{createDisMatrix}} (e.g. \code{parallel}, \code{chunk_size},
#'   \code{verbose}).
#' @param seed Optional integer for reproducible ensemble fits.
#' @param ... Additional arguments passed to the ensemble engine
#'   (\code{ranger::ranger} or \code{randomForest::randomForest}). The
#'   arguments \code{formula}, \code{data}, \code{ntree}, \code{num.trees} and
#'   \code{importance} are managed internally and cannot be overridden here.
#'
#' @return An object of class \code{"e2panel"}: a list with
#'   \item{between, within}{Per-component lists: \code{tree} (the
#'     \code{\link{e2tree}}), \code{ensemble}, \code{fidelity} (vs full-ensemble
#'     predictions), \code{degenerate}, \code{pred}, \code{variables} (split
#'     variables), \code{varimp} (decomposed importance); plus \code{D} and
#'     \code{data} if requested. The \code{within} tree is \code{NULL} when no
#'     time-varying feature survives.}
#'   \item{outcome_var_recovered}{\eqn{\mathrm{cor}(\hat y, y)^2} of the
#'     additive reconstruction against the explanandum (observed outcome, or
#'     pooled predictions under \code{target = "pooled"}).}
#'   \item{r2_panel}{\eqn{1 - \mathrm{SSE}/\mathrm{SST}} of the same
#'     reconstruction (honest \eqn{R^2}, sensitive to scale/bias; the headline
#'     metric).}
#'   \item{fidelity_panel}{Deprecated alias of \code{outcome_var_recovered},
#'     kept for backward compatibility.}
#'   \item{vs_outcome}{Under \code{target = "pooled"}: the same two metrics
#'     computed against the *observed* outcome (descriptive).}
#'   \item{icc}{Intraclass correlation of the explanandum across units.}
#'   \item{predictions}{\code{data.frame(.row, unit, [time], outcome,
#'     [.pooled], .between, [.timeeffect], .within, .panel)}; \code{.row} is
#'     the row index in the original \code{data}.}
#'   \item{unit_means, bpred}{Training unit means and per-unit between
#'     predictions, used by \code{predict()}.}
#'   \item{time_means}{Under \code{within = "twoway"}: period means (of the
#'     unit-demeaned explanandum and features), used by \code{predict()}.}
#'   \item{within_signal}{Mean within-variance share of the features.}
#'   \item{dropped}{Features dropped (constant) from each representation.}
#'   \item{dropped_units, n_singleton}{Units excluded by \code{min_periods};
#'     number of single-observation units retained.}
#'   \item{features, outcome, unit, time, engine, target, within_type,
#'     na.action, call}{Bookkeeping.}
#'
#' @references Mundlak, Y. (1978). On the pooling of time series and cross
#'   section data. \emph{Econometrica}, 46(1), 69--85.
#'
#' @seealso \code{\link{e2tree}}, \code{\link{createDisMatrix}},
#'   \code{\link{vimp}}; methods \code{\link{predict.e2panel}},
#'   \code{\link{plot.e2panel}}.
#'
#' @examples
#' \donttest{
#' ## A simulated country x year panel of life-expectancy determinants.
#' ## BETWEEN drivers (gdp_pc, health_exp) distinguish countries; WITHIN drivers
#' ## (immunization, undernourish) move the outcome over time within a country.
#' data(panel_health)
#' if (requireNamespace("ranger", quietly = TRUE)) {
#'   m <- panel_e2tree(
#'     life_expectancy ~ gdp_pc + health_exp + schooling +
#'       immunization + sanitation + undernourish,
#'     data = panel_health, unit = "country", time = "year",
#'     engine = "ranger", ntree = 300, seed = 7)
#'   print(m)
#'   summary(m)
#'   predict(m, newdata = panel_health[1:3, ])
#' }
#' }
#'
#' @export
panel_e2tree <- function(formula, data, unit, time = NULL,
                         engine = c("ranger", "randomForest"),
                         ntree = 500,
                         setting_between = list(impTotal = 0.10, maxDec = 1e-6, n = 2, level = 5),
                         setting_within  = list(impTotal = 0.05, maxDec = 1e-7, n = 5, level = 6),
                         target = c("outcome", "pooled"),
                         within = c("unit", "twoway"),
                         na.action = c("listwise", "unit.available"),
                         min_periods = 1L,
                         pooled_ensemble = NULL,
                         keep_D = FALSE,
                         keep_data = FALSE,
                         dis_args = list(),
                         seed = NULL, ...) {

  engine    <- match.arg(engine)
  target    <- match.arg(target)
  within    <- match.arg(within)
  na.action <- match.arg(na.action)

  # ---- engine dots: reject arguments managed internally ---------------------
  dot_names <- names(list(...))
  reserved  <- intersect(dot_names,
                         c("formula", "data", "ntree", "num.trees", "importance"))
  if (length(reserved)) {
    stop(sprintf(paste0("Argument(s) %s are managed internally by panel_e2tree() ",
                        "and cannot be passed through '...'. Use the dedicated ",
                        "parameters (e.g. 'ntree')."),
                 paste(sQuote(reserved), collapse = ", ")), call. = FALSE)
  }

  # ---- formula / column resolution -----------------------------------------
  if (!inherits(formula, "formula")) {
    stop("'formula' must be a valid formula object.", call. = FALSE)
  }
  if (!is.data.frame(data) || nrow(data) == 0) {
    stop("'data' must be a non-empty data frame.", call. = FALSE)
  }
  data <- as.data.frame(data)
  if (!is.character(unit) || length(unit) != 1 || !(unit %in% names(data))) {
    stop("'unit' must be a single column name present in 'data'.", call. = FALSE)
  }
  if (!is.null(time) && (!is.character(time) || length(time) != 1 || !(time %in% names(data)))) {
    stop("'time' must be NULL or a single column name present in 'data'.", call. = FALSE)
  }
  if (within == "twoway" && is.null(time)) {
    stop("within = 'twoway' requires the 'time' column (period effects).", call. = FALSE)
  }
  if (!is.numeric(min_periods) || length(min_periods) != 1 || min_periods < 1) {
    stop("'min_periods' must be a single integer >= 1.", call. = FALSE)
  }
  min_periods <- as.integer(min_periods)

  outcome  <- all.vars(formula)[1]
  tt       <- stats::terms(formula, data = data)
  features <- setdiff(all.vars(stats::delete.response(tt)), c(unit, time))
  if (!(outcome %in% names(data))) {
    stop(sprintf("Outcome '%s' is not a column of 'data'.", outcome), call. = FALSE)
  }
  if (length(features) == 0) {
    stop("No predictors found in 'formula' (after removing unit/time).", call. = FALSE)
  }
  miss <- setdiff(features, names(data))
  if (length(miss)) {
    stop(sprintf("Predictor(s) not found in 'data': %s.",
                 paste(miss, collapse = ", ")), call. = FALSE)
  }

  # ---- scope checks: regression, numeric predictors ------------------------
  if (!is.numeric(data[[outcome]])) {
    stop(paste0("panel_e2tree() supports regression only: outcome '", outcome,
                "' must be numeric. A categorical 'within deviation' is not yet ",
                "defined (open direction)."), call. = FALSE)
  }
  non_num <- features[!vapply(data[features], is.numeric, logical(1))]
  if (length(non_num)) {
    stop(sprintf(paste0("The between/within decomposition is defined on numeric ",
                        "predictors; non-numeric feature(s): %s."),
                 paste(non_num, collapse = ", ")), call. = FALSE)
  }

  if (!is.null(seed)) set.seed(seed)

  # ---- clean panel ----------------------------------------------------------
  keep_cols <- c(unit, time, outcome, features)
  d_all <- data[, keep_cols, drop = FALSE]
  d_all$.row <- seq_len(nrow(data))
  cc <- stats::complete.cases(d_all[, c(unit, time, outcome, features), drop = FALSE])
  d  <- d_all[cc, , drop = FALSE]
  if (nrow(d) == 0) stop("No complete cases in 'data' for the selected columns.", call. = FALSE)
  d[[unit]] <- as.character(d[[unit]])
  vars <- c(outcome, features)

  # ---- min_periods filter / singleton diagnostics ---------------------------
  tab <- table(d[[unit]])
  dropped_units <- names(tab)[tab < min_periods]
  if (length(dropped_units)) {
    d <- d[!(d[[unit]] %in% dropped_units), , drop = FALSE]
    if (nrow(d) == 0) {
      stop("All units were excluded by 'min_periods'; lower the threshold.", call. = FALSE)
    }
    tab <- table(d[[unit]])
  }
  n_singleton <- sum(tab == 1L)
  if (n_singleton > 0) {
    warning(sprintf(paste0("%d unit(s) have a single observation: their within ",
                           "rows are identically zero and carry no within ",
                           "information. Consider min_periods = 2."),
                    n_singleton), call. = FALSE)
  }

  n_units <- length(unique(d[[unit]]))
  if (n_units < 2) stop("Need at least two units for a panel decomposition.", call. = FALSE)

  # ---- duplicated (unit, time) rows -----------------------------------------
  if (!is.null(time)) {
    ndup <- sum(duplicated(d[, c(unit, time)]))
    if (ndup > 0) {
      warning(sprintf(paste0("%d duplicated (unit, time) row(s) found; unit and ",
                             "period means will average duplicates."), ndup),
              call. = FALSE)
    }
  }

  # ---- explanandum: observed outcome, or pooled-ensemble predictions --------
  y_obs <- d[[outcome]]
  pooled_pred <- NULL
  if (target == "pooled") {
    if (is.null(pooled_ensemble)) {
      pooled_ensemble <- .fit_ensemble(d[, vars, drop = FALSE], outcome,
                                       engine, ntree, ...)
    }
    pooled_pred <- get_ensemble_predictions(pooled_ensemble,
                                            d[, vars, drop = FALSE],
                                            type = "regression", oob = FALSE)
    d[[outcome]] <- pooled_pred          # decompose the model's predictions
  }
  y_work <- d[[outcome]]
  icc    <- .icc(y_work, d[[unit]])

  # original feature scales: reference for the constant-column tolerance
  ref_scale <- vapply(d[features], stats::sd, numeric(1), na.rm = TRUE)

  # ---- unit means (between representation source) ----------------------------
  # listwise: means over the retained complete rows (d).
  # unit.available: per-variable means over all available observations in the
  #   original data (less biased in unbalanced panels with missingness).
  if (na.action == "unit.available" && target == "outcome") {
    src <- data[, c(unit, vars), drop = FALSE]
    src[[unit]] <- as.character(src[[unit]])
    src <- src[src[[unit]] %in% unique(d[[unit]]), , drop = FALSE]
    um  <- stats::aggregate(src[vars], by = list(.unit = src[[unit]]),
                            FUN = function(z) mean(z, na.rm = TRUE))
  } else {
    if (na.action == "unit.available" && target == "pooled") {
      warning(paste0("na.action = 'unit.available' is not available with ",
                     "target = 'pooled' (the explanandum is defined on complete ",
                     "rows only); falling back to 'listwise'."), call. = FALSE)
      na.action <- "listwise"
    }
    um <- stats::aggregate(d[vars], by = list(.unit = d[[unit]]),
                           FUN = function(z) mean(z, na.rm = TRUE))
  }
  unit_ids <- as.character(um$.unit)

  # ---- BETWEEN representation: one row per unit (group means) ---------------
  bw_full <- um[, vars, drop = FALSE]
  bw_df   <- .drop_constant(bw_full, outcome, ref_scale)
  dropped_between <- setdiff(features, setdiff(names(bw_df), outcome))
  comp_b <- .panel_fit_component(bw_df, outcome, engine, ntree, setting_between,
                                 label = "between", dis_args = dis_args, ...)

  # ---- WITHIN representation: group-demeaned (Mundlak within) ---------------
  um_aligned <- um[match(d[[unit]], um$.unit), vars, drop = FALSE]
  u_dev <- as.data.frame(d[vars]) - as.data.frame(um_aligned)  # unit-demeaned

  time_means <- NULL
  tau_y <- rep(0, nrow(d))
  if (within == "twoway") {
    # period means of the unit-demeaned values: tau_t. Row-wise identity:
    # y_it = ybar_i + tau_t + (y_it - ybar_i - tau_t).
    tm <- stats::aggregate(u_dev, by = list(.time = d[[time]]),
                           FUN = function(z) mean(z, na.rm = TRUE))
    tm_aligned <- tm[match(d[[time]], tm$.time), vars, drop = FALSE]
    tau_y      <- as.numeric(tm_aligned[[outcome]])
    wn_full    <- u_dev - as.data.frame(tm_aligned)
    time_means <- tm
  } else {
    wn_full <- u_dev
  }
  wn_df <- .drop_constant(wn_full, outcome, ref_scale)
  dropped_within <- setdiff(features, setdiff(names(wn_df), outcome))

  within_signal <- .within_signal(d, features, um_aligned)
  if (is.finite(within_signal) && within_signal < 0.10) {
    warning(sprintf(paste0("Weak within signal (mean within-variance share = %.2f): ",
                           "the within tree may degenerate. Consider adding ",
                           "time-varying features."), within_signal), call. = FALSE)
  }

  within_feats <- setdiff(names(wn_df), outcome)
  if (length(within_feats) == 0) {
    warning(paste0("No time-varying feature survived within-demeaning; the within ",
                   "component is empty (within prediction = 0)."), call. = FALSE)
    comp_w <- .empty_component(nrow(d))
    comp_w$data <- wn_df
  } else {
    comp_w <- .panel_fit_component(wn_df, outcome, engine, ntree, setting_within,
                                   label = "within", dis_args = dis_args, ...)
  }

  # ---- panel reconstruction: between(+ period effect) + within --------------
  bpred       <- stats::setNames(comp_b$pred, unit_ids)
  between_obs <- as.numeric(bpred[d[[unit]]])
  panel_pred  <- between_obs + tau_y + comp_w$pred
  fid_panel   <- if (stats::sd(panel_pred) < 1e-12) NA_real_ else
    stats::cor(panel_pred, y_work)^2
  sse         <- sum((y_work - panel_pred)^2)
  sst         <- sum((y_work - mean(y_work))^2)
  r2_panel    <- if (sst > 0) 1 - sse / sst else NA_real_

  vs_outcome <- NULL
  if (target == "pooled") {
    sse_o <- sum((y_obs - panel_pred)^2)
    sst_o <- sum((y_obs - mean(y_obs))^2)
    vs_outcome <- list(
      outcome_var_recovered = if (stats::sd(panel_pred) < 1e-12) NA_real_ else
        stats::cor(panel_pred, y_obs)^2,
      r2 = if (sst_o > 0) 1 - sse_o / sst_o else NA_real_
    )
  }

  preds_df <- data.frame(.row = d$.row, unit = d[[unit]],
                         stringsAsFactors = FALSE, check.names = FALSE)
  if (!is.null(time)) preds_df[[time]] <- d[[time]]
  preds_df$outcome  <- y_obs
  if (target == "pooled") preds_df$.pooled <- pooled_pred
  preds_df$.between <- between_obs
  if (within == "twoway") preds_df$.timeeffect <- tau_y
  preds_df$.within  <- comp_w$pred
  preds_df$.panel   <- panel_pred

  # ---- strip heavy elements unless requested ---------------------------------
  if (!keep_D)    comp_b$D    <- comp_w$D    <- NULL
  if (!keep_data) comp_b$data <- comp_w$data <- NULL

  out <- list(
    between               = comp_b,
    within                = comp_w,
    outcome_var_recovered = fid_panel,
    r2_panel              = r2_panel,
    fidelity_panel        = fid_panel,   # deprecated alias
    vs_outcome            = vs_outcome,
    icc                   = icc,
    predictions           = preds_df,
    unit_means            = um,
    bpred                 = bpred,
    time_means            = time_means,
    within_signal         = within_signal,
    dropped               = list(between = dropped_between, within = dropped_within),
    dropped_units         = dropped_units,
    n_singleton           = n_singleton,
    features              = features,
    outcome               = outcome,
    unit                  = unit,
    time                  = time,
    engine                = engine,
    target                = target,
    within_type           = within,
    na.action             = na.action,
    pooled_ensemble       = if (target == "pooled") pooled_ensemble else NULL,
    call                  = match.call()
  )
  class(out) <- "e2panel"
  out
}


# ============================================================================
# S3 methods
# ============================================================================

.fid_str <- function(comp) {
  if (is.null(comp$tree)) return("-- (empty: no time-varying feature)")
  if (isTRUE(comp$degenerate)) return("-- (degenerate: single-leaf surrogate)")
  sprintf("%.3f", comp$fidelity)
}

#' @describeIn panel_e2tree Compact fidelity report.
#' @param x,object An \code{e2panel} object.
#' @method print e2panel
#' @export
print.e2panel <- function(x, ...) {
  explanandum <- if (x$target == "pooled") "pooled ensemble predictions" else "observed outcome"
  cat("Panel e2tree (between/within decomposition)\n")
  cat(sprintf("  target: %s | within: %s | engine: %s\n",
              x$target, x$within_type, x$engine))
  cat(sprintf("  units: %d | observations: %d | ICC of explanandum: %.2f\n",
              length(x$bpred), nrow(x$predictions), x$icc))
  cat(sprintf("  between fidelity (vs full ensemble): %s  | split vars: %s\n",
              .fid_str(x$between),
              paste(x$between$variables, collapse = ", ")))
  cat(sprintf("  within  fidelity (vs full ensemble): %s  | split vars: %s\n",
              .fid_str(x$within),
              paste(x$within$variables, collapse = ", ")))
  cat(sprintf("  PANEL reconstruction vs %s: R2 = %.3f (cor^2 = %.3f)\n",
              explanandum, x$r2_panel, x$outcome_var_recovered))
  if (!is.null(x$vs_outcome)) {
    cat(sprintf("  ... and vs observed outcome:  R2 = %.3f (cor^2 = %.3f)\n",
                x$vs_outcome$r2, x$vs_outcome$outcome_var_recovered))
  }
  if (is.finite(x$within_signal) && x$within_signal < 0.10) {
    cat(sprintf("  ! weak within signal (mean share = %.2f)\n", x$within_signal))
  }
  invisible(x)
}

#' @describeIn panel_e2tree Detailed summary: fidelity, decomposed importance,
#'   dropped features, panel shape.
#' @method summary e2panel
#' @export
summary.e2panel <- function(object, ...) {
  cat("Panel e2tree summary\n")
  cat(paste(rep("-", 60), collapse = ""), "\n")
  n_obs   <- nrow(object$predictions)
  n_units <- length(object$bpred)
  tab     <- table(object$predictions$unit)
  balanced <- length(unique(tab)) == 1
  cat(sprintf("Panel: %d units x ~%.1f periods = %d obs (%s)\n",
              n_units, n_obs / n_units, n_obs,
              if (balanced) "balanced" else "unbalanced"))
  cat(sprintf("Outcome: %s | Engine: %s | Target: %s | Within: %s | NA: %s\n",
              object$outcome, object$engine, object$target,
              object$within_type, object$na.action))
  cat(sprintf("ICC of explanandum: %.3f\n", object$icc))
  if (object$n_singleton > 0) {
    cat(sprintf("Single-observation units retained: %d (their within rows are zero)\n",
                object$n_singleton))
  }
  if (length(object$dropped_units)) {
    cat(sprintf("Units excluded by min_periods: %d\n", length(object$dropped_units)))
  }
  cat("\nFidelity (vs full-ensemble predictions)\n")
  cat(sprintf("  between: %s\n", .fid_str(object$between)))
  cat(sprintf("  within : %s\n", .fid_str(object$within)))
  cat(sprintf("  panel reconstruction: R2 = %.3f (cor^2 = %.3f)\n",
              object$r2_panel, object$outcome_var_recovered))
  if (!is.null(object$vs_outcome)) {
    cat(sprintf("  vs observed outcome : R2 = %.3f (cor^2 = %.3f)\n",
                object$vs_outcome$r2, object$vs_outcome$outcome_var_recovered))
  }
  cat(sprintf("  mean within-variance share of features: %.3f\n\n",
              object$within_signal))

  imp_tab <- function(comp, label) {
    if (is.null(comp$varimp) || is.null(comp$varimp$vimp) ||
        nrow(comp$varimp$vimp) == 0) {
      cat(sprintf("  %s: (none)\n", label)); return(invisible())
    }
    vi <- comp$varimp$vimp
    cat(sprintf("  %s:\n", label))
    for (i in seq_len(nrow(vi))) {
      cat(sprintf("    %-24s %.4f\n", vi[[1]][i], vi[[2]][i]))
    }
  }
  cat("Decomposed variable importance (mean impurity decrease)\n")
  imp_tab(object$between, "between")
  imp_tab(object$within,  "within")

  dropped <- object$dropped
  if (length(dropped$between) || length(dropped$within)) {
    cat("\nDropped (constant) features\n")
    if (length(dropped$between))
      cat(sprintf("  between: %s\n", paste(dropped$between, collapse = ", ")))
    if (length(dropped$within))
      cat(sprintf("  within : %s\n", paste(dropped$within, collapse = ", ")))
  }
  invisible(object)
}

#' Plot a Panel E2Tree
#'
#' Plots the between and/or within surrogate via \code{\link{plot_e2tree}}.
#'
#' @param x An \code{e2panel} object.
#' @param component Which surrogate to draw: \code{"both"} (default, drawn
#'   sequentially), \code{"between"}, or \code{"within"}.
#' @param ... Passed to \code{\link{plot_e2tree}} (e.g. \code{main}).
#' @return Invisibly, the underlying \code{rpart} object(s).
#' @method plot e2panel
#' @export
plot.e2panel <- function(x, component = c("both", "between", "within"), ...) {
  component <- match.arg(component)
  draw <- function(comp, main) {
    if (is.null(comp$tree)) {
      warning("within component is empty; nothing to plot.", call. = FALSE)
      return(NULL)
    }
    plot_e2tree(comp$tree, comp$ensemble, main = main, ...)
  }
  res <- switch(component,
    between = draw(x$between, "Panel e2tree - BETWEEN"),
    within  = draw(x$within,  "Panel e2tree - WITHIN"),
    both    = list(between = draw(x$between, "Panel e2tree - BETWEEN"),
                   within  = draw(x$within,  "Panel e2tree - WITHIN"))
  )
  invisible(res)
}

#' Panel E2Tree Predictions
#'
#' Reconstructs the additive panel explanation for new observations:
#' \code{between(unit mean) [+ period effect] + within(deviation)}. New
#' observations are demeaned using the *training* unit means (and, under
#' \code{within = "twoway"}, the training period means). For units unseen in
#' training the between prediction falls back to the mean over training units
#' and the within contribution is set to zero (with a warning). Missing
#' feature values in \code{newdata} are treated as "at the unit mean"
#' (deviation zero), with a warning.
#'
#' @param object An \code{e2panel} object.
#' @param newdata A \code{data.frame} in the same long panel format (must
#'   contain the \code{unit} column, the predictors and -- for
#'   \code{within = "twoway"} -- the \code{time} column).
#' @param ... Ignored.
#' @return \code{data.frame(unit, .between, [.timeeffect], .within, .panel)}.
#' @method predict e2panel
#' @export
predict.e2panel <- function(object, newdata, ...) {
  unit     <- object$unit
  time     <- object$time
  features <- object$features
  nd       <- as.data.frame(newdata)
  if (!(unit %in% names(nd))) {
    stop(sprintf("'newdata' must contain the unit column '%s'.", unit), call. = FALSE)
  }
  miss <- setdiff(features, names(nd))
  if (length(miss)) {
    stop(sprintf("'newdata' is missing predictor(s): %s.",
                 paste(miss, collapse = ", ")), call. = FALSE)
  }
  non_num <- features[!vapply(nd[features], is.numeric, logical(1))]
  if (length(non_num)) {
    stop(sprintf("Non-numeric predictor(s) in 'newdata': %s.",
                 paste(non_num, collapse = ", ")), call. = FALSE)
  }
  twoway <- identical(object$within_type, "twoway")
  if (twoway && !(time %in% names(nd))) {
    stop(sprintf("within = 'twoway' predictions require the time column '%s' in 'newdata'.",
                 time), call. = FALSE)
  }

  u  <- as.character(nd[[unit]])
  bp <- object$bpred

  # between: per-unit prediction; unseen units -> mean of training between preds
  between_obs <- bp[u]
  unseen <- is.na(between_obs)
  if (any(unseen)) {
    warning(sprintf("%d observation(s) from unit(s) unseen in training; between = mean, within deviation = 0.",
                    sum(unseen)), call. = FALSE)
    between_obs[unseen] <- mean(bp)
  }
  between_obs <- as.numeric(between_obs)

  # within: demean newdata features by stored unit (and period) means;
  # missing values -> deviation 0 ("at the unit mean"), with a warning
  um  <- object$unit_means
  ua  <- um[match(u, um$.unit), features, drop = FALSE]
  dev <- as.data.frame(nd[features]) - as.data.frame(ua)

  tau_y <- rep(0, nrow(nd))
  if (twoway) {
    tm  <- object$time_means
    ti  <- match(nd[[time]], tm$.time)
    unseen_t <- is.na(ti)
    if (any(unseen_t)) {
      warning(sprintf("%d observation(s) from period(s) unseen in training; period effect = 0.",
                      sum(unseen_t)), call. = FALSE)
    }
    ta <- tm[ti, features, drop = FALSE]
    ta[unseen_t, ] <- 0
    dev <- dev - as.data.frame(ta)
    tau <- tm[[object$outcome]][ti]
    tau[unseen_t] <- 0
    tau_y <- as.numeric(tau)
  }

  # NA deviations of unseen units are handled (and warned about) above; only
  # genuinely missing feature values of known units are zero-imputed here.
  n_imputed <- sum(is.na(dev[!unseen, , drop = FALSE]))
  if (n_imputed > 0) {
    warning(sprintf(paste0("%d missing feature value(s) in 'newdata'; their within ",
                           "deviation is set to 0 (i.e. 'at the unit mean')."),
                    n_imputed), call. = FALSE)
  }
  dev[is.na(dev)] <- 0

  if (is.null(object$within$tree)) {
    within_pred <- rep(0, nrow(nd))
  } else {
    within_pred <- as.numeric(predict(object$within$tree, newdata = dev)$fit)
  }
  # Unseen units have no training mean, so their within deviation is undefined;
  # the within contribution is set to zero (the panel value reduces to the
  # between fallback) rather than injecting the surrogate's value-at-origin.
  within_pred[unseen] <- 0

  out <- data.frame(unit = u, .between = between_obs, check.names = FALSE)
  if (twoway) out$.timeeffect <- tau_y
  out$.within <- within_pred
  out$.panel  <- between_obs + tau_y + within_pred
  out
}
