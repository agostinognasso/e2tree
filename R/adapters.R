# ============================================================================
# Adapter Layer for Ensemble Models
#
# Three S3 generics provide a unified interface over diverse ensemble backends.
# To add support for a new model class, implement the three methods below for
# that class — no changes to the core algorithm are required.
#
#   1. get_ensemble_type(ensemble)           → "classification" | "regression"
#   2. extract_terminal_nodes(ensemble, data) → data.frame (n_obs × n_trees)
#   3. get_ensemble_predictions(ensemble, data, type) → numeric vector (n_obs)
#
# All three generics share the same `default` failure mode and the same list
# of supported classes; `.unsupported_backend()` centralises that message so
# the supported-classes list is maintained in exactly one place.
# ============================================================================

.SUPPORTED_BACKENDS <- c(
  "randomForest", "ranger", "xgb.Booster",
  "lgb.Booster", "gbm", "catboost.CatBoost", "catboost.Model"
)

.unsupported_backend <- function(generic, ensemble) {
  stop(sprintf(
    "%s(): unsupported ensemble class '%s'.\nSupported classes: %s.",
    generic,
    paste(class(ensemble), collapse = ", "),
    paste(.SUPPORTED_BACKENDS, collapse = ", ")
  ), call. = FALSE)
}


# ============================================================================
# 1.  get_ensemble_type()
# ============================================================================

#' Determine Task Type from a Trained Ensemble Model
#'
#' Returns \code{"classification"} or \code{"regression"} depending on the
#' objective used to train the ensemble.
#'
#' @param ensemble A trained ensemble model. Supported classes:
#'   \code{randomForest}, \code{ranger}, \code{xgb.Booster},
#'   \code{lgb.Booster}, \code{gbm}, \code{catboost.CatBoost}.
#' @return Character scalar: \code{"classification"} or \code{"regression"}.
#' @export
get_ensemble_type <- function(ensemble) {
  UseMethod("get_ensemble_type")
}

#' @export
#' @method get_ensemble_type randomForest
get_ensemble_type.randomForest <- function(ensemble) {
  ensemble$type
}

#' @export
#' @method get_ensemble_type ranger
get_ensemble_type.ranger <- function(ensemble) {
  tolower(ensemble$treetype)
}

#' @export
#' @method get_ensemble_type xgb.Booster
get_ensemble_type.xgb.Booster <- function(ensemble) {
  check_package("xgboost")
  # $params may be empty in xgboost >= 2.x; fall back to xgb.config()
  obj <- ensemble$params$objective
  if (is.null(obj) || identical(obj, "")) {
    obj <- tryCatch({
      cfg <- xgboost::xgb.config(ensemble)
      if (is.list(cfg)) {
        # xgboost >= 2.x: config is an R list
        cfg[["learner"]][["learner_train_param"]][["objective"]]
      } else {
        # xgboost < 2.x: config is a JSON character string
        m <- regexpr('"objective"\\s*:\\s*"([^"]+)"', cfg, perl = TRUE)
        if (m == -1L) NULL else sub('.*"objective"\\s*:\\s*"([^"]+)".*', "\\1",
                                    regmatches(cfg, m))
      }
    }, error = function(e) NULL)
  }
  if (is.null(obj) || identical(obj, "")) {
    obj <- tryCatch(xgboost::xgb.attr(ensemble, "objective"), error = function(e) NULL)
  }
  if (is.null(obj) || identical(obj, "")) {
    stop("Cannot determine objective from xgb.Booster.", call. = FALSE)
  }
  cls_prefixes <- c("binary:", "multi:")
  if (any(sapply(cls_prefixes, function(p) startsWith(obj, p)))) {
    return("classification")
  }
  "regression"
}

#' @export
#' @method get_ensemble_type lgb.Booster
get_ensemble_type.lgb.Booster <- function(ensemble) {
  check_package("lightgbm")
  obj <- ensemble$params$objective
  if (is.null(obj)) stop("Cannot determine objective from lgb.Booster.", call. = FALSE)
  cls_objs <- c("binary", "multiclass", "softmax", "multiclassova",
                "ovr", "multiclass_ova", "cross_entropy")
  if (any(sapply(cls_objs, function(x) startsWith(tolower(obj), x)))) {
    return("classification")
  }
  "regression"
}

#' @export
#' @method get_ensemble_type gbm
get_ensemble_type.gbm <- function(ensemble) {
  check_package("gbm")
  dist <- ensemble$distribution$name
  reg_dists <- c("gaussian", "laplace", "tdist", "quantile",
                 "huberized", "tweedie", "poisson", "gamma", "coxph")
  if (dist %in% reg_dists) "regression" else "classification"
}

#' @export
#' @method get_ensemble_type catboost.CatBoost
get_ensemble_type.catboost.CatBoost <- function(ensemble) {
  .catboost_get_ensemble_type(ensemble)
}

#' @export
#' @method get_ensemble_type catboost.Model
get_ensemble_type.catboost.Model <- function(ensemble) {
  .catboost_get_ensemble_type(ensemble)
}

# Internal worker shared by both CatBoost class names.  Older releases of the
# catboost R package returned objects of class \code{catboost.CatBoost}; from
# release 1.2.x the class is \code{catboost.Model}.  Both names are supported.
.catboost_get_ensemble_type <- function(ensemble) {
  check_package("catboost")
  params <- tryCatch(
    catboost::catboost.get_model_params(ensemble),
    error = function(e) NULL
  )
  loss <- .catboost_extract_loss(params)
  if (is.null(loss)) {
    # Last-resort fallback: inspect the user-side params attribute that
    # `catboost.train()` typically attaches to the returned model.
    loss <- .catboost_extract_loss(attr(ensemble, "params"))
  }
  if (is.null(loss) || !is.character(loss) || length(loss) == 0L) {
    stop("Cannot determine 'loss_function' from CatBoost model.", call. = FALSE)
  }
  cls_losses <- c("Logloss", "CrossEntropy", "MultiClass", "MultiClassOneVsAll")
  if (any(startsWith(loss[1L], cls_losses))) {
    return("classification")
  }
  "regression"
}

# Pull the loss function name out of a CatBoost params list.  Across catboost
# R-package versions the entry can be:
#   * a plain character scalar      ("MultiClass")
#   * a named list with "type"      (list(type = "MultiClass", ...))
#   * a list whose first element is the name (legacy odd serialisations)
#   * absent entirely (NULL params)
# This helper canonicalises all four shapes to a length-one character or NULL.
.catboost_extract_loss <- function(params) {
  if (is.null(params)) return(NULL)
  loss <- params[["loss_function"]]
  if (is.null(loss) && !is.null(params[["loss_function_params"]])) {
    loss <- params[["loss_function_params"]]
  }
  if (is.null(loss)) return(NULL)
  if (is.list(loss)) {
    nm <- loss[["type"]]
    if (is.null(nm)) nm <- loss[["name"]]
    if (is.null(nm) && length(loss) > 0L) nm <- loss[[1L]]
    loss <- nm
  }
  if (is.null(loss)) return(NULL)
  loss <- as.character(loss)
  if (length(loss) == 0L || all(is.na(loss)) || all(!nzchar(loss))) return(NULL)
  loss[1L]
}

#' @export
#' @method get_ensemble_type default
get_ensemble_type.default <- function(ensemble) {
  .unsupported_backend("get_ensemble_type", ensemble)
}


# ============================================================================
# 2.  extract_terminal_nodes()
# ============================================================================

#' Extract Terminal Node Assignments from an Ensemble Model
#'
#' Returns a \code{data.frame} with \code{n_obs} rows and \code{n_trees}
#' columns where each cell is the terminal-node index assigned to that
#' observation by that tree.
#'
#' @param ensemble A trained ensemble model.
#' @param data A \code{data.frame} of observations (may include the response
#'   column; it is ignored internally).
#' @return A \code{data.frame} with \code{n_obs} rows and \code{n_trees} columns
#'   of integer terminal-node identifiers.
#' @export
extract_terminal_nodes <- function(ensemble, data) {
  UseMethod("extract_terminal_nodes")
}

#' @export
#' @method extract_terminal_nodes randomForest
extract_terminal_nodes.randomForest <- function(ensemble, data) {
  as.data.frame(
    attr(predict(ensemble, newdata = data, nodes = TRUE), "nodes")
  )
}

#' @export
#' @method extract_terminal_nodes ranger
extract_terminal_nodes.ranger <- function(ensemble, data) {
  if (is.null(ensemble$forest)) {
    stop("ranger model must be trained with write.forest = TRUE.", call. = FALSE)
  }
  as.data.frame(
    predict(ensemble, data, type = "terminalNodes", num.threads = 1L)$predictions
  )
}

#' @export
#' @method extract_terminal_nodes xgb.Booster
extract_terminal_nodes.xgb.Booster <- function(ensemble, data) {
  check_package("xgboost")
  dm <- .xgb_dmatrix(ensemble, data)
  as.data.frame(predict(.xgb_strip_wrapper(ensemble), dm, predleaf = TRUE))
}

#' @export
#' @method extract_terminal_nodes lgb.Booster
extract_terminal_nodes.lgb.Booster <- function(ensemble, data) {
  check_package("lightgbm")
  feat <- .lgb_feature_names(ensemble)
  X    <- as.matrix(data[, feat, drop = FALSE])
  as.data.frame(ensemble$predict(X, predleaf = TRUE))
}

#' @export
#' @method extract_terminal_nodes gbm
extract_terminal_nodes.gbm <- function(ensemble, data) {
  check_package("gbm")
  n_trees <- ensemble$n.trees
  n_obs   <- nrow(data)
  X       <- data[, ensemble$var.names, drop = FALSE]

  # Convert factor columns to factors with the same levels as in training
  for (v in ensemble$var.names) {
    if (is.character(X[[v]])) X[[v]] <- as.factor(X[[v]])
  }

  leaf_mat <- matrix(0L, nrow = n_obs, ncol = n_trees)
  for (t in seq_len(n_trees)) {
    leaf_mat[, t] <- .gbm_leaf_indices(ensemble$trees[[t]], X, ensemble$c.splits)
  }
  as.data.frame(leaf_mat)
}

#' @export
#' @method extract_terminal_nodes catboost.CatBoost
extract_terminal_nodes.catboost.CatBoost <- function(ensemble, data) {
  .catboost_extract_terminal_nodes(ensemble, data)
}

#' @export
#' @method extract_terminal_nodes catboost.Model
extract_terminal_nodes.catboost.Model <- function(ensemble, data) {
  .catboost_extract_terminal_nodes(ensemble, data)
}

.catboost_extract_terminal_nodes <- function(ensemble, data) {
  check_package("catboost")

  # Build a catboost pool (without label)
  pool <- .catboost_pool_from_data(ensemble, data)

  n_trees <- .catboost_n_trees(ensemble)
  n_obs   <- nrow(data)
  leaf_mat <- matrix(0L, nrow = n_obs, ncol = n_trees)

  # Predict raw score from each individual tree and discretise into leaf IDs.
  # CatBoost R API does not expose leaf indices directly; raw scores per tree
  # are unique per leaf and serve as a faithful proxy.  For multi-class
  # objectives the per-tree prediction is a matrix of shape
  # (n_obs x n_classes); we collapse it to a single key per observation by
  # concatenating the rounded raw scores, so two observations sharing the
  # same leaf still receive the same integer identifier.
  for (t in seq_len(n_trees)) {
    raw <- catboost::catboost.predict(
      ensemble, pool,
      prediction_type = "RawFormulaVal",
      ntree_start     = t - 1L,
      ntree_end       = t
    )
    if (is.matrix(raw) || (!is.null(dim(raw)) && length(dim(raw)) == 2L)) {
      # Multi-class: catboost.predict() returns an (n_obs x n_classes) matrix.
      raw_mat <- as.matrix(raw)
      if (nrow(raw_mat) != n_obs && ncol(raw_mat) == n_obs) {
        raw_mat <- t(raw_mat)
      }
      raw_mat <- round(raw_mat, digits = 10L)
      key <- apply(raw_mat, 1L, paste, collapse = "|")
    } else {
      raw_vec <- as.numeric(raw)
      if (length(raw_vec) == n_obs) {
        key <- as.character(round(raw_vec, digits = 10L))
      } else {
        stop(sprintf(
          "extract_terminal_nodes(catboost): unexpected raw-score length (%d) for tree %d (expected %d or an n_obs-row matrix).",
          length(raw_vec), t, n_obs), call. = FALSE)
      }
    }
    leaf_mat[, t] <- as.integer(factor(key))
  }
  as.data.frame(leaf_mat)
}

#' @export
#' @method extract_terminal_nodes default
extract_terminal_nodes.default <- function(ensemble, data) {
  .unsupported_backend("extract_terminal_nodes", ensemble)
}


# ============================================================================
# 3.  get_ensemble_predictions()
# ============================================================================

#' Get Training Predictions from an Ensemble Model
#'
#' Returns a numeric vector of length \code{n_obs} with the ensemble's
#' prediction for every training observation. For models that store
#' out-of-bag (OOB) predictions (\code{randomForest}, \code{ranger}) the
#' stored OOB vector is returned; for other models in-sample predictions
#' are computed from the training data.
#'
#' @param ensemble A trained ensemble model.
#' @param data The training \code{data.frame} that was used to fit the model.
#' @param type Character: \code{"classification"} or \code{"regression"}.
#' @return Numeric vector of length \code{nrow(data)}.
#' @export
get_ensemble_predictions <- function(ensemble, data, type) {
  UseMethod("get_ensemble_predictions")
}

#' @export
#' @method get_ensemble_predictions randomForest
get_ensemble_predictions.randomForest <- function(ensemble, data, type) {
  as.numeric(ensemble$predicted)
}

#' @export
#' @method get_ensemble_predictions ranger
get_ensemble_predictions.ranger <- function(ensemble, data, type) {
  as.numeric(ensemble$predictions)
}

#' @export
#' @method get_ensemble_predictions xgb.Booster
get_ensemble_predictions.xgb.Booster <- function(ensemble, data, type) {
  check_package("xgboost")
  as.numeric(predict(.xgb_strip_wrapper(ensemble), .xgb_dmatrix(ensemble, data)))
}

#' @export
#' @method get_ensemble_predictions lgb.Booster
get_ensemble_predictions.lgb.Booster <- function(ensemble, data, type) {
  check_package("lightgbm")
  feat <- .lgb_feature_names(ensemble)
  X <- as.matrix(data[, feat, drop = FALSE])
  as.numeric(ensemble$predict(X))
}

#' @export
#' @method get_ensemble_predictions gbm
get_ensemble_predictions.gbm <- function(ensemble, data, type) {
  check_package("gbm")
  as.numeric(
    gbm::predict.gbm(ensemble, newdata = data,
                     n.trees = ensemble$n.trees, type = "response")
  )
}

#' @export
#' @method get_ensemble_predictions catboost.CatBoost
get_ensemble_predictions.catboost.CatBoost <- function(ensemble, data, type) {
  .catboost_get_ensemble_predictions(ensemble, data, type)
}

#' @export
#' @method get_ensemble_predictions catboost.Model
get_ensemble_predictions.catboost.Model <- function(ensemble, data, type) {
  .catboost_get_ensemble_predictions(ensemble, data, type)
}

.catboost_get_ensemble_predictions <- function(ensemble, data, type) {
  check_package("catboost")
  pool      <- .catboost_pool_from_data(ensemble, data)
  pred_type <- if (type == "classification") "Class" else "RawFormulaVal"
  as.numeric(catboost::catboost.predict(ensemble, pool, prediction_type = pred_type))
}

#' @export
#' @method get_ensemble_predictions default
get_ensemble_predictions.default <- function(ensemble, data, type) {
  .unsupported_backend("get_ensemble_predictions", ensemble)
}


# ============================================================================
# Internal helpers
# ============================================================================

# Convenience wrapper: build a double-precision predictor matrix and wrap it
# in an xgb.DMatrix in one call. Used by both extract_terminal_nodes() and
# get_ensemble_predictions() for the xgb.Booster backend.
.xgb_dmatrix <- function(ensemble, data) {
  X <- .xgb_predictor_matrix(ensemble, data)
  xgboost::xgb.DMatrix(data = X)
}

# In xgboost >= 3.x, models built via xgboost::xgboost() carry the additional
# class "xgboost" with its own predict method that rejects xgb.DMatrix inputs.
# To force dispatch to predict.xgb.Booster (which we need for `predleaf`), we
# strip the wrapper class for the duration of the prediction call.
.xgb_strip_wrapper <- function(ensemble) {
  cls <- class(ensemble)
  if ("xgboost" %in% cls && "xgb.Booster" %in% cls) {
    class(ensemble) <- setdiff(cls, "xgboost")
  }
  ensemble
}


# Build a double-precision predictor matrix for xgb.DMatrix.
# In xgboost >= 3.x, `ensemble$feature_names` is NULL; we fall back to all
# numeric columns, trimming to `num_feature` if data contains extra columns
# (e.g., the response variable appended by the user).
.xgb_predictor_matrix <- function(ensemble, data) {
  feat <- ensemble$feature_names
  if (length(feat) > 0L) {
    X <- as.matrix(data[, feat, drop = FALSE])
  } else {
    # xgboost >= 3.x: feature names not stored — use num_feature from config
    num_feat <- tryCatch({
      cfg <- xgboost::xgb.config(ensemble)
      as.integer(cfg[["learner"]][["learner_model_param"]][["num_feature"]])
    }, error = function(e) -1L)

    num_cols <- vapply(as.data.frame(data), is.numeric, logical(1L))
    X <- as.matrix(as.data.frame(data)[, num_cols, drop = FALSE])

    if (num_feat > 0L && ncol(X) > num_feat) {
      # Extra numeric columns present (e.g., response appended at the end)
      X <- X[, seq_len(num_feat), drop = FALSE]
    }
  }
  storage.mode(X) <- "double"
  X
}


# Extract feature names from an lgb.Booster model.
# In lightgbm < 4.x they are in private$used_feature_names; in 4.x they must
# be parsed from the dump_model() JSON string.
.lgb_feature_names <- function(ensemble) {
  # Old API (lightgbm < 4.x)
  feat <- tryCatch(
    ensemble$.__enclos_env__$private$used_feature_names,
    error = function(e) NULL
  )
  if (!is.null(feat) && length(feat) > 0L) return(feat)

  # New API (lightgbm >= 4.x): parse feature_names from the model JSON dump
  dmp <- tryCatch(ensemble$dump_model(), error = function(e) NULL)
  if (is.null(dmp)) {
    stop("Cannot determine feature names from lgb.Booster model.", call. = FALSE)
  }
  # Locate "feature_names":[...] and extract the array content
  tag_pos <- regexpr('"feature_names"', dmp, fixed = TRUE)
  if (tag_pos == -1L) {
    stop("Cannot parse feature_names from lgb.Booster dump.", call. = FALSE)
  }
  rest        <- substring(dmp, tag_pos)
  open_pos    <- regexpr("[", rest, fixed = TRUE)
  close_pos   <- regexpr("]", rest, fixed = TRUE)
  array_inner <- substring(rest, open_pos + 1L, close_pos - 1L)
  # array_inner is like: "cyl","disp","hp",...
  vals <- gsub('"', "", unlist(strsplit(array_inner, ",")), fixed = TRUE)
  trimws(vals)
}


# Traverse a single gbm tree and return the (1-indexed) leaf node for each obs.
# tree     : data.frame from ensemble$trees[[t]]
# X        : data.frame of predictor columns (already subset to var.names)
# c_splits : ensemble$c.splits (list of categorical split directions)
.gbm_leaf_indices <- function(tree, X, c_splits) {
  n_obs <- nrow(X)
  # Root node is row 1 (gbm uses 0-based indices internally).
  # ensemble$trees[[t]] is a plain list (not a data.frame): access by position.
  # [[1]]=SplitVar, [[2]]=SplitCodePred, [[3]]=LeftNode,
  # [[4]]=RightNode, [[5]]=MissingNode
  split_var  <- tree[[1L]]
  split_code <- tree[[2L]]
  left_node  <- tree[[3L]]
  right_node <- tree[[4L]]
  miss_node  <- tree[[5L]]

  current_nodes <- rep(0L, n_obs)

  repeat {
    sv <- split_var[current_nodes + 1L]
    # All observations have reached a terminal node
    if (all(sv == -1L)) break

    next_nodes <- current_nodes

    # Vectorised traversal: process each distinct non-terminal node
    non_terminal <- unique(current_nodes[sv != -1L])

    for (node_idx in non_terminal) {
      in_node  <- which(current_nodes == node_idx)
      sv_val   <- split_var[node_idx + 1L]       # variable index (0-based)
      sc_val   <- split_code[node_idx + 1L]       # split value / cat index
      left_n   <- left_node[node_idx + 1L]
      right_n  <- right_node[node_idx + 1L]
      miss_n   <- miss_node[node_idx + 1L]

      var_name <- names(X)[sv_val + 1L]  # convert to 1-based
      x_vals   <- X[[var_name]][in_node]

      if (is.numeric(x_vals)) {
        is_miss   <- is.na(x_vals)
        goes_left <- !is_miss & (x_vals < sc_val)
      } else {
        # Categorical variable: sc_val indexes into c_splits
        cat_split <- c_splits[[as.integer(sc_val) + 1L]]
        lvl_idx   <- match(as.character(x_vals), levels(X[[var_name]]))
        is_miss   <- is.na(lvl_idx)
        goes_left <- !is_miss & (cat_split[lvl_idx] == -1L)
      }

      next_nodes[in_node[is_miss]]              <- miss_n
      next_nodes[in_node[!is_miss & goes_left]] <- left_n
      next_nodes[in_node[!is_miss & !goes_left]] <- right_n
    }

    current_nodes <- next_nodes
  }

  current_nodes + 1L  # return 1-indexed leaf IDs
}


# Build a catboost.Pool from a data.frame, stripping the response column if
# present.  The label column name is stored as an attribute on the model by
# e2tree/createDisMatrix callers; if absent we strip nothing.
.catboost_pool_from_data <- function(ensemble, data) {
  label_col <- attr(ensemble, "e2tree_label")
  if (!is.null(label_col) && label_col %in% colnames(data)) {
    data <- data[, setdiff(colnames(data), label_col), drop = FALSE]
  }
  catboost::catboost.load_pool(data = data)
}


# Return the number of trees in a CatBoost model.
.catboost_n_trees <- function(ensemble) {
  params <- tryCatch(
    catboost::catboost.get_model_params(ensemble),
    error = function(e) NULL
  )
  n <- .catboost_extract_n_trees(params)
  if (is.null(n)) n <- .catboost_extract_n_trees(attr(ensemble, "params"))
  if (is.null(n)) {
    # Some catboost versions expose the count via dedicated helpers.
    n <- tryCatch(
      as.integer(catboost::catboost.get_model_params(ensemble)$boosting_options$iterations),
      error = function(e) NULL
    )
  }
  if (is.null(n) || !is.finite(n) || n <= 0L) {
    stop("Cannot determine number of trees from catboost model.", call. = FALSE)
  }
  as.integer(n)
}

# Walk a CatBoost params list looking for an `iterations` field.  Different
# catboost releases place it at the top level, under `boosting_options`, or
# under `tree_learner_options`; this helper handles all three.
.catboost_extract_n_trees <- function(params) {
  if (is.null(params) || !is.list(params)) return(NULL)
  if (!is.null(params$iterations)) return(as.integer(params$iterations[1L]))
  if (is.list(params$boosting_options) && !is.null(params$boosting_options$iterations)) {
    return(as.integer(params$boosting_options$iterations[1L]))
  }
  if (is.list(params$tree_learner_options) && !is.null(params$tree_learner_options$iterations)) {
    return(as.integer(params$tree_learner_options$iterations[1L]))
  }
  NULL
}
