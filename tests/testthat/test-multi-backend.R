library(testthat)
library(e2tree)

# ============================================================================
# Multi-backend integration tests for createDisMatrix() and the adapter layer.
#
# These tests assert the *invariants* that any supported backend must satisfy:
#   - D is a square dense matrix of the right size,
#   - D has zeros on the diagonal,
#   - D is approximately symmetric,
#   - D is non-degenerate (i.e. it is not the all-zero matrix — this would
#     indicate the silent leaf-extraction bug fixed for gbm),
#   - D carries the ensemble_backend attribute set by createDisMatrix(),
#   - validate_terminal_nodes() rejects malformed adapter output.
#
# Each backend is skipped automatically when the corresponding R package is
# not available, so the file is safe to run in any CI environment.
# ============================================================================

assert_dis_invariants <- function(D, n, backend) {
  expect_true(is.matrix(D))
  expect_equal(dim(D), c(n, n))
  expect_equal(as.numeric(diag(D)), rep(0, n),
               info = sprintf("backend=%s: diagonal must be zero", backend))
  expect_lt(max(abs(D - t(D))), 1e-8,
            label = sprintf("backend=%s: D must be symmetric", backend))
  # Non-degenerate: at least one off-diagonal entry must be non-zero. An
  # all-zero D is the smoking gun of an extract_terminal_nodes() failure.
  expect_gt(sum(D), 0,
            label = sprintf("backend=%s: D must be non-degenerate", backend))
  expect_equal(attr(D, "ensemble_backend"), backend)
}

prep_iris <- function() {
  set.seed(42)
  data(iris)
  idx <- sample(seq_len(nrow(iris)), size = floor(0.6 * nrow(iris)))
  iris[idx, ]
}

prep_mtcars <- function() {
  set.seed(42)
  data(mtcars)
  idx <- sample(seq_len(nrow(mtcars)), size = floor(0.7 * nrow(mtcars)))
  mtcars[idx, ]
}

# ----------------------------------------------------------------------------
# randomForest (regression) — baseline sanity (covered elsewhere for clf)
# ----------------------------------------------------------------------------
test_that("randomForest regression: D invariants hold", {
  skip_if_not_installed("randomForest")
  tr <- prep_mtcars()
  ens <- randomForest::randomForest(mpg ~ ., data = tr, ntree = 50)
  D <- createDisMatrix(ens, data = tr, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "randomForest")
})

# ----------------------------------------------------------------------------
# ranger
# ----------------------------------------------------------------------------
test_that("ranger classification: D invariants hold", {
  skip_if_not_installed("ranger")
  tr <- prep_iris()
  ens <- ranger::ranger(Species ~ ., data = tr, num.trees = 100,
                        write.forest = TRUE)
  D <- createDisMatrix(ens, data = tr, label = "Species",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "ranger")
})

test_that("ranger regression: D invariants hold", {
  skip_if_not_installed("ranger")
  tr <- prep_mtcars()
  ens <- ranger::ranger(mpg ~ ., data = tr, num.trees = 100,
                        write.forest = TRUE)
  D <- createDisMatrix(ens, data = tr, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "ranger")
})

# ----------------------------------------------------------------------------
# xgboost
# ----------------------------------------------------------------------------
test_that("xgb.Booster regression: D invariants hold", {
  skip_if_not_installed("xgboost")
  tr <- prep_mtcars()
  X <- as.matrix(tr[, setdiff(names(tr), "mpg")])
  y <- tr$mpg
  # xgb.train returns a plain xgb.Booster (no `xgboost` wrapper class) and is
  # API-stable across xgboost 1.x/2.x/3.x; use it for clean test output.
  dtrain <- xgboost::xgb.DMatrix(data = X, label = y)
  ens <- xgboost::xgb.train(
    params = list(objective = "reg:squarederror"),
    data = dtrain, nrounds = 30, verbose = 0
  )
  D <- createDisMatrix(ens, data = tr, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "xgb.Booster")
})

test_that("xgb.Booster (high-level wrapper) is also handled", {
  skip_if_not_installed("xgboost")
  tr <- prep_mtcars()
  X <- as.matrix(tr[, setdiff(names(tr), "mpg")])
  y <- tr$mpg
  # The high-level xgboost::xgboost() returns class c("xgboost","xgb.Booster")
  # in xgboost >= 3.x. The adapter must dispatch to the booster method even
  # when the wrapper class is present.
  suppressWarnings(
    ens <- xgboost::xgboost(data = X, label = y, nrounds = 20,
                            objective = "reg:squarederror", verbose = 0)
  )
  D <- createDisMatrix(ens, data = tr, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "xgb.Booster")
})

# ----------------------------------------------------------------------------
# gbm — historically the failing backend; explicit guard
# ----------------------------------------------------------------------------
test_that("gbm regression: D is not the all-zero degenerate matrix", {
  skip_if_not_installed("gbm")
  tr <- prep_mtcars()
  # Loosen gbm defaults so the small sample is acceptable.
  ens <- gbm::gbm(mpg ~ ., data = tr, distribution = "gaussian",
                  n.trees = 50, interaction.depth = 3, verbose = FALSE,
                  bag.fraction = 0.9, n.minobsinnode = 2)
  D <- createDisMatrix(ens, data = tr, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "gbm")
})

# ----------------------------------------------------------------------------
# lightgbm
# ----------------------------------------------------------------------------
test_that("lgb.Booster regression: D invariants hold", {
  skip_if_not_installed("lightgbm")
  tr <- prep_mtcars()
  X <- as.matrix(tr[, setdiff(names(tr), "mpg")])
  y <- tr$mpg
  dtrain <- lightgbm::lgb.Dataset(data = X, label = y)
  # Default min_data_in_leaf=20 collapses tiny samples to a single leaf,
  # which would (correctly) be flagged as degenerate by the validator.
  ens <- lightgbm::lgb.train(
    params = list(objective = "regression", verbose = -1,
                  min_data_in_leaf = 2L, min_data_in_bin = 1L),
    data = dtrain, nrounds = 30
  )
  D <- createDisMatrix(ens, data = tr, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "lgb.Booster")
})

# ----------------------------------------------------------------------------
# Adapter contract (no backend dependency required)
# ----------------------------------------------------------------------------
test_that("validate_terminal_nodes rejects degenerate output", {
  bad <- data.frame(Tree1 = rep(0L, 10), Tree2 = rep(0L, 10))
  expect_error(
    e2tree:::validate_terminal_nodes(bad, data.frame(x = 1:10), backend = "fake"),
    "degenerate"
  )
})

test_that("validate_terminal_nodes rejects wrong row count", {
  bad <- data.frame(Tree1 = 1:5)
  expect_error(
    e2tree:::validate_terminal_nodes(bad, data.frame(x = 1:10), backend = "fake"),
    "rows"
  )
})

test_that("validate_terminal_nodes rejects non-numeric columns", {
  bad <- data.frame(Tree1 = letters[1:10], stringsAsFactors = FALSE)
  expect_error(
    e2tree:::validate_terminal_nodes(bad, data.frame(x = 1:10), backend = "fake"),
    "non-numeric"
  )
})

test_that("default adapter methods raise an informative error", {
  bogus <- structure(list(), class = "not_a_real_ensemble")
  expect_error(get_ensemble_type(bogus), "Supported classes")
  expect_error(extract_terminal_nodes(bogus, data.frame(x = 1:3)), "Supported classes")
  expect_error(get_ensemble_predictions(bogus, data.frame(x = 1:3), "regression"),
               "Supported classes")
})

# ----------------------------------------------------------------------------
# End-to-end pipeline: D → e2tree → as.rpart for boosting backends.
# Catches regressions in the full surrogate-tree workflow when the underlying
# ensemble is not random-forest-like.
# ----------------------------------------------------------------------------
test_that("xgboost full pipeline (D -> e2tree -> as.rpart) works", {
  skip_if_not_installed("xgboost")
  tr <- prep_mtcars()
  X <- as.matrix(tr[, setdiff(names(tr), "mpg")])
  y <- tr$mpg
  dtrain <- xgboost::xgb.DMatrix(data = X, label = y)
  ens <- xgboost::xgb.train(
    params = list(objective = "reg:squarederror"),
    data = dtrain, nrounds = 20, verbose = 0
  )
  D <- createDisMatrix(ens, data = tr, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)
  fit <- e2tree(mpg ~ ., tr, D, ens, setting)
  expect_s3_class(fit, "e2tree")
  expect_equal(attr(fit, "ensemble_backend"), "xgb.Booster")
  rp <- as.rpart(fit, ens)
  expect_true(all(c("frame", "where") %in% names(rp)))
})

test_that("xgboost binary classification: D invariants hold", {
  skip_if_not_installed("xgboost")
  set.seed(7)
  data(iris)
  df <- iris[iris$Species != "virginica", ]
  df$Species <- factor(df$Species)
  idx <- sample(seq_len(nrow(df)), floor(0.6 * nrow(df)))
  tr <- df[idx, ]
  X <- as.matrix(tr[, 1:4])
  y <- as.integer(tr$Species) - 1L
  dtrain <- xgboost::xgb.DMatrix(data = X, label = y)
  ens <- xgboost::xgb.train(
    params = list(objective = "binary:logistic"),
    data = dtrain, nrounds = 20, verbose = 0
  )
  D <- createDisMatrix(ens, data = tr, label = "Species",
                       parallel = list(active = FALSE, no_cores = 1))
  assert_dis_invariants(D, nrow(tr), "xgb.Booster")
})

test_that("createDisMatrix rejects unsupported ensembles before adapter call", {
  expect_error(
    createDisMatrix(structure(list(), class = "fake_model"),
                    data = mtcars, label = "mpg"),
    "unsupported ensemble class"
  )
})

test_that("createDisMatrix rejects non-data.frame data", {
  skip_if_not_installed("randomForest")
  ens <- randomForest::randomForest(mpg ~ ., data = mtcars, ntree = 10)
  expect_error(
    createDisMatrix(ens, data = as.matrix(mtcars), label = "mpg"),
    "must be a valid data frame"
  )
})

test_that("createDisMatrix rejects label not in data", {
  skip_if_not_installed("randomForest")
  ens <- randomForest::randomForest(mpg ~ ., data = mtcars, ntree = 10)
  expect_error(
    createDisMatrix(ens, data = mtcars, label = "not_a_column"),
    "valid column name"
  )
})

test_that("regression createDisMatrix requires non-NULL label", {
  skip_if_not_installed("randomForest")
  ens <- randomForest::randomForest(mpg ~ ., data = mtcars, ntree = 10)
  expect_error(
    createDisMatrix(ens, data = mtcars, label = NULL),
    "label.*required"
  )
})

test_that("D from chunked path is bit-identical to non-chunked path", {
  skip_if_not_installed("randomForest")
  set.seed(42)
  ens <- randomForest::randomForest(mpg ~ ., data = mtcars, ntree = 30)
  D_full <- createDisMatrix(ens, data = mtcars, label = "mpg",
                            parallel = list(active = FALSE, no_cores = 1))
  D_chunk <- createDisMatrix(ens, data = mtcars, label = "mpg",
                             parallel = list(active = FALSE, no_cores = 1),
                             chunk_size = 8)
  expect_equal(unname(D_full), unname(D_chunk))
})

test_that("ensemble_backend() returns NA for unknown classes", {
  expect_true(is.na(e2tree:::ensemble_backend(list())))
  expect_true(is.na(e2tree:::ensemble_backend(NULL)))
  expect_equal(e2tree:::ensemble_backend(
    structure(list(), class = "randomForest")), "randomForest")
})

test_that("C++ entry points reject non-numeric tree columns", {
  bad <- data.frame(OBS = as.character(1:4),
                    Tree1 = c("a","b","a","b"),
                    resp = c(1,2,3,4),
                    stringsAsFactors = FALSE)
  expect_error(
    e2tree:::compute_all_cooccurrences_cpp("regression", bad, 1L, 1L, 1.0),
    "must be integer or double"
  )
})

test_that("e2tree warns when D and ensemble come from different backends", {
  skip_if_not_installed("randomForest")
  skip_if_not_installed("ranger")
  tr <- prep_iris()
  rf <- randomForest::randomForest(Species ~ ., data = tr, ntree = 50)
  rg <- ranger::ranger(Species ~ ., data = tr, num.trees = 50,
                       write.forest = TRUE)
  D_rf <- createDisMatrix(rf, data = tr, label = "Species",
                          parallel = list(active = FALSE, no_cores = 1))
  setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)
  expect_warning(
    e2tree(Species ~ ., tr, D_rf, rg, setting),
    "different backends|backend|consistent"
  )
})
