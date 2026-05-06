library(testthat)
library(e2tree)

# ===========================================================================
# Helpers
# ===========================================================================

.make_rf_clf <- function() {
  skip_if_not_installed("randomForest")
  library(randomForest)
  set.seed(1)
  randomForest(Species ~ ., data = iris, ntree = 50)
}

.make_rf_reg <- function() {
  skip_if_not_installed("randomForest")
  library(randomForest)
  set.seed(1)
  randomForest(mpg ~ ., data = mtcars, ntree = 50)
}

.make_ranger_clf <- function() {
  skip_if_not_installed("ranger")
  library(ranger)
  set.seed(1)
  ranger(Species ~ ., data = iris, num.trees = 50, write.forest = TRUE)
}

.make_ranger_reg <- function() {
  skip_if_not_installed("ranger")
  library(ranger)
  set.seed(1)
  ranger(mpg ~ ., data = mtcars, num.trees = 50, write.forest = TRUE)
}

.make_xgb_clf <- function() {
  skip_if_not_installed("xgboost")
  library(xgboost)
  set.seed(1)
  X  <- as.matrix(iris[, 1:4])
  y  <- as.integer(iris$Species) - 1L
  dm <- xgb.DMatrix(data = X, label = y)
  xgb.train(
    params = list(objective = "multi:softmax", num_class = 3,
                  max_depth = 3, eta = 0.3),
    data = dm, nrounds = 20, verbose = 0
  )
}

.make_xgb_reg <- function() {
  skip_if_not_installed("xgboost")
  library(xgboost)
  set.seed(1)
  X  <- as.matrix(mtcars[, -1])
  y  <- mtcars$mpg
  dm <- xgb.DMatrix(data = X, label = y)
  xgb.train(
    params = list(objective = "reg:squarederror", max_depth = 3, eta = 0.3),
    data = dm, nrounds = 20, verbose = 0
  )
}

.make_gbm_clf <- function() {
  skip_if_not_installed("gbm")
  library(gbm)
  set.seed(1)
  df <- iris
  df$Species <- as.integer(df$Species == "setosa")
  gbm(Species ~ ., data = df, distribution = "bernoulli",
      n.trees = 50, interaction.depth = 3, verbose = FALSE)
}

.make_gbm_reg <- function() {
  skip_if_not_installed("gbm")
  library(gbm)
  set.seed(1)
  # mtcars has only 32 rows; the gbm defaults (bag.fraction=0.5,
  # n.minobsinnode=10) violate `nTrain*bag.fraction > 2*n.minobsinnode+1`,
  # so we relax both for the test fixture.
  gbm(mpg ~ ., data = mtcars, distribution = "gaussian",
      n.trees = 50, interaction.depth = 3, verbose = FALSE,
      bag.fraction = 0.9, n.minobsinnode = 2)
}

.make_lgb_clf <- function() {
  skip_if_not_installed("lightgbm")
  library(lightgbm)
  set.seed(1)
  X  <- as.matrix(iris[, 1:4])
  y  <- as.integer(iris$Species) - 1L
  ds <- lgb.Dataset(X, label = y)
  lgb.train(
    params = list(objective = "multiclass", num_class = 3,
                  num_leaves = 8, verbose = -1),
    data = ds, nrounds = 20
  )
}

.make_lgb_reg <- function() {
  skip_if_not_installed("lightgbm")
  library(lightgbm)
  set.seed(1)
  X  <- as.matrix(mtcars[, -1])
  y  <- mtcars$mpg
  ds <- lgb.Dataset(X, label = y)
  lgb.train(
    params = list(objective = "regression", num_leaves = 8, verbose = -1),
    data = ds, nrounds = 20
  )
}


# ===========================================================================
# Tests: get_ensemble_type()
# ===========================================================================

test_that("get_ensemble_type — randomForest", {
  m <- .make_rf_clf()
  expect_equal(get_ensemble_type(m), "classification")

  m2 <- .make_rf_reg()
  expect_equal(get_ensemble_type(m2), "regression")
})

test_that("get_ensemble_type — ranger", {
  m <- .make_ranger_clf()
  expect_equal(get_ensemble_type(m), "classification")

  m2 <- .make_ranger_reg()
  expect_equal(get_ensemble_type(m2), "regression")
})

test_that("get_ensemble_type — xgboost", {
  m <- .make_xgb_clf()
  expect_equal(get_ensemble_type(m), "classification")

  m2 <- .make_xgb_reg()
  expect_equal(get_ensemble_type(m2), "regression")
})

test_that("get_ensemble_type — gbm", {
  m <- .make_gbm_clf()
  expect_equal(get_ensemble_type(m), "classification")

  m2 <- .make_gbm_reg()
  expect_equal(get_ensemble_type(m2), "regression")
})

test_that("get_ensemble_type — lightgbm", {
  m <- .make_lgb_clf()
  expect_equal(get_ensemble_type(m), "classification")

  m2 <- .make_lgb_reg()
  expect_equal(get_ensemble_type(m2), "regression")
})

test_that("get_ensemble_type — unsupported class throws error", {
  expect_error(get_ensemble_type(list()), "unsupported ensemble class")
})


# ===========================================================================
# Tests: extract_terminal_nodes()
# ===========================================================================

test_that("extract_terminal_nodes — randomForest returns correct dimensions", {
  m   <- .make_rf_clf()
  res <- extract_terminal_nodes(m, iris)
  expect_true(is.data.frame(res))
  expect_equal(nrow(res), nrow(iris))
  expect_equal(ncol(res), m$ntree)
  expect_true(all(sapply(res, is.numeric)))
})

test_that("extract_terminal_nodes — ranger returns correct dimensions", {
  m   <- .make_ranger_clf()
  res <- extract_terminal_nodes(m, iris)
  expect_true(is.data.frame(res))
  expect_equal(nrow(res), nrow(iris))
  expect_equal(ncol(res), m$num.trees)
})

test_that("extract_terminal_nodes — xgboost returns correct dimensions", {
  m   <- .make_xgb_clf()
  res <- extract_terminal_nodes(m, iris[, 1:4])
  expect_true(is.data.frame(res))
  expect_equal(nrow(res), nrow(iris))
  # number of columns = n_rounds * n_class for multi:softmax with predleaf
  expect_gt(ncol(res), 0)
})

test_that("extract_terminal_nodes — gbm returns correct dimensions", {
  m   <- .make_gbm_reg()
  res <- extract_terminal_nodes(m, mtcars)
  expect_true(is.data.frame(res))
  expect_equal(nrow(res), nrow(mtcars))
  expect_equal(ncol(res), m$n.trees)
  expect_true(all(unlist(res) > 0))  # 1-indexed leaf IDs
})

test_that("extract_terminal_nodes — lightgbm returns correct dimensions", {
  m   <- .make_lgb_reg()
  res <- extract_terminal_nodes(m, as.data.frame(as.matrix(mtcars[, -1])))
  expect_true(is.data.frame(res))
  expect_equal(nrow(res), nrow(mtcars))
  expect_gt(ncol(res), 0)
})


# ===========================================================================
# Tests: get_ensemble_predictions()
# ===========================================================================

test_that("get_ensemble_predictions — randomForest returns numeric vector", {
  m   <- .make_rf_reg()
  preds <- get_ensemble_predictions(m, mtcars, "regression")
  expect_true(is.numeric(preds))
  expect_equal(length(preds), nrow(mtcars))
})

test_that("get_ensemble_predictions — ranger returns numeric vector", {
  m     <- .make_ranger_reg()
  preds <- get_ensemble_predictions(m, mtcars, "regression")
  expect_true(is.numeric(preds))
  expect_equal(length(preds), nrow(mtcars))
})

test_that("get_ensemble_predictions — xgboost returns numeric vector", {
  m     <- .make_xgb_reg()
  preds <- get_ensemble_predictions(m, mtcars[, -1], "regression")
  expect_true(is.numeric(preds))
  expect_equal(length(preds), nrow(mtcars))
})

test_that("get_ensemble_predictions — gbm returns numeric vector", {
  m     <- .make_gbm_reg()
  preds <- get_ensemble_predictions(m, mtcars, "regression")
  expect_true(is.numeric(preds))
  expect_equal(length(preds), nrow(mtcars))
})

test_that("get_ensemble_predictions — lightgbm returns numeric vector", {
  m     <- .make_lgb_reg()
  preds <- get_ensemble_predictions(m, as.data.frame(as.matrix(mtcars[, -1])), "regression")
  expect_true(is.numeric(preds))
  expect_equal(length(preds), nrow(mtcars))
})


# ===========================================================================
# Tests: createDisMatrix() backward-compatibility (RF and ranger unchanged)
# ===========================================================================

test_that("createDisMatrix — randomForest classification still works", {
  skip_if_not_installed("randomForest")
  library(randomForest)
  set.seed(42)
  n  <- floor(0.75 * nrow(iris))
  tr <- iris[sample(nrow(iris), n), ]
  m  <- randomForest(Species ~ ., data = tr, ntree = 50)
  D  <- createDisMatrix(m, data = tr, label = "Species",
                        parallel = list(active = FALSE, no_cores = 1))
  expect_true(is.matrix(D))
  expect_equal(dim(D), c(nrow(tr), nrow(tr)))
  expect_true(all(diag(D) == 0))
})

test_that("createDisMatrix — randomForest regression still works", {
  skip_if_not_installed("randomForest")
  library(randomForest)
  set.seed(42)
  n  <- floor(0.75 * nrow(mtcars))
  tr <- mtcars[sample(nrow(mtcars), n), ]
  m  <- randomForest(mpg ~ ., data = tr, ntree = 50)
  D  <- createDisMatrix(m, data = tr, label = "mpg",
                        parallel = list(active = FALSE, no_cores = 1))
  expect_true(is.matrix(D))
  expect_equal(dim(D), c(nrow(tr), nrow(tr)))
})
