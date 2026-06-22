# Tests for the quantile binning of numeric split thresholds (ordSplit /
# setting$max_thresholds) and for the oob argument of
# get_ensemble_predictions().

skip_if_not_installed("randomForest")

test_that("ordSplit caps candidate thresholds via quantile binning", {
  set.seed(1)
  x <- rnorm(1000)                       # ~1000 unique values
  S_full <- ordSplit(x, max_thresholds = Inf)
  expect_equal(ncol(S_full), length(unique(x)) - 1L)

  S_cap <- ordSplit(x, max_thresholds = 64L)
  expect_lte(ncol(S_cap), 64L)
  # thresholds are observed values and every split is non-trivial
  cs <- colSums(S_cap)
  expect_true(all(cs >= 1 & cs <= length(x) - 1))

  # discrete variables below the cap are untouched
  xd <- sample(1:10, 500, replace = TRUE)
  expect_equal(ncol(ordSplit(xd, max_thresholds = 64L)), 9L)
})

test_that("e2tree with capped thresholds still recovers structure", {
  set.seed(42)
  n  <- 150
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- 5 * (df$x1 > 0) + rnorm(n, sd = 0.3)
  ens <- randomForest::randomForest(y ~ ., data = df, ntree = 100)
  D   <- createDisMatrix(ens, data = df, label = "y",
                         parallel = list(active = FALSE, no_cores = 1))
  tr  <- e2tree(y ~ ., df, D, ens,
                setting = list(impTotal = 0.1, maxDec = 1e-6, n = 5, level = 4,
                               max_thresholds = 32))
  expect_true("x1" %in% stats::na.omit(tr$tree$variable))
})

test_that("get_ensemble_predictions distinguishes full-ensemble from OOB", {
  set.seed(42)
  n  <- 200
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.2)

  ens <- randomForest::randomForest(y ~ ., data = df, ntree = 100)
  p_full <- get_ensemble_predictions(ens, df, type = "regression", oob = FALSE)
  p_oob  <- get_ensemble_predictions(ens, df, type = "regression", oob = TRUE)
  expect_length(p_full, n)
  expect_equal(p_oob, as.numeric(ens$predicted))
  expect_false(isTRUE(all.equal(p_full, p_oob)))   # different quantities
  # full-ensemble in-sample predictions fit the data more closely than OOB
  expect_lt(mean((df$y - p_full)^2), mean((df$y - p_oob)^2))

  # OOB on data of a different size is a misalignment -> error
  expect_error(
    get_ensemble_predictions(ens, df[1:10, ], type = "regression", oob = TRUE),
    "misaligning|must be the training set"
  )

  skip_if_not_installed("ranger")
  rg <- ranger::ranger(y ~ ., data = df, num.trees = 100)
  r_full <- get_ensemble_predictions(rg, df, type = "regression", oob = FALSE)
  r_oob  <- get_ensemble_predictions(rg, df, type = "regression", oob = TRUE)
  expect_equal(r_oob, as.numeric(rg$predictions))
  expect_false(isTRUE(all.equal(r_full, r_oob)))
})
