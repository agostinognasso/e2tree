library(testthat)
library(e2tree)
library(randomForest)

test_that("eValidation works correctly with valid inputs (classification case)", {
  set.seed(42)

  # Prepare data
  data(iris)
  train_idx <- sample(seq_len(nrow(iris)), size = 0.75 * nrow(iris))
  training <- iris[train_idx, ]

  # Train Random Forest
  ensemble <- randomForest(Species ~ ., data = training, importance = TRUE, proximity = TRUE)

  # Create dissimilarity matrix
  D <- createDisMatrix(ensemble, data = training, label = "Species", parallel = list(active=FALSE, no_cores = 1))

  # Define settings
  setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)

  # Generate e2tree model
  fit <- e2tree(Species ~ ., training, D, ensemble, setting)

  # Run eValidation
  comparison <- eValidation(training, fit, D, graph=FALSE)

  # Tests
  expect_type(comparison, "list")  # Should return a list
  expect_true("mantel_test" %in% names(comparison))
})

test_that("eValidation handles incorrect input types (classification case)", {
  set.seed(42)

  data(iris)
  train_idx <- sample(seq_len(nrow(iris)), size = 0.75 * nrow(iris))
  training <- iris[train_idx, ]

  ensemble <- randomForest(Species ~ ., data = training, importance = TRUE, proximity = TRUE)
  D <- createDisMatrix(ensemble, data = training, label = "Species", parallel = list(active=FALSE, no_cores = 1))
  setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)
  fit <- e2tree(Species ~ ., training, D, ensemble, setting)

  # Test incorrect inputs
  expect_error(eValidation(NULL, fit, D, graph=FALSE),
               "Error: 'data' must be a non-empty data frame.")

  expect_error(eValidation(training, NULL, D, graph=FALSE),
               "Error: 'fit' must be an 'e2tree' object.")

  expect_error(eValidation(training, fit, NULL, graph=FALSE),
               "Error: 'D' must be a square dissimilarity matrix.")

  expect_error(eValidation(training, fit, matrix(1, 5, 5), graph=FALSE),
               "Error: The number of rows in 'data' must match the dimensions of 'D'.")
})


test_that("eValidation works correctly with valid inputs (regression case)", {
  set.seed(42)

  # Prepare data
  data(mtcars)
  train_idx <- sample(seq_len(nrow(mtcars)), size = 0.75 * nrow(mtcars))
  training <- mtcars[train_idx, ]

  # Train Random Forest
  ensemble <- randomForest(mpg ~ ., data = training, importance = TRUE, proximity = TRUE)

  # Create dissimilarity matrix
  D <- createDisMatrix(ensemble, data = training, label = "mpg", parallel = list(active=FALSE, no_cores = 1))

  # Define settings
  setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)

  # Generate e2tree model
  fit <- e2tree(mpg ~ ., training, D, ensemble, setting)

  # Run eValidation
  comparison <- eValidation(training, fit, D, graph=FALSE)

  # Tests
  expect_type(comparison, "list")  # Should return a list
  expect_true("mantel_test" %in% names(comparison))
})

test_that("eValidation handles incorrect input types (regression case)", {
  set.seed(42)

  data(mtcars)
  train_idx <- sample(seq_len(nrow(mtcars)), size = 0.75 * nrow(mtcars))
  training <- mtcars[train_idx, ]

  ensemble <- randomForest(mpg ~ ., data = training, importance = TRUE, proximity = TRUE)
  D <- createDisMatrix(ensemble, data = training, label = "mpg", parallel = list(active=FALSE, no_cores = 1))
  setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)
  fit <- e2tree(mpg ~ ., training, D, ensemble, setting)

  # Test incorrect inputs
  expect_error(eValidation(NULL, fit, D, graph=FALSE),
               "Error: 'data' must be a non-empty data frame.")

  expect_error(eValidation(training, NULL, D, graph=FALSE),
               "Error: 'fit' must be an 'e2tree' object.")

  expect_error(eValidation(training, fit, NULL, graph=FALSE),
               "Error: 'D' must be a square dissimilarity matrix.")

  expect_error(eValidation(training, fit, matrix(1, 5, 5), graph=FALSE),
               "Error: The number of rows in 'data' must match the dimensions of 'D'.")
})

test_that("eValidation: transform, kappa, bootstrap and equivalence", {
  set.seed(42)
  data(iris)
  training <- iris[sample(seq_len(nrow(iris)), 105), ]
  ensemble <- randomForest(Species ~ ., data = training, proximity = TRUE)
  D <- createDisMatrix(ensemble, data = training, label = "Species",
                       parallel = list(active = FALSE, no_cores = 1))
  fit <- e2tree(Species ~ ., training, D, ensemble,
                list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5))

  v <- eValidation(training, fit, D, graph = FALSE, test = "measures",
                   n_perm = 49, n_boot = 49, delta = 0.5, seed = 1,
                   transform = "none")
  expect_equal(v$transform, "none")
  expect_equal(eValidation(training, fit, D, graph = FALSE, test = "measures",
                           n_perm = 0)$transform, "sqrt")
  expect_true(all(c("kappa", "ci_lower", "ci_upper", "p_value") %in% names(v$measures)))
  expect_equal(v$measures$observed[v$measures$method == "sqrt_nLoI"],
               sqrt(v$measures$observed[v$measures$method == "nLoI"]))
  expect_true(v$equivalence$fidelity)
  expect_output(print(v), "kappa")

  vs <- eValidation(training, fit, D, graph = FALSE, test = "measures",
                    n_perm = 0, transform = "sqrt")
  expect_equal(vs$Proximity_matrix_ensemble, sqrt(v$Proximity_matrix_ensemble))
  expect_warning(eValidation(training, fit, D, graph = FALSE, test = "measures",
                             n_perm = 0, delta = 0.1), "delta is ignored")
})
