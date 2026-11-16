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
                           n_perm = 0)$transform, "none")
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

test_that("eValidation: default scale is the proximity scale, leaves follow the rows", {
  set.seed(42)
  data(iris)
  training <- iris[sample(seq_len(nrow(iris)), 105), ]
  ensemble <- randomForest(Species ~ ., data = training, proximity = TRUE)
  D <- createDisMatrix(ensemble, data = training, label = "Species",
                       parallel = list(active = FALSE, no_cores = 1))
  fit <- e2tree(Species ~ ., training, D, ensemble,
                list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5))

  v <- eValidation(training, fit, D, graph = FALSE, test = "measures", n_perm = 0)
  ord <- as.integer(rownames(v$Proximity_matrix_ensemble))

  # default: the proximities themselves, 1 - D, in hclust order
  expect_equal(unname(v$Proximity_matrix_ensemble), unname(1 - D[ord, ord]))

  # leaves: one per row, named by the original index, consistent with fit
  expect_length(v$leaves, nrow(training))
  expect_equal(names(v$leaves), rownames(v$Proximity_matrix_e2tree))
  td <- fit$tree[fit$tree$terminal, ]
  leaf_data <- integer(nrow(training))
  for (k in seq_len(nrow(td))) leaf_data[unlist(td$obs[k])] <- td$node[k]
  expect_equal(unname(v$leaves), leaf_data[ord])

  # the decomposition stored in the object is loi(leaves = v$leaves)
  l <- loi(v$Proximity_matrix_ensemble, v$Proximity_matrix_e2tree,
           leaves = v$leaves)
  expect_equal(l$loi_in, v$loi$loi_in)
  expect_equal(l$loi_out, v$loi$loi_out)

  # the print labels the two Mantel statistics apart
  vb <- eValidation(training, fit, D, graph = FALSE, n_perm = 19, seed = 1)
  expect_output(print(vb), "Mantel test \\(ape")
  expect_output(print(vb), "Mantel row: Pearson correlation")
})

test_that("eValidation (regression): stored decomposition uses the real leaves", {
  set.seed(42)
  data(mtcars)
  ensemble <- randomForest(mpg ~ ., data = mtcars, proximity = TRUE)
  D <- createDisMatrix(ensemble, data = mtcars, label = "mpg",
                       parallel = list(active = FALSE, no_cores = 1))
  fit <- e2tree(mpg ~ ., mtcars, D, ensemble,
                list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5))
  v <- eValidation(mtcars, fit, D, graph = FALSE, test = "measures", n_perm = 0)

  # leaf weights come from Wt, within [0, 1]
  expect_true(all(v$Proximity_matrix_e2tree >= 0 & v$Proximity_matrix_e2tree <= 1))
  l <- loi(v$Proximity_matrix_ensemble, v$Proximity_matrix_e2tree,
           leaves = v$leaves)
  expect_equal(l$n_within, v$loi$n_within)
  expect_equal(v$loi$n_within,
               sum(outer(v$leaves, v$leaves, "==")[lower.tri(diag(nrow(mtcars)))]))
})
