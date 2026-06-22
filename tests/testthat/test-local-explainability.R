# Tests for the local-explainability layer: localLoI(), eContribution(),
# eNeighbors(). These build a small E2Tree from a randomForest ensemble.

skip_if_not_installed("randomForest")

build_iris_tree <- function() {
  set.seed(42)
  data(iris)
  ensemble <- randomForest::randomForest(Species ~ ., data = iris,
    importance = TRUE, proximity = TRUE)
  D <- createDisMatrix(ensemble, data = iris, label = "Species",
    parallel = list(active = FALSE, no_cores = 1))
  setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)
  tree <- e2tree(Species ~ ., iris, D, ensemble, setting)
  list(ensemble = ensemble, D = D, tree = tree, data = iris)
}

# ---------------------------------------------------------------------------
# localLoI()
# ---------------------------------------------------------------------------

test_that("localLoI per-observation mean equals the global nLoI", {
  obj  <- build_iris_tree()
  vs   <- eValidation(obj$data, obj$tree, obj$D)
  prox <- proximity(vs)

  g  <- loi(prox$ensemble, prox$e2tree)
  ll <- localLoI(prox$ensemble, prox$e2tree, fit = obj$tree)

  expect_s3_class(ll, "localLoI")
  expect_equal(mean(ll$obs$loi), g$nloi)
  expect_equal(ll$nloi, g$nloi)
})

test_that("localLoI recovers the leaf partition from O_hat alone", {
  # synthetic block-structured proximity
  blocks <- c(rep(1, 5), rep(2, 4), rep(3, 3))
  O_hat  <- outer(blocks, blocks, function(a, b) as.numeric(a == b))
  diag(O_hat) <- 1
  set.seed(1); M <- matrix(runif(144), 12, 12)
  O <- (M + t(M)) / 2; diag(O) <- 1

  ll <- localLoI(O, O_hat)            # no fit -> block detection
  # detected components must match the true blocks up to relabelling
  expect_equal(length(unique(ll$obs$node)), 3L)
  expect_true(all(tapply(ll$obs$node, blocks, function(z) length(unique(z))) == 1))
})

test_that("localLoI node table is consistent", {
  obj  <- build_iris_tree()
  vs   <- eValidation(obj$data, obj$tree, obj$D)
  prox <- proximity(vs)
  ll   <- localLoI(prox$ensemble, prox$e2tree, fit = obj$tree)

  expect_equal(sum(ll$node$n), nrow(ll$obs))
  expect_setequal(ll$node$node, nodes(obj$tree, terminal = TRUE)$node)
  expect_true(all(ll$node$loi_in >= 0))
  expect_true(all(ll$node$loi_out >= 0))
})

# ---------------------------------------------------------------------------
# eContribution()
# ---------------------------------------------------------------------------

test_that("eContribution is exactly additive (classification)", {
  obj <- build_iris_tree()
  ec  <- eContribution(obj$tree, newdata = obj$data[c(1, 60, 120), ])
  expect_s3_class(ec, "e2contribution")
  recon <- as.numeric(ec$baseline + rowSums(ec$contributions))
  expect_equal(recon, ec$prediction, tolerance = 1e-8)
})

test_that("eContribution predicted class matches predict.e2tree", {
  obj <- build_iris_tree()
  idx <- c(1, 60, 120)
  ec  <- eContribution(obj$tree, newdata = obj$data[idx, ])
  pp  <- predict(obj$tree, newdata = obj$data[idx, ])
  expect_equal(as.character(ec$predicted), as.character(pp$fit))
})

test_that("eContribution gives distinct attributions for distinct leaves", {
  obj <- build_iris_tree()
  ec  <- eContribution(obj$tree, newdata = obj$data[c(1, 150), ])
  expect_false(isTRUE(all.equal(ec$contributions[1, ], ec$contributions[2, ])))
})

test_that("plot.e2contribution: single bars and all multi-obs views", {
  skip_if_not_installed("ggplot2")
  obj <- build_iris_tree()
  ec  <- eContribution(obj$tree, newdata = obj$data[c(1, 60, 120), ])
  expect_s3_class(plot(ec, obs = 2), "ggplot")          # waterfall (single-obs)
  for (ty in c("heatmap", "summary", "importance"))
    expect_s3_class(plot(ec, obs = 1:3, type = ty), "ggplot")
  expect_error(plot(ec, obs = c(1, 99)), "between 1 and")
})

test_that("eContribution is additive for regression", {
  skip_if_not_installed("randomForest")
  set.seed(7); data(mtcars)
  ens <- randomForest::randomForest(mpg ~ ., data = mtcars, ntree = 300,
    proximity = TRUE)
  D <- createDisMatrix(ens, data = mtcars, label = "mpg",
    parallel = list(active = FALSE, no_cores = 1))
  tree <- e2tree(mpg ~ ., mtcars, D, ens,
    list(impTotal = 0.1, maxDec = 1e-6, n = 2, level = 5))
  ec <- eContribution(tree, newdata = mtcars[1:5, ])
  recon <- as.numeric(ec$baseline + rowSums(ec$contributions))
  expect_equal(recon, ec$prediction, tolerance = 1e-8)
})

# ---------------------------------------------------------------------------
# eNeighbors()
# ---------------------------------------------------------------------------

test_that("eNeighbors returns k neighbours in the query's own leaf set", {
  obj <- build_iris_tree()
  nb  <- eNeighbors(obj$tree, obj$ensemble, query = obj$data[c(1, 80, 130), ],
                    k = 5)
  expect_s3_class(nb, "e2neighbors")
  expect_length(nb$neighbors, 3L)
  expect_equal(nrow(nb$neighbors[[1]]), 5L)
  # neighbour proximities are sorted descending and within [0, 1]
  pr <- nb$neighbors[[2]]$proximity
  expect_true(all(pr >= 0 & pr <= 1))
  expect_false(is.unsorted(rev(pr)))
})

test_that("eNeighbors leaf assignment matches predict routing", {
  obj <- build_iris_tree()
  q   <- obj$data[c(5, 75, 145), ]
  nb  <- eNeighbors(obj$tree, obj$ensemble, query = q, k = 3)
  # query_pred must equal the e2tree prediction for the same rows
  pp  <- predict(obj$tree, newdata = q)
  expect_equal(as.character(nb$query_pred), as.character(pp$fit))
})

# ---------------------------------------------------------------------------
# eHeterogeneity()
# ---------------------------------------------------------------------------

test_that("eHeterogeneity (classification) describes each region", {
  obj <- build_iris_tree()
  h <- eHeterogeneity(obj$tree, alpha = 0.1)
  expect_s3_class(h, "e2heterogeneity")
  expect_setequal(h$node$node, nodes(obj$tree, terminal = TRUE)$node)
  expect_true(all(h$node$entropy >= 0 & h$node$entropy <= 1))
  expect_true(all(h$node$set_size >= 1L))
  expect_true(all(h$node$purity > 0 & h$node$purity <= 1))
  # per-region class proportions sum to one and feed the composition view
  expect_equal(nrow(h$proportions), nrow(h$node))
  expect_true(all(abs(rowSums(h$proportions) - 1) < 1e-8))
})

test_that("plot.e2heterogeneity: entropy and composition views render", {
  skip_if_not_installed("ggplot2")
  obj <- build_iris_tree()
  h <- eHeterogeneity(obj$tree, alpha = 0.1)
  expect_s3_class(plot(h), "ggplot")
  expect_s3_class(plot(h, type = "composition"), "ggplot")
  expect_error(plot(h, type = "nope"), "should be one of")
})

test_that("eHeterogeneity (regression) reports a valid central band", {
  skip_if_not_installed("randomForest")
  set.seed(7); data(mtcars)
  ens <- randomForest::randomForest(mpg ~ ., data = mtcars, ntree = 300,
    proximity = TRUE)
  D <- createDisMatrix(ens, data = mtcars, label = "mpg",
    parallel = list(active = FALSE, no_cores = 1))
  tree <- e2tree(mpg ~ ., mtcars, D, ens,
    list(impTotal = 0.1, maxDec = 1e-6, n = 2, level = 5))
  h <- eHeterogeneity(tree, alpha = 0.1)
  expect_s3_class(h, "e2heterogeneity")
  expect_true(all(h$node$lower <= h$node$upper))
  expect_true(all(h$node$width >= 0))
  expect_true(all(h$node$sd >= 0))
})

test_that("eHeterogeneity dominant class matches the leaf majority", {
  obj <- build_iris_tree()
  h <- eHeterogeneity(obj$tree)
  term <- nodes(obj$tree, terminal = TRUE)
  # the e2tree's own leaf prediction is the majority class -> must match dominant
  m <- match(h$node$node, term$node)
  expect_equal(as.character(h$node$dominant), as.character(term$pred[m]))
})

# ---------------------------------------------------------------------------
# nodeStats()
# ---------------------------------------------------------------------------

test_that("nodeStats works on the root (internal) node", {
  obj <- build_iris_tree()
  ns <- nodeStats(obj$tree, node = 1)
  expect_s3_class(ns, "e2nodeStats")
  expect_false(ns$meta$terminal)              # root is internal
  expect_equal(ns$meta$n, nrow(obj$data))     # root holds all observations
  expect_equal(nrow(ns$numeric), 4L)          # iris has 4 numeric predictors
  expect_true(all(c("cohen_d", "mean_node", "mean_global") %in% names(ns$numeric)))
  expect_equal(sum(ns$response$proportions), 1, tolerance = 1e-8)
})

test_that("nodeStats works on a terminal node and matches its size", {
  obj <- build_iris_tree()
  term <- nodes(obj$tree, terminal = TRUE)
  k <- term$node[which.max(term$n)]
  ns <- nodeStats(obj$tree, node = k)
  expect_true(ns$meta$terminal)
  expect_equal(ns$meta$n, term$n[term$node == k])
  expect_equal(length(ns$obs), ns$meta$n)
})

test_that("nodeStats handles an all-categorical dataset (credit)", {
  skip_if_not_installed("randomForest")
  data(credit); credit <- as.data.frame(credit)
  credit$Type_of_client <- factor(credit$Type_of_client)
  set.seed(5)
  ens <- randomForest::randomForest(Type_of_client ~ ., data = credit,
    proximity = TRUE)
  D <- createDisMatrix(ens, data = credit, label = "Type_of_client",
    parallel = list(active = FALSE, no_cores = 1))
  tree <- e2tree(Type_of_client ~ ., credit, D, ens,
    list(impTotal = 0.1, maxDec = 0.01, n = 5, level = 6))
  ns <- nodeStats(tree, node = 1)
  expect_null(ns$numeric)                       # no numeric predictors
  expect_true(nrow(ns$categorical) >= 1)
  expect_true(all(ns$categorical$cramers_v >= 0 & ns$categorical$cramers_v <= 1))
})

test_that("nodeStats rejects a non-existent node", {
  obj <- build_iris_tree()
  expect_error(nodeStats(obj$tree, node = 9999), "not a node")
})

test_that("plotNodeComparison returns a ranked comparison and runs", {
  obj <- build_iris_tree()
  term <- nodes(obj$tree, terminal = TRUE)$node
  pdf(NULL)
  cmp <- plotNodeComparison(obj$tree, term[1], term[2])
  dev.off()
  expect_s3_class(cmp, "data.frame")
  expect_true(all(c("variable", "type", "score", "node1", "node2") %in% names(cmp)))
  expect_false(is.unsorted(rev(cmp$score)))      # ranked by descending score
  expect_error(plotNodeComparison(obj$tree, 1, 1), "must differ")
})


# ---------------------------------------------------------------------------
# eCounterfactual()  (A: proximity-verified contrastive explanation)
# ---------------------------------------------------------------------------

test_that("eCounterfactual changes re-route the instance to the target leaf", {
  obj <- build_iris_tree()
  nd  <- obj$data[c(1, 80, 120), ]
  cf  <- eCounterfactual(obj$tree, nd, ensemble = obj$ensemble)

  expect_s3_class(cf, "e2counterfactual")
  expect_true(all(cf$cost[!is.na(cf$cost)] >= 0))

  route <- e2tree:::.e2_route_leaf
  for (i in which(!is.na(cf$target_leaf))) {
    row <- nd[i, , drop = FALSE]
    ch  <- cf$changes[[i]]
    if (nrow(ch) > 0) for (r in seq_len(nrow(ch))) {
      v <- ch$variable[r]; col <- row[[v]]
      row[[v]] <- if (is.factor(col)) factor(ch$to[r], levels = levels(col))
                  else if (is.numeric(col)) as.numeric(ch$to[r]) else ch$to[r]
    }
    expect_equal(route(obj$tree, row), cf$target_leaf[i])
  }
})

test_that("eCounterfactual reports the ensemble verification in [0, 1]", {
  obj <- build_iris_tree()
  cf  <- eCounterfactual(obj$tree, obj$data[c(80, 120), ], ensemble = obj$ensemble)
  pt  <- cf$prox_to_target; ps <- cf$prox_to_source
  expect_true(all(is.na(pt) | (pt >= 0 & pt <= 1)))
  expect_true(all(is.na(ps) | (ps >= 0 & ps <= 1)))
  expect_true(all(is.na(cf$validated) | is.logical(cf$validated)))
})

test_that("eCounterfactual without an ensemble leaves validation NA", {
  obj <- build_iris_tree()
  cf  <- eCounterfactual(obj$tree, obj$data[80, , drop = FALSE])
  expect_null(cf$validated)
  expect_s3_class(cf, "e2counterfactual")
})

test_that("plot.e2counterfactual: single bars and all multi-obs views", {
  skip_if_not_installed("ggplot2")
  obj <- build_iris_tree()
  cf  <- eCounterfactual(obj$tree, obj$data[c(60, 80, 120), ],
                         ensemble = obj$ensemble)
  for (ty in c("heatmap", "frequency", "beeswarm"))
    expect_s3_class(plot(cf, obs = 1:3, type = ty), "ggplot")
  expect_error(plot(cf, obs = c(1, 99)), "between 1 and")
})


# ---------------------------------------------------------------------------
# eStability()  (B: tree-bootstrap confidence / stability)
# ---------------------------------------------------------------------------

test_that("eStability returns confidence and stability in [0, 1], reproducibly", {
  obj <- build_iris_tree()
  nd  <- obj$data[c(1, 80, 120), ]
  st  <- eStability(obj$tree, obj$ensemble, newdata = nd, B = 20, k = 5, seed = 1)

  expect_s3_class(st, "e2stability")
  expect_true(all(st$confidence >= 0 & st$confidence <= 1))
  expect_true(all(vapply(st$neighbor_stability,
    function(d) all(d$stability >= 0 & d$stability <= 1), logical(1))))
  expect_true(all(st$prototype_prox$lower <= st$prototype_prox$mean + 1e-9))
  expect_true(all(st$prototype_prox$mean  <= st$prototype_prox$upper + 1e-9))

  st2 <- eStability(obj$tree, obj$ensemble, newdata = nd, B = 20, k = 5, seed = 1)
  expect_identical(st$confidence, st2$confidence)
})

test_that("eStability exposes the reconstructed region per instance", {
  obj <- build_iris_tree()
  nd  <- obj$data[c(1, 80, 120), ]
  st  <- eStability(obj$tree, obj$ensemble, newdata = nd, B = 20, k = 5, seed = 1)
  expect_length(st$region, nrow(nd))
  expect_true(all(!is.na(st$region)))
})

test_that("plot.e2stability: all view types render, with and without highlight", {
  skip_if_not_installed("ggplot2")
  obj <- build_iris_tree()
  nd  <- obj$data[c(1, 80, 120), ]
  st  <- eStability(obj$tree, obj$ensemble, newdata = nd, B = 20, k = 5, seed = 1)
  for (ty in c("profile", "beeswarm", "scatter", "forest"))
    expect_s3_class(plot(st, type = ty), "ggplot")
  expect_s3_class(plot(st, highlight = c(1, 3), labels = c("a", "c")), "ggplot")
  expect_error(plot(st, type = "nope"), "should be one of")
})


# ---------------------------------------------------------------------------
# explain()  (C: unified per-instance explanation)
# ---------------------------------------------------------------------------

test_that("explain composes the local layer and attaches A and B on request", {
  obj <- build_iris_tree()
  nd  <- obj$data[c(1, 80, 120), ]
  ex  <- explain(obj$tree, obj$ensemble, newdata = nd,
                 counterfactual = TRUE, stability = TRUE, B = 15, seed = 2)

  expect_s3_class(ex, "e2explanation")
  expect_s3_class(ex$contribution, "e2contribution")
  expect_s3_class(ex$neighbors, "e2neighbors")
  expect_s3_class(ex$counterfactual, "e2counterfactual")
  expect_s3_class(ex$stability, "e2stability")
  expect_equal(length(ex$leaf), nrow(nd))
  expect_output(print(ex), "E2Tree Local Explanation")
})

test_that("explain handles a single-row newdata", {
  obj <- build_iris_tree()
  ex  <- explain(obj$tree, obj$ensemble, newdata = obj$data[60, , drop = FALSE])
  expect_s3_class(ex, "e2explanation")
  expect_equal(length(ex$leaf), 1L)
})
