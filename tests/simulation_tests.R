#!/usr/bin/env Rscript
# =============================================================================
# e2tree — Comprehensive Simulation Test Suite
# =============================================================================
# Covers:
#   1.  Adapter correctness  (get_ensemble_type, extract_terminal_nodes,
#                              get_ensemble_predictions)
#   2.  Backward compatibility — RF/ranger leaf extraction matches native APIs;
#       D matrix is identical across two runs with the same model
#   3.  Dissimilarity matrix mathematical properties (symmetry, [0,1], zero
#       diagonal, non-trivial) for all available backends
#   4.  Core logic consistency — weighted co-occurrence formula is respected;
#       intra-class pairs have lower mean D than inter-class pairs
#   5.  Full e2tree pipeline — tree builds, predict() works, accuracy/RMSE
#       better than a trivial baseline
#   6.  Cross-backend coherence — D matrices from different backends on the
#       same data should be positively correlated (Spearman > 0.5)
#   7.  Determinism — same model + same data → identical D every call
#
# Usage:
#   Rscript tests/simulation_tests.R
# =============================================================================

suppressPackageStartupMessages(library(e2tree))

# ─── Test runner ─────────────────────────────────────────────────────────────

.results  <- list()
.pass_cnt <- 0L
.fail_cnt <- 0L

PASS <- function(name) {
  cat(sprintf("  \033[32m[PASS]\033[0m  %s\n", name))
  .pass_cnt <<- .pass_cnt + 1L
  .results[[length(.results) + 1L]] <<- list(name = name, status = "PASS")
}

FAIL <- function(name, reason = "") {
  cat(sprintf("  \033[31m[FAIL]\033[0m  %s", name))
  if (nzchar(reason)) cat(sprintf("  →  %s", reason))
  cat("\n")
  .fail_cnt <<- .fail_cnt + 1L
  .results[[length(.results) + 1L]] <<- list(name = name, status = "FAIL",
                                              reason = reason)
}

SKIP <- function(name, reason = "") {
  cat(sprintf("  \033[33m[SKIP]\033[0m  %s  →  %s\n", name, reason))
  .results[[length(.results) + 1L]] <<- list(name = name, status = "SKIP",
                                              reason = reason)
}

check <- function(name, expr) {
  ok <- tryCatch(isTRUE(expr), error = function(e) FALSE)
  if (ok) PASS(name) else FAIL(name, deparse(substitute(expr)))
}

section <- function(title) {
  cat(sprintf("\n\033[1;34m══════  %s  ══════\033[0m\n", title))
}

# ─── Shared data & settings ──────────────────────────────────────────────────

set.seed(42)

# Train / validation split — iris (classification)
n_iris     <- nrow(iris)
idx_iris   <- sample(n_iris, floor(0.75 * n_iris))
tr_iris    <- iris[idx_iris, ]
te_iris    <- iris[-idx_iris, ]

# Train / validation split — iris regression (Sepal.Length ~ rest)
tr_iris_reg <- tr_iris
te_iris_reg <- te_iris

# Train / validation split — mtcars (regression)
n_cars    <- nrow(mtcars)
idx_cars  <- sample(n_cars, floor(0.75 * n_cars))
tr_cars   <- mtcars[idx_cars, ]
te_cars   <- mtcars[-idx_cars, ]

SETTING_CLF <- list(impTotal = 0.1, maxDec = 0.01,  n = 5, level = 5)
SETTING_REG <- list(impTotal = 0.1, maxDec = 1e-6,  n = 3, level = 5)
PAR_OFF     <- list(active = FALSE, no_cores = 1)


# =============================================================================
# SECTION 1 — randomForest
# =============================================================================
section("1 · randomForest")

if (requireNamespace("randomForest", quietly = TRUE)) {
  suppressPackageStartupMessages(library(randomForest))
  set.seed(42)
  rf_clf <- randomForest(Species ~ ., data = tr_iris,  ntree = 100)
  rf_reg <- randomForest(mpg    ~ ., data = tr_cars, ntree = 100)

  # ── 1a. Adapter: get_ensemble_type ──
  check("RF clf  → 'classification'",
        get_ensemble_type(rf_clf) == "classification")
  check("RF reg  → 'regression'",
        get_ensemble_type(rf_reg) == "regression")

  # ── 1b. Adapter: extract_terminal_nodes ──
  nodes_rf <- extract_terminal_nodes(rf_clf, tr_iris)
  check("RF extract nodes — data.frame",       is.data.frame(nodes_rf))
  check("RF extract nodes — nrow = n_train",   nrow(nodes_rf) == nrow(tr_iris))
  check("RF extract nodes — ncol = ntree",     ncol(nodes_rf) == rf_clf$ntree)
  check("RF extract nodes — no NA",            !anyNA(nodes_rf))
  check("RF extract nodes — positive integers",
        all(unlist(nodes_rf) > 0) && all(unlist(nodes_rf) == floor(unlist(nodes_rf))))

  # ── 1c. Backward compatibility: adapter matches native RF predict(nodes=TRUE) ──
  native_nodes <- attr(predict(rf_clf, tr_iris, nodes = TRUE), "nodes")
  check("RF nodes identical to native predict(nodes=TRUE)",
        max(abs(as.matrix(nodes_rf) - unname(native_nodes))) == 0)

  # ── 1d. Adapter: get_ensemble_predictions ──
  preds_rf <- get_ensemble_predictions(rf_reg, tr_cars, "regression")
  check("RF preds — numeric vector",       is.numeric(preds_rf))
  check("RF preds — length = n_train",     length(preds_rf) == nrow(tr_cars))
  check("RF preds — no NA / Inf",          all(is.finite(preds_rf)))

  # ── 1e. createDisMatrix properties (classification) ──
  D_rf_clf <- createDisMatrix(rf_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("RF D clf — square matrix",        is.matrix(D_rf_clf) && nrow(D_rf_clf) == ncol(D_rf_clf))
  check("RF D clf — dim = n_train × n",    all(dim(D_rf_clf) == nrow(tr_iris)))
  check("RF D clf — diagonal = 0",         all(diag(D_rf_clf) == 0))
  check("RF D clf — symmetric",            isSymmetric(D_rf_clf))
  check("RF D clf — values in [0,1]",      all(D_rf_clf >= 0) && all(D_rf_clf <= 1))
  check("RF D clf — non-trivial",          sum(D_rf_clf) > 0)

  # ── 1f. createDisMatrix properties (regression) ──
  D_rf_reg <- createDisMatrix(rf_reg, tr_cars, "mpg", parallel = PAR_OFF)
  check("RF D reg — diagonal = 0",         all(diag(D_rf_reg) == 0))
  check("RF D reg — symmetric",            isSymmetric(D_rf_reg))
  check("RF D reg — values in [0,1]",      all(D_rf_reg >= 0) && all(D_rf_reg <= 1))

  # ── 1g. Determinism — same model, same data → identical D ──
  D_rf_clf2 <- createDisMatrix(rf_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("RF D clf — deterministic (run 1 == run 2)",
        identical(D_rf_clf, D_rf_clf2))

  # ── 1h. Core logic: intra-class D < inter-class D ──
  species <- tr_iris$Species
  idx_same  <- which(outer(species, species, `==`) & upper.tri(D_rf_clf))
  idx_diff  <- which(!outer(species, species, `==`) & upper.tri(D_rf_clf))
  check("RF D clf — intra-class mean D < inter-class mean D",
        mean(D_rf_clf[idx_same]) < mean(D_rf_clf[idx_diff]))

  # ── 1i. Full pipeline (classification) ──
  tree_rf_clf <- e2tree(Species ~ ., tr_iris, D_rf_clf, rf_clf, SETTING_CLF)
  check("RF e2tree clf — builds",          inherits(tree_rf_clf, "e2tree"))
  check("RF e2tree clf — at least 1 node", sum(!is.na(tree_rf_clf$tree$node)) >= 1)
  pred_rf_clf <- predict(tree_rf_clf, te_iris)
  check("RF e2tree clf — predict returns data.frame", is.data.frame(pred_rf_clf))
  acc_rf <- mean(pred_rf_clf$fit == te_iris$Species)
  check("RF e2tree clf — accuracy > 60%",  acc_rf > 0.60)

  # ── 1j. Full pipeline (regression) ──
  tree_rf_reg <- e2tree(mpg ~ ., tr_cars, D_rf_reg, rf_reg, SETTING_REG)
  pred_rf_reg <- predict(tree_rf_reg, te_cars)
  rmse_rf <- sqrt(mean((as.numeric(pred_rf_reg$fit) - te_cars$mpg)^2))
  baseline_rmse <- sd(tr_cars$mpg)
  check("RF e2tree reg — RMSE < sd(y_train)",  rmse_rf < baseline_rmse)

} else {
  SKIP("randomForest", "package not installed")
}


# =============================================================================
# SECTION 2 — ranger
# =============================================================================
section("2 · ranger")

if (requireNamespace("ranger", quietly = TRUE)) {
  suppressPackageStartupMessages(library(ranger))
  set.seed(42)
  rng_clf <- ranger(Species ~ ., data = tr_iris, num.trees = 100,
                    write.forest = TRUE)
  rng_reg <- ranger(mpg    ~ ., data = tr_cars, num.trees = 100,
                    write.forest = TRUE)

  # ── 2a. Adapter types ──
  check("ranger clf → 'classification'",   get_ensemble_type(rng_clf) == "classification")
  check("ranger reg → 'regression'",       get_ensemble_type(rng_reg) == "regression")

  # ── 2b. extract_terminal_nodes ──
  nodes_rng <- extract_terminal_nodes(rng_clf, tr_iris)
  check("ranger nodes — nrow = n_train",   nrow(nodes_rng) == nrow(tr_iris))
  check("ranger nodes — ncol = num.trees", ncol(nodes_rng) == rng_clf$num.trees)

  # Backward compatibility: matches native ranger predict(type="terminalNodes")
  native_rng <- predict(rng_clf, tr_iris, type = "terminalNodes")$predictions
  check("ranger nodes identical to native terminalNodes",
        max(abs(as.matrix(nodes_rng) - unname(native_rng))) == 0)

  # ── 2c. Dissimilarity matrix properties ──
  D_rng_clf <- createDisMatrix(rng_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("ranger D clf — diagonal = 0",     all(diag(D_rng_clf) == 0))
  check("ranger D clf — symmetric",        isSymmetric(D_rng_clf))
  check("ranger D clf — values in [0,1]",  all(D_rng_clf >= 0) && all(D_rng_clf <= 1))

  D_rng_reg <- createDisMatrix(rng_reg, tr_cars, "mpg", parallel = PAR_OFF)
  check("ranger D reg — diagonal = 0",     all(diag(D_rng_reg) == 0))
  check("ranger D reg — symmetric",        isSymmetric(D_rng_reg))
  check("ranger D reg — values in [0,1]",  all(D_rng_reg >= 0) && all(D_rng_reg <= 1))

  # ── 2d. Determinism ──
  D_rng_clf2 <- createDisMatrix(rng_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("ranger D clf — deterministic",    identical(D_rng_clf, D_rng_clf2))

  # ── 2e. Cross-backend coherence: RF vs ranger D should be correlated ──
  if (exists("D_rf_clf")) {
    r_spearman <- cor(as.vector(D_rf_clf[upper.tri(D_rf_clf)]),
                      as.vector(D_rng_clf[upper.tri(D_rng_clf)]),
                      method = "spearman")
    check("RF vs ranger D clf — Spearman > 0.50", r_spearman > 0.50)
  }

  # ── 2f. Full pipeline ──
  tree_rng_clf <- e2tree(Species ~ ., tr_iris, D_rng_clf, rng_clf, SETTING_CLF)
  acc_rng <- mean(predict(tree_rng_clf, te_iris)$fit == te_iris$Species)
  check("ranger e2tree clf — accuracy > 60%", acc_rng > 0.60)

  tree_rng_reg <- e2tree(mpg ~ ., tr_cars, D_rng_reg, rng_reg, SETTING_REG)
  rmse_rng <- sqrt(mean((as.numeric(predict(tree_rng_reg, te_cars)$fit) - te_cars$mpg)^2))
  check("ranger e2tree reg — RMSE < sd(y_train)",  rmse_rng < sd(tr_cars$mpg))

} else {
  SKIP("ranger", "package not installed")
}


# =============================================================================
# SECTION 3 — XGBoost
# =============================================================================
section("3 · XGBoost")

if (requireNamespace("xgboost", quietly = TRUE)) {
  suppressPackageStartupMessages(library(xgboost))

  # ── Build models ──
  set.seed(42)
  X_clf  <- as.matrix(tr_iris[, 1:4])
  y_clf  <- as.integer(tr_iris$Species) - 1L
  xgb_clf <- xgb.train(
    params = list(objective = "multi:softmax", num_class = 3,
                  max_depth = 4, eta = 0.3, nthread = 1),
    data = xgb.DMatrix(X_clf, label = y_clf),
    nrounds = 50, verbose = 0
  )

  X_reg  <- as.matrix(tr_iris[, 2:4])
  y_reg  <- tr_iris$Sepal.Length
  xgb_reg <- xgb.train(
    params = list(objective = "reg:squarederror",
                  max_depth = 4, eta = 0.3, nthread = 1),
    data = xgb.DMatrix(X_reg, label = y_reg),
    nrounds = 50, verbose = 0
  )

  # ── 3a. Adapter types ──
  check("XGB clf  → 'classification'",     get_ensemble_type(xgb_clf) == "classification")
  check("XGB reg  → 'regression'",         get_ensemble_type(xgb_reg) == "regression")

  # ── 3b. extract_terminal_nodes ──
  nodes_xgb <- extract_terminal_nodes(xgb_clf, tr_iris[, 1:4])
  check("XGB nodes — is data.frame",       is.data.frame(nodes_xgb))
  check("XGB nodes — nrow = n_train",      nrow(nodes_xgb) == nrow(tr_iris))
  check("XGB nodes — ncol > 0",            ncol(nodes_xgb) > 0)
  check("XGB nodes — no NA",               !anyNA(nodes_xgb))

  # ── 3c. Dissimilarity matrix properties ──
  D_xgb_clf <- createDisMatrix(xgb_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("XGB D clf — diagonal = 0",        all(diag(D_xgb_clf) == 0))
  check("XGB D clf — symmetric",           isSymmetric(D_xgb_clf))
  check("XGB D clf — values in [0,1]",     all(D_xgb_clf >= 0) && all(D_xgb_clf <= 1))
  check("XGB D clf — non-trivial",         sum(D_xgb_clf) > 0)

  tr_iris_r <- tr_iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")]
  tr_iris_r$Sepal.Length_y <- tr_iris$Sepal.Length
  D_xgb_reg <- createDisMatrix(xgb_reg,
                                cbind(Sepal.Length = tr_iris$Sepal.Length,
                                      tr_iris[, 2:4]),
                                "Sepal.Length",
                                parallel = PAR_OFF)
  check("XGB D reg — diagonal = 0",        all(diag(D_xgb_reg) == 0))
  check("XGB D reg — symmetric",           isSymmetric(D_xgb_reg))
  check("XGB D reg — values in [0,1]",     all(D_xgb_reg >= 0) && all(D_xgb_reg <= 1))

  # ── 3d. Determinism ──
  D_xgb_clf2 <- createDisMatrix(xgb_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("XGB D clf — deterministic",       identical(D_xgb_clf, D_xgb_clf2))

  # ── 3e. Core logic: intra-class D < inter-class D ──
  sp2 <- tr_iris$Species
  idx_s <- which(outer(sp2, sp2, `==`) & upper.tri(D_xgb_clf))
  idx_d <- which(!outer(sp2, sp2, `==`) & upper.tri(D_xgb_clf))
  check("XGB D clf — intra-class D < inter-class D",
        mean(D_xgb_clf[idx_s]) < mean(D_xgb_clf[idx_d]))

  # ── 3f. Full pipeline ──
  tree_xgb_clf <- e2tree(Species ~ ., tr_iris, D_xgb_clf, xgb_clf, SETTING_CLF)
  check("XGB e2tree clf — builds",         inherits(tree_xgb_clf, "e2tree"))
  acc_xgb <- mean(predict(tree_xgb_clf, te_iris)$fit == te_iris$Species)
  check("XGB e2tree clf — accuracy > 60%", acc_xgb > 0.60)

  # ── 3g. Cross-backend coherence with RF ──
  if (exists("D_rf_clf")) {
    r_xgb_rf <- cor(as.vector(D_rf_clf[upper.tri(D_rf_clf)]),
                    as.vector(D_xgb_clf[upper.tri(D_xgb_clf)]),
                    method = "spearman")
    check("XGB vs RF D clf — Spearman > 0.30", r_xgb_rf > 0.30)
  }

} else {
  SKIP("XGBoost", "package not installed")
}


# =============================================================================
# SECTION 4 — GBM
# =============================================================================
section("4 · GBM")

if (requireNamespace("gbm", quietly = TRUE)) {
  suppressPackageStartupMessages(library(gbm))

  set.seed(42)
  gbm_reg <- gbm(mpg ~ ., data = tr_cars,
                 distribution = "gaussian",
                 n.trees = 100, interaction.depth = 2, shrinkage = 0.1,
                 n.minobsinnode = 1, bag.fraction = 0.8, verbose = FALSE)

  # Binary classification: is_setosa
  tr_bin            <- tr_iris
  tr_bin$is_setosa  <- as.integer(tr_iris$Species == "setosa")
  gbm_clf <- gbm(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    data = tr_bin, distribution = "bernoulli",
    n.trees = 100, interaction.depth = 2, shrinkage = 0.1,
    n.minobsinnode = 1, bag.fraction = 0.8, verbose = FALSE
  )

  # ── 4a. Types ──
  check("GBM reg  → 'regression'",         get_ensemble_type(gbm_reg) == "regression")
  check("GBM clf  → 'classification'",     get_ensemble_type(gbm_clf) == "classification")

  # ── 4b. extract_terminal_nodes ──
  nodes_gbm <- extract_terminal_nodes(gbm_reg, tr_cars)
  check("GBM nodes — is data.frame",       is.data.frame(nodes_gbm))
  check("GBM nodes — nrow = n_train",      nrow(nodes_gbm) == nrow(tr_cars))
  check("GBM nodes — ncol = n.trees",      ncol(nodes_gbm) == gbm_reg$n.trees)
  check("GBM nodes — positive integers",
        all(unlist(nodes_gbm) > 0) && all(unlist(nodes_gbm) == floor(unlist(nodes_gbm))))

  # ── 4c. Dissimilarity matrix properties ──
  D_gbm_reg <- createDisMatrix(gbm_reg, tr_cars, "mpg", parallel = PAR_OFF)
  check("GBM D reg — diagonal = 0",        all(diag(D_gbm_reg) == 0))
  check("GBM D reg — symmetric",           isSymmetric(D_gbm_reg))
  check("GBM D reg — values in [0,1]",     all(D_gbm_reg >= 0) && all(D_gbm_reg <= 1))

  # For GBM clf: coerce is_setosa to factor BEFORE createDisMatrix
  tr_bin$is_setosa  <- factor(tr_bin$is_setosa)
  D_gbm_clf <- createDisMatrix(gbm_clf, tr_bin, "is_setosa", parallel = PAR_OFF)
  check("GBM D clf — diagonal = 0",        all(diag(D_gbm_clf) == 0))
  check("GBM D clf — symmetric",           isSymmetric(D_gbm_clf))
  check("GBM D clf — values in [0,1]",     all(D_gbm_clf >= 0) && all(D_gbm_clf <= 1))

  # ── 4d. Determinism ──
  D_gbm_reg2 <- createDisMatrix(gbm_reg, tr_cars, "mpg", parallel = PAR_OFF)
  check("GBM D reg — deterministic",       identical(D_gbm_reg, D_gbm_reg2))

  # ── 4e. get_ensemble_predictions ──
  preds_gbm <- get_ensemble_predictions(gbm_reg, tr_cars, "regression")
  check("GBM preds — numeric, no NA",
        is.numeric(preds_gbm) && length(preds_gbm) == nrow(tr_cars) && !anyNA(preds_gbm))

  # ── 4f. Full pipeline (regression) ──
  tree_gbm_reg <- e2tree(mpg ~ ., tr_cars, D_gbm_reg, gbm_reg, SETTING_REG)
  check("GBM e2tree reg — builds",         inherits(tree_gbm_reg, "e2tree"))
  rmse_gbm <- sqrt(mean((as.numeric(predict(tree_gbm_reg, te_cars)$fit) - te_cars$mpg)^2))
  check("GBM e2tree reg — RMSE < sd(y_train)",  rmse_gbm < sd(tr_cars$mpg))

  # ── 4g. Full pipeline (classification) ──
  tree_gbm_clf <- e2tree(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    tr_bin, D_gbm_clf, gbm_clf, SETTING_CLF
  )
  check("GBM e2tree clf — builds",         inherits(tree_gbm_clf, "e2tree"))

} else {
  SKIP("GBM", "package not installed")
}


# =============================================================================
# SECTION 5 — LightGBM
# =============================================================================
section("5 · LightGBM")

if (requireNamespace("lightgbm", quietly = TRUE)) {
  suppressPackageStartupMessages(library(lightgbm))

  # Use iris for regression (150 obs; avoids 1-tree collapse on mtcars 32 obs)
  set.seed(42)
  X_lgb_clf <- as.matrix(tr_iris[, 1:4])
  y_lgb_clf <- as.integer(tr_iris$Species) - 1L
  lgb_clf <- lgb.train(
    params = list(objective = "multiclass", num_class = 3,
                  num_leaves = 15, min_data_in_leaf = 3, verbose = -1),
    data = lgb.Dataset(X_lgb_clf, label = y_lgb_clf,
                       colnames = colnames(X_lgb_clf)),
    nrounds = 50
  )

  X_lgb_reg <- as.matrix(tr_iris[, 2:4])
  y_lgb_reg <- tr_iris$Sepal.Length
  lgb_reg <- lgb.train(
    params = list(objective = "regression", num_leaves = 8,
                  min_data_in_leaf = 3, verbose = -1),
    data = lgb.Dataset(X_lgb_reg, label = y_lgb_reg,
                       colnames = colnames(X_lgb_reg)),
    nrounds = 50
  )

  # ── 5a. Types ──
  check("LGB clf → 'classification'",      get_ensemble_type(lgb_clf) == "classification")
  check("LGB reg → 'regression'",          get_ensemble_type(lgb_reg) == "regression")

  # ── 5b. extract_terminal_nodes ──
  nodes_lgb <- extract_terminal_nodes(lgb_clf, tr_iris[, 1:4])
  check("LGB nodes — is data.frame",       is.data.frame(nodes_lgb))
  check("LGB nodes — nrow = n_train",      nrow(nodes_lgb) == nrow(tr_iris))
  check("LGB nodes — ncol = num_trees",    ncol(nodes_lgb) == lgb_clf$num_trees())
  check("LGB nodes — no NA",               !anyNA(nodes_lgb))

  nodes_lgb_r <- extract_terminal_nodes(lgb_reg, tr_iris[, 2:4])
  check("LGB reg nodes — nrow = n_train",  nrow(nodes_lgb_r) == nrow(tr_iris))
  check("LGB reg nodes — ncol = num_trees", ncol(nodes_lgb_r) == lgb_reg$num_trees())

  # ── 5c. Dissimilarity matrix properties ──
  D_lgb_clf <- createDisMatrix(lgb_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("LGB D clf — diagonal = 0",        all(diag(D_lgb_clf) == 0))
  check("LGB D clf — symmetric",           isSymmetric(D_lgb_clf))
  check("LGB D clf — values in [0,1]",     all(D_lgb_clf >= 0) && all(D_lgb_clf <= 1))
  check("LGB D clf — non-trivial",         sum(D_lgb_clf) > 0)

  D_lgb_reg <- createDisMatrix(lgb_reg,
                                cbind(Sepal.Length = tr_iris$Sepal.Length,
                                      tr_iris[, 2:4]),
                                "Sepal.Length",
                                parallel = PAR_OFF)
  check("LGB D reg — diagonal = 0",        all(diag(D_lgb_reg) == 0))
  check("LGB D reg — symmetric",           isSymmetric(D_lgb_reg))
  check("LGB D reg — values in [0,1]",     all(D_lgb_reg >= 0) && all(D_lgb_reg <= 1))

  # ── 5d. Determinism ──
  D_lgb_clf2 <- createDisMatrix(lgb_clf, tr_iris, "Species", parallel = PAR_OFF)
  check("LGB D clf — deterministic",       identical(D_lgb_clf, D_lgb_clf2))

  # ── 5e. Core logic: intra-class D < inter-class D ──
  sp3 <- tr_iris$Species
  idx_s3 <- which(outer(sp3, sp3, `==`) & upper.tri(D_lgb_clf))
  idx_d3 <- which(!outer(sp3, sp3, `==`) & upper.tri(D_lgb_clf))
  check("LGB D clf — intra-class D < inter-class D",
        mean(D_lgb_clf[idx_s3]) < mean(D_lgb_clf[idx_d3]))

  # ── 5f. Cross-backend coherence with RF ──
  if (exists("D_rf_clf")) {
    r_lgb_rf <- cor(as.vector(D_rf_clf[upper.tri(D_rf_clf)]),
                    as.vector(D_lgb_clf[upper.tri(D_lgb_clf)]),
                    method = "spearman")
    check("LGB vs RF D clf — Spearman > 0.30", r_lgb_rf > 0.30)
  }

  # ── 5g. Full pipeline (classification) ──
  tree_lgb_clf <- e2tree(Species ~ ., tr_iris, D_lgb_clf, lgb_clf, SETTING_CLF)
  check("LGB e2tree clf — builds",         inherits(tree_lgb_clf, "e2tree"))
  acc_lgb <- mean(predict(tree_lgb_clf, te_iris)$fit == te_iris$Species)
  check("LGB e2tree clf — accuracy > 60%", acc_lgb > 0.60)

  # ── 5h. Full pipeline (regression) ──
  te_iris_r <- cbind(Sepal.Length = te_iris$Sepal.Length, te_iris[, 2:4])
  tree_lgb_reg <- e2tree(
    Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
    cbind(Sepal.Length = tr_iris$Sepal.Length, tr_iris[, 2:4]),
    D_lgb_reg, lgb_reg, SETTING_REG
  )
  check("LGB e2tree reg — builds",         inherits(tree_lgb_reg, "e2tree"))
  pred_lgb_r <- predict(tree_lgb_reg, as.data.frame(te_iris_r))
  rmse_lgb <- sqrt(mean((as.numeric(pred_lgb_r$fit) - te_iris$Sepal.Length)^2))
  check("LGB e2tree reg — RMSE < sd(y_train)",
        rmse_lgb < sd(tr_iris$Sepal.Length))

} else {
  SKIP("LightGBM", "package not installed")
}


# =============================================================================
# SECTION 6 — Cross-backend logic consistency (shared iris classification)
# =============================================================================
section("6 · Cross-backend consistency")

# All available D matrices for iris classification should share the same
# qualitative structure: same-class pairs should cluster together.
backends <- list()
if (exists("D_rf_clf"))  backends[["RF"]]       <- D_rf_clf
if (exists("D_rng_clf")) backends[["ranger"]]   <- D_rng_clf
if (exists("D_xgb_clf")) backends[["XGBoost"]]  <- D_xgb_clf
if (exists("D_lgb_clf")) backends[["LightGBM"]] <- D_lgb_clf

sp_all <- tr_iris$Species
idx_same_all <- which(outer(sp_all, sp_all, `==`) & upper.tri(D_rf_clf))
idx_diff_all <- which(!outer(sp_all, sp_all, `==`) & upper.tri(D_rf_clf))

for (nm in names(backends)) {
  D_b <- backends[[nm]]
  check(sprintf("%s  — intra-class mean D < inter-class mean D", nm),
        mean(D_b[idx_same_all]) < mean(D_b[idx_diff_all]))
}

if (length(backends) >= 2) {
  bnames <- names(backends)
  for (i in seq_len(length(bnames) - 1)) {
    for (j in seq(i + 1, length(bnames))) {
      a <- backends[[bnames[i]]]
      b <- backends[[bnames[j]]]
      r <- cor(as.vector(a[upper.tri(a)]),
               as.vector(b[upper.tri(b)]),
               method = "spearman")
      check(sprintf("%s vs %s — Spearman > 0.30", bnames[i], bnames[j]),
            r > 0.30)
    }
  }
}


# =============================================================================
# SECTION 7 — Error handling and input validation
# =============================================================================
section("7 · Input validation")

check("get_ensemble_type unsupported class → error",
      tryCatch({ get_ensemble_type(list()); FALSE },
               error = function(e) TRUE))

check("createDisMatrix NULL ensemble → error",
      tryCatch({ createDisMatrix(NULL, iris, "Species", PAR_OFF); FALSE },
               error = function(e) TRUE))

check("createDisMatrix non-data.frame data → error",
      tryCatch({
        if (exists("rf_clf")) createDisMatrix(rf_clf, as.matrix(iris[, 1:4]),
                                              "Species", PAR_OFF)
        FALSE
      }, error = function(e) TRUE))

check("createDisMatrix bad label → error",
      tryCatch({
        if (exists("rf_clf")) createDisMatrix(rf_clf, iris, "NOTACOL", PAR_OFF)
        FALSE
      }, error = function(e) TRUE))

check("e2tree bad setting → error",
      tryCatch({
        if (exists("rf_clf") && exists("D_rf_clf"))
          e2tree(Species ~ ., tr_iris, D_rf_clf, rf_clf,
                 list(minsplit = 20))
        FALSE
      }, error = function(e) TRUE))


# =============================================================================
# SUMMARY
# =============================================================================
section("SUMMARY")

total <- .pass_cnt + .fail_cnt
skipped <- length(.results) - total
cat(sprintf(
  "\n  \033[32m%d PASS\033[0m   \033[31m%d FAIL\033[0m   \033[33m%d SKIP\033[0m   (total checked: %d)\n\n",
  .pass_cnt, .fail_cnt, skipped, total
))

if (.fail_cnt > 0) {
  cat("  Failed tests:\n")
  for (r in .results) {
    if (identical(r$status, "FAIL"))
      cat(sprintf("    - %s\n", r$name))
  }
  cat("\n")
  quit(status = 1L)
} else {
  cat("  \033[32mAll tests passed.\033[0m\n\n")
}
