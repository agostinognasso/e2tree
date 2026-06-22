# Tests for panel_e2tree(): the between/within (Mundlak) decomposition.
# A synthetic panel with KNOWN roles is used so we can assert that the between
# tree recovers the unit-level driver and the within tree recovers the
# time-varying driver, and that the decomposed reconstruction is no worse than a
# single pooled e2tree (the necessity argument).

skip_if_not_installed("ranger")
skip_if_not_installed("randomForest")

# ---------------------------------------------------------------------------
# Synthetic known-roles panel:
#   x1 = A_i  (time-invariant)  -> drives BETWEEN
#   x2 = B_it (mean-zero within) -> drives WITHIN
#   x3       = noise
#   y = 3*A + 2*B + small noise   (high ICC: between dominates)
# ---------------------------------------------------------------------------
make_panel <- function(N = 40, Tt = 6, seed = 1) {
  set.seed(seed)
  unit <- rep(seq_len(N), each = Tt)
  A    <- rep(rnorm(N, sd = 2), each = Tt)              # unit-level latent
  B    <- rnorm(N * Tt)
  B    <- B - rep(tapply(B, unit, mean), each = Tt)     # demean within unit
  data.frame(
    id = unit,
    x1 = A,                                             # time-invariant
    x2 = B,                                             # time-varying
    x3 = rnorm(N * Tt),                                 # noise
    y  = 3 * A + 2 * B + rnorm(N * Tt, sd = 0.1)
  )
}

fit_panel <- function(df) {
  panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id",
               engine = "ranger", ntree = 200, seed = 7)
}

# ---------------------------------------------------------------------------

test_that("panel_e2tree returns an e2panel with the expected structure", {
  m <- fit_panel(make_panel())
  expect_s3_class(m, "e2panel")
  expect_true(all(c("between", "within", "outcome_var_recovered", "r2_panel",
                    "fidelity_panel", "icc", "predictions", "unit_means",
                    "bpred") %in% names(m)))
  # deprecated alias still mirrors the new name
  expect_identical(m$fidelity_panel, m$outcome_var_recovered)
  expect_equal(nrow(m$predictions), 40 * 6)
  expect_setequal(names(m$predictions),
                  c(".row", "unit", "outcome", ".between", ".within", ".panel"))
  # .row maps back to the original data rows
  expect_equal(m$predictions$outcome,
               make_panel()$y[m$predictions$.row], tolerance = 1e-12)
  # heavy elements are stripped by default
  expect_null(m$between$D)
  expect_null(m$within$D)
})

test_that("between recovers the unit driver, within recovers the time driver", {
  m <- fit_panel(make_panel())
  # between tree splits on x1 (the time-invariant unit driver)
  expect_true("x1" %in% m$between$variables)
  # within tree splits on x2 (the time-varying driver)
  expect_true("x2" %in% m$within$variables)
  # x1 is constant after demeaning -> dropped from the within representation
  expect_true("x1" %in% m$dropped$within)
  # x2 has ~zero between-unit variance -> dropped from the between representation
  expect_true("x2" %in% m$dropped$between)
})

test_that("decomposed panel fidelity is high and not worse than pooled e2tree", {
  df <- make_panel()
  m  <- fit_panel(df)
  expect_gt(m$fidelity_panel, 0.80)

  # Pooled baseline: a single e2tree on the raw features, reconstruction vs y.
  set.seed(7)
  ens  <- ranger::ranger(y ~ x1 + x2 + x3, data = df, num.trees = 200,
                         importance = "permutation")
  D    <- createDisMatrix(ens, data = df, label = "y",
                          parallel = list(active = FALSE, no_cores = 1))
  tree <- e2tree(y ~ x1 + x2 + x3, df, D, ens,
                 setting = list(impTotal = 0.1, maxDec = 1e-6, n = 2, level = 5))
  pooled_fit <- as.numeric(predict(tree, newdata = df)$fit)
  pooled_fid <- stats::cor(pooled_fit, df$y)^2

  expect_gte(m$fidelity_panel, pooled_fid)
})

test_that("predict reproduces the training reconstruction", {
  df <- make_panel()
  m  <- fit_panel(df)
  pr <- predict(m, newdata = df)
  expect_equal(pr$.panel, m$predictions$.panel, tolerance = 1e-8)
  expect_equal(pr$.between, m$predictions$.between, tolerance = 1e-8)
  expect_equal(pr$.within, m$predictions$.within, tolerance = 1e-8)
})

test_that("predict warns and falls back for unseen units", {
  df <- make_panel()
  m  <- fit_panel(df)
  nd <- df[1:3, ]
  nd$id <- "brand_new_unit"
  expect_warning(pr <- predict(m, newdata = nd), "unseen")
  expect_equal(pr$.within, rep(0, 3))           # deviation 0 for unseen unit
  expect_equal(pr$.between, rep(mean(m$bpred), 3))
})

test_that("a factor outcome is rejected (regression only)", {
  df <- make_panel()
  df$y <- factor(df$y > median(df$y))
  expect_error(
    panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id", engine = "ranger"),
    "regression only"
  )
})

test_that("non-numeric predictors are rejected", {
  df <- make_panel()
  df$x3 <- factor(sample(letters[1:3], nrow(df), replace = TRUE))
  expect_error(
    panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id", engine = "ranger"),
    "numeric"
  )
})

test_that("print and summary run without error", {
  m <- fit_panel(make_panel())
  expect_output(print(m), "Panel e2tree")
  expect_output(summary(m), "Fidelity")
})

test_that("engine arguments managed internally are rejected from '...'", {
  df <- make_panel()
  expect_error(
    panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id",
                 engine = "ranger", importance = "permutation"),
    "managed internally"
  )
  expect_error(
    panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id",
                 engine = "ranger", num.trees = 100),
    "managed internally"
  )
})

test_that("single-observation units are flagged and min_periods filters them", {
  df <- make_panel()
  extra <- df[1, ]
  extra$id <- 9999L          # one unit with a single observation
  df2 <- rbind(df, extra)
  expect_warning(m <- fit_panel(df2), "single observation")
  expect_equal(m$n_singleton, 1L)

  m2 <- suppressWarnings(
    panel_e2tree(y ~ x1 + x2 + x3, data = df2, unit = "id",
                 engine = "ranger", ntree = 200, seed = 7, min_periods = 2)
  )
  expect_equal(m2$dropped_units, "9999")
  expect_equal(length(m2$bpred), 40L)
})

test_that("duplicated (unit, time) rows raise a warning when time is given", {
  df <- make_panel()
  df$tt <- rep(seq_len(6), times = 40)
  df2 <- rbind(df, df[1, ])
  expect_warning(
    panel_e2tree(y ~ x1 + x2 + x3, data = df2, unit = "id", time = "tt",
                 engine = "ranger", ntree = 100, seed = 7),
    "duplicated"
  )
})

test_that("predict warns when missing newdata features are zero-imputed", {
  df <- make_panel()
  m  <- fit_panel(df)
  nd <- df[1:4, ]
  nd$x2[2] <- NA
  expect_warning(pr <- predict(m, newdata = nd), "missing feature")
  expect_false(anyNA(pr$.panel))
})

test_that("within = 'twoway' removes period effects and reconstructs additively", {
  df <- make_panel()
  df$tt <- rep(seq_len(6), times = 40)
  # inject a strong common period shock
  shock <- c(0, 3, -2, 5, 1, -4)
  df$y  <- df$y + shock[df$tt]

  m <- panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id", time = "tt",
                    within = "twoway", engine = "ranger", ntree = 200, seed = 7)
  expect_identical(m$within_type, "twoway")
  expect_true(".timeeffect" %in% names(m$predictions))
  # the estimated period effects track the injected shock (up to centering)
  est <- m$time_means$y[order(m$time_means$.time)]
  expect_gt(stats::cor(est, shock), 0.99)
  # period effects are (weighted) centered: unit means + period effects + within
  # residuals reproduce y by construction, so the taus must average to ~0
  expect_lt(abs(mean(m$predictions$.timeeffect)), 1e-8)
  # predict() reproduces the training reconstruction under twoway as well
  pr <- predict(m, newdata = df)
  expect_equal(pr$.panel, m$predictions$.panel, tolerance = 1e-8)
  # twoway prediction requires the time column
  expect_error(predict(m, newdata = df[, setdiff(names(df), "tt")]), "time column")
})

test_that("target = 'pooled' explains a given pooled ensemble", {
  df <- make_panel()
  set.seed(7)
  pooled <- ranger::ranger(y ~ x1 + x2 + x3, data = df, num.trees = 200)

  m <- panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id",
                    target = "pooled", pooled_ensemble = pooled,
                    engine = "ranger", ntree = 200, seed = 7)
  expect_identical(m$target, "pooled")
  expect_true(".pooled" %in% names(m$predictions))
  # the explanandum is the pooled model's predictions ...
  fhat <- as.numeric(predict(pooled, data = df)$predictions)
  expect_equal(m$predictions$.pooled, fhat, tolerance = 1e-8)
  # ... and the decomposed explanation reproduces it faithfully
  expect_gt(m$r2_panel, 0.8)
  # metrics vs the observed outcome are also reported
  expect_false(is.null(m$vs_outcome))
  expect_true(is.finite(m$vs_outcome$r2))
})

test_that("keep_D retains the dissimilarity matrices on request", {
  df <- make_panel(N = 12, Tt = 4)
  m <- suppressWarnings(
    panel_e2tree(y ~ x1 + x2 + x3, data = df, unit = "id",
                 engine = "ranger", ntree = 100, seed = 7, keep_D = TRUE)
  )
  expect_true(is.matrix(m$between$D))
  expect_equal(dim(m$between$D), c(12L, 12L))
})

test_that("bundled panel_health dataset works end to end", {
  data(panel_health, package = "e2tree")
  expect_s3_class(panel_health, "data.frame")
  expect_equal(dim(panel_health), c(480L, 9L))
  expect_true(all(c("country", "year", "life_expectancy") %in% names(panel_health)))

  m <- panel_e2tree(
    life_expectancy ~ gdp_pc + health_exp + schooling +
      immunization + sanitation + undernourish,
    data = panel_health, unit = "country", time = "year",
    engine = "ranger", ntree = 200, seed = 7)

  expect_s3_class(m, "e2panel")
  expect_gt(m$between$fidelity, 0.8)            # between is strongly recovered
  expect_false(is.na(m$within$fidelity))        # within is not degenerate
  # the designed roles surface: income level drives between, immunization within
  expect_true("gdp_pc" %in% m$between$variables)
  expect_true("immunization" %in% m$within$variables)
})
