make_pair <- function(n = 40, K = 4, seed = 1) {
  set.seed(seed)
  g <- sample(rep_len(seq_len(K), n))
  O <- matrix(stats::runif(n * n, 0, 0.3), n, n)
  same <- outer(g, g, "==")
  O[same] <- stats::runif(sum(same), 0.5, 0.9)
  O[lower.tri(O)] <- t(O)[lower.tri(O)]
  diag(O) <- 1
  w <- stats::runif(K, 0.6, 1)
  Oh <- same * w[g][row(same)]
  Oh[!same] <- 0
  diag(Oh) <- 1
  list(O = O, Oh = Oh)
}

test_that("loi terms follow the 0/0 = 0 convention and the decomposition", {
  expect_equal(e2tree:::.loi_terms(c(0, 0.5, 1), c(0, 0, 0.5)), c(0, 0.5, 0.25))
  p <- make_pair()
  r <- loi(p$O, p$Oh)
  expect_equal(r$loi, r$loi_in + r$loi_out)
  expect_equal(r$sqrt_nloi, sqrt(r$nloi))
})

test_that("nLoI violates the triangle inequality, sqrt(nLoI) does not", {
  f <- function(a, b) e2tree:::.loi_terms(a, b)
  expect_gt(f(0, 1), f(0, 0.5) + f(0.5, 1))
  set.seed(2)
  x <- matrix(stats::runif(3 * 5000), ncol = 3)
  lhs <- sqrt(f(x[, 1], x[, 3]))
  rhs <- sqrt(f(x[, 1], x[, 2])) + sqrt(f(x[, 2], x[, 3]))
  expect_true(all(lhs <= rhs + 1e-12))
})

test_that("exact null mean of nLoI matches the permutation average", {
  p <- make_pair(n = 30)
  li <- which(lower.tri(p$O))
  exact <- e2tree:::.nloi_null_mean(p$O[li], p$Oh[li])
  # enumerate the permutation mean by brute force over all pair images
  o <- p$O[li]; oh <- p$Oh[li]
  brute <- mean(outer(o, oh, function(a, b) e2tree:::.loi_terms(a, b)))
  expect_equal(exact, brute)
  set.seed(3)
  mc <- mean(replicate(2000, {
    q <- sample.int(30)
    mean(e2tree:::.loi_terms(o, p$Oh[q, q][li]))
  }))
  expect_equal(exact, mc, tolerance = 0.01)
})

test_that("measures are invariant to simultaneous relabelling (except SSIM)", {
  p <- make_pair()
  li <- which(lower.tri(p$O))
  q <- sample.int(nrow(p$O))
  a <- e2tree:::.fid_all(p$O[li], p$Oh[li], ssim = FALSE)
  Oq <- p$O[q, q]; Ohq <- p$Oh[q, q]
  b <- e2tree:::.fid_all(Oq[li], Ohq[li], ssim = FALSE)
  expect_equal(a, b)
})

test_that("RV and Mantel give identical permutation p-values", {
  p <- make_pair()
  set.seed(4)
  pm <- e2tree:::.fid_perm(p$O, p$Oh, 199, ssim = FALSE)
  li <- which(lower.tri(p$O))
  obs <- e2tree:::.fid_all(p$O[li], p$Oh[li], ssim = FALSE)
  expect_equal(rank(pm["RV", ]), rank(pm["Mantel", ]))
  expect_equal(mean(pm["RV", ] >= obs[["RV"]]), mean(pm["Mantel", ] >= obs[["Mantel"]]))
})

test_that("kappa is 1 at perfect agreement and about 0 under random relabelling", {
  p <- make_pair()
  li <- which(lower.tri(p$O))
  o <- p$O[li]
  expect_equal(e2tree:::.fid_kappa(0, e2tree:::.nloi_null_mean(o, o), TRUE), 1)
  set.seed(5)
  k <- replicate(200, {
    q <- sample.int(nrow(p$O)); ohq <- p$Oh[q, q][li]
    e2tree:::.fid_kappa(mean(e2tree:::.loi_terms(o, ohq)),
                        e2tree:::.nloi_null_mean(o, ohq), TRUE)
  })
  expect_lt(abs(mean(k)), 0.02)
})

test_that("bootstrap excludes pairs of copies and returns sensible intervals", {
  p <- make_pair(n = 50)
  set.seed(6)
  bm <- e2tree:::.fid_boot(p$O, p$Oh, 99)
  expect_equal(dim(bm), c(99L, length(e2tree:::.fid_methods) + 1L))
  expect_true(all(is.na(bm[, "SSIM"])))
  li <- which(lower.tri(p$O))
  obs <- mean(e2tree:::.loi_terms(p$O[li], p$Oh[li]))
  ci <- stats::quantile(bm[, "nLoI"], c(0.025, 0.975))
  expect_true(obs > ci[1] && obs < ci[2])
})

test_that("loi_perm returns a bootstrap CI only when requested", {
  p <- make_pair()
  r0 <- loi_perm(p$O, p$Oh, n_perm = 49, seed = 1)
  expect_true(all(is.na(r0$ci)))
  r1 <- loi_perm(p$O, p$Oh, n_perm = 49, seed = 1, n_boot = 49)
  expect_true(all(is.finite(r1$ci)) && r1$ci[1] <= r1$ci[2])
  expect_equal(r1$exact_null_mean, loi(p$O, p$Oh)$null_mean)
})

test_that("loi uses the partition when leaves are supplied", {
  O <- matrix(c(1, .5, .2, .5, 1, .3, .2, .3, 1), 3)
  Oh <- matrix(0, 3, 3); diag(Oh) <- 1          # leaf {1,2} with zero weight
  r0 <- loi(O, Oh)
  r1 <- loi(O, Oh, leaves = c(1, 1, 2))
  expect_equal(r0$loi, r1$loi)
  expect_equal(r1$n_within, 1)
  expect_equal(r0$n_within, 0)
  expect_equal(r1$loi_in, 0.5)
})

test_that("loi rejects a partition with missing values", {
  p <- make_pair(n = 10)
  expect_error(loi(p$O, p$Oh, leaves = c(NA, rep(1, 9))), "no missing values")
  expect_error(loi(p$O, p$Oh, leaves = 1:3), "length nrow")
})

test_that("loi_perm: no empty CI in the output, leaves passed to the decomposition", {
  p <- make_pair()
  r0 <- loi_perm(p$O, p$Oh, n_perm = 49, seed = 1)
  out0 <- capture.output(print(r0))
  expect_true(any(grepl("not computed", out0)))
  expect_false(any(grepl("NA", out0)))
  expect_equal(r0$n_boot, 0)

  r1 <- loi_perm(p$O, p$Oh, n_perm = 49, seed = 1, n_boot = 49)
  expect_true(any(grepl("bootstrap CI", capture.output(print(r1)))))

  # zero-weight leaf: the decomposition follows the partition, the test does not
  O  <- matrix(c(1, .5, .2, .5, 1, .3, .2, .3, 1), 3)
  Oh <- matrix(0, 3, 3); diag(Oh) <- 1
  a <- loi_perm(O, Oh, n_perm = 19, seed = 2)
  b <- loi_perm(O, Oh, n_perm = 19, seed = 2, leaves = c(1, 1, 2))
  expect_equal(b$observed$n_within, 1)
  expect_equal(a$observed$n_within, 0)
  expect_equal(a$null_dist, b$null_dist)
  expect_equal(a$p.value, b$p.value)
})

test_that("loi_perm null distribution is the nLoI of the permuted matrices", {
  p <- make_pair(n = 30)
  r <- loi_perm(p$O, p$Oh, n_perm = 499, seed = 3)
  expect_equal(r$statistic, loi(p$O, p$Oh)$nloi)
  expect_true(all(is.finite(r$null_dist)))
  expect_equal(mean(r$null_dist), r$exact_null_mean, tolerance = 0.02)
})
