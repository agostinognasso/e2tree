# ===========================================================================
# Internal computational core for reconstruction-fidelity measures
#
# All element-wise measures are computed on the vectors of the M = n(n-1)/2
# off-diagonal pairs (lower triangle). Only SSIM needs the full matrices,
# because it is a windowed (order-dependent) measure.
# ===========================================================================

.fid_methods <- c("nLoI", "sqrt_nLoI", "Hellinger", "wRMSE", "RV", "SSIM", "Mantel")
.fid_is_div  <- c(nLoI = TRUE, sqrt_nLoI = TRUE, Hellinger = TRUE, wRMSE = TRUE,
                  RV = FALSE, SSIM = FALSE, Mantel = FALSE)

# Pairwise LoI contribution h(a, b) = (a - b)^2 / max(a, b), with 0/0 = 0
.loi_terms <- function(o, oh) {
  den <- pmax(o, oh)
  out <- numeric(length(o))
  pos <- den > 0
  out[pos] <- (o[pos] - oh[pos])^2 / den[pos]
  out
}

.fid_nloi <- function(o, oh) mean(.loi_terms(o, oh))

.fid_hellinger <- function(o, oh) sqrt(mean((sqrt(o) - sqrt(oh))^2))

.fid_wrmse <- function(o, oh, epsilon = .Machine$double.eps) {
  w <- pmax(o, oh, epsilon)
  sqrt(sum(w * (o - oh)^2) / sum(w))
}

# Uncentred RV (congruence) coefficient on the off-diagonal entries.
# Because both matrices are symmetric with zeroed diagonal,
# tr(Oc Ohc) = 2 * sum_{i<j} o_ij oh_ij, so the factor 2 cancels.
.fid_rv <- function(o, oh) {
  den <- sqrt(sum(o * o) * sum(oh * oh))
  if (den < .Machine$double.eps) return(0)
  sum(o * oh) / den
}

# Mantel statistic: Pearson correlation of the off-diagonal entries
.fid_mantel <- function(o, oh) {
  if (stats::sd(o) == 0 || stats::sd(oh) == 0) return(0)
  stats::cor(o, oh)
}

# Exact permutation mean of the nLoI under simultaneous row/column
# permutation of O_hat. Under a uniform random permutation the image of an
# unordered pair is uniform over the M unordered pairs, hence
#   E_pi[nLoI] = (1/M^2) sum_a sum_b h(o_a, oh_b),
# which only requires the distinct values of oh (few for a tree).
.nloi_null_mean <- function(o, oh, max_unique = 5000L) {
  tab <- table(oh)
  if (length(tab) > max_unique) return(NA_real_)
  vals <- as.numeric(names(tab))
  wts  <- as.numeric(tab) / length(oh)
  sum(vapply(seq_along(vals),
             function(k) wts[k] * mean(.loi_terms(o, rep(vals[k], length(o)))),
             numeric(1)))
}

# Chance-corrected fidelity (kappa-type): 1 = perfect agreement,
# 0 = no better than a random relabelling of the observations.
.fid_kappa <- function(obs, null_mean, is_div) {
  if (is.na(null_mean)) return(NA_real_)
  if (is_div) {
    if (null_mean <= 0) return(NA_real_)
    1 - obs / null_mean
  } else {
    if (null_mean >= 1) return(NA_real_)
    (obs - null_mean) / (1 - null_mean)
  }
}

# All measures for one pair of matrices.
# o, oh: off-diagonal vectors; O, Oh: matrices (needed only for SSIM)
.fid_all <- function(o, oh, O = NULL, Oh = NULL, ssim = TRUE, ssim_prep = NULL) {
  nl <- .fid_nloi(o, oh)
  ssim_val <- NA_real_
  if (ssim && !is.null(Oh)) {
    if (is.null(ssim_prep)) ssim_prep <- .ssim_prep(O)
    ssim_val <- .ssim_eval(ssim_prep, Oh)
  }
  c(nLoI      = nl,
    sqrt_nLoI = sqrt(nl),
    Hellinger = .fid_hellinger(o, oh),
    wRMSE     = .fid_wrmse(o, oh),
    RV        = .fid_rv(o, oh),
    SSIM      = ssim_val,
    Mantel    = .fid_mantel(o, oh))
}

# Matrix transformation applied before computing the measures
.fid_transform <- function(M, transform = c("none", "sqrt")) {
  transform <- match.arg(transform)
  M <- pmin(pmax(M, 0), 1)
  if (transform == "sqrt") sqrt(M) else M
}

# ---------------------------------------------------------------------------
# Unified row/column permutation test.
# A single set of permutations is shared by all measures; the permuted
# off-diagonal vector is extracted once and reused by every element-wise
# measure (the only shared work: the asymptotic cost is unchanged).
# ---------------------------------------------------------------------------
.fid_perm <- function(O, Oh, n_perm, ssim = TRUE) {
  n  <- nrow(O)
  li <- which(lower.tri(O))
  o  <- O[li]
  sp <- if (ssim) .ssim_prep(O) else NULL   # O-only SSIM terms, computed once
  one <- function(dummy) {
    p   <- sample.int(n)
    Ohp <- Oh[p, p]
    .fid_all(o, Ohp[li], O, Ohp, ssim = ssim, ssim_prep = sp)
  }
  if (requireNamespace("future.apply", quietly = TRUE)) {
    future.apply::future_sapply(seq_len(n_perm), one, future.seed = TRUE)
  } else {
    vapply(seq_len(n_perm), one, numeric(length(.fid_methods)))
  }
}

# ---------------------------------------------------------------------------
# Observation-level (pairs) bootstrap.
# Observations are resampled with replacement; the statistic is recomputed
# on the off-diagonal pairs of the resampled matrices, excluding pairs formed
# by two copies of the same observation (U-statistic bootstrap).
# Conditional on the fitted ensemble and E2Tree, this quantifies the sampling
# variability of the population pairwise discrepancy. SSIM is excluded
# because it depends on the ordering of the observations.
# Returns an n_boot x (measures + kappa_nLoI) matrix.
# ---------------------------------------------------------------------------
.fid_boot <- function(O, Oh, n_boot) {
  n  <- nrow(O)
  lt <- lower.tri(O)
  rr <- row(O)[lt]
  cc <- col(O)[lt]
  one <- function(dummy) {
    b    <- sample.int(n, n, replace = TRUE)
    keep <- b[rr] != b[cc]
    ib   <- (b[cc] - 1L) * n + b[rr]
    o    <- O[ib[keep]]
    oh   <- Oh[ib[keep]]
    v    <- .fid_all(o, oh, ssim = FALSE)
    c(v, kappa_nLoI = .fid_kappa(v[["nLoI"]], .nloi_null_mean(o, oh), TRUE))
  }
  if (requireNamespace("future.apply", quietly = TRUE)) {
    t(future.apply::future_sapply(seq_len(n_boot), one, future.seed = TRUE))
  } else {
    t(vapply(seq_len(n_boot), one, numeric(length(.fid_methods) + 1L)))
  }
}
