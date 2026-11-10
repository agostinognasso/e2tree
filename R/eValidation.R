utils::globalVariables("tree") # to avoid CRAN check errors for tidyverse programming

#' Validate an E2Tree Model via Proximity Matrix Comparison
#'
#' Compares the ensemble proximity matrix \eqn{O} with the E2Tree proximity
#' matrix \eqn{\hat{O}} through a family of agreement measures (nLoI and its
#' metric version \eqn{\sqrt{nLoI}}, Hellinger distance, wRMSE, RV
#' (congruence) coefficient, SSIM) and the Mantel correlation as an
#' association baseline. Three complementary outputs are returned:
#' \enumerate{
#'   \item \strong{Point estimates} of each measure and the chance-corrected
#'     fidelity \eqn{\kappa = 1 - \tau/\mu_0} (divergences) or
#'     \eqn{\kappa = (\tau - \mu_0)/(1 - \mu_0)} (similarities), where
#'     \eqn{\mu_0} is the mean under random relabelling (exact, closed form,
#'     for the nLoI).
#'   \item A \strong{unified row/column permutation test} of the null
#'     hypothesis of no association between the two structures (a minimal
#'     requirement, not evidence of high fidelity).
#'   \item Optional \strong{bootstrap confidence intervals} (observation-level
#'     resampling, conditional on the fitted models) and an optional
#'     \strong{equivalence test} \eqn{H_0: nLoI \ge \delta}.
#' }
#'
#' @param data A data frame containing the variables in the model.
#' @param fit An e2tree object.
#' @param D The dissimilarity matrix obtained with \code{\link{createDisMatrix}}.
#' @param test Character string specifying which tests to perform. One of
#'   \code{"both"} (default), \code{"mantel"} (Mantel test only), or
#'   \code{"measures"} (agreement measures only).
#' @param graph Logical (default TRUE). If TRUE, heatmaps are displayed.
#' @param n_perm Integer. Number of permutations. Default 999. Set to 0 to
#'   skip the permutation test.
#' @param conf.level Numeric. Confidence level for bootstrap intervals and
#'   the equivalence test. Default 0.95.
#' @param seed Integer or NULL. Random seed for reproducibility.
#' @param transform Character. Scale on which the measures are computed:
#'   \code{"sqrt"} (default, as in e2tree versions up to 1.0.0.9000) uses
#'   \eqn{\sqrt{O}} and \eqn{\sqrt{\hat{O}}}; \code{"none"} uses the
#'   proximities \eqn{O} and \eqn{\hat{O}} directly, the scale on which the
#'   measures are defined in Aria et al. (nLoI paper).
#' @param n_boot Integer. Number of bootstrap replicates for confidence
#'   intervals. Default 0 (no bootstrap).
#' @param delta Numeric or NULL. Equivalence margin for the nLoI. If given
#'   (and \code{n_boot > 0}), fidelity is declared when the one-sided upper
#'   bootstrap bound of the nLoI is below \code{delta}.
#'
#' @return An object of class \code{"eValidation"} containing:
#'   \describe{
#'     \item{Proximity_matrix_ensemble}{Ensemble proximity matrix (reordered, transformed)}
#'     \item{Proximity_matrix_e2tree}{E2Tree proximity matrix (reordered, transformed)}
#'     \item{mantel_test}{Mantel test result (NULL if \code{test = "measures"})}
#'     \item{loi}{LoI object with decomposition (NULL if \code{test = "mantel"})}
#'     \item{measures}{Data frame with observed values, null moments, Z, p-values,
#'       kappa and bootstrap intervals}
#'     \item{permutation}{Permutation results (null distributions)}
#'     \item{bootstrap}{Bootstrap replicates (if \code{n_boot > 0})}
#'     \item{equivalence}{Equivalence test for the nLoI (if \code{delta} is given)}
#'   }
#'
#' @details
#' The nLoI is a symmetric Csiszar f-divergence with generator
#' \eqn{f(t) = (t-1)^2/\max(1,t)}; it is not a metric, whereas
#' \eqn{\sqrt{nLoI}} satisfies the triangle inequality. Being a monotone
#' transformation, \eqn{\sqrt{nLoI}} has the same permutation p-value as the
#' nLoI. SSIM is computed under the canonical ordering given by hierarchical
#' clustering of \code{D}; unlike the other measures it is not invariant to
#' the ordering of the observations and is excluded from the bootstrap.
#'
#' @examples
#' \donttest{
#' ## Classification:
#' data(iris)
#' smp_size <- floor(0.75 * nrow(iris))
#' train_ind <- sample(seq_len(nrow(iris)), size = smp_size)
#' training <- iris[train_ind, ]
#'
#' ensemble <- randomForest::randomForest(Species ~ ., data=training,
#'   importance=TRUE, proximity=TRUE)
#'
#' D <- createDisMatrix(ensemble, data=training, label = "Species",
#'   parallel = list(active=FALSE, no_cores = 1))
#'
#' setting <- list(impTotal=0.1, maxDec=0.01, n=2, level=5)
#' tree <- e2tree(Species ~ ., training, D, ensemble, setting)
#'
#' val <- eValidation(training, tree, D, n_perm = 199, n_boot = 199,
#'                    delta = 0.1)
#' print(val)
#' summary(val)
#' plot(val)
#' }
#'
#' @export
eValidation <- function(data, fit, D, test = c("both", "mantel", "measures"),
                        graph = TRUE,
                        n_perm = 999, conf.level = 0.95, seed = NULL,
                        transform = c("sqrt", "none"),
                        n_boot = 0, delta = NULL) {

  if (!is.null(seed)) set.seed(seed)
  test <- match.arg(test)
  transform <- match.arg(transform)

  # === Input Validation ===
  if (!is.data.frame(data) || nrow(data) == 0) {
    stop("Error: 'data' must be a non-empty data frame.")
  }
  if (!inherits(fit, "e2tree")) {
    stop("Error: 'fit' must be an 'e2tree' object.")
  }
  if (!is.matrix(D) || nrow(D) != ncol(D)) {
    stop("Error: 'D' must be a square dissimilarity matrix.")
  }
  if (nrow(data) != nrow(D)) {
    stop("Error: The number of rows in 'data' must match the dimensions of 'D'.")
  }
  if (conf.level <= 0 || conf.level >= 1) {
    stop("conf.level must be between 0 and 1")
  }
  if (!is.null(delta) && (!is.numeric(delta) || length(delta) != 1 || delta <= 0)) {
    stop("delta must be a single positive number")
  }

  n <- nrow(data)
  df <- fit$tree
  terminal_nodes <- df$node[df$terminal]

  # === Build E2Tree proximity matrix Ps ===
  # Leaf weights W_t: correct-classification rate (column `prob`) in
  # classification, normalised-variance weight (column `Wt`) in regression.
  # (In regression `prob` stores the residual sum of squares, not a weight.)
  wcol <- if (!is.null(df$Wt)) "Wt" else "prob"
  Ps <- matrix(0, n, n)
  leaf <- integer(n)
  for (i in terminal_nodes) {
    obs <- unlist(df$obs[df$node == i])
    Ps[obs, obs] <- df[[wcol]][df$node == i]
    leaf[obs] <- i
  }
  diag(Ps) <- 1
  rownames(Ps) <- 1:n
  colnames(Ps) <- 1:n

  # === Canonical ordering by hierarchical clustering of D ===
  # (irrelevant for all measures except the windowed SSIM)
  clusD <- hclust(as.dist(D))
  ord <- clusD$order

  Ps_ord <- Ps[ord, ord]
  rownames(Ps_ord) <- ord
  colnames(Ps_ord) <- ord

  # === Mantel test ===
  mantel_test <- NULL
  if (test %in% c("both", "mantel")) {
    mantel_test <- ape::mantel.test(
      Ps_ord,
      1 - D[ord, ord],
      graph = graph,
      main = "Mantel test",
      xlab = "z-statistic", ylab = "Density"
    )
  }

  # === Proximity matrices on the requested scale ===
  prox_e2tree <- .fid_transform(Ps_ord, transform)
  prox_ens    <- .fid_transform(1 - D[ord, ord], transform)

  if (graph) {
    e2heatmap(prox_e2tree)
    e2heatmap(prox_ens)
  }

  loi_result <- NULL
  measures_df <- NULL
  perm_result <- NULL
  boot_result <- NULL
  equivalence <- NULL

  if (test %in% c("both", "measures")) {
    O <- prox_ens
    O_hat <- prox_e2tree
    li <- which(lower.tri(O))
    o <- O[li]
    oh <- O_hat[li]

    loi_result <- loi(O, O_hat, leaves = leaf[ord])
    obs_vals <- .fid_all(o, oh, O, O_hat)
    is_div <- .fid_is_div[.fid_methods]

    measures_df <- data.frame(
      method   = .fid_methods,
      type     = ifelse(is_div, "divergence", "similarity"),
      observed = unname(obs_vals),
      stringsAsFactors = FALSE
    )

    # Null means: exact for nLoI, permutation-based for the others
    null_means <- rep(NA_real_, length(.fid_methods))
    names(null_means) <- .fid_methods
    null_means["nLoI"] <- .nloi_null_mean(o, oh)

    if (n_perm > 0) {
      perm_mat <- .fid_perm(O, O_hat, n_perm)
      rownames(perm_mat) <- .fid_methods
      pm <- rowMeans(perm_mat, na.rm = TRUE)
      null_means[is.na(null_means)] <- pm[is.na(null_means)]
      null_sds <- apply(perm_mat, 1, sd, na.rm = TRUE)
      p_values <- vapply(seq_along(.fid_methods), function(k) {
        if (is_div[k]) {
          (1 + sum(perm_mat[k, ] <= obs_vals[k], na.rm = TRUE)) / (1 + n_perm)
        } else {
          (1 + sum(perm_mat[k, ] >= obs_vals[k], na.rm = TRUE)) / (1 + n_perm)
        }
      }, numeric(1))
      measures_df$null_mean <- unname(null_means)
      measures_df$null_sd   <- unname(null_sds)
      measures_df$z_stat    <- unname((obs_vals - null_means) / null_sds)
      measures_df$p_value   <- p_values
      perm_result <- list(null_dists = perm_mat, null_means = null_means,
                          null_sds = null_sds, p_values = p_values,
                          method_names = .fid_methods)
    } else {
      measures_df$null_mean <- unname(null_means)
    }

    measures_df$kappa <- vapply(seq_along(.fid_methods), function(k)
      .fid_kappa(obs_vals[k], null_means[k], is_div[k]), numeric(1))

    # === Bootstrap confidence intervals ===
    if (n_boot > 0) {
      bm <- .fid_boot(O, O_hat, n_boot)
      alpha <- 1 - conf.level
      qs <- apply(bm, 2, quantile, probs = c(alpha / 2, 1 - alpha / 2),
                  na.rm = TRUE, names = FALSE)
      measures_df$ci_lower <- unname(qs[1, .fid_methods])
      measures_df$ci_upper <- unname(qs[2, .fid_methods])
      measures_df$ci_lower[measures_df$method == "SSIM"] <- NA
      measures_df$ci_upper[measures_df$method == "SSIM"] <- NA
      kappa_ci <- qs[, "kappa_nLoI"]
      boot_result <- list(replicates = bm, kappa_nLoI_ci = kappa_ci,
                          n_boot = n_boot)

      if (!is.null(delta)) {
        upper <- quantile(bm[, "nLoI"], probs = conf.level, na.rm = TRUE,
                          names = FALSE)
        equivalence <- list(delta = delta, upper_bound = upper,
                            conf.level = conf.level,
                            fidelity = upper < delta)
      }
    } else if (!is.null(delta)) {
      warning("delta is ignored when n_boot = 0")
    }
  }

  results <- list(
    Proximity_matrix_ensemble = prox_ens,
    Proximity_matrix_e2tree   = prox_e2tree,
    mantel_test               = mantel_test,
    loi                       = loi_result,
    measures                  = measures_df,
    permutation               = perm_result,
    bootstrap                 = boot_result,
    equivalence               = equivalence,
    n                         = n,
    n_perm                    = n_perm,
    n_boot                    = n_boot,
    conf.level                = conf.level,
    transform                 = transform,
    test                      = test
  )
  class(results) <- "eValidation"
  results
}


# ===========================================================================
# INTERNAL: SSIM
# ===========================================================================
.e2val_ssim <- function(O, O_hat, window_size = NULL) {
  .ssim_eval(.ssim_prep(O, window_size), O_hat)
}

# SSIM is split in two steps so that the quantities depending only on O
# (local means and variances) are computed once and reused across
# permutations. Only the interior windows enter the average, so local sums
# are computed on that region with vectorised cumulative sums (identical to
# the zero-padded box filter of earlier versions on the interior).
.ssim_prep <- function(O, window_size = NULL) {
  n <- nrow(O)
  if (is.null(window_size)) {
    window_size <- min(11L, max(3L, floor(n / 3)))
  }
  w <- as.integer(window_size)
  if (w %% 2 == 0) w <- w + 1L
  prep <- list(n = n, w = w, k = w * w, C1 = (0.01)^2, C2 = (0.03)^2, O = O,
               fallback = n < w)
  if (prep$fallback) return(prep)
  nv <- n - w + 1L
  k <- prep$k
  # interior window sums via vectorised column cumulative sums
  win_cols <- function(M) {
    cs <- matrix(cumsum(M), n)
    if (n > 1) cs <- cs - rep(c(0, cs[n, -ncol(cs)]), each = n)
    cs <- rbind(0, cs)
    cs[(w + 1):(n + 1), , drop = FALSE] - cs[1:nv, , drop = FALSE]
  }
  box <- function(M) t(win_cols(t(win_cols(M)))) / k
  prep$box <- box
  prep$mu_O <- box(O)
  prep$s_O2 <- pmax(box(O * O) - prep$mu_O^2, 0)
  prep
}

.ssim_eval <- function(prep, O_hat) {
  C1 <- prep$C1; C2 <- prep$C2; O <- prep$O
  if (prep$fallback) {
    o_vec <- O[lower.tri(O)]
    oh_vec <- O_hat[lower.tri(O_hat)]
    mu_o <- mean(o_vec); mu_oh <- mean(oh_vec)
    s_o2 <- var(o_vec); s_oh2 <- var(oh_vec)
    s_cross <- cov(o_vec, oh_vec)
    return(((2 * mu_o * mu_oh + C1) * (2 * s_cross + C2)) /
             ((mu_o^2 + mu_oh^2 + C1) * (s_o2 + s_oh2 + C2)))
  }
  mu_O <- prep$mu_O
  mu_Oh <- prep$box(O_hat)
  s_Oh2 <- pmax(prep$box(O_hat * O_hat) - mu_Oh^2, 0)
  s_OOh <- prep$box(O * O_hat) - mu_O * mu_Oh
  num <- (2 * mu_O * mu_Oh + C1) * (2 * s_OOh + C2)
  den <- (mu_O^2 + mu_Oh^2 + C1) * (prep$s_O2 + s_Oh2 + C2)
  mean(num / den, na.rm = TRUE)
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print eValidation
#' @export
print.eValidation <- function(x, digits = 4, ...) {

  rule <- paste0("  ", strrep("-", 92), "\n")
  cat("\n  E2Tree Validation\n")
  cat(sprintf("  Matrix dimension: %d x %d   Pairs: %d   Scale: %s\n",
              x$n, x$n, as.integer(x$n * (x$n - 1) / 2),
              if (identical(x$transform, "sqrt")) "sqrt(O)" else "O"))

  if (!is.null(x$mantel_test)) {
    cat(sprintf("  Mantel test (ape): z = %.2f, p = %.4f\n",
                x$mantel_test$z.stat, x$mantel_test$p))
  }

  if (!is.null(x$measures)) {
    tbl <- x$measures
    has_p <- !is.null(tbl$p_value)
    has_ci <- !is.null(tbl$ci_lower)
    fmt <- function(v) if (is.na(v)) sprintf("%9s", "--") else sprintf("%9.*f", digits, v)

    cat("\n", rule, sep = "")
    hdr <- sprintf("  %-10s %-5s %9s %9s %8s", "Measure", "", "Observed", "Null mean", "kappa")
    if (has_p) hdr <- paste0(hdr, sprintf(" %9s %9s", "Z", "p-value"))
    if (has_ci) hdr <- paste0(hdr, sprintf("  %s", sprintf("%d%% bootstrap CI", round(x$conf.level * 100))))
    cat(hdr, "\n", rule, sep = "")

    for (i in seq_len(nrow(tbl))) {
      dir <- if (tbl$type[i] == "divergence") "[div]" else "[sim]"
      line <- sprintf("  %-10s %-5s %s %s %s", tbl$method[i], dir,
                      fmt(tbl$observed[i]), fmt(tbl$null_mean[i]),
                      if (is.na(tbl$kappa[i])) sprintf("%8s", "--") else sprintf("%8.3f", tbl$kappa[i]))
      if (has_p) {
        p_str <- if (tbl$p_value[i] < 0.001) "< 0.001" else sprintf("%.3f", tbl$p_value[i])
        line <- paste0(line, sprintf(" %+9.2f %9s", tbl$z_stat[i], p_str))
      }
      if (has_ci) {
        line <- paste0(line, if (is.na(tbl$ci_lower[i])) "  --" else
          sprintf("  [%.*f, %.*f]", digits, tbl$ci_lower[i], digits, tbl$ci_upper[i]))
      }
      cat(line, "\n")
    }
    cat(rule)

    if (has_p) cat(sprintf("  Permutations: %d (row/column, shared by all measures)\n", x$n_perm))
    if (!is.null(x$bootstrap)) {
      kc <- x$bootstrap$kappa_nLoI_ci
      cat(sprintf("  Bootstrap: %d replicates (observations resampled); kappa(nLoI) CI [%.3f, %.3f]\n",
                  x$bootstrap$n_boot, kc[1], kc[2]))
    }
    if (!is.null(x$equivalence)) {
      eq <- x$equivalence
      cat(sprintf("  Equivalence test H0: nLoI >= %.4f -> upper %d%% bound = %.4f: %s\n",
                  eq$delta, round(eq$conf.level * 100), eq$upper_bound,
                  if (eq$fidelity) "fidelity established" else "fidelity NOT established"))
    }
    if (!is.null(x$loi)) {
      cat(sprintf("  LoI decomposition (per-pair averages): mean_in = %.4f, mean_out = %.4f\n",
                  x$loi$mean_in, x$loi$mean_out))
    }
    cat("\n  [div] lower = better; [sim] higher = better. kappa: 1 = perfect, 0 = random relabelling.\n")
    cat("  The permutation test assesses association only; fidelity is judged by the\n")
    cat("  observed values, kappa and their confidence intervals.\n\n")
  }

  invisible(x)
}


# ===========================================================================
# SUMMARY METHOD
# ===========================================================================

#' @method summary eValidation
#' @export
summary.eValidation <- function(object, digits = 4, ...) {

  print(object, digits = digits, ...)
  if (!is.null(object$loi)) {
    cat("\n")
    summary(object$loi, digits = digits)
  }

  invisible(object)
}


# ===========================================================================
# PLOT METHOD
# ===========================================================================

#' @method plot eValidation
#' @export
plot.eValidation <- function(x, ...) {

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))

  has_perm <- !is.null(x$permutation)
  has_measures <- !is.null(x$measures)

  # Determine layout based on available components
  n_panels <- 2L  # always show heatmaps
  if (has_measures && has_perm) n_panels <- 4L
  else if (has_measures) n_panels <- 3L

  if (n_panels == 4L) {
    layout(matrix(c(1, 2, 3, 4), 2, 2, byrow = TRUE))
  } else {
    par(mfrow = c(1, n_panels))
  }

  par(mar = c(4, 4, 3, 1))

  # --- Panel 1: Ensemble heatmap ---
  image(x$Proximity_matrix_ensemble,
        main = "Ensemble Proximity (O)",
        col = colorRampPalette(c("white", "black"))(100),
        axes = FALSE, xlab = "", ylab = "")

  # --- Panel 2: E2Tree heatmap ---
  image(x$Proximity_matrix_e2tree,
        main = expression("E2Tree Proximity (" * hat(O) * ")"),
        col = colorRampPalette(c("white", "black"))(100),
        axes = FALSE, xlab = "", ylab = "")

  if (has_measures && has_perm) {
    # --- Panel 3: Null distribution of nLoI ---
    null_nloi <- x$permutation$null_dists[1, ]
    obs_nloi <- x$measures$observed[1]

    xlim_range <- range(c(null_nloi, obs_nloi), na.rm = TRUE)
    h <- hist(null_nloi, plot = FALSE)
    d <- density(null_nloi, na.rm = TRUE)
    ylim_max <- max(max(h$density), max(d$y)) * 1.15

    hist(null_nloi,
         main = "Null Distribution of nLoI",
         xlab = "nLoI", col = "lightgray", border = "white",
         xlim = xlim_range, ylim = c(0, ylim_max), freq = FALSE)
    lines(d, col = "darkgray", lwd = 2)
    abline(v = obs_nloi, col = "red", lwd = 2)
    abline(v = mean(null_nloi), col = "darkgray", lwd = 1.5, lty = 2)

    p_val <- x$measures$p_value[1]
    p_str <- if (p_val < 0.0001) "p < 0.0001" else sprintf("p = %.4f", p_val)
    legend("topright",
           legend = c(sprintf("Observed (%.4f)", obs_nloi), p_str),
           col = c("red", NA), lty = c(1, NA), lwd = c(2, NA),
           bty = "n", cex = 0.85)
  }

  # --- LoI Decomposition (per-pair) ---
  if (has_measures && !is.null(x$loi)) {
    lo <- x$loi
    means <- c(lo$mean_in, lo$mean_out)
    nms <- c("Within\n(mean_in)", "Between\n(mean_out)")
    cols <- c("#3498db", "#e74c3c")

    bp <- barplot(means, names.arg = nms, col = cols, border = NA,
                  main = sprintf("Per-Pair Average Loss\nnLoI = %.4f", lo$nloi),
                  ylab = "Mean loss per pair",
                  ylim = c(0, max(means) * 1.35))
    text(bp, means, labels = sprintf("%.4f", means),
         pos = 3, cex = 0.9, font = 2)
  }
}


# ===========================================================================
# HEATMAP HELPER (internal)
# ===========================================================================

e2heatmap <- function(data_matrix) {
  heatmap(
    data_matrix,
    Rowv = NA,
    Colv = NA,
    scale = "none",
    col = colorRampPalette(c("white", "black"))(100)
  )
}
