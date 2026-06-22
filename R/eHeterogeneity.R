utils::globalVariables(c("node_lab"))

#' Outcome Heterogeneity per E2Tree Region
#'
#' Describes how heterogeneous the ensemble's outcomes are \emph{within each
#' reconstructed region} (terminal node) of an E2Tree. This is a purely
#' \strong{descriptive, interpretive} diagnostic: it characterises how decisive
#' or ambiguous the ensemble's behaviour is inside each group of observations it
#' treated as similar. It makes no prediction and offers no coverage guarantee.
#'
#' @details
#' Each terminal node collects the training observations the ensemble routed to
#' the same region (\code{fit$tree$obs}); their responses (\code{fit$y}) describe
#' the outcome distribution of that region. For region \eqn{k}:
#'
#' \strong{Regression} — the centre (\code{mean}), spread (\code{sd},
#' \code{iqr}) and a central band covering the middle \eqn{1-\alpha} of the
#' regional outcomes, \eqn{[\,q_{\alpha/2},\, q_{1-\alpha/2}\,]}, with its
#' \code{width}. A wide band marks a region where the ensemble groups together
#' observations with very different outcomes — an inherently uncertain part of
#' the ensemble's behaviour.
#'
#' \strong{Classification} — the regional class proportions summarised by the
#' \code{dominant} class and its \code{purity}, the normalised \code{entropy}
#' and \code{gini} impurity, and the set of \emph{non-negligible} classes
#' (proportion \eqn{\ge \alpha}) with its size (\code{set_size}). A region with
#' \code{set_size = 1} is one the ensemble treats decisively; \code{set_size > 1}
#' marks an ambiguous region.
#'
#' This complements \code{\link{localLoI}}: \code{localLoI} measures how
#' \emph{faithfully} the tree reconstructs a region, whereas
#' \code{eHeterogeneity} measures how \emph{decisive} the ensemble itself is
#' within that region. A region can be faithfully reconstructed yet
#' intrinsically heterogeneous, or vice versa.
#'
#' @param fit An \code{e2tree} object (must carry \code{$y}).
#' @param alpha Tail level for the descriptive summary. For regression it sets
#'   the central band to the middle \eqn{1-\alpha} of outcomes; for
#'   classification it is the proportion threshold below which a class is
#'   considered negligible. Default \code{0.1}.
#'
#' @return An object of class \code{"e2heterogeneity"} with a \code{node} data
#'   frame (one row per terminal node), the fields \code{is_class},
#'   \code{alpha}, and (classification only) a \code{proportions} matrix of
#'   per-region class proportions used by the \code{"composition"} plot view.
#'
#' @seealso \code{\link{localLoI}} for reconstruction fidelity per region,
#'   \code{\link{eNeighbors}}, \code{\link{eContribution}}.
#'
#' @examples
#' \donttest{
#' data(iris)
#' ensemble <- randomForest::randomForest(Species ~ ., data = iris,
#'   importance = TRUE, proximity = TRUE)
#' D <- createDisMatrix(ensemble, data = iris, label = "Species",
#'   parallel = list(active = FALSE, no_cores = 1))
#' setting <- list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5)
#' tree <- e2tree(Species ~ ., iris, D, ensemble, setting)
#'
#' h <- eHeterogeneity(tree)
#' print(h)
#' plot(h)                          # "entropy": per-region heterogeneity bars
#' plot(h, type = "composition")    # per-region class mix
#' }
#'
#' @importFrom stats sd quantile IQR
#' @importFrom graphics barplot par
#' @export
eHeterogeneity <- function(fit, alpha = 0.1) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  if (alpha <= 0 || alpha >= 1) stop("'alpha' must be in (0, 1).")

  td   <- fit$tree
  term <- td[td$terminal == TRUE, , drop = FALSE]
  ylevels  <- attr(fit, "ylevels")
  is_class <- !is.null(ylevels) && length(ylevels) > 0
  y <- fit$y
  if (is.null(y)) stop("fit$y is missing; refit with a current version of e2tree().")

  rows  <- vector("list", nrow(term))
  props <- vector("list", nrow(term))      # per-region class proportions (classification)
  for (k in seq_len(nrow(term))) {
    idx <- suppressWarnings(as.integer(unlist(term$obs[k])))
    idx <- idx[!is.na(idx)]
    yk  <- y[idx]
    nk  <- length(idx)

    if (is_class) {
      p <- as.numeric(table(factor(yk, levels = ylevels))) / max(nk, 1L)
      names(p) <- ylevels
      props[[k]] <- p
      pp  <- p[p > 0]
      ent <- if (length(pp) <= 1L) 0 else -sum(pp * log(pp)) / log(length(ylevels))
      amb <- ylevels[p >= alpha]
      if (length(amb) == 0L) amb <- ylevels[which.max(p)]
      rows[[k]] <- data.frame(
        node = term$node[k], n = nk,
        dominant = ylevels[which.max(p)], purity = max(p),
        entropy = ent, gini = 1 - sum(p^2),
        ambiguous_set = paste(amb, collapse = ","), set_size = length(amb),
        stringsAsFactors = FALSE)
    } else {
      qb <- if (nk >= 1L) stats::quantile(yk, c(alpha / 2, 1 - alpha / 2),
                                          names = FALSE, na.rm = TRUE) else c(NA, NA)
      rows[[k]] <- data.frame(
        node = term$node[k], n = nk,
        mean = mean(yk), sd = if (nk > 1L) stats::sd(yk) else 0,
        lower = qb[1], upper = qb[2], width = qb[2] - qb[1],
        iqr = if (nk > 1L) stats::IQR(yk) else 0,
        stringsAsFactors = FALSE)
    }
  }

  node <- do.call(rbind, rows)
  row.names(node) <- NULL
  res <- list(node = node, is_class = is_class, alpha = alpha, ylevels = ylevels)
  if (is_class) {
    proportions <- do.call(rbind, props)
    dimnames(proportions) <- list(node$node, ylevels)
    res$proportions <- proportions
  }
  class(res) <- "e2heterogeneity"
  res
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print e2heterogeneity
#' @export
print.e2heterogeneity <- function(x, digits = 3, ...) {
  cat("\n")
  cat("  E2Tree Outcome Heterogeneity per region (descriptive)\n")
  cat("  ------------------------------------------------------------\n")
  cat(sprintf("  Task: %s   |   regions: %d   |   alpha: %.2f\n\n",
              if (x$is_class) "Classification" else "Regression",
              nrow(x$node), x$alpha))

  if (x$is_class) {
    nd <- x$node[order(-x$node$entropy), , drop = FALSE]
    cat(sprintf("  %-6s %-5s %-14s %-8s %-8s %-8s\n",
                "Node", "n", "dominant", "purity", "entropy", "set_sz"))
    for (i in seq_len(nrow(nd)))
      cat(sprintf("  %-6s %-5d %-14s %-8.*f %-8.*f %-8d\n",
                  nd$node[i], nd$n[i], substr(nd$dominant[i], 1, 14),
                  digits, nd$purity[i], digits, nd$entropy[i], nd$set_size[i]))
    cat("\n  Most ambiguous regions sit at the top (highest entropy / set_size > 1):\n")
    cat("  there the ensemble does not commit to a single class.\n")
  } else {
    nd <- x$node[order(-x$node$width), , drop = FALSE]
    cat(sprintf("  %-6s %-5s %-10s %-10s %-18s %-8s\n",
                "Node", "n", "mean", "sd", "central band", "width"))
    for (i in seq_len(nrow(nd)))
      cat(sprintf("  %-6s %-5d %-10.*f %-10.*f [%6.*f, %6.*f] %-8.*f\n",
                  nd$node[i], nd$n[i], digits, nd$mean[i], digits, nd$sd[i],
                  digits, nd$lower[i], digits, nd$upper[i], digits, nd$width[i]))
    cat("\n  Widest-band regions sit at the top: there the ensemble groups\n")
    cat("  together observations with very different outcomes.\n")
  }
  cat("\n")
  invisible(x)
}


# ===========================================================================
# PLOT METHOD -- regional heterogeneity views selected by `type`
#   classification: "entropy" (default bars) | "composition" (class mix)
#   regression:     "spread"  (default band width) | "interval" (central band)
# ===========================================================================

#' @method plot e2heterogeneity
#' @export
plot.e2heterogeneity <- function(x, type = NULL, ...) {
  if (x$is_class) {
    if (is.null(type)) type <- "entropy"
    type <- match.arg(type, c("entropy", "composition"))
    switch(type,
      entropy     = .e2het_entropy(x),
      composition = .e2het_composition(x))
  } else {
    if (is.null(type)) type <- "spread"
    type <- match.arg(type, c("spread", "interval"))
    switch(type,
      spread   = .e2het_spread(x),
      interval = .e2het_interval(x))
  }
}

.e2het_theme <- function() {
  ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle    = ggplot2::element_text(colour = "grey40", size = 10),
      panel.grid.minor = ggplot2::element_blank(),
      axis.text.y      = ggplot2::element_text(size = 8))
}

# ---- classification: normalised-entropy bars (regions sorted) --------------
.e2het_entropy <- function(x) {
  nd <- x$node
  nd$node <- factor(nd$node, levels = nd$node[order(nd$entropy)])
  ggplot2::ggplot(nd, ggplot2::aes(x = entropy, y = node, fill = entropy)) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::scale_fill_gradient(low = "#f1c40f", high = "#c0392b",
                                 limits = c(0, 1), name = "entropy") +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::labs(title = "Regional outcome heterogeneity",
                  subtitle = "normalised entropy (0 = pure, 1 = uniform)",
                  x = "entropy", y = "region (terminal node)") +
    .e2het_theme()
}

# ---- classification: class composition per region --------------------------
.e2het_composition <- function(x) {
  if (is.null(x$proportions))
    stop("composition view needs class proportions; refit with a current eHeterogeneity().")
  P  <- x$proportions
  ord <- rownames(P)[order(x$node$entropy[match(rownames(P), x$node$node)])]
  long <- data.frame(
    region = factor(rep(rownames(P), times = ncol(P)), levels = ord),
    class  = factor(rep(colnames(P), each = nrow(P)), levels = colnames(P)),
    prop   = as.numeric(P), stringsAsFactors = FALSE)
  ggplot2::ggplot(long, ggplot2::aes(x = prop, y = region, fill = class)) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::scale_fill_viridis_d(name = "class", end = 0.9) +
    ggplot2::scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    ggplot2::labs(title = "Regional class composition",
                  subtitle = "outcome mix per region (regions sorted by entropy)",
                  x = "proportion", y = "region (terminal node)") +
    .e2het_theme()
}

# ---- regression: central-band width bars -----------------------------------
.e2het_spread <- function(x) {
  nd <- x$node
  nd$node <- factor(nd$node, levels = nd$node[order(nd$width)])
  ggplot2::ggplot(nd, ggplot2::aes(x = width, y = node, fill = width)) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::scale_fill_gradient(low = "#f1c40f", high = "#c0392b", name = "width") +
    ggplot2::labs(
      title = "Regional outcome spread",
      subtitle = sprintf("central %.0f%% band width", 100 * (1 - x$alpha)),
      x = "band width", y = "region (terminal node)") +
    .e2het_theme()
}

# ---- regression: central interval per region -------------------------------
.e2het_interval <- function(x) {
  nd <- x$node
  nd$node <- factor(nd$node, levels = nd$node[order(nd$mean)])
  ggplot2::ggplot(nd, ggplot2::aes(x = mean, y = node)) +
    ggplot2::geom_segment(ggplot2::aes(x = lower, xend = upper, y = node, yend = node),
                          colour = "grey70", linewidth = 0.5) +
    ggplot2::geom_point(colour = "#c0392b", size = 1.8) +
    ggplot2::labs(
      title = "Regional outcome interval",
      subtitle = sprintf("region mean and central %.0f%% band", 100 * (1 - x$alpha)),
      x = "outcome", y = "region (terminal node)") +
    .e2het_theme()
}
