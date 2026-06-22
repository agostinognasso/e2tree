#' Unified Per-Instance Local Explanation
#'
#' Composes E2Tree's local-explanation layer into a single, coherent per-instance
#' account. Rather than calling \code{\link{eContribution}},
#' \code{\link{eNeighbors}}, \code{\link{localLoI}}, \code{\link{eHeterogeneity}}
#' (and optionally \code{\link{eCounterfactual}} / \code{\link{eStability}})
#' separately, \code{explain()} assembles them into one object whose \code{print}
#' method reads as a narrative: where the ensemble routed the instance, why
#' (additive attribution), which cases it is grouped with, how reliable that
#' region is, how dispersed its outcome is, and -- on request -- what would move
#' it elsewhere and how stable the account is.
#'
#' @details
#' Every component is purely interpretive: the leaf value is the ensemble's
#' \emph{reconstructed} grouping outcome (never framed as a prediction), the
#' attribution decomposes that reconstruction, and the neighbours / prototypes
#' come from the ensemble's own leaf co-occurrence proximity. Components are
#' computed once over \code{newdata} and sliced per instance, so the cost is that
#' of the underlying functions.
#'
#' @param fit An \code{e2tree} object (must carry \code{$data} and \code{$y}).
#' @param ensemble The trained ensemble used to build \code{fit}.
#' @param newdata A data frame of instances to explain.
#' @param reliability Optional \code{\link{localLoI}} object; when supplied the
#'   destination region's fidelity is attached.
#' @param k Integer. Number of ensemble neighbours. Default \code{5}.
#' @param alpha Tail level passed to \code{\link{eHeterogeneity}}. Default \code{0.1}.
#' @param counterfactual Logical. Also compute \code{\link{eCounterfactual}}.
#'   Default \code{FALSE}.
#' @param stability Logical. Also compute \code{\link{eStability}}. Default
#'   \code{FALSE}.
#' @param B Integer. Resamples for \code{stability}. Default \code{100}.
#' @param ... Passed to \code{\link{eStability}} (e.g. \code{seed}).
#'
#' @return An object of class \code{"e2explanation"}:
#'   \item{leaf, predicted}{Destination terminal node and its reconstructed
#'     outcome per instance.}
#'   \item{contribution}{The \code{\link{eContribution}} object.}
#'   \item{neighbors}{The \code{\link{eNeighbors}} object.}
#'   \item{heterogeneity}{Per-instance destination-region dispersion summary.}
#'   \item{reliability}{(If supplied) per-instance destination-region fidelity.}
#'   \item{counterfactual, stability}{(If requested) the \code{e2counterfactual}
#'     / \code{e2stability} objects.}
#'   \item{is_class}{Logical.}
#'
#' @seealso The components: \code{\link{eContribution}}, \code{\link{eNeighbors}},
#'   \code{\link{localLoI}}, \code{\link{eHeterogeneity}},
#'   \code{\link{eCounterfactual}}, \code{\link{eStability}}.
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
#' ex <- explain(tree, ensemble, newdata = iris[c(1, 80, 120), ],
#'               counterfactual = TRUE)
#' print(ex)
#' }
#'
#' @export
explain <- function(fit, ensemble, newdata, reliability = NULL, k = 5,
                    alpha = 0.1, counterfactual = FALSE, stability = FALSE,
                    B = 100, ...) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  if (!is.data.frame(newdata) || nrow(newdata) == 0)
    stop("'newdata' must be a non-empty data frame.")
  if (!is.null(reliability) && !inherits(reliability, "localLoI"))
    stop("'reliability', when supplied, must be a localLoI object.")
  row.names(newdata) <- NULL

  ylevels  <- attr(fit, "ylevels")
  is_class <- !is.null(ylevels) && length(ylevels) > 0
  n_obs    <- nrow(newdata)

  # ---- Components (computed once) ------------------------------------------
  contribution <- eContribution(fit, newdata, reliability = reliability)
  neighbors    <- eNeighbors(fit, ensemble, query = newdata, k = k)
  het          <- eHeterogeneity(fit, alpha = alpha)

  leaf      <- contribution$leaf
  predicted <- contribution$predicted

  # destination-region heterogeneity, sliced per instance
  hrow <- het$node[match(leaf, het$node$node), , drop = FALSE]

  out <- list(
    leaf          = leaf,
    predicted     = predicted,
    contribution  = contribution,
    neighbors     = neighbors,
    heterogeneity = hrow,
    het_is_class  = het$is_class,
    newdata       = newdata,
    is_class      = is_class)

  if (!is.null(reliability) && !is.null(contribution$reliability))
    out$reliability <- contribution$reliability

  if (isTRUE(counterfactual))
    out$counterfactual <- eCounterfactual(fit, newdata, ensemble = ensemble)
  if (isTRUE(stability))
    out$stability <- eStability(fit, ensemble, newdata = newdata, B = B, k = k, ...)

  class(out) <- "e2explanation"
  out
}


# ===========================================================================
# PRINT METHOD -- one narrative card per instance
# ===========================================================================

#' @method print e2explanation
#' @export
print.e2explanation <- function(x, obs = NULL, top = 4, digits = 3, ...) {
  n_obs <- nrow(x$newdata)
  if (is.null(obs)) obs <- seq_len(min(n_obs, 3L))
  obs <- intersect(obs, seq_len(n_obs))

  cat("\n")
  cat("  E2Tree Local Explanation\n")
  cat("  ============================================================\n")
  cat(sprintf("  %d instance(s)  |  Task: %s\n",
              n_obs, if (x$is_class) "Classification" else "Regression"))

  for (i in obs) {
    cat("\n")
    cat(sprintf("  ------ Instance %d ------------------------------------------\n", i))
    cat(sprintf("  Ensemble groups it in region %d  ->  reconstructed outcome: %s\n",
                x$leaf[i], as.character(x$predicted[i])))

    # reliability
    if (!is.null(x$reliability)) {
      cat(sprintf("  Region fidelity:  loi_out = %.*f   mean_loi = %.*f\n",
                  digits, x$reliability$loi_out[i], digits, x$reliability$mean_loi[i]))
    }

    # contribution (top drivers)
    cv <- x$contribution$contributions[i, ]
    cv <- cv[order(-abs(cv))]; cv <- cv[abs(cv) > 1e-12]
    cv <- utils::head(cv, top)
    if (length(cv) > 0) {
      cat("  Why (top drivers of the reconstruction):\n")
      for (v in names(cv)) cat(sprintf("      %-20s %+.*f\n", v, digits, cv[v]))
    }

    # neighbours
    nb <- x$neighbors$neighbors[[i]]
    if (!is.null(nb) && nrow(nb) > 0) {
      cat(sprintf("  Most similar cases (ensemble proximity): %s\n",
                  paste(sprintf("%d(%.2f)", nb$train_id, nb$proximity),
                        collapse = "  ")))
    }

    # heterogeneity
    hr <- x$heterogeneity[i, , drop = FALSE]
    if (x$het_is_class) {
      cat(sprintf("  Region outcome dispersion: purity %.*f, entropy %.*f, set {%s}\n",
                  digits, hr$purity, digits, hr$entropy, hr$ambiguous_set))
    } else {
      cat(sprintf("  Region outcome dispersion: mean %.*f, sd %.*f, central band [%.*f, %.*f]\n",
                  digits, hr$mean, digits, hr$sd, digits, hr$lower, digits, hr$upper))
    }

    # optional A / B
    if (!is.null(x$counterfactual) && !is.na(x$counterfactual$target_leaf[i])) {
      cf <- x$counterfactual
      v  <- if (!is.null(cf$validated))
              sprintf(" [%s]", if (isTRUE(cf$validated[i])) "ensemble-validated"
                               else "not validated") else ""
      cat(sprintf("  Counterfactual: -> region %d (%s), cost %.*f%s\n",
                  cf$target_leaf[i], as.character(cf$target_pred[i]),
                  digits, cf$cost[i], v))
    }
    if (!is.null(x$stability)) {
      st <- x$stability
      cat(sprintf("  Explanation confidence: %.*f  (B = %d tree resamples)\n",
                  digits, st$confidence[i], st$B))
    }
  }
  cat("\n")
  if (n_obs > length(obs))
    cat(sprintf("  ... %d more (use print(x, obs = ...))\n", n_obs - length(obs)))
  invisible(x)
}


# ===========================================================================
# PLOT METHOD -- composite panel for one instance
# ===========================================================================

#' @method plot e2explanation
#' @export
plot.e2explanation <- function(x, obs = 1L, ...) {
  if (obs < 1L || obs > nrow(x$newdata))
    stop(sprintf("'obs' must be between 1 and %d.", nrow(x$newdata)))

  old <- graphics::par(no.readonly = TRUE); on.exit(graphics::par(old))
  graphics::par(mfrow = c(1, 2), mar = c(5, 7, 4, 1) + 0.1)

  # Panel 1: top contributions
  cv <- x$contribution$contributions[obs, ]
  cv <- cv[abs(cv) > 1e-12]
  if (length(cv) > 0) {
    cv <- cv[order(abs(cv))]
    cols <- ifelse(cv >= 0, "tomato", "steelblue")
    graphics::barplot(cv, names.arg = names(cv), horiz = TRUE, las = 1,
                      col = cols, border = NA,
                      main = sprintf("Drivers (instance %d -> %s)",
                                     obs, as.character(x$predicted[obs])),
                      xlab = "Contribution")
    graphics::abline(v = 0, col = "grey40")
  } else {
    graphics::plot.new(); graphics::title("No path splits")
  }

  # Panel 2: nearest ensemble neighbours
  nb <- x$neighbors$neighbors[[obs]]
  if (!is.null(nb) && nrow(nb) > 0) {
    o <- order(nb$proximity)
    graphics::barplot(nb$proximity[o], names.arg = nb$train_id[o], horiz = TRUE,
                      las = 1, col = "#2980b9", border = NA, xlim = c(0, 1),
                      main = "Ensemble neighbours", xlab = "Proximity")
  } else {
    graphics::plot.new(); graphics::title("No neighbours")
  }
  invisible(NULL)
}
