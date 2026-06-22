#' Local Loss of Interpretability (Local LoI)
#'
#' Disaggregates the global Loss of Interpretability (\code{\link{loi}}) into
#' a \strong{per-observation} and a \strong{per-node} component, so that the
#' fidelity of the E2Tree reconstruction can be inspected locally instead of
#' as a single global number.
#'
#' @details
#' For a pair of observations \eqn{(i, j)} the LoI contribution is
#' \deqn{c_{ij} = \frac{(o_{ij} - \hat{o}_{ij})^2}{\max(o_{ij}, \hat{o}_{ij})},}
#' where \eqn{o_{ij}} is the ensemble proximity and \eqn{\hat{o}_{ij}} the
#' E2Tree proximity. As in \code{\link{loi}}, \emph{between-node} pairs
#' (\eqn{\hat{o}_{ij} = 0}) reduce to \eqn{c_{ij} = o_{ij}}.
#'
#' \strong{Per observation} \eqn{i}:
#' \deqn{\ell_i = \frac{1}{n-1} \sum_{j \neq i} c_{ij},}
#' decomposed into a within-node part (mean of \eqn{c_{ij}} over the
#' leaf-mates of \eqn{i}) and a between-node part (mean ensemble proximity
#' that \eqn{i} shares with observations placed in other leaves). A high
#' \eqn{\ell_i} flags an observation whose local explanation (path,
#' neighbours, attribution) is \emph{less trustworthy}.
#'
#' \strong{Per terminal node} \eqn{k} with member set \eqn{\mathcal{I}_k}:
#' \itemize{
#'   \item \code{loi_in}: mean \eqn{c_{ij}} over pairs both inside the node
#'     (internal calibration error of the node);
#'   \item \code{loi_out}: mean \eqn{o_{ij}} over pairs with one member inside
#'     and one outside the node (ensemble proximity lost by the separation).
#' }
#'
#' The per-observation values average exactly to the normalised global index:
#' \eqn{\mathrm{mean}_i(\ell_i) = \mathrm{nLoI}}. Local LoI is therefore an
#' exact spatial decomposition of the published statistic, not a separate
#' metric.
#'
#' The leaf partition is recovered directly from the block structure of
#' \code{O_hat} (two observations share a leaf iff \eqn{\hat{o}_{ij} > 0}), so
#' the function works without \code{fit}. When \code{fit} is supplied, the
#' detected blocks are relabelled with the actual E2Tree terminal-node ids.
#'
#' @param O Ensemble proximity matrix (\eqn{n \times n}), values in [0, 1].
#'   Typically \code{proximity(eValidation(...))$ensemble}.
#' @param O_hat E2Tree proximity matrix (\eqn{n \times n}), values in [0, 1].
#'   Typically \code{proximity(eValidation(...))$e2tree}.
#' @param fit Optional \code{e2tree} object, used only to label nodes with
#'   their real terminal-node ids. Default \code{NULL}.
#'
#' @return An object of class \code{"localLoI"}:
#'   \item{obs}{Data frame: \code{obs}, \code{node}, \code{loi},
#'     \code{loi_in}, \code{loi_out}, \code{n_within}, \code{n_between}.}
#'   \item{node}{Data frame: \code{node}, \code{n}, \code{loi_in},
#'     \code{loi_out}, \code{mean_loi}, \code{n_within}, \code{n_between}.}
#'   \item{nloi}{Global normalised LoI (equals \code{mean(obs$loi)}).}
#'   \item{n}{Matrix dimension.}
#'
#' @seealso \code{\link{loi}} for the global index, \code{\link{eValidation}}
#'   and \code{\link{proximity}} for obtaining \code{O} and \code{O_hat}.
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
#' vs    <- eValidation(iris, tree, D)
#' prox  <- proximity(vs)
#' ll    <- localLoI(prox$ensemble, prox$e2tree, fit = tree)
#' print(ll)
#' summary(ll)
#' plot(ll)
#' }
#'
#' @importFrom graphics barplot hist abline text par
#' @export
localLoI <- function(O, O_hat, fit = NULL) {

  # ---- Input validation (mirrors loi()) -----------------------------------
  if (!is.matrix(O) || !is.matrix(O_hat)) {
    stop("Both O and O_hat must be matrices")
  }
  if (nrow(O) != ncol(O) || nrow(O_hat) != ncol(O_hat)) {
    stop("Matrices must be square")
  }
  if (!all(dim(O) == dim(O_hat))) {
    stop("O and O_hat must have the same dimensions")
  }
  if (!is.null(fit) && !inherits(fit, "e2tree")) {
    stop("'fit', when supplied, must be an e2tree object")
  }

  n <- nrow(O)

  # ---- Per-pair contribution matrix ---------------------------------------
  # c_ij = (o - o_hat)^2 / max(o, o_hat); 0 where both are 0.
  denom <- pmax(O, O_hat, na.rm = FALSE)
  C <- (O - O_hat)^2 / denom
  C[denom == 0] <- 0
  C[is.na(C)] <- 0
  diag(C) <- 0

  within_mask <- (O_hat > 0)
  diag(within_mask) <- FALSE

  # ---- Per-observation decomposition --------------------------------------
  # loi_i      : mean c_ij over all j != i
  # loi_in_i   : mean c_ij over within-node neighbours
  # loi_out_i  : mean o_ij over between-node neighbours (c_ij == o_ij there)
  n_within  <- rowSums(within_mask)
  n_between <- (n - 1L) - n_within

  sum_all     <- rowSums(C)
  sum_within  <- rowSums(C * within_mask)
  sum_between <- sum_all - sum_within

  loi_obs     <- sum_all / (n - 1L)
  loi_in_obs  <- ifelse(n_within  > 0, sum_within  / n_within,  0)
  loi_out_obs <- ifelse(n_between > 0, sum_between / n_between, 0)

  # ---- Leaf membership ----------------------------------------------------
  membership <- .localLoI_membership(O_hat, fit, n)

  obs_df <- data.frame(
    obs       = seq_len(n),
    node      = membership,
    loi       = loi_obs,
    loi_in    = loi_in_obs,
    loi_out   = loi_out_obs,
    n_within  = n_within,
    n_between = n_between,
    stringsAsFactors = FALSE
  )

  # ---- Per-node decomposition ---------------------------------------------
  nodes_u <- sort(unique(membership))
  node_rows <- lapply(nodes_u, function(k) {
    idx <- which(membership == k)
    nk  <- length(idx)
    out_idx <- setdiff(seq_len(n), idx)

    # within: pairs (i<j) both in node
    if (nk > 1L) {
      sub <- C[idx, idx, drop = FALSE]
      n_in_pairs <- nk * (nk - 1L) / 2
      loi_in_k <- sum(sub[lower.tri(sub)]) / n_in_pairs
    } else {
      n_in_pairs <- 0L
      loi_in_k   <- 0
    }

    # between: pairs (i in node, j outside)
    if (nk >= 1L && length(out_idx) > 0L) {
      sub_out <- O[idx, out_idx, drop = FALSE]
      n_out_pairs <- nk * length(out_idx)
      loi_out_k <- sum(sub_out) / n_out_pairs
    } else {
      n_out_pairs <- 0L
      loi_out_k   <- 0
    }

    data.frame(
      node      = k,
      n         = nk,
      loi_in    = loi_in_k,
      loi_out   = loi_out_k,
      mean_loi  = mean(loi_obs[idx]),
      n_within  = as.integer(n_in_pairs),
      n_between = as.integer(n_out_pairs),
      stringsAsFactors = FALSE
    )
  })
  node_df <- do.call(rbind, node_rows)
  row.names(node_df) <- NULL

  result <- list(
    obs   = obs_df,
    node  = node_df,
    nloi  = mean(loi_obs),
    n     = n
  )
  class(result) <- "localLoI"
  result
}


# ---------------------------------------------------------------------------
# Internal: recover leaf membership
# ---------------------------------------------------------------------------
# Preference order:
#   1. If `fit` is supplied and its terminal-node observation sets form a valid
#      partition of seq_len(n), use the real node ids.
#   2. Otherwise, recover blocks as connected components of (O_hat > 0).
.localLoI_membership <- function(O_hat, fit, n) {

  if (!is.null(fit)) {
    td <- fit$tree
    if (!is.null(td) && "terminal" %in% names(td) && "obs" %in% names(td)) {
      term <- td[td$terminal == TRUE, , drop = FALSE]
      m <- rep(NA_integer_, n)
      ok <- TRUE
      for (k in seq_len(nrow(term))) {
        idx <- suppressWarnings(as.integer(unlist(term$obs[k])))
        idx <- idx[!is.na(idx)]
        if (length(idx) == 0L || any(idx < 1L) || any(idx > n)) { ok <- FALSE; break }
        m[idx] <- as.integer(term$node[k])
      }
      if (ok && !any(is.na(m))) return(m)
      warning("localLoI: fit$tree$obs does not align with the proximity matrix; ",
              "falling back to block detection from O_hat.", call. = FALSE)
    }
  }

  .localLoI_blocks(O_hat)
}

# Connected components of the (O_hat > 0) graph. Leaves are cliques in O_hat,
# so a BFS recovers the exact leaf partition. Returns an integer label vector.
.localLoI_blocks <- function(O_hat) {
  n <- nrow(O_hat)
  adj <- (O_hat > 0)
  diag(adj) <- FALSE
  comp <- integer(n)
  cur <- 0L
  for (s in seq_len(n)) {
    if (comp[s] != 0L) next
    cur <- cur + 1L
    stack <- s
    while (length(stack) > 0L) {
      v <- stack[length(stack)]
      stack <- stack[-length(stack)]
      if (comp[v] != 0L) next
      comp[v] <- cur
      nb <- which(adj[v, ] & comp == 0L)
      if (length(nb) > 0L) stack <- c(stack, nb)
    }
  }
  comp
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print localLoI
#' @export
print.localLoI <- function(x, digits = 4, ...) {
  cat("\n")
  cat("  Local Loss of Interpretability (Local LoI)\n")
  cat("  -------------------------------------------------\n")
  cat(sprintf("  Global nLoI:          %.*f\n", digits, x$nloi))
  cat(sprintf("  Observations:         %d   |   Terminal nodes: %d\n",
              x$n, nrow(x$node)))
  cat("\n")

  # Least reliable nodes (highest loi_out = most ensemble proximity lost)
  nd <- x$node[order(-x$node$loi_out), , drop = FALSE]
  show_n <- min(5L, nrow(nd))
  cat("  Least reliable terminal nodes (highest loi_out):\n")
  cat(sprintf("  %-8s %-6s %-12s %-12s %-10s\n",
              "Node", "n", "loi_in", "loi_out", "mean_loi"))
  for (i in seq_len(show_n)) {
    cat(sprintf("  %-8s %-6d %-12.*f %-12.*f %-10.*f\n",
                nd$node[i], nd$n[i],
                digits, nd$loi_in[i], digits, nd$loi_out[i],
                digits, nd$mean_loi[i]))
  }
  cat("\n")

  # Least reliable observations
  ob <- x$obs[order(-x$obs$loi), , drop = FALSE]
  show_o <- min(5L, nrow(ob))
  cat("  Least reliable observations (highest loi):\n")
  cat(sprintf("  %-8s %-8s %-12s\n", "Obs", "Node", "loi"))
  for (i in seq_len(show_o)) {
    cat(sprintf("  %-8d %-8s %-12.*f\n",
                ob$obs[i], ob$node[i], digits, ob$loi[i]))
  }
  cat("\n")
  invisible(x)
}


# ===========================================================================
# SUMMARY METHOD
# ===========================================================================

#' @method summary localLoI
#' @export
summary.localLoI <- function(object, digits = 4, ...) {
  x <- object
  cat("\n")
  cat("##############################################################################\n")
  cat("   Local Loss of Interpretability (Local LoI) -- Node decomposition\n")
  cat("##############################################################################\n\n")
  cat(sprintf("  Global nLoI: %.*f   (= mean of per-observation loi)\n\n", digits, x$nloi))

  nd <- x$node[order(x$node$node), , drop = FALSE]
  cat("------------------------------------------------------------------------------\n")
  cat(sprintf("  %-8s %-6s %-12s %-12s %-10s\n",
              "Node", "n", "loi_in", "loi_out", "mean_loi"))
  cat("------------------------------------------------------------------------------\n")
  for (i in seq_len(nrow(nd))) {
    cat(sprintf("  %-8s %-6d %-12.*f %-12.*f %-10.*f\n",
                nd$node[i], nd$n[i],
                digits, nd$loi_in[i], digits, nd$loi_out[i],
                digits, nd$mean_loi[i]))
  }
  cat("------------------------------------------------------------------------------\n\n")

  cat("  Interpretation:\n")
  cat("    loi_in  : within-node calibration error (lower is better).\n")
  cat("    loi_out : ensemble proximity lost by separating the node (lower is better).\n")
  cat("    High loi_out nodes split apart observations the ensemble sees as similar;\n")
  cat("    local explanations for their members should be treated with caution.\n\n")
  cat("##############################################################################\n\n")
  invisible(x)
}


# ===========================================================================
# PLOT METHOD
# ===========================================================================

#' @method plot localLoI
#' @export
plot.localLoI <- function(x, ...) {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  par(mfrow = c(1, 2), mar = c(5, 4, 4, 2) + 0.1)

  # Panel 1: per-node loi_out ranked
  nd <- x$node[order(-x$node$loi_out), , drop = FALSE]
  bp <- barplot(nd$loi_out,
                names.arg = nd$node,
                col = "#e74c3c", border = NA,
                main = "Node reliability\n(loi_out: ensemble proximity lost)",
                xlab = "Terminal node", ylab = "loi_out",
                las = 2)

  # Panel 2: distribution of per-observation loi
  hist(x$obs$loi,
       main = sprintf("Per-observation LoI\nnLoI = %.4f", x$nloi),
       xlab = "loi_i", col = "#3498db", border = "white")
  abline(v = x$nloi, col = "red", lwd = 2, lty = 2)
}
