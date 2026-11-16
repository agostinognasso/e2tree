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
#' \strong{Leaf partition.} As in \code{\link{loi}} with its \code{leaves}
#' argument, the within/between split is defined by the actual leaves of the
#' E2Tree. They are taken, in order of preference, from the \code{leaves}
#' argument, from the terminal nodes of \code{fit}, or from the block structure
#' of \code{O_hat} (two observations share a leaf iff
#' \eqn{\hat{o}_{ij} > 0}). The last option cannot recognise leaves of weight
#' zero, whose members have \eqn{\hat{o}_{ij} = 0} even though they share a
#' leaf (this happens in regression); supply \code{fit} or \code{leaves}
#' whenever possible. A warning is issued when the blocks are used and some
#' observations have no within-leaf neighbour.
#'
#' \strong{Row order.} \code{\link{eValidation}} returns its proximity matrices
#' in hierarchical-clustering order, with the original observation indices as
#' row names. When \code{O_hat} carries such row names, they are used to match
#' the leaves of \code{fit} to the rows and are reported in \code{obs}; without
#' them the rows are assumed to follow the data order.
#'
#' \strong{Scale.} The decomposition is computed on the matrices it receives.
#' The nLoI is defined on the proximities themselves, which is what
#' \code{\link{eValidation}} returns by default (\code{transform = "none"});
#' with \code{transform = "sqrt"} the matrices are square-rooted and the index
#' is that of the square-root scale.
#'
#' @param O Ensemble proximity matrix (\eqn{n \times n}), values in \eqn{[0, 1]}.
#'   Typically \code{proximity(eValidation(...))$ensemble}.
#' @param O_hat E2Tree proximity matrix (\eqn{n \times n}), values in \eqn{[0, 1]}.
#'   Typically \code{proximity(eValidation(...))$e2tree}.
#' @param fit Optional \code{e2tree} object. Its terminal nodes define the leaf
#'   partition and supply the node ids. Default \code{NULL}.
#' @param leaves Optional vector of length \code{nrow(O)} giving the leaf of
#'   each row of the matrices, with the same meaning as in \code{\link{loi}}
#'   (for matrices taken from \code{\link{eValidation}}, its \code{leaves}
#'   component). It takes precedence over \code{fit}. Default \code{NULL}.
#'
#' @return An object of class \code{"localLoI"}:
#'   \item{obs}{Data frame: \code{obs} (original observation index), \code{node},
#'     \code{loi}, \code{loi_in}, \code{loi_out}, \code{n_within},
#'     \code{n_between}.}
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
#' vs    <- eValidation(iris, tree, D, graph = FALSE)
#' prox  <- proximity(vs)
#' ll    <- localLoI(prox$ensemble, prox$e2tree, fit = tree)
#' print(ll)
#' summary(ll)
#' plot(ll)
#' }
#'
#' @importFrom graphics barplot hist abline text par
#' @export
localLoI <- function(O, O_hat, fit = NULL, leaves = NULL) {

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
  if (!is.null(leaves) && (length(leaves) != nrow(O) || anyNA(leaves))) {
    stop("'leaves' must have length nrow(O) and no missing values")
  }

  n <- nrow(O)

  # Original observation index of each row (the matrices of eValidation() are
  # in hclust order and carry the original indices as row names)
  row_id <- .localLoI_row_index(O, O_hat, n)

  # ---- Per-pair contribution matrix ---------------------------------------
  # c_ij = (o - o_hat)^2 / max(o, o_hat); 0 where both are 0.
  denom <- pmax(O, O_hat, na.rm = FALSE)
  C <- (O - O_hat)^2 / denom
  C[denom == 0] <- 0
  C[is.na(C)] <- 0
  diag(C) <- 0

  # ---- Leaf partition and within/between mask ------------------------------
  # With a known partition (leaves or fit) within-leaf pairs are those sharing a
  # leaf, as in loi(leaves =); only without one does O_hat > 0 stand in for it.
  part <- .localLoI_membership(O_hat, fit, leaves, row_id)
  membership <- part$membership
  within_mask <- if (part$known) {
    outer(membership, membership, "==")
  } else {
    (O_hat > 0)
  }
  diag(within_mask) <- FALSE

  if (!part$known) {
    n_iso <- sum(rowSums(within_mask) == 0L)
    if (n_iso > 0L) {
      warning("localLoI: ", n_iso, " observation(s) have no within-leaf neighbour ",
              "(O_hat is 0 for all their pairs). If they belong to leaves of weight ",
              "zero the within/between split is wrong: supply `fit` or `leaves` to ",
              "use the real leaf partition.", call. = FALSE)
    }
  }

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

  obs_df <- data.frame(
    obs       = row_id,
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
# Internal: original observation index of each row of the matrices
# ---------------------------------------------------------------------------
# eValidation() orders its proximity matrices by hierarchical clustering and
# stores the original observation indices as row names. If O_hat carries row
# names that are a permutation of 1..n they give the index of each row;
# otherwise the rows are taken to be in data order.
.localLoI_row_index <- function(O, O_hat, n) {
  as_id <- function(M) {
    rn <- rownames(M)
    if (is.null(rn)) return(NULL)
    id <- suppressWarnings(as.integer(rn))
    if (anyNA(id) || !identical(as.character(id), rn) ||
        anyDuplicated(id) || !setequal(id, seq_len(n))) return(NULL)
    id
  }
  id_hat <- as_id(O_hat)
  id_o   <- as_id(O)
  if (!is.null(id_hat) && !is.null(id_o) && !identical(id_hat, id_o)) {
    warning("localLoI: O and O_hat have different row names (different row ",
            "orders); the order of O_hat is used.", call. = FALSE)
  }
  if (is.null(id_hat)) seq_len(n) else id_hat
}

# ---------------------------------------------------------------------------
# Internal: leaf membership of each row of the matrices
# ---------------------------------------------------------------------------
# Returns list(membership, known). Preference order:
#   1. `leaves`, one leaf label per row, as in loi(leaves =).
#   2. `fit`: the terminal nodes of the e2tree, whose observation sets must form
#      a valid partition of 1..n; they are in data order, so they are mapped to
#      the rows through `row_id`.
#   3. Otherwise the blocks of (O_hat > 0), flagged as not known: they cannot
#      recognise leaves of weight zero.
.localLoI_membership <- function(O_hat, fit, leaves, row_id) {
  n <- length(row_id)

  if (!is.null(leaves)) return(list(membership = leaves, known = TRUE))

  if (!is.null(fit)) {
    td <- fit$tree
    if (!is.null(td) && "terminal" %in% names(td) && "obs" %in% names(td)) {
      term <- td[td$terminal == TRUE, , drop = FALSE]
      m <- rep(NA_integer_, n)          # leaf of each observation, in data order
      ok <- TRUE
      for (k in seq_len(nrow(term))) {
        idx <- suppressWarnings(as.integer(unlist(term$obs[k])))
        idx <- idx[!is.na(idx)]
        if (length(idx) == 0L || any(idx < 1L) || any(idx > n)) { ok <- FALSE; break }
        m[idx] <- as.integer(term$node[k])
      }
      if (ok && !any(is.na(m))) return(list(membership = m[row_id], known = TRUE))
      warning("localLoI: fit$tree$obs does not align with the proximity matrix; ",
              "falling back to block detection from O_hat.", call. = FALSE)
    }
  }

  list(membership = .localLoI_blocks(O_hat), known = FALSE)
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
