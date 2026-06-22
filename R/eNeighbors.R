#' Proximity-Native Case-Based Explanation
#'
#' Explains a prediction by the cases the \emph{ensemble} considers most
#' similar to the query, using the ensemble's own learned proximity (leaf
#' co-occurrence) rather than a generic feature-space distance. This is the
#' explanation form that is unique to E2Tree: the dissimilarity that drives the
#' tree is the ensemble's co-occurrence geometry, so neighbours and prototypes
#' reflect how the ensemble groups observations, not how close they are in raw
#' predictor space.
#'
#' @details
#' For each query observation the function:
#' \enumerate{
#'   \item routes it through the E2Tree to its terminal node (leaf);
#'   \item computes its \strong{ensemble proximity} to every training
#'     observation as the fraction of trees in which the two share a terminal
#'     node (\code{extract_terminal_nodes()}), so any supported ensemble
#'     backend is handled identically;
#'   \item returns the \code{k} nearest training observations (with their
#'     outcome), and the leaf \strong{prototypes} -- the members of the query's
#'     leaf that are most central under the same proximity (highest mean
#'     proximity to the other members).
#' }
#'
#' Proximity here uses unweighted leaf co-occurrence, which coincides with the
#' classification dissimilarity in \code{\link{createDisMatrix}}
#' (\eqn{O = 1 - D}); for regression ensembles the package's \code{D} adds a
#' variance weighting, so neighbours are ranked on the unweighted co-occurrence
#' for consistency across new and training instances.
#'
#' @param fit An \code{e2tree} object (must carry \code{$data} and \code{$y},
#'   stored by current versions of \code{e2tree()}).
#' @param ensemble The trained ensemble used to build \code{fit}.
#' @param query A data frame of observations to explain. Default \code{NULL}
#'   uses the training data in \code{fit$data}.
#' @param k Integer. Number of nearest neighbours to return. Default \code{5}.
#' @param n_proto Integer. Number of leaf prototypes to return. Default \code{3}.
#' @param data Optional training data frame. Default \code{NULL} uses
#'   \code{fit$data}.
#'
#' @return An object of class \code{"e2neighbors"}:
#'   \item{query_leaf}{Terminal node id reached by each query observation.}
#'   \item{query_pred}{E2Tree prediction for each query observation.}
#'   \item{neighbors}{List (one element per query) of data frames with columns
#'     \code{train_id}, \code{proximity}, \code{outcome}, \code{leaf}.}
#'   \item{prototypes}{List (one per query) of data frames with the most central
#'     members of the query's leaf: \code{train_id}, \code{centrality},
#'     \code{outcome}.}
#'   \item{is_class}{Logical.}
#'
#' @seealso \code{\link{eContribution}} for feature attribution,
#'   \code{\link{localLoI}} for leaf reliability.
#'
#' @references
#' Wilms, I. et al. (2024). A-PETE: Adaptive Prototype Explanations of Tree
#' Ensembles. arXiv:2405.21036.
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
#' nb <- eNeighbors(tree, ensemble, query = iris[c(1, 80), ], k = 5)
#' print(nb)
#' }
#'
#' @export
eNeighbors <- function(fit, ensemble, query = NULL, k = 5, n_proto = 3,
                       data = NULL) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  if (is.null(data)) data <- fit$data
  if (is.null(data)) stop("Training data not found; pass 'data' explicitly.")
  if (is.null(query)) query <- data
  if (!is.data.frame(query) || nrow(query) == 0)
    stop("'query' must be a non-empty data frame.")

  response <- fit$y
  if (is.null(response)) stop("fit$y is missing; refit with a current e2tree().")
  ylevels  <- attr(fit, "ylevels")
  is_class <- !is.null(ylevels) && length(ylevels) > 0

  n_train <- nrow(data)
  n_query <- nrow(query)
  k       <- min(k, n_train)

  # ---- Ensemble proximity (leaf co-occurrence), backend-agnostic ----------
  leaves_train <- as.matrix(extract_terminal_nodes(ensemble, data))
  leaves_query <- as.matrix(extract_terminal_nodes(ensemble, query))
  ntree <- ncol(leaves_train)

  # P[q, t] = fraction of trees where query q and train t share a leaf
  P <- matrix(0.0, n_query, n_train)
  for (tt in seq_len(ntree)) {
    P <- P + outer(leaves_query[, tt], leaves_train[, tt], `==`)
  }
  P <- P / ntree

  # ---- Route query to E2Tree leaves ---------------------------------------
  query_leaf <- .e2_route_leaf(fit, query)
  term <- fit$tree[fit$tree$terminal == TRUE, , drop = FALSE]
  leaf_pred <- term$pred[match(query_leaf, term$node)]

  # Training-observation leaf membership (for neighbour annotation)
  train_leaf <- .e2_route_leaf(fit, data)

  # ---- Per-query neighbours and prototypes --------------------------------
  neighbors  <- vector("list", n_query)
  prototypes <- vector("list", n_query)

  for (i in seq_len(n_query)) {
    pr <- P[i, ]
    ord <- order(-pr)
    # drop self-match when query is a training row (proximity == 1 with itself)
    top <- utils::head(ord, k)
    neighbors[[i]] <- data.frame(
      train_id  = top,
      proximity = pr[top],
      outcome   = response[top],
      leaf      = train_leaf[top],
      stringsAsFactors = FALSE
    )

    # prototypes: members of the query's leaf, ranked by centrality
    leaf_members <- which(train_leaf == query_leaf[i])
    if (length(leaf_members) >= 1L) {
      if (length(leaf_members) == 1L) {
        cent <- 1.0
      } else {
        Pmm <- P_train_block(leaves_train, leaf_members, ntree)
        cent <- rowMeans(Pmm)
      }
      ord_p <- order(-cent)
      sel <- utils::head(leaf_members[ord_p], n_proto)
      prototypes[[i]] <- data.frame(
        train_id   = sel,
        centrality = cent[ord_p][seq_along(sel)],
        outcome    = response[sel],
        stringsAsFactors = FALSE
      )
    } else {
      prototypes[[i]] <- data.frame(train_id = integer(0),
                                    centrality = numeric(0),
                                    outcome = response[integer(0)])
    }
  }

  result <- list(
    query_leaf = query_leaf,
    query_pred = leaf_pred,
    neighbors  = neighbors,
    prototypes = prototypes,
    is_class   = is_class,
    k          = k
  )
  class(result) <- "e2neighbors"
  result
}


# Proximity block among a set of training members (member x member).
P_train_block <- function(leaves_train, members, ntree) {
  Lm <- leaves_train[members, , drop = FALSE]
  B <- matrix(0.0, length(members), length(members))
  for (tt in seq_len(ntree)) {
    B <- B + outer(Lm[, tt], Lm[, tt], `==`)
  }
  B / ntree
}


# Route observations to their E2Tree terminal node. Returns an integer leaf id
# per row of `newdata`. Shares the parsing helpers used by ePredTree().
.e2_route_leaf <- function(fit, newdata) {
  td <- fit$tree
  internal <- td[td$terminal == FALSE, , drop = FALSE]
  split_cache <- parse_all_splits(internal$splitLabel)
  names(split_cache) <- as.character(internal$node)

  n <- nrow(newdata)
  leaf <- integer(n)
  for (i in seq_len(n)) {
    cur <- 1L
    while (cur %in% internal$node) {
      rule <- split_cache[[as.character(cur)]]
      if (is.null(rule) || rule$type == "unknown") break
      goes_left <- apply_split_rule(newdata, i, rule)
      cur <- if (goes_left) cur * 2L else cur * 2L + 1L
    }
    leaf[i] <- cur
  }
  leaf
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print e2neighbors
#' @export
print.e2neighbors <- function(x, obs = NULL, digits = 3, ...) {
  n_query <- length(x$query_leaf)
  if (is.null(obs)) obs <- seq_len(min(n_query, 3L))
  obs <- intersect(obs, seq_len(n_query))

  cat("\n")
  cat("  E2Tree Case-Based Explanation (ensemble proximity)\n")
  cat("  ------------------------------------------------------------\n")
  cat(sprintf("  %d query observation(s)  |  Task: %s\n\n",
              n_query, if (x$is_class) "Classification" else "Regression"))

  for (i in obs) {
    cat(sprintf("  Query %d  ->  Prediction: %s  (leaf %d)\n",
                i, as.character(x$query_pred[i]), x$query_leaf[i]))
    nb <- x$neighbors[[i]]
    cat(sprintf("    %d nearest ensemble neighbours:\n", nrow(nb)))
    cat(sprintf("    %-10s %-12s %-12s %-6s\n",
                "train_id", "proximity", "outcome", "leaf"))
    for (r in seq_len(nrow(nb))) {
      cat(sprintf("    %-10d %-12.*f %-12s %-6d\n",
                  nb$train_id[r], digits, nb$proximity[r],
                  as.character(nb$outcome[r]), nb$leaf[r]))
    }
    pp <- x$prototypes[[i]]
    if (nrow(pp) > 0) {
      cat(sprintf("    leaf prototypes (most central): %s\n",
                  paste(pp$train_id, collapse = ", ")))
    }
    cat("\n")
  }
  if (n_query > length(obs))
    cat(sprintf("  ... %d more (use print(x, obs = ...))\n", n_query - length(obs)))
  invisible(x)
}


# ===========================================================================
# PLOT METHOD -- neighbour proximities for one query
# ===========================================================================

#' Plot method for e2neighbors
#'
#' Horizontal bar chart of the nearest ensemble neighbours of one query
#' observation, with bars ordered by proximity and (for classification)
#' coloured by the neighbour's outcome.
#'
#' @param x An \code{e2neighbors} object.
#' @param obs Integer. Index of the query observation to plot. Default \code{1}.
#' @param ... Additional arguments (ignored).
#' @method plot e2neighbors
#' @export
plot.e2neighbors <- function(x, obs = 1L, ...) {
  if (obs < 1L || obs > length(x$neighbors))
    stop(sprintf("'obs' must be between 1 and %d.", length(x$neighbors)))
  nb <- x$neighbors[[obs]]
  if (nrow(nb) == 0) {
    message("No neighbours for query ", obs, ".")
    return(invisible(NULL))
  }

  o <- order(nb$proximity)
  prox <- nb$proximity[o]
  lab  <- nb$train_id[o]

  if (x$is_class) {
    out <- factor(as.character(nb$outcome[o]))
    pal <- grDevices::hcl.colors(nlevels(out), "Dark 3")
    cols <- pal[as.integer(out)]
    barplot(prox, names.arg = lab, horiz = TRUE, las = 1, col = cols,
            border = NA, xlim = c(0, 1),
            main = sprintf("Ensemble neighbours of query %d (pred: %s)",
                           obs, as.character(x$query_pred[obs])),
            xlab = "Ensemble proximity")
    legend("bottomright", legend = levels(out), fill = pal, bty = "n",
           title = "outcome")
  } else {
    barplot(prox, names.arg = lab, horiz = TRUE, las = 1, col = "#3498db",
            border = NA, xlim = c(0, 1),
            main = sprintf("Ensemble neighbours of query %d (pred: %.3g)",
                           obs, as.numeric(x$query_pred[obs])),
            xlab = "Ensemble proximity")
  }
  invisible(NULL)
}
