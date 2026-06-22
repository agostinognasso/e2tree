#' Inspect a Single E2Tree Node
#'
#' Produces a complete descriptive profile of one node of an E2Tree --
#' \strong{terminal or internal} -- characterising the observations the ensemble
#' routed there and how they differ from the overall population. This is a
#' purely interpretive description (no prediction): it answers "who is in this
#' region the ensemble carved out, and what makes it distinctive?".
#'
#' @details
#' The node's observations are taken from \code{fit$tree$obs} (populated for
#' every node, internal nodes included). For each predictor the within-node
#' distribution is compared against the rest of the data:
#' \itemize{
#'   \item \strong{numeric}: mean, sd, median, IQR, range, plus \emph{Cohen's d}
#'     (node vs the complement) measuring how far the node sits from the rest of
#'     the population on that variable;
#'   \item \strong{categorical}: per-level proportions (node vs global), mode,
#'     normalised entropy, plus \emph{Cramer's V} measuring the association
#'     between node membership and the variable.
#' }
#' The response distribution within the node is summarised too (class
#' proportions, purity and entropy for classification; mean, sd and quantiles
#' for regression). Effect sizes (Cohen's d, Cramer's V) make variables
#' comparable, so the profile immediately surfaces \emph{what separates this
#' region}.
#'
#' @param fit An \code{e2tree} object (must carry \code{$data} and \code{$y}).
#' @param node Integer node id (as in \code{nodes(fit)}). May be terminal or
#'   internal.
#' @param data Optional data frame. Default \code{NULL} uses \code{fit$data}.
#'
#' @return An object of class \code{"e2nodeStats"}:
#'   \item{meta}{List: \code{node}, \code{terminal}, \code{depth}, \code{n},
#'     \code{parent}, \code{children}, \code{rule}, \code{pred}, \code{prob},
#'     \code{impTotal}.}
#'   \item{obs}{Integer indices of the observations in the node.}
#'   \item{numeric}{Data frame, one row per numeric predictor, with node and
#'     global statistics and \code{cohen_d}.}
#'   \item{categorical}{Data frame, one row per categorical predictor, with
#'     \code{mode}, \code{n_levels}, \code{entropy}, \code{cramers_v}.}
#'   \item{cat_levels}{Named list of per-variable data frames (\code{level},
#'     \code{p_node}, \code{p_global}).}
#'   \item{response}{List summarising the response distribution in the node.}
#'   \item{is_class}{Logical.}
#'
#' @seealso \code{\link{nodes}}, \code{\link{localLoI}} (region fidelity),
#'   \code{\link{eHeterogeneity}} (outcome dispersion), \code{\link{eNeighbors}}.
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
#' ns <- nodeStats(tree, node = 1)   # the root
#' print(ns); plot(ns)
#' nodeStats(tree, node = nodes(tree, terminal = TRUE)$node[1])
#' }
#'
#' @importFrom stats var median IQR quantile chisq.test
#' @importFrom graphics boxplot barplot par legend
#' @export
nodeStats <- function(fit, node, data = NULL) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  td <- fit$tree
  if (!(node %in% td$node)) stop("'node' ", node, " is not a node of this e2tree.")
  if (is.null(data)) data <- fit$data
  if (is.null(data)) stop("Training data not found; pass 'data' explicitly.")
  data <- as.data.frame(data)

  ylevels  <- attr(fit, "ylevels")
  is_class <- !is.null(ylevels) && length(ylevels) > 0
  y <- fit$y

  resp_name <- .e2_resp_name(fit)
  preds <- setdiff(names(data), resp_name)

  row <- td[td$node == node, , drop = FALSE]
  idx <- suppressWarnings(as.integer(unlist(row$obs)))
  idx <- idx[!is.na(idx)]
  n_node <- length(idx)
  rest   <- setdiff(seq_len(nrow(data)), idx)

  ## ---- metadata ----------------------------------------------------------
  children <- if (isTRUE(row$terminal)) integer(0)
    else if (!is.null(row$children) && length(unlist(row$children)) > 0)
      unlist(row$children) else c(node * 2L, node * 2L + 1L)
  children <- children[!is.na(children)]
  meta <- list(
    node = node, terminal = isTRUE(row$terminal),
    depth = floor(log2(node)), n = n_node,
    parent = if (node == 1L) NA_integer_ else node %/% 2L,
    children = children,
    rule = if ("path" %in% names(row)) paste(unlist(row$path), collapse = " & ") else NA_character_,
    pred = row$pred, prob = row$prob, impTotal = row$impTotal)

  ## ---- numeric predictors ------------------------------------------------
  num_vars <- preds[vapply(data[preds], is.numeric, logical(1))]
  numeric_df <- NULL; num_raw <- list()
  if (length(num_vars) > 0) {
    numeric_df <- do.call(rbind, lapply(num_vars, function(v) {
      xn <- data[idx, v]; xr <- data[rest, v]
      num_raw[[v]] <<- list(node = xn, rest = xr)
      data.frame(
        variable = v, n = n_node,
        mean_node = mean(xn, na.rm = TRUE),
        sd_node = stats::sd(xn, na.rm = TRUE),
        median_node = stats::median(xn, na.rm = TRUE),
        iqr_node = stats::IQR(xn, na.rm = TRUE),
        min_node = min(xn, na.rm = TRUE), max_node = max(xn, na.rm = TRUE),
        mean_global = mean(data[[v]], na.rm = TRUE),
        sd_global = stats::sd(data[[v]], na.rm = TRUE),
        cohen_d = .cohen_d(xn, xr),
        stringsAsFactors = FALSE)
    }))
    numeric_df <- numeric_df[order(-abs(numeric_df$cohen_d)), ]
    row.names(numeric_df) <- NULL
  }

  ## ---- categorical predictors --------------------------------------------
  cat_vars <- preds[vapply(data[preds], function(x)
    is.factor(x) || is.character(x), logical(1))]
  categorical_df <- NULL; cat_levels <- list()
  if (length(cat_vars) > 0) {
    member <- seq_len(nrow(data)) %in% idx
    rows <- lapply(cat_vars, function(v) {
      xv <- factor(data[[v]])
      p_node <- prop.table(table(factor(data[idx, v], levels = levels(xv))))
      p_glob <- prop.table(table(xv))
      cat_levels[[v]] <<- data.frame(
        level = levels(xv), p_node = as.numeric(p_node),
        p_global = as.numeric(p_glob), stringsAsFactors = FALSE)
      pp <- as.numeric(p_node); pp <- pp[pp > 0]
      ent <- if (length(pp) <= 1) 0 else -sum(pp * log(pp)) / log(nlevels(xv))
      data.frame(
        variable = v, mode = levels(xv)[which.max(p_node)],
        n_levels = nlevels(xv), entropy = ent,
        cramers_v = .cramers_v(member, xv), stringsAsFactors = FALSE)
    })
    categorical_df <- do.call(rbind, rows)
    categorical_df <- categorical_df[order(-categorical_df$cramers_v), ]
    row.names(categorical_df) <- NULL
  }

  ## ---- response distribution ---------------------------------------------
  yn <- y[idx]
  if (is_class) {
    p <- prop.table(table(factor(yn, levels = ylevels)))
    pp <- as.numeric(p); pp <- pp[pp > 0]
    response <- list(
      type = "classification",
      proportions = as.numeric(p),
      levels = ylevels,
      dominant = ylevels[which.max(p)], purity = max(p),
      entropy = if (length(pp) <= 1) 0 else -sum(pp * log(pp)) / log(length(ylevels)))
  } else {
    response <- list(
      type = "regression",
      mean = mean(yn), sd = stats::sd(yn),
      quantiles = stats::quantile(yn, c(.1, .25, .5, .75, .9), names = TRUE),
      mean_global = mean(y), cohen_d = .cohen_d(yn, y[rest]))
  }

  res <- list(meta = meta, obs = idx, numeric = numeric_df,
              categorical = categorical_df, cat_levels = cat_levels,
              num_raw = num_raw, response = response, is_class = is_class)
  class(res) <- "e2nodeStats"
  res
}


# ---- internal helpers -------------------------------------------------------

.e2_resp_name <- function(fit) {
  tt <- fit$terms
  if (is.null(tt)) return(NULL)
  vars <- as.character(attr(tt, "variables"))[-1]
  ri <- attr(tt, "response")
  if (is.null(ri) || ri < 1L || ri > length(vars)) return(vars[1])
  vars[ri]
}

.cohen_d <- function(x1, x2) {
  x1 <- x1[!is.na(x1)]; x2 <- x2[!is.na(x2)]
  n1 <- length(x1); n2 <- length(x2)
  if (n1 < 1L || n2 < 1L) return(0)
  sp <- sqrt(((n1 - 1) * stats::var(x1) + (n2 - 1) * stats::var(x2)) /
               max(n1 + n2 - 2, 1))
  if (is.na(sp) || sp == 0) return(0)
  (mean(x1) - mean(x2)) / sp
}

.cramers_v <- function(member, x) {
  tab <- table(member, x)
  if (any(dim(tab) < 2)) return(0)
  chi <- suppressWarnings(stats::chisq.test(tab, correct = FALSE)$statistic)
  n <- sum(tab); k <- min(dim(tab))
  v <- sqrt(as.numeric(chi) / (n * (k - 1)))
  if (is.na(v)) 0 else v
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print e2nodeStats
#' @export
print.e2nodeStats <- function(x, top = 6, digits = 3, ...) {
  m <- x$meta
  cat("\n")
  cat(sprintf("  E2Tree node %s  (%s, depth %d)\n", m$node,
              if (m$terminal) "terminal" else "internal", m$depth))
  cat("  ------------------------------------------------------------\n")
  cat(sprintf("  size: %d obs   parent: %s   children: %s\n",
              m$n, ifelse(is.na(m$parent), "-", m$parent),
              if (length(m$children)) paste(m$children, collapse = ", ") else "-"))
  if (!is.na(m$rule) && nzchar(m$rule)) cat(sprintf("  rule: %s\n", m$rule))
  cat(sprintf("  node prediction: %s   (prob/level %.*f)\n",
              as.character(m$pred), digits, as.numeric(m$prob)))

  cat("\n  Response in this node:\n")
  if (x$is_class) {
    pr <- x$response$proportions; names(pr) <- x$response$levels
    for (l in names(pr)) cat(sprintf("    %-16s %.3f\n", l, pr[l]))
    cat(sprintf("    dominant: %s | purity: %.3f | entropy: %.3f\n",
                x$response$dominant, x$response$purity, x$response$entropy))
  } else {
    q <- x$response$quantiles
    cat(sprintf("    mean: %.*f  sd: %.*f  (global mean %.*f, Cohen's d %.*f)\n",
                digits, x$response$mean, digits, x$response$sd,
                digits, x$response$mean_global, digits, x$response$cohen_d))
    cat(sprintf("    quantiles 10/25/50/75/90: %s\n",
                paste(round(q, digits), collapse = " / ")))
  }

  if (!is.null(x$numeric) && nrow(x$numeric) > 0) {
    cat("\n  Most distinctive numeric predictors (|Cohen's d|):\n")
    nd <- utils::head(x$numeric, top)
    cat(sprintf("    %-18s %-10s %-10s %-8s\n", "variable", "node mean", "global", "d"))
    for (i in seq_len(nrow(nd)))
      cat(sprintf("    %-18s %-10.*f %-10.*f %-+8.*f\n",
                  substr(nd$variable[i], 1, 18), digits, nd$mean_node[i],
                  digits, nd$mean_global[i], digits, nd$cohen_d[i]))
  }
  if (!is.null(x$categorical) && nrow(x$categorical) > 0) {
    cat("\n  Most distinctive categorical predictors (Cramer's V):\n")
    cd <- utils::head(x$categorical, top)
    cat(sprintf("    %-26s %-16s %-8s\n", "variable", "node mode", "V"))
    for (i in seq_len(nrow(cd)))
      cat(sprintf("    %-26s %-16s %-8.*f\n",
                  substr(cd$variable[i], 1, 26), substr(cd$mode[i], 1, 16),
                  digits, cd$cramers_v[i]))
  }
  cat("\n")
  invisible(x)
}


# ===========================================================================
# PLOT METHOD — node vs rest of the population, for the most distinctive vars
# ===========================================================================

#' @method plot e2nodeStats
#' @export
plot.e2nodeStats <- function(x, top = 6, ...) {
  ## select the most distinctive variables (mixing numeric + categorical)
  specs <- list()
  if (!is.null(x$numeric))
    for (i in seq_len(min(nrow(x$numeric), top)))
      specs[[length(specs) + 1L]] <- list(var = x$numeric$variable[i],
                                           type = "num", score = abs(x$numeric$cohen_d[i]))
  if (!is.null(x$categorical))
    for (i in seq_len(min(nrow(x$categorical), top)))
      specs[[length(specs) + 1L]] <- list(var = x$categorical$variable[i],
                                           type = "cat", score = x$categorical$cramers_v[i])
  if (length(specs) == 0) { message("No predictors to plot."); return(invisible(NULL)) }
  specs <- specs[order(-vapply(specs, function(s) s$score, numeric(1)))]
  specs <- utils::head(specs, top)

  np <- length(specs)
  old <- par(no.readonly = TRUE); on.exit(par(old))
  par(mfrow = c(ceiling(np / 2), min(np, 2)), mar = c(4, 4, 3, 1))

  for (s in specs) {
    if (s$type == "num") {
      nr <- x$numeric[x$numeric$variable == s$var, ]
      raw <- x$num_raw[[s$var]]
      boxplot(list(node = raw$node, rest = raw$rest),
              col = c("#e67e22", "grey80"), border = "grey30",
              main = sprintf("%s (d = %+.2f)", s$var, nr$cohen_d),
              ylab = s$var)
    } else {
      lv <- x$cat_levels[[s$var]]
      M <- rbind(node = lv$p_node, global = lv$p_global)
      colnames(M) <- lv$level
      barplot(M, beside = TRUE, col = c("#e67e22", "grey70"), border = NA,
              las = 2, cex.names = 0.7,
              main = sprintf("%s (V = %.2f)", s$var,
                             x$categorical$cramers_v[x$categorical$variable == s$var]))
      legend("topright", c("node", "global"), fill = c("#e67e22", "grey70"),
             bty = "n", cex = 0.7)
    }
  }
  invisible(NULL)
}


# ===========================================================================
# Compare two nodes
# ===========================================================================

#' Compare Two E2Tree Nodes
#'
#' Side-by-side comparison of two nodes (terminal or internal), highlighting the
#' predictors on which the two regions differ most. Purely descriptive: it shows
#' how the ensemble's two groups of observations are distributed, not any
#' prediction.
#'
#' @details
#' For every predictor a between-node distinctiveness score is computed:
#' \emph{Cohen's d} between the two nodes for numeric variables, and the
#' \emph{total variation distance} between the two regional class distributions
#' for categorical variables. The \code{top} most differing variables are shown
#' as node-1-vs-node-2 boxplots (numeric) or grouped bar charts (categorical).
#'
#' @param fit An \code{e2tree} object.
#' @param node1,node2 Integer node ids to compare (must differ).
#' @param data Optional data frame. Default \code{NULL} uses \code{fit$data}.
#' @param top Maximum number of variables (panels) to display. Default \code{6}.
#'
#' @return Invisibly, a data frame ranking all predictors by the between-node
#'   distinctiveness \code{score}, with the per-node summary (mean for numeric,
#'   mode for categorical).
#'
#' @seealso \code{\link{nodeStats}} for a single-node profile.
#'
#' @examples
#' \donttest{
#' data(iris)
#' ensemble <- randomForest::randomForest(Species ~ ., data = iris,
#'   importance = TRUE, proximity = TRUE)
#' D <- createDisMatrix(ensemble, data = iris, label = "Species",
#'   parallel = list(active = FALSE, no_cores = 1))
#' tree <- e2tree(Species ~ ., iris, D, ensemble,
#'   list(impTotal = 0.1, maxDec = 0.01, n = 2, level = 5))
#' term <- nodes(tree, terminal = TRUE)$node
#' plotNodeComparison(tree, term[1], term[2])
#' }
#'
#' @importFrom graphics boxplot barplot par legend
#' @importFrom stats prop.table
#' @export
plotNodeComparison <- function(fit, node1, node2, data = NULL, top = 6) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  td <- fit$tree
  if (!(node1 %in% td$node)) stop("'node1' ", node1, " is not a node of this e2tree.")
  if (!(node2 %in% td$node)) stop("'node2' ", node2, " is not a node of this e2tree.")
  if (node1 == node2) stop("'node1' and 'node2' must differ.")
  if (is.null(data)) data <- fit$data
  if (is.null(data)) stop("Training data not found; pass 'data' explicitly.")
  data <- as.data.frame(data)

  preds <- setdiff(names(data), .e2_resp_name(fit))
  idx1 <- suppressWarnings(as.integer(unlist(td$obs[td$node == node1])))
  idx1 <- idx1[!is.na(idx1)]
  idx2 <- suppressWarnings(as.integer(unlist(td$obs[td$node == node2])))
  idx2 <- idx2[!is.na(idx2)]

  rows <- list(); raw <- list()
  for (v in preds) {
    if (is.numeric(data[[v]])) {
      x1 <- data[idx1, v]; x2 <- data[idx2, v]
      raw[[v]] <- list(type = "num", node1 = x1, node2 = x2)
      rows[[v]] <- data.frame(variable = v, type = "numeric",
                              score = abs(.cohen_d(x1, x2)),
                              node1 = round(mean(x1, na.rm = TRUE), 3),
                              node2 = round(mean(x2, na.rm = TRUE), 3),
                              stringsAsFactors = FALSE)
    } else {
      xv <- factor(data[[v]])
      p1 <- as.numeric(prop.table(table(factor(data[idx1, v], levels = levels(xv)))))
      p2 <- as.numeric(prop.table(table(factor(data[idx2, v], levels = levels(xv)))))
      raw[[v]] <- list(type = "cat", levels = levels(xv), p1 = p1, p2 = p2)
      rows[[v]] <- data.frame(variable = v, type = "categorical",
                              score = 0.5 * sum(abs(p1 - p2)),
                              node1 = levels(xv)[which.max(p1)],
                              node2 = levels(xv)[which.max(p2)],
                              stringsAsFactors = FALSE)
    }
  }
  cmp <- do.call(rbind, rows)
  cmp <- cmp[order(-cmp$score), ]; row.names(cmp) <- NULL

  sel <- utils::head(cmp$variable, top)
  np  <- length(sel)
  old <- par(no.readonly = TRUE); on.exit(par(old))
  par(mfrow = c(ceiling(np / 2), min(np, 2)), mar = c(4, 4, 3, 1))
  cols <- c("#3498db", "#e67e22")
  for (v in sel) {
    r <- raw[[v]]; sc <- cmp$score[cmp$variable == v]
    if (r$type == "num") {
      bx <- list(r$node1, r$node2); names(bx) <- c(paste0("node ", node1), paste0("node ", node2))
      boxplot(bx, col = cols, border = "grey30", ylab = v,
              main = sprintf("%s (|d| = %.2f)", v, sc))
    } else {
      M <- rbind(r$p1, r$p2); colnames(M) <- r$levels
      barplot(M, beside = TRUE, col = cols, border = NA, las = 2, cex.names = 0.7,
              main = sprintf("%s (TVD = %.2f)", v, sc))
      legend("topright", c(paste0("node ", node1), paste0("node ", node2)),
             fill = cols, bty = "n", cex = 0.7)
    }
  }
  invisible(cmp)
}
