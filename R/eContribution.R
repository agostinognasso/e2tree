utils::globalVariables(c("variable_name", "contrib_val", "sign_fill"))

#' Per-Instance Feature Attribution (Saabas Decomposition)
#'
#' Computes an exact, additive, \emph{per-instance} feature attribution for the
#' predictions of an E2Tree, following the Saabas / "tree interpreter"
#' decomposition. Unlike an impurity-share attribution (which is identical for
#' every observation that follows the same path), this attribution depends on
#' the observation's own routing and yields different magnitudes \emph{and
#' signs} for observations that end up in different leaves.
#'
#' @details
#' Let \eqn{v(t)} be the value of node \eqn{t}: the mean response for
#' regression, or the vector of class proportions for classification, computed
#' from the training observations falling in \eqn{t}. For an observation
#' \eqn{x} routed from the root to its leaf, the prediction decomposes as
#' \deqn{\hat{f}(x) = v(\mathrm{root}) + \sum_{t \in \mathrm{path}(x)}
#'   \big[ v(\mathrm{child}(t, x)) - v(t) \big],}
#' where \eqn{\mathrm{child}(t, x)} is the child that \eqn{x} enters at node
#' \eqn{t}. The bracketed term is the local contribution of the variable used
#' to split node \eqn{t}; contributions are summed per variable. The
#' decomposition is exact: \code{baseline + rowSums(contributions)} equals the
#' E2Tree prediction.
#'
#' For classification the decomposition is carried out per class; the reported
#' \code{contributions} are those toward the predicted class (the full per-class
#' array is returned in \code{contributions_byclass}).
#'
#' Because the attribution explains the \emph{surrogate tree}, it is only as
#' faithful to the ensemble as the tree is in that region. Pass a
#' \code{\link{localLoI}} object via \code{reliability} to annotate each
#' explanation with the fidelity (\code{loi_out}, \code{mean_loi}) of its
#' destination leaf.
#'
#' @param fit An \code{e2tree} object.
#' @param newdata A data frame of observations to explain.
#' @param reliability Optional \code{localLoI} object. When supplied, the
#'   destination-leaf reliability is attached to the output.
#'
#' @return An object of class \code{"e2contribution"}:
#'   \item{contributions}{Numeric matrix (\eqn{n_{obs} \times p}) of per-variable
#'     contributions toward the prediction (predicted-class probability for
#'     classification).}
#'   \item{baseline}{Root-node value (predicted-class root probability for
#'     classification; global mean for regression).}
#'   \item{prediction}{E2Tree prediction value reconstructed at the leaf.}
#'   \item{predicted}{Predicted label (classification) or value (regression).}
#'   \item{leaf}{Terminal node id reached by each observation.}
#'   \item{contributions_byclass}{(Classification only) list of
#'     \eqn{n_{obs} \times p} matrices, one per class.}
#'   \item{is_class}{Logical.}
#'   \item{reliability}{(If \code{reliability} supplied) data frame with the
#'     destination-leaf \code{loi_out} and \code{mean_loi} per observation.}
#'
#' @seealso \code{\link{localLoI}} for the reliability layer, \code{\link{eNeighbors}}
#'   for case-based explanations.
#'
#' @references
#' Saabas, A. (2014). Interpreting random forests.
#' Lundberg, S.M. & Lee, S.I. (2020). From local explanations to global
#' understanding with explainable AI for trees. \emph{Nature Machine
#' Intelligence}, 2(1), 56-67.
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
#' ec <- eContribution(tree, newdata = iris[c(1, 60, 120), ])
#' print(ec)
#' plot(ec, obs = 3)                          # single-observation attribution bars
#' plot(ec, obs = 1:3)                        # "heatmap" across several observations
#' plot(ec, obs = 1:3, type = "summary")      # SHAP-style per-feature distribution
#' plot(ec, obs = 1:3, type = "importance")   # mean |contribution| per feature
#' }
#'
#' @export
eContribution <- function(fit, newdata, reliability = NULL) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  if (!is.data.frame(newdata) || nrow(newdata) == 0)
    stop("'newdata' must be a non-empty data frame.")
  if (!is.null(reliability) && !inherits(reliability, "localLoI"))
    stop("'reliability', when supplied, must be a localLoI object.")

  row.names(newdata) <- NULL
  td <- fit$tree
  ylevels  <- attr(fit, "ylevels")
  is_class <- !is.null(ylevels) && length(ylevels) > 0
  y <- fit$y
  if (is.null(y))
    stop("fit$y is missing; refit with a current version of e2tree().")

  internal <- td[td$terminal == FALSE, , drop = FALSE]
  terminal <- td[td$terminal == TRUE,  , drop = FALSE]

  split_cache <- parse_all_splits(internal$splitLabel)
  names(split_cache) <- as.character(internal$node)

  split_vars <- sort(unique(internal$variable[!is.na(internal$variable)]))

  # ---- Node values from training membership -------------------------------
  node_ids <- as.character(td$node)
  if (is_class) {
    yf <- factor(y, levels = ylevels)
    vmat <- matrix(0.0, nrow = nrow(td), ncol = length(ylevels),
                   dimnames = list(node_ids, ylevels))
    for (r in seq_len(nrow(td))) {
      idx <- suppressWarnings(as.integer(unlist(td$obs[r])))
      idx <- idx[!is.na(idx)]
      if (length(idx) == 0L) next
      tb <- table(yf[idx])
      vmat[r, ] <- as.numeric(tb) / sum(tb)
    }
  } else {
    vvec <- stats::setNames(rep(NA_real_, nrow(td)), node_ids)
    for (r in seq_len(nrow(td))) {
      idx <- suppressWarnings(as.integer(unlist(td$obs[r])))
      idx <- idx[!is.na(idx)]
      if (length(idx) > 0L) vvec[r] <- mean(y[idx])
    }
  }

  n_obs <- nrow(newdata)
  p     <- length(split_vars)
  leaf  <- integer(n_obs)

  if (is_class) {
    byclass <- lapply(ylevels, function(z)
      matrix(0.0, n_obs, p, dimnames = list(seq_len(n_obs), split_vars)))
    names(byclass) <- ylevels
  } else {
    contrib <- matrix(0.0, n_obs, p, dimnames = list(seq_len(n_obs), split_vars))
  }

  for (i in seq_len(n_obs)) {
    cur <- 1L
    while (cur %in% internal$node) {
      rule <- split_cache[[as.character(cur)]]
      if (is.null(rule) || rule$type == "unknown") break
      var_here  <- as.character(td$variable[td$node == cur])
      goes_left <- apply_split_rule(newdata, i, rule)
      child <- if (goes_left) cur * 2L else cur * 2L + 1L

      if (is_class) {
        delta <- vmat[as.character(child), ] - vmat[as.character(cur), ]
        for (z in ylevels) byclass[[z]][i, var_here] <-
          byclass[[z]][i, var_here] + delta[z]
      } else {
        delta <- vvec[as.character(child)] - vvec[as.character(cur)]
        contrib[i, var_here] <- contrib[i, var_here] + delta
      }
      cur <- child
    }
    leaf[i] <- cur
  }

  # ---- Assemble baseline / prediction / predicted -------------------------
  if (is_class) {
    root_v <- vmat["1", ]
    pred_lab <- character(n_obs)
    baseline <- numeric(n_obs)
    prediction <- numeric(n_obs)
    contributions <- matrix(0.0, n_obs, p,
                            dimnames = list(seq_len(n_obs), split_vars))
    for (i in seq_len(n_obs)) {
      leaf_v <- vmat[as.character(leaf[i]), ]
      pc <- ylevels[which.max(leaf_v)]
      pred_lab[i]   <- pc
      baseline[i]   <- root_v[pc]
      prediction[i] <- leaf_v[pc]
      contributions[i, ] <- byclass[[pc]][i, ]
    }
    predicted <- pred_lab
  } else {
    baseline   <- rep(vvec["1"], n_obs)
    prediction <- vvec[as.character(leaf)]
    contributions <- contrib
    predicted  <- as.numeric(prediction)
  }

  result <- list(
    contributions = contributions,
    baseline      = as.numeric(baseline),
    prediction    = as.numeric(prediction),
    predicted     = predicted,
    leaf          = leaf,
    is_class      = is_class,
    split_vars    = split_vars,
    newdata       = newdata
  )
  if (is_class) result$contributions_byclass <- byclass

  if (!is.null(reliability)) {
    nd <- reliability$node
    m  <- match(leaf, nd$node)
    result$reliability <- data.frame(
      obs      = seq_len(n_obs),
      leaf     = leaf,
      loi_out  = nd$loi_out[m],
      mean_loi = nd$mean_loi[m],
      stringsAsFactors = FALSE
    )
  }

  class(result) <- "e2contribution"
  result
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print e2contribution
#' @export
print.e2contribution <- function(x, obs = NULL, digits = 4, ...) {
  n_obs <- nrow(x$newdata)
  if (is.null(obs)) obs <- seq_len(min(n_obs, 5L))
  obs <- intersect(obs, seq_len(n_obs))

  cat("\n")
  cat("  E2Tree Per-Instance Attribution (Saabas decomposition)\n")
  cat("  ------------------------------------------------------------\n")
  cat(sprintf("  %d observation(s)  |  Task: %s\n\n",
              n_obs, if (x$is_class) "Classification" else "Regression"))

  for (i in obs) {
    cat(sprintf("  Observation %d  ->  Prediction: %s  (leaf %d)\n",
                i, as.character(x$predicted[i]), x$leaf[i]))
    cat(sprintf("    baseline = %.*f   prediction = %.*f\n",
                digits, x$baseline[i], digits, x$prediction[i]))
    cv <- x$contributions[i, ]
    cv <- cv[order(-abs(cv))]
    cv <- cv[abs(cv) > 1e-12]
    if (length(cv) > 0) {
      for (v in names(cv)) {
        cat(sprintf("    %-22s %+.*f\n", v, digits, cv[v]))
      }
    } else {
      cat("    (no splits on path)\n")
    }
    if (!is.null(x$reliability)) {
      cat(sprintf("    leaf reliability: loi_out = %.4f, mean_loi = %.4f\n",
                  x$reliability$loi_out[i], x$reliability$mean_loi[i]))
    }
    cat("\n")
  }
  if (n_obs > length(obs))
    cat(sprintf("  ... %d more (use print(x, obs = ...))\n", n_obs - length(obs)))
  invisible(x)
}


# ===========================================================================
# PLOT METHOD -- signed contribution bars (one obs) or heatmap (many obs)
# ===========================================================================

#' @method plot e2contribution
#' @export
plot.e2contribution <- function(x, obs = 1L, type = NULL, ...) {
  if (any(obs < 1L) || any(obs > nrow(x$newdata)))
    stop(sprintf("'obs' must be between 1 and %d.", nrow(x$newdata)))

  # Default view: a single waterfall for one observation, a heatmap for many.
  if (is.null(type)) type <- if (length(obs) == 1L) "waterfall" else "heatmap"
  type <- match.arg(type, c("waterfall", "heatmap", "summary", "importance"))
  switch(type,
    waterfall  = .e2contribution_waterfall(x, obs[1]),
    heatmap    = .e2contribution_heatmap(x, obs),
    summary    = .e2contribution_summary(x, obs),
    importance = .e2contribution_importance(x, obs))
}

# Signed-contribution bars for a single observation (waterfall reading).
.e2contribution_waterfall <- function(x, obs) {
  cv <- x$contributions[obs, ]
  cv <- cv[abs(cv) > 1e-12]
  if (length(cv) == 0) {
    message("Observation ", obs, " has no splits on its path.")
    return(invisible(NULL))
  }
  plot_data <- data.frame(
    variable_name = names(cv),
    contrib_val   = as.numeric(cv),
    stringsAsFactors = FALSE)
  plot_data$sign_fill <- ifelse(plot_data$contrib_val >= 0, "increases", "decreases")
  plot_data <- plot_data[order(plot_data$contrib_val), ]
  plot_data$variable_name <- factor(plot_data$variable_name,
                                     levels = plot_data$variable_name)

  ggplot2::ggplot(plot_data,
                  ggplot2::aes(x = variable_name, y = contrib_val, fill = sign_fill)) +
    ggplot2::geom_col(width = 0.65) +
    ggplot2::scale_fill_manual(values = c(increases = "tomato",
                                          decreases = "steelblue"), name = NULL) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey40") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title    = sprintf("Per-Instance Attribution - Observation %d", obs),
      subtitle = sprintf("Prediction: %s   (baseline %.3f -> %.3f)",
                         as.character(x$predicted[obs]),
                         x$baseline[obs], x$prediction[obs]),
      x = NULL, y = "Contribution to prediction") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle = ggplot2::element_text(colour = "grey40", size = 10))
}

# SHAP-style summary: per feature, the spread of signed contributions across the
# selected observations, each point coloured by that observation's feature value.
.e2contribution_summary <- function(x, obs) {
  M <- x$contributions[obs, , drop = FALSE]
  keep <- colSums(abs(M) > 1e-12) > 0
  M <- M[, keep, drop = FALSE]
  if (ncol(M) == 0L) {
    message("None of the selected observations split on any feature.")
    return(invisible(NULL))
  }
  feats <- colnames(M)
  ord   <- names(sort(colMeans(abs(M))))          # important features on top
  long  <- do.call(rbind, lapply(seq_along(feats), function(j) {
    v  <- suppressWarnings(as.numeric(x$newdata[obs, feats[j]]))
    rg <- range(v, na.rm = TRUE)
    vn <- if (!is.finite(diff(rg)) || diff(rg) == 0) rep(0.5, length(v)) else (v - rg[1]) / diff(rg)
    data.frame(feature = feats[j], contrib = M[, j], value = vn)
  }))
  long$feature <- factor(long$feature, levels = ord)

  ggplot2::ggplot(long, ggplot2::aes(x = contrib, y = feature, colour = value)) +
    ggplot2::geom_vline(xintercept = 0, colour = "grey70") +
    ggplot2::geom_jitter(width = 0, height = 0.2, alpha = 0.55,
                         size = if (length(obs) > 60) 1.0 else 1.6) +
    ggplot2::scale_colour_gradient(low = "steelblue", high = "tomato",
                                   name = "feature value", breaks = c(0, 1),
                                   labels = c("low", "high")) +
    ggplot2::labs(title = "Per-feature contribution distribution",
                  subtitle = sprintf("%d observations; features ordered by mean |contribution|",
                                     length(obs)),
                  x = "contribution to prediction", y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle = ggplot2::element_text(colour = "grey40", size = 10),
      panel.grid.minor = ggplot2::element_blank())
}

# Local feature importance: mean |contribution| per feature across observations.
.e2contribution_importance <- function(x, obs) {
  M   <- x$contributions[obs, , drop = FALSE]
  imp <- colMeans(abs(M)); imp <- imp[imp > 1e-12]
  if (!length(imp)) {
    message("None of the selected observations split on any feature.")
    return(invisible(NULL))
  }
  d <- data.frame(feature = factor(names(imp), levels = names(sort(imp))),
                  imp = as.numeric(imp))
  ggplot2::ggplot(d, ggplot2::aes(x = imp, y = feature)) +
    ggplot2::geom_col(fill = "#34495e", width = 0.7) +
    ggplot2::labs(title = "Local feature importance",
                  subtitle = sprintf("mean |contribution| across %d observations", length(obs)),
                  x = "mean |contribution|", y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle = ggplot2::element_text(colour = "grey40", size = 10),
      panel.grid.minor = ggplot2::element_blank())
}

# Compact attribution overview for many observations at once: a heatmap with
# observations on the x-axis (grouped by reconstructed region) and features on
# the y-axis, the fill encoding the signed contribution toward the prediction.
.e2contribution_heatmap <- function(x, obs) {
  ord <- order(x$leaf[obs])                       # group columns by region/leaf
  obs <- obs[ord]
  M   <- x$contributions[obs, , drop = FALSE]
  keep <- colSums(abs(M) > 1e-12) > 0             # drop never-used features
  M <- M[, keep, drop = FALSE]
  if (ncol(M) == 0L) {
    message("None of the selected observations split on any feature.")
    return(invisible(NULL))
  }
  inst <- as.character(obs)
  pd <- data.frame(
    instance = factor(rep(inst, times = ncol(M)), levels = inst),
    variable = factor(rep(colnames(M), each = nrow(M)), levels = colnames(M)),
    contrib  = as.numeric(M),                     # column-major matches above
    stringsAsFactors = FALSE)
  lim   <- max(abs(pd$contrib))
  many  <- length(obs) > 40L
  ggplot2::ggplot(pd, ggplot2::aes(x = instance, y = variable, fill = contrib)) +
    ggplot2::geom_tile(colour = if (many) NA else "grey92") +
    ggplot2::scale_fill_gradient2(low = "steelblue", mid = "white",
                                  high = "tomato", midpoint = 0,
                                  limits = c(-lim, lim),
                                  name = "Contribution") +
    ggplot2::labs(
      title    = "Per-instance attribution across observations",
      subtitle = sprintf("%d observations, ordered by reconstructed region",
                         length(obs)),
      x = "Observation", y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle = ggplot2::element_text(colour = "grey40", size = 10),
      panel.grid    = ggplot2::element_blank(),
      axis.text.x   = if (many) ggplot2::element_blank()
                      else ggplot2::element_text(angle = 90, vjust = 0.5,
                                                 hjust = 1, size = 7),
      axis.ticks.x  = if (many) ggplot2::element_blank()
                      else ggplot2::element_line())
}
