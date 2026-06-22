utils::globalVariables(c("idx_lab", "value", "stat"))

#' Stability and Confidence of a Local Explanation
#'
#' Quantifies how \strong{stable} a local, proximity-based explanation is by
#' resampling the ensemble's trees. Most local explainers (Saabas, SHAP) report
#' a point estimate with no notion of uncertainty; because E2Tree reconstructs
#' grouping from the ensemble's leaf co-occurrence, the trees can be bootstrapped
#' to turn every local quantity into a \emph{distribution}, yielding a confidence
#' for the explanation and a stability score for its neighbours.
#'
#' @details
#' Let \eqn{L} be the \eqn{n_{\text{train}} \times T} leaf-membership matrix of
#' the ensemble (\code{\link{extract_terminal_nodes}}). The ensemble proximity
#' between a query and a training case is the fraction of trees in which they
#' share a leaf. For \eqn{b = 1, \dots, B} the function draws a bootstrap sample
#' of the \eqn{T} trees (columns of \eqn{L}, with replacement) and recomputes,
#' for each query:
#' \itemize{
#'   \item the \strong{grouping vote} -- the outcome of the proximity-weighted
#'     \code{k} nearest training cases; its modal value across the \eqn{B}
#'     resamples, and the frequency with which that mode is attained, define the
#'     \strong{confidence} \eqn{\in [0, 1]};
#'   \item the \strong{neighbour stability} -- for each of the query's nominal
#'     top-\code{k} neighbours (computed on the full tree set), the frequency
#'     with which it remains in the top-\code{k} across resamples;
#'   \item the proximity to the query's own leaf prototype, summarised by its
#'     mean and a \code{conf} central interval.
#' }
#'
#' High confidence with high neighbour stability means the ensemble's local
#' grouping of the instance is robust, so its E2Tree explanation can be trusted;
#' a low score flags an instance whose explanation hinges on a few trees. This is
#' a fidelity-uncertainty statement, not a predictive confidence interval.
#'
#' @param fit An \code{e2tree} object (must carry \code{$data} and \code{$y}).
#' @param ensemble The trained ensemble used to build \code{fit}.
#' @param newdata A data frame of instances to assess. Default \code{NULL} uses
#'   \code{fit$data}.
#' @param B Integer. Number of tree-bootstrap resamples. Default \code{100}.
#' @param k Integer. Neighbourhood size. Default \code{5}.
#' @param conf Numeric in (0, 1). Central-interval level for the prototype
#'   proximity. Default \code{0.95}.
#' @param seed Optional integer for reproducibility. Default \code{NULL}.
#' @param data Optional training data frame. Default \code{NULL} uses
#'   \code{fit$data}.
#'
#' @return An object of class \code{"e2stability"}:
#'   \item{confidence}{Per-instance confidence in the modal grouping outcome.}
#'   \item{vote}{Modal grouping outcome per instance.}
#'   \item{region}{Reconstructed e2tree region (terminal node) per instance,
#'     used by the by-region plotting views.}
#'   \item{prototype_prox}{Data frame: \code{mean}, \code{lower}, \code{upper}
#'     proximity of each instance to its leaf prototype.}
#'   \item{neighbor_stability}{List (one per instance) of data frames with
#'     \code{train_id} and \code{stability} (top-k retention frequency).}
#'   \item{B, k, conf, is_class}{Call settings.}
#'
#' @seealso \code{\link{eNeighbors}} for the point-estimate neighbours,
#'   \code{\link{localLoI}} for region-level fidelity.
#'
#' @references
#' Molnar, C. (2022). \emph{Interpretable Machine Learning}, 2nd ed. -- chapter
#' on the stability of explanations.
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
#' st <- eStability(tree, ensemble, newdata = iris[c(1, 80, 120), ], B = 50, seed = 1)
#' print(st)
#' plot(st)                                  # "profile": instances sorted by proximity
#' plot(st, type = "beeswarm")               # per-region point cloud
#' plot(st, type = "scatter")                # proximity vs confidence
#' plot(st, type = "forest")                 # per-region proximity +/- CI
#' plot(st, highlight = c(1, 3), labels = c("setosa", "virginica"))
#' }
#'
#' @export
eStability <- function(fit, ensemble, newdata = NULL, B = 100, k = 5,
                       conf = 0.95, seed = NULL, data = NULL) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  if (is.null(data)) data <- fit$data
  if (is.null(data)) stop("Training data not found; pass 'data' explicitly.")
  if (is.null(newdata)) newdata <- data
  if (!is.data.frame(newdata) || nrow(newdata) == 0)
    stop("'newdata' must be a non-empty data frame.")
  if (B < 2L) stop("'B' must be at least 2.")
  if (conf <= 0 || conf >= 1) stop("'conf' must be in (0, 1).")
  if (!is.null(seed)) set.seed(seed)

  response <- fit$y
  if (is.null(response)) stop("fit$y is missing; refit with a current e2tree().")
  ylevels  <- attr(fit, "ylevels")
  is_class <- !is.null(ylevels) && length(ylevels) > 0

  n_train <- nrow(data)
  n_query <- nrow(newdata)
  k <- min(k, n_train)

  leaves_train <- as.matrix(extract_terminal_nodes(ensemble, data))
  leaves_query <- as.matrix(extract_terminal_nodes(ensemble, newdata))
  ntree <- ncol(leaves_train)

  # Nominal (full-tree) proximity and top-k neighbours, for the stability anchor
  P_full <- .stab_prox(leaves_query, leaves_train, seq_len(ntree))
  nominal_topk <- lapply(seq_len(n_query), function(q)
    utils::head(order(-P_full[q, ]), k))

  # Accumulators: the B grouping votes per instance (character for
  # classification, numeric for regression), neighbour retention, prototype prox.
  votes      <- lapply(seq_len(n_query), function(q)
                  vector(if (is_class) "character" else "numeric", B))
  topk_hit   <- lapply(nominal_topk, function(z) stats::setNames(integer(length(z)),
                                                                 as.character(z)))
  proto_prox <- matrix(NA_real_, n_query, B)

  for (b in seq_len(B)) {
    cols <- sample.int(ntree, ntree, replace = TRUE)
    Pb   <- .stab_prox(leaves_query, leaves_train, cols)
    for (q in seq_len(n_query)) {
      ord  <- order(-Pb[q, ])
      topk <- utils::head(ord, k)
      w    <- Pb[q, topk]
      votes[[q]][b] <- .stab_vote(response[topk], w, is_class)
      # neighbour retention
      hit <- intersect(as.character(nominal_topk[[q]]), as.character(topk))
      topk_hit[[q]][hit] <- topk_hit[[q]][hit] + 1L
      # prototype proximity: mean proximity to the resampled top-k
      proto_prox[q, b] <- mean(w)
    }
  }

  alpha <- (1 - conf) / 2
  sdy   <- if (!is_class) { s <- stats::sd(response); if (is.na(s) || s == 0) 1 else s } else NA_real_
  confidence <- numeric(n_query)
  vote_out   <- vector(if (is_class) "character" else "numeric", n_query)
  vote_lo    <- rep(NA_real_, n_query)
  vote_hi    <- rep(NA_real_, n_query)
  nb_stab    <- vector("list", n_query)
  proto_df   <- data.frame(mean = numeric(n_query), lower = numeric(n_query),
                           upper = numeric(n_query))
  for (q in seq_len(n_query)) {
    vv <- votes[[q]]
    if (is_class) {
      # confidence = frequency of the modal grouping outcome across resamples
      tb <- table(vv)
      confidence[q] <- max(tb) / sum(tb)
      vote_out[q]   <- names(tb)[which.max(tb)]
    } else {
      # confidence = 1 - dispersion of the (continuous) vote, scaled by sd(y);
      # tight resample-to-resample agreement -> high confidence.
      vv <- as.numeric(vv)
      vote_out[q]   <- stats::median(vv)
      confidence[q] <- max(0, min(1, 1 - stats::sd(vv) / sdy))
      vote_lo[q]    <- stats::quantile(vv, alpha,     names = FALSE)
      vote_hi[q]    <- stats::quantile(vv, 1 - alpha, names = FALSE)
    }
    nb_stab[[q]] <- data.frame(
      train_id  = nominal_topk[[q]],
      stability = as.numeric(topk_hit[[q]][as.character(nominal_topk[[q]])]) / B,
      stringsAsFactors = FALSE)
    pq <- proto_prox[q, ]
    proto_df[q, ] <- c(mean(pq), stats::quantile(pq, alpha, names = FALSE),
                       stats::quantile(pq, 1 - alpha, names = FALSE))
  }

  # Reconstructed e2tree region (terminal node) per query: enables the
  # by-region plotting views and links stability to the heterogeneity layer.
  region <- tryCatch(.e2_route_leaf(fit, newdata), error = function(e) NULL)

  out <- list(
    confidence         = confidence,
    vote               = vote_out,
    region             = region,
    vote_interval      = if (is_class) NULL else
                           data.frame(lower = vote_lo, upper = vote_hi),
    prototype_prox     = proto_df,
    neighbor_stability = nb_stab,
    B = B, k = k, conf = conf, is_class = is_class)
  class(out) <- "e2stability"
  out
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Query x train proximity using a (possibly bootstrapped) set of tree columns.
.stab_prox <- function(leaves_query, leaves_train, cols) {
  nq <- nrow(leaves_query); nt <- nrow(leaves_train)
  P <- matrix(0.0, nq, nt)
  for (tt in cols) P <- P + outer(leaves_query[, tt], leaves_train[, tt], `==`)
  P / length(cols)
}

# Proximity-weighted grouping vote among a neighbour set.
.stab_vote <- function(outcomes, w, is_class) {
  if (is_class) {
    agg <- tapply(w, factor(as.character(outcomes)), sum)
    names(agg)[which.max(agg)]
  } else {
    sw <- sum(w); if (sw == 0) mean(outcomes) else sum(w * outcomes) / sw
  }
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print e2stability
#' @export
print.e2stability <- function(x, obs = NULL, digits = 3, ...) {
  n_query <- length(x$confidence)
  if (is.null(obs)) obs <- seq_len(min(n_query, 5L))
  obs <- intersect(obs, seq_len(n_query))

  cat("\n")
  cat("  E2Tree Explanation Stability (tree-bootstrap)\n")
  cat("  ------------------------------------------------------------\n")
  cat(sprintf("  %d instance(s)  |  B = %d resamples, k = %d  |  Task: %s\n\n",
              n_query, x$B, x$k, if (x$is_class) "Classification" else "Regression"))

  for (i in obs) {
    if (x$is_class) {
      cat(sprintf("  Instance %d  ->  grouping outcome: %s   confidence: %.*f\n",
                  i, as.character(x$vote[i]), digits, x$confidence[i]))
    } else {
      cat(sprintf("  Instance %d  ->  grouping value: %.*f  [%.*f, %.*f]   confidence: %.*f\n",
                  i, digits, as.numeric(x$vote[i]),
                  digits, x$vote_interval$lower[i], digits, x$vote_interval$upper[i],
                  digits, x$confidence[i]))
    }
    pp <- x$prototype_prox
    cat(sprintf("    prototype proximity: %.*f  [%.*f, %.*f]\n",
                digits, pp$mean[i], digits, pp$lower[i], digits, pp$upper[i]))
    ns <- x$neighbor_stability[[i]]
    ns <- ns[order(-ns$stability), , drop = FALSE]
    cat(sprintf("    neighbour stability (top-%d retention): %s\n", x$k,
                paste(sprintf("%d:%.2f", ns$train_id, ns$stability), collapse = "  ")))
    cat("\n")
  }
  if (n_query > length(obs))
    cat(sprintf("  ... %d more (use print(x, obs = ...))\n", n_query - length(obs)))
  invisible(x)
}


# ===========================================================================
# PLOT METHOD -- stability views selected by `type`
#   "profile"  : all instances sorted by prototype proximity (default)
#   "beeswarm" : per-region point cloud, colour = confidence
#   "scatter"  : the two stability axes at once (proximity vs confidence)
#   "forest"   : small-multiples forest plot of proximity +/- CI per region
# `highlight` marks instance positions (query order) and `labels` names them.
# ===========================================================================

#' @method plot e2stability
#' @export
plot.e2stability <- function(x, type = c("profile", "beeswarm", "scatter", "forest"),
                             highlight = NULL, labels = NULL, ...) {
  type <- match.arg(type)
  n  <- length(x$confidence)
  pp <- x$prototype_prox
  df <- data.frame(
    id = seq_len(n), mean = pp$mean, lower = pp$lower, upper = pp$upper,
    confidence = x$confidence,
    region = if (!is.null(x$region)) as.character(x$region) else NA_character_,
    stringsAsFactors = FALSE)
  if (type %in% c("beeswarm", "scatter", "forest") && all(is.na(df$region)))
    stop("type='", type, "' needs per-instance regions; refit with a current eStability().")

  hl <- .stab_resolve_highlight(df$id, highlight, labels)
  conf_const <- (max(df$confidence) - min(df$confidence)) < 1e-9
  switch(type,
    profile  = .stab_plot_profile(df, hl, conf_const, x$conf),
    beeswarm = .stab_plot_beeswarm(df, hl, conf_const),
    scatter  = .stab_plot_scatter(df, hl, conf_const),
    forest   = .stab_plot_forest(df, hl, x$conf))
}

# Resolve highlighted instance positions into a small id/label table.
.stab_resolve_highlight <- function(ids, highlight, labels) {
  if (is.null(highlight)) return(NULL)
  keep <- highlight %in% ids
  highlight <- highlight[keep]
  if (!length(highlight)) return(NULL)
  labels <- if (is.null(labels)) as.character(highlight) else labels[keep]
  data.frame(id = highlight, lab = labels, stringsAsFactors = FALSE)
}

.stab_theme <- function(drop_x = FALSE) {
  th <- ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle    = ggplot2::element_text(colour = "grey40", size = 10),
      panel.grid.minor = ggplot2::element_blank())
  if (drop_x)
    th <- th + ggplot2::theme(axis.text.x = ggplot2::element_blank(),
                              axis.ticks.x = ggplot2::element_blank())
  th
}

# ---- "profile": instances sorted by prototype proximity --------------------
.stab_plot_profile <- function(df, hl, conf_const, conf) {
  df <- df[order(df$mean), ]; df$rank <- seq_len(nrow(df))
  big <- nrow(df) > 60L
  sub <- if (conf_const)
    sprintf("explanation confidence = %.2f for all %d instances; band = %.0f%% interval",
            df$confidence[1], nrow(df), 100 * conf)
  else sprintf("colour = explanation confidence; band = %.0f%% interval", 100 * conf)

  g <- ggplot2::ggplot(df, ggplot2::aes(x = rank, y = mean)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lower, ymax = upper),
                         fill = "#2980b9", alpha = 0.18)
  if (conf_const)
    g <- g + ggplot2::geom_line(colour = "#2980b9", linewidth = 0.5) +
             ggplot2::geom_point(colour = "#2980b9", size = if (big) 0.7 else 1.8)
  else
    g <- g + ggplot2::geom_line(colour = "grey70", linewidth = 0.4) +
             ggplot2::geom_point(ggplot2::aes(colour = confidence),
                                 size = if (big) 1.0 else 2.0) +
             ggplot2::scale_colour_viridis_c(name = "confidence", limits = c(0, 1))
  if (!is.null(hl)) {
    H <- merge(df, hl, by = "id")
    g <- g + ggplot2::geom_point(data = H, colour = "#c0392b", size = 2.6) +
             ggplot2::geom_text(data = H,
               ggplot2::aes(label = sprintf("%s (%.2f)", lab, mean)),
               colour = "#c0392b", size = 3, vjust = -1, hjust = 0.5)
  }
  g + ggplot2::scale_y_continuous(limits = c(0, NA),
        expand = ggplot2::expansion(mult = c(0.02, 0.10))) +
    ggplot2::labs(title = "Explanation stability across observations", subtitle = sub,
      x = "instances ordered by prototype proximity (peripheral → central)",
      y = "proximity to region prototype") +
    .stab_theme(drop_x = TRUE)
}

# ---- "beeswarm": per-region point cloud ------------------------------------
.stab_plot_beeswarm <- function(df, hl, conf_const) {
  med <- tapply(df$mean, df$region, stats::median)
  df$region <- factor(df$region, levels = names(sort(med)))
  big <- nrow(df) > 60L
  g <- ggplot2::ggplot(df, ggplot2::aes(x = region, y = mean))
  if (conf_const)
    g <- g + ggplot2::geom_jitter(width = 0.25, height = 0, colour = "#2980b9",
                                  alpha = 0.55, size = if (big) 1.1 else 1.8)
  else
    g <- g + ggplot2::geom_jitter(ggplot2::aes(colour = confidence), width = 0.25,
                                  height = 0, alpha = 0.8, size = if (big) 1.3 else 2.0) +
             ggplot2::scale_colour_viridis_c(name = "confidence", limits = c(0, 1))
  if (!is.null(hl)) {
    H <- merge(df, hl, by = "id")
    g <- g + ggplot2::geom_point(data = H, colour = "#c0392b", size = 2.6) +
             ggplot2::geom_text(data = H, ggplot2::aes(label = lab),
                                colour = "#c0392b", size = 3, vjust = -1, hjust = 0.5)
  }
  sub <- if (conf_const)
    sprintf("confidence = %.2f for all instances; points jittered within region", df$confidence[1])
  else "colour = explanation confidence; points jittered within region"
  g + ggplot2::scale_y_continuous(limits = c(0, NA),
        expand = ggplot2::expansion(mult = c(0.02, 0.10))) +
    ggplot2::labs(title = "Stability by reconstructed region", subtitle = sub,
      x = "reconstructed region (leaf), ordered by median proximity",
      y = "proximity to region prototype") +
    .stab_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5,
                                                       hjust = 1, size = 8))
}

# ---- "scatter": the two stability axes at once -----------------------------
.stab_plot_scatter <- function(df, hl, conf_const) {
  nreg <- length(unique(df$region))
  g <- ggplot2::ggplot(df, ggplot2::aes(x = mean, y = confidence)) +
    ggplot2::geom_rug(sides = "b", colour = "grey60", alpha = 0.4)
  g <- g + (if (conf_const)
              ggplot2::geom_jitter(ggplot2::aes(colour = region), width = 0,
                                   height = 0.012, alpha = 0.8, size = 1.8)
            else
              ggplot2::geom_point(ggplot2::aes(colour = region), alpha = 0.8, size = 1.8)) +
    ggplot2::scale_colour_viridis_d(name = "region",
                                    guide = if (nreg > 12) "none" else "legend")
  if (!is.null(hl)) {
    H <- merge(df, hl, by = "id")
    g <- g + ggplot2::geom_point(data = H, colour = "#c0392b", size = 2.8) +
             ggplot2::geom_text(data = H, ggplot2::aes(label = lab),
                                colour = "#c0392b", size = 3, vjust = -1, hjust = 0.5)
  }
  sub <- if (conf_const)
    "every instance at confidence 1.00 (jittered vertically); colour = region"
  else "the two stability axes at once; colour = region"
  g + ggplot2::scale_y_continuous(limits = c(0, NA),
        expand = ggplot2::expansion(mult = c(0.02, 0.08))) +
    ggplot2::labs(title = "Stability: proximity vs confidence", subtitle = sub,
      x = "proximity to region prototype", y = "explanation confidence") +
    .stab_theme()
}

# ---- "forest": small-multiples forest plot per region ----------------------
.stab_plot_forest <- function(df, hl, conf) {
  med <- tapply(df$mean, df$region, stats::median)
  df$region <- factor(df$region, levels = names(sort(med)))
  df <- df[order(df$region, df$mean), ]
  df$pos <- stats::ave(df$mean, df$region, FUN = function(z) seq_along(z))
  g <- ggplot2::ggplot(df, ggplot2::aes(x = mean, y = pos)) +
    ggplot2::geom_segment(ggplot2::aes(x = lower, xend = upper, y = pos, yend = pos),
                          colour = "grey70", linewidth = 0.3) +
    ggplot2::geom_point(colour = "#2980b9", size = 0.9) +
    ggplot2::facet_wrap(~ region, scales = "free_y")
  if (!is.null(hl)) {
    H <- merge(df, hl, by = "id")
    g <- g + ggplot2::geom_point(data = H, colour = "#c0392b", size = 2.2) +
             ggplot2::geom_text(data = H, ggplot2::aes(label = lab),
                                colour = "#c0392b", size = 2.6, hjust = -0.15)
  }
  g + ggplot2::labs(
      title = sprintf("Stability by region (%.0f%% intervals)", 100 * conf),
      subtitle = "each row is one aspirate; one panel per reconstructed region",
      x = "proximity to region prototype", y = NULL) +
    .stab_theme() +
    ggplot2::theme(axis.text.y = ggplot2::element_blank(),
                   axis.ticks.y = ggplot2::element_blank())
}
