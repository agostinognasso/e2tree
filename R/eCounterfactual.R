utils::globalVariables(c("variable", "delta_std", "dir_fill"))

#' Proximity-Native Contrastive (Counterfactual) Explanation
#'
#' Answers a contrastive question that stays within E2Tree's interpretive
#' mission: \emph{what is the smallest change to an instance's features that
#' would make the ensemble group it with a different region?} Unlike generic
#' counterfactual methods (which perturb a model's decision boundary), the
#' contrast here is anchored to the ensemble's own grouping geometry: a
#' candidate change is only reported once it has been \strong{verified against
#' the ensemble proximity} (leaf co-occurrence), so the explanation reflects how
#' the \emph{forest} regroups the instance, not merely how the surrogate tree
#' reroutes it.
#'
#' @details
#' For each instance the function:
#' \enumerate{
#'   \item routes it to its E2Tree terminal node (the \emph{source region}) and
#'     reads the region's reconstructed label;
#'   \item enumerates \emph{candidate target regions} -- terminal nodes whose
#'     reconstructed label differs (classification) or whose reconstructed value
#'     lies on the opposite side of the global median (regression). A specific
#'     target node id may be forced via \code{target};
#'   \item rebuilds each candidate's \strong{decision box} by walking the path
#'     from the root: the node id encodes the direction taken at every ancestor
#'     (left child \eqn{2t}, right child \eqn{2t+1}), and the ancestor's split
#'     rule (\code{\link{parse_all_splits}}) gives the constraint (numeric
#'     \eqn{x \le \theta} / \eqn{x > \theta}; categorical \eqn{x \in S} /
#'     \eqn{x \notin S}). Constraints on the same variable are intersected;
#'   \item computes the \strong{minimal change} that moves the instance into the
#'     box -- numeric variables are shifted to the violated boundary, categorical
#'     variables to a representative admissible level -- and scores it by a
#'     standardised cost (\eqn{\sum |\Delta| / \mathrm{sd}} for numeric, plus one
#'     per categorical change). The cheapest candidate is the counterfactual;
#'   \item (if \code{ensemble} is supplied) \strong{verifies} the counterfactual
#'     against the ensemble: the changed instance's leaf co-occurrence proximity
#'     to the members of the target region is compared with its proximity to the
#'     source region. \code{validated} is \code{TRUE} when the ensemble places
#'     the counterfactual closer to the target.
#' }
#'
#' The cost is a transparent, monotone distance, not a probability; the
#' explanation is descriptive (how to move across the ensemble's grouping), never
#' a prescription or a prediction.
#'
#' @param fit An \code{e2tree} object (must carry \code{$data} and \code{$y}).
#' @param newdata A data frame of instances to explain.
#' @param target Either \code{"nearest_different"} (default; the cheapest region
#'   with a different reconstructed outcome) or an integer terminal-node id to
#'   aim for explicitly.
#' @param ensemble Optional trained ensemble used to verify the counterfactual on
#'   the ensemble proximity. When \code{NULL}, only the surrogate-tree
#'   counterfactual is returned (\code{validated = NA}).
#' @param standardize Logical. Standardise numeric changes by the predictor's
#'   standard deviation when scoring cost. Default \code{TRUE}.
#' @param data Optional training data frame. Default \code{NULL} uses
#'   \code{fit$data}.
#'
#' @return An object of class \code{"e2counterfactual"}:
#'   \item{source_leaf, target_leaf}{Source / target terminal-node ids per
#'     instance.}
#'   \item{source_pred, target_pred}{Reconstructed outcomes of the two regions.}
#'   \item{cost}{Standardised change cost per instance (\code{NA} when no
#'     differing region exists).}
#'   \item{changes}{List (one per instance) of data frames with columns
#'     \code{variable}, \code{from}, \code{to}, \code{delta}, \code{type}.}
#'   \item{prox_to_target, prox_to_source, validated}{(If \code{ensemble} given)
#'     ensemble-proximity verification of each counterfactual.}
#'   \item{is_class}{Logical.}
#'
#' @seealso \code{\link{eContribution}} for additive attribution,
#'   \code{\link{eNeighbors}} for case-based explanation, \code{\link{localLoI}}
#'   for region reliability.
#'
#' @references
#' Wachter, S., Mittelstadt, B. & Russell, C. (2017). Counterfactual explanations
#' without opening the black box. \emph{Harvard Journal of Law & Technology},
#' 31, 841-887.
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
#' cf <- eCounterfactual(tree, newdata = iris[c(80, 120), ], ensemble = ensemble)
#' print(cf)
#' plot(cf, obs = 1)                          # single-observation change bars
#' plot(cf, obs = 1:2)                        # "heatmap" of required changes
#' plot(cf, obs = 1:2, type = "frequency")    # how often each feature is the lever
#' plot(cf, obs = 1:2, type = "beeswarm")     # signed-change distribution per feature
#' }
#'
#' @export
eCounterfactual <- function(fit, newdata, target = "nearest_different",
                            ensemble = NULL, standardize = TRUE, data = NULL) {

  if (!inherits(fit, "e2tree")) stop("'fit' must be an e2tree object.")
  if (!is.data.frame(newdata) || nrow(newdata) == 0)
    stop("'newdata' must be a non-empty data frame.")
  if (is.null(data)) data <- fit$data
  if (is.null(data)) stop("Training data not found; pass 'data' explicitly.")
  data <- as.data.frame(data)
  row.names(newdata) <- NULL

  ylevels  <- attr(fit, "ylevels")
  is_class <- !is.null(ylevels) && length(ylevels) > 0
  y <- fit$y
  if (is.null(y)) stop("fit$y is missing; refit with a current e2tree().")

  td       <- fit$tree
  internal <- td[td$terminal == FALSE, , drop = FALSE]
  term     <- td[td$terminal == TRUE,  , drop = FALSE]

  split_cache <- parse_all_splits(internal$splitLabel)
  names(split_cache) <- as.character(internal$node)
  internal_nodes <- internal$node

  resp_name <- .e2_resp_name(fit)
  preds     <- setdiff(names(data), resp_name)
  num_vars  <- preds[vapply(data[preds], is.numeric, logical(1))]
  sds <- vapply(num_vars, function(v) {
    s <- stats::sd(data[[v]], na.rm = TRUE); if (is.na(s) || s == 0) 1 else s
  }, numeric(1))
  names(sds) <- num_vars

  # ---- Route instances and training data ----------------------------------
  source_leaf <- .e2_route_leaf(fit, newdata)
  train_leaf  <- .e2_route_leaf(fit, data)
  src_pred    <- term$pred[match(source_leaf, term$node)]

  # Reconstructed value per terminal node (for the regression target rule)
  term_pred <- term$pred
  names(term_pred) <- as.character(term$node)
  med_y <- if (!is_class) stats::median(y, na.rm = TRUE) else NA_real_

  n_obs <- nrow(newdata)
  out <- list(
    source_leaf = source_leaf,
    target_leaf = rep(NA_integer_, n_obs),
    source_pred = src_pred,
    target_pred = rep(NA, n_obs),
    cost        = rep(NA_real_, n_obs),
    changes     = vector("list", n_obs),
    is_class    = is_class
  )
  do_validate <- !is.null(ensemble)
  if (do_validate) {
    out$prox_to_target <- rep(NA_real_, n_obs)
    out$prox_to_source <- rep(NA_real_, n_obs)
    out$validated      <- rep(NA, n_obs)
    leaves_train <- as.matrix(extract_terminal_nodes(ensemble, data))
  }

  for (i in seq_len(n_obs)) {
    # candidate target leaves
    if (is.numeric(target) && length(target) == 1L) {
      cand <- term$node[term$node == as.integer(target)]
    } else {
      if (is_class) {
        cand <- term$node[term$pred != src_pred[i]]
      } else {
        src_val  <- as.numeric(term_pred[as.character(source_leaf[i])])
        side_src <- src_val >= med_y
        cand_val <- as.numeric(term_pred)
        cand <- term$node[(cand_val >= med_y) != side_src]
      }
    }
    cand <- cand[!is.na(cand)]
    if (length(cand) == 0L) {
      out$changes[[i]] <- .cf_empty_changes()
      next
    }

    # cheapest candidate box
    best <- NULL
    for (leaf in cand) {
      box <- .cf_box(leaf, split_cache)
      ch  <- .cf_min_change(newdata[i, , drop = FALSE], box, data, train_leaf,
                            leaf, sds, standardize)
      if (is.null(best) || ch$cost < best$cost) {
        best <- ch; best$leaf <- leaf
      }
    }

    out$target_leaf[i] <- best$leaf
    out$target_pred[i] <- term_pred[as.character(best$leaf)]
    out$cost[i]        <- best$cost
    out$changes[[i]]   <- best$changes

    # ---- ensemble verification --------------------------------------------
    if (do_validate) {
      cf_row <- newdata[i, , drop = FALSE]
      if (nrow(best$changes) > 0L) {
        for (r in seq_len(nrow(best$changes))) {
          v <- best$changes$variable[r]
          cf_row[[v]] <- .cf_coerce(cf_row[[v]], best$changes$to[r])
        }
      }
      lv_cf  <- as.integer(extract_terminal_nodes(ensemble, cf_row)[1, ])
      tgt_mem <- which(train_leaf == best$leaf)
      src_mem <- which(train_leaf == source_leaf[i])
      out$prox_to_target[i] <- .cf_prox(lv_cf, leaves_train, tgt_mem)
      out$prox_to_source[i] <- .cf_prox(lv_cf, leaves_train, src_mem)
      out$validated[i] <- isTRUE(out$prox_to_target[i] > out$prox_to_source[i])
    }
  }

  class(out) <- "e2counterfactual"
  out
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.cf_empty_changes <- function()
  data.frame(variable = character(0), from = character(0), to = character(0),
             delta = numeric(0), type = character(0), stringsAsFactors = FALSE)

# Decision box of a terminal node: per-variable constraints recovered from the
# path encoded in the node id. Returns a named list keyed by variable; each
# element is list(type, lower, upper [numeric] | allowed [categorical levels]).
.cf_box <- function(leaf, split_cache) {
  chain <- integer(0); cur <- leaf
  while (cur > 1L) { chain <- c(cur, chain); cur <- cur %/% 2L }
  chain <- c(1L, chain)                       # root ... leaf

  box <- list()
  for (i in seq_len(length(chain) - 1L)) {
    parent <- chain[i]; child <- chain[i + 1L]
    rule <- split_cache[[as.character(parent)]]
    if (is.null(rule) || rule$type == "unknown") next
    goes_left <- (child == parent * 2L)
    v <- rule$var
    if (rule$type == "numeric") {
      cur_box <- box[[v]]
      if (is.null(cur_box))
        cur_box <- list(type = "numeric", lower = -Inf, upper = Inf)
      if (goes_left) {                         # x <= thr
        cur_box$upper <- min(cur_box$upper, rule$thr)
      } else {                                 # x > thr
        cur_box$lower <- max(cur_box$lower, rule$thr)
      }
      box[[v]] <- cur_box
    } else {                                   # categorical
      cur_box <- box[[v]]
      if (is.null(cur_box))
        cur_box <- list(type = "categorical", allowed = NULL, inc = list(), exc = list())
      if (goes_left) cur_box$inc[[length(cur_box$inc) + 1L]] <- rule$cats
      else           cur_box$exc[[length(cur_box$exc) + 1L]] <- rule$cats
      box[[v]] <- cur_box
    }
  }
  box
}

# Minimal change that moves a single-row instance into the decision box.
.cf_min_change <- function(row, box, data, train_leaf, leaf, sds, standardize) {
  changes <- list()
  cost <- 0
  for (v in names(box)) {
    b <- box[[v]]
    if (b$type == "numeric") {
      x <- as.numeric(row[[v]])
      lo <- b$lower; up <- b$upper
      eps <- (if (is.finite(up) && is.finite(lo)) (up - lo) else 1) * 1e-6 + 1e-8
      if (!is.na(x) && x > lo && x <= up) next            # already inside
      new <- if (!is.na(x) && x <= lo) lo + eps else up    # nearest boundary
      d <- new - x
      sc <- if (standardize) abs(d) / sds[[v]] else abs(d)
      cost <- cost + sc
      changes[[length(changes) + 1L]] <- data.frame(
        variable = v, from = as.character(round(x, 6)), to = as.character(round(new, 6)),
        delta = d, type = "numeric", stringsAsFactors = FALSE)
    } else {
      levs    <- levels(factor(data[[v]]))
      allowed <- levs
      for (s in b$inc) allowed <- intersect(allowed, s)
      for (s in b$exc) allowed <- setdiff(allowed, s)
      x <- as.character(row[[v]])
      if (length(allowed) == 0L || x %in% allowed) next    # inside / infeasible
      # representative admissible level: most frequent among target-leaf members
      mem <- which(train_leaf == leaf)
      tab <- table(factor(data[mem, v], levels = allowed))
      new <- if (sum(tab) > 0) names(tab)[which.max(tab)] else allowed[1]
      cost <- cost + 1
      changes[[length(changes) + 1L]] <- data.frame(
        variable = v, from = x, to = new, delta = NA_real_, type = "categorical",
        stringsAsFactors = FALSE)
    }
  }
  changes_df <- if (length(changes)) do.call(rbind, changes) else .cf_empty_changes()
  list(cost = cost, changes = changes_df)
}

# Coerce a replacement value back to the column's type.
.cf_coerce <- function(col, value) {
  if (is.factor(col)) factor(value, levels = levels(col))
  else if (is.numeric(col)) as.numeric(value)
  else value
}

# Mean leaf co-occurrence proximity of a query leaf-vector to a set of training
# members (columns = trees).
.cf_prox <- function(lv_query, leaves_train, members) {
  if (length(members) == 0L) return(NA_real_)
  Lm <- leaves_train[members, , drop = FALSE]
  mean(rowMeans(sweep(Lm, 2, lv_query, FUN = "==")))
}


# ===========================================================================
# PRINT METHOD
# ===========================================================================

#' @method print e2counterfactual
#' @export
print.e2counterfactual <- function(x, obs = NULL, digits = 3, ...) {
  n_obs <- length(x$source_leaf)
  if (is.null(obs)) obs <- seq_len(min(n_obs, 5L))
  obs <- intersect(obs, seq_len(n_obs))

  cat("\n")
  cat("  E2Tree Contrastive Explanation (proximity-verified counterfactual)\n")
  cat("  ------------------------------------------------------------------\n")
  cat(sprintf("  %d instance(s)  |  Task: %s\n\n",
              n_obs, if (x$is_class) "Classification" else "Regression"))

  for (i in obs) {
    if (is.na(x$target_leaf[i])) {
      cat(sprintf("  Instance %d  ->  region %d (%s): no differing region available.\n\n",
                  i, x$source_leaf[i], as.character(x$source_pred[i])))
      next
    }
    cat(sprintf("  Instance %d:  region %d (%s)  =>  region %d (%s)   [cost %.*f]\n",
                i, x$source_leaf[i], as.character(x$source_pred[i]),
                x$target_leaf[i], as.character(x$target_pred[i]), digits, x$cost[i]))
    ch <- x$changes[[i]]
    if (nrow(ch) > 0) {
      for (r in seq_len(nrow(ch)))
        cat(sprintf("      %-20s %s -> %s\n", ch$variable[r], ch$from[r], ch$to[r]))
    } else {
      cat("      (already satisfies the target region's rules)\n")
    }
    if (!is.null(x$validated)) {
      cat(sprintf("      ensemble check: prox(target) = %.*f vs prox(source) = %.*f  ->  %s\n",
                  digits, x$prox_to_target[i], digits, x$prox_to_source[i],
                  if (isTRUE(x$validated[i])) "VALIDATED" else "not validated"))
    }
    cat("\n")
  }
  if (n_obs > length(obs))
    cat(sprintf("  ... %d more (use print(x, obs = ...))\n", n_obs - length(obs)))
  invisible(x)
}


# ===========================================================================
# PLOT METHOD -- standardised change per variable for one instance
# ===========================================================================

#' @method plot e2counterfactual
#' @export
plot.e2counterfactual <- function(x, obs = 1L, type = NULL, ...) {
  if (any(obs < 1L) || any(obs > length(x$source_leaf)))
    stop(sprintf("'obs' must be between 1 and %d.", length(x$source_leaf)))

  # Default view: change bars for one instance, a heatmap for many.
  if (is.null(type)) type <- if (length(obs) == 1L) "bars" else "heatmap"
  type <- match.arg(type, c("bars", "heatmap", "frequency", "beeswarm"))
  switch(type,
    bars      = .e2counterfactual_bars(x, obs[1]),
    heatmap   = .e2counterfactual_heatmap(x, obs),
    frequency = .e2counterfactual_frequency(x, obs),
    beeswarm  = .e2counterfactual_beeswarm(x, obs))
}

# Standardised change per variable for a single instance.
.e2counterfactual_bars <- function(x, obs) {
  ch <- x$changes[[obs]]
  if (is.null(ch) || nrow(ch) == 0) {
    message("Instance ", obs, " has no required changes (or no target region).")
    return(invisible(NULL))
  }
  pd <- data.frame(
    variable  = ch$variable,
    delta_std = ifelse(ch$type == "numeric", abs(ch$delta), 1),
    dir_fill  = ifelse(ch$type == "numeric" & ch$delta >= 0, "increase",
                ifelse(ch$type == "numeric", "decrease", "switch")),
    stringsAsFactors = FALSE)
  pd <- pd[order(pd$delta_std), ]
  pd$variable <- factor(pd$variable, levels = pd$variable)

  ggplot2::ggplot(pd, ggplot2::aes(x = variable, y = delta_std, fill = dir_fill)) +
    ggplot2::geom_col(width = 0.65) +
    ggplot2::scale_fill_manual(values = c(increase = "tomato", decrease = "steelblue",
                                          switch = "darkorchid"), name = NULL) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title    = sprintf("Counterfactual change - instance %d", obs),
      subtitle = sprintf("region %s -> region %s%s",
                         as.character(x$source_pred[obs]),
                         as.character(x$target_pred[obs]),
                         if (!is.null(x$validated))
                           sprintf("  (%s)", if (isTRUE(x$validated[obs]))
                             "ensemble-validated" else "not validated") else ""),
      x = NULL, y = "Change magnitude (numeric: |delta|/sd; categorical: switch)") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle = ggplot2::element_text(colour = "grey40", size = 10))
}

# Collect the per-instance change records over a set of observations.
.cf_collect <- function(x, obs, value = TRUE) {
  do.call(rbind, lapply(obs, function(i) {
    d <- x$changes[[i]]
    if (is.null(d) || nrow(d) == 0L) return(NULL)
    out <- data.frame(variable = d$variable, stringsAsFactors = FALSE)
    if (value) out$value <- ifelse(d$type == "numeric", d$delta, 1)
    out$dir <- ifelse(d$type != "numeric", "switch",
               ifelse(d$delta >= 0, "increase", "decrease"))
    out
  }))
}

# How often each feature is the lever, split by direction of change.
.e2counterfactual_frequency <- function(x, obs) {
  recs <- .cf_collect(x, obs, value = FALSE)
  if (is.null(recs)) {
    message("None of the selected observations require any change.")
    return(invisible(NULL))
  }
  n_chg <- sum(vapply(obs, function(i) {
    d <- x$changes[[i]]; !is.null(d) && nrow(d) > 0 }, logical(1)))
  recs$variable <- factor(recs$variable, levels = names(sort(table(recs$variable))))
  recs$dir <- factor(recs$dir, levels = c("decrease", "increase", "switch"))
  ggplot2::ggplot(recs, ggplot2::aes(y = variable, fill = dir)) +
    ggplot2::geom_bar(width = 0.7) +
    ggplot2::scale_fill_manual(values = c(decrease = "steelblue", increase = "tomato",
                                          switch = "darkorchid"), name = NULL, drop = FALSE) +
    ggplot2::labs(title = "How often each feature is the lever",
                  subtitle = sprintf("across %d aspirates requiring a change", n_chg),
                  x = "number of aspirates", y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle = ggplot2::element_text(colour = "grey40", size = 10),
      panel.grid.minor = ggplot2::element_blank())
}

# Distribution of the signed change magnitude per feature across observations.
.e2counterfactual_beeswarm <- function(x, obs) {
  recs <- .cf_collect(x, obs, value = TRUE)
  if (is.null(recs)) {
    message("None of the selected observations require any change.")
    return(invisible(NULL))
  }
  recs$variable <- factor(recs$variable, levels = names(sort(table(recs$variable))))
  lim <- max(stats::quantile(abs(recs$value), 0.95, names = FALSE), 1e-6)
  recs$value <- pmax(pmin(recs$value, lim), -lim)
  ggplot2::ggplot(recs, ggplot2::aes(x = value, y = variable, colour = value)) +
    ggplot2::geom_vline(xintercept = 0, colour = "grey70") +
    ggplot2::geom_jitter(width = 0, height = 0.2, alpha = 0.6,
                         size = if (length(obs) > 60) 1.0 else 1.6) +
    ggplot2::scale_colour_gradient2(low = "steelblue", mid = "white", high = "tomato",
                                    midpoint = 0, limits = c(-lim, lim),
                                    name = "change\n(signed)") +
    ggplot2::labs(title = "Counterfactual change distribution per feature",
                  subtitle = "signed change required, by feature (95th-percentile colour cap)",
                  x = "change (signed)", y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 13),
      plot.subtitle = ggplot2::element_text(colour = "grey40", size = 10),
      panel.grid.minor = ggplot2::element_blank())
}

# Compact counterfactual overview for many observations at once: a heatmap with
# observations on the x-axis (grouped by source -> target region) and features on
# the y-axis. The fill encodes the signed change required on each feature; blank
# cells mark features that need no change for that observation. Categorical
# switches are shown with unit magnitude in the sign of nothing (set to +1).
.e2counterfactual_heatmap <- function(x, obs) {
  recs <- do.call(rbind, lapply(obs, function(i) {
    d <- x$changes[[i]]
    if (is.null(d) || nrow(d) == 0L) return(NULL)
    data.frame(
      instance = i,
      variable = d$variable,
      value    = ifelse(d$type == "numeric", d$delta, 1),
      stringsAsFactors = FALSE)
  }))
  if (is.null(recs) || nrow(recs) == 0L) {
    message("None of the selected observations require any change.")
    return(invisible(NULL))
  }
  obs_ord <- obs[order(as.character(x$source_pred[obs]),
                       as.character(x$target_pred[obs]))]
  recs$instance <- factor(as.character(recs$instance),
                          levels = as.character(obs_ord))
  vars <- sort(unique(recs$variable))
  recs$variable <- factor(recs$variable, levels = vars)
  # Robust colour limit: a few large-cost outliers would otherwise wash out the
  # typical one- or two-step levers. Cap at the 95th percentile of |change| so
  # larger moves saturate the scale (clamp the data to avoid a scales dependency).
  lim  <- max(stats::quantile(abs(recs$value), 0.95, names = FALSE), 1e-6)
  recs$value <- pmax(pmin(recs$value, lim), -lim)
  many <- length(obs_ord) > 40L
  ggplot2::ggplot(recs, ggplot2::aes(x = instance, y = variable, fill = value)) +
    ggplot2::geom_tile(colour = if (many) NA else "grey92") +
    ggplot2::scale_fill_gradient2(low = "steelblue", mid = "white",
                                  high = "tomato", midpoint = 0,
                                  limits = c(-lim, lim), name = "Change\n(signed)") +
    ggplot2::labs(
      title    = "Counterfactual changes across observations",
      subtitle = sprintf("%d instances, ordered by source -> target region; blank = unchanged",
                         length(obs_ord)),
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
